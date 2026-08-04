"""Shared driver for the research-demo console smoke tests.

Plumbing only — no test classes, so pytest does not collect it. Used by both
``test_research_demo_pty.py`` (source checkout, ``python -m research_demo``)
and ``test_research_demo_wheel.py`` (built wheel, the installed
``research-demo`` console script), so the two exercise *the same* interaction
against different installations and different entry points.

The demo is a full-screen prompt_toolkit application, so it needs a real
terminal: it is driven over a pty with ``pexpect`` and its output is
ANSI-stripped before matching. Matching is on short, stable fragments rather
than a snapshot of raw terminal bytes — the screen is redrawn repeatedly and
line-wrapped to the terminal width, so raw output is not stable.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def project_version() -> str:
    """The version declared in the repository's ``pyproject.toml``.

    Derived, never hardcoded: a release-specific literal would pin the
    acceptance suite to one release, going red on every branch that predates a
    version bump and needing an edit at each bump. Reading the declared version
    instead means these assertions validate *whatever* is being released.
    """
    with (_REPO_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)["project"]["version"]

# CSI/OSC escape sequences, charset selects, and bare CRs, so matching sees
# the text a human reads rather than the bytes that drew it.
_ANSI = re.compile(
    r"\x1b\[[0-9;?]*[a-zA-Z]"      # CSI
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC
    r"|\x1b[()][A-Za-z0-9]"        # charset select
    r"|\x1b[=>]"                   # keypad mode
    r"|\r"
)

#: The *only* parent variables allowed through to the child. Everything else —
#: provider credentials, cloud authentication, ``RESEARCH_DEMO_*``/``AGENTIC_*``
#: settings overrides, custom config paths, ``PYTHONPATH`` — is dropped, so the
#: smoke measures the shipped defaults rather than whatever the developer (or
#: CI) happens to export. These are what it takes to *launch* an interpreter and
#: a terminal, nothing about the application.
ALLOWED_PARENT_VARS = (
    "PATH",         # find the interpreter/console script and anything it execs
    "LANG",         # terminal text encoding
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",       # POSIX temp location (macOS gives every user its own)
    "SYSTEMROOT",   # Windows needs these to start a process at all
    "SystemRoot",
    "COMSPEC",
)

#: Set explicitly on every child, never inherited.
EXPLICIT_CHILD_VARS = (
    "HOME",
    "TERM",
    "NO_COLOR",
    "PYTHONUNBUFFERED",
    "PYTHONDONTWRITEBYTECODE",
)

#: Terminal size. Wide enough that the help table's cells do not wrap, which is
#: what makes short-fragment matching reliable.
TERM_SIZE = (40, 200)

#: Every wait is bounded. Startup measured ~1s on a warm checkout; 60s is
#: headroom for a cold import on CI, not an invitation to hang.
STARTUP_TIMEOUT = 60
STEP_TIMEOUT = 30


@dataclass
class SmokeResult:
    """What the console session did, for the caller to assert on."""

    transcript: str
    exit_status: int | None
    signal_status: int | None
    argv: list[str]
    steps: list[str] = field(default_factory=list)

    def saw(self, fragment: str) -> bool:
        return fragment in self.transcript


@dataclass
class ChildImports:
    """Where a child process resolved the packages under test."""

    demo: Path
    framework: Path
    version: str
    executable: Path


def child_env(home: Path) -> dict[str, str]:
    """A minimal, allowlisted environment for the child process.

    Built up from nothing rather than filtered down from ``os.environ``: a
    deny-list only removes the hazards someone thought of, and the ones that
    matter here (``RESEARCH_DEMO_*``, ``AGENTIC_*``, ``GOOGLE_APPLICATION_
    CREDENTIALS``, ``VERTEX_*``) are open-ended.

    ``HOME`` is redirected so the demo's user config, user KB and
    ``~/.research_demo/.env`` all resolve inside the temp dir; ``PYTHONPATH`` is
    absent so the child cannot pick up a checkout the caller did not intend.
    """
    env = {k: os.environ[k] for k in ALLOWED_PARENT_VARS if k in os.environ}
    env.update(
        HOME=str(home),
        TERM="xterm-256color",
        NO_COLOR="1",
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    return env


def probe_child_imports(python: str, cwd: Path, home: Path) -> ChildImports:
    """Ask a child process — same interpreter, cwd and env — what it imports.

    The parent pytest process is *not* evidence: it has the editable install on
    ``sys.path`` and whatever the developer's environment supplies. Only the
    child can say where the child resolves ``research_demo``.
    """
    code = (
        "import json, sys, research_demo, agentic_cli; "
        "print(json.dumps({"
        "'demo': research_demo.__file__, "
        "'framework': agentic_cli.__file__, "
        "'version': agentic_cli.__version__, "
        "'executable': sys.executable}))"
    )
    result = subprocess.run(
        [python, "-c", code],
        cwd=str(cwd),
        env=child_env(home),
        capture_output=True,
        text=True,
        timeout=STEP_TIMEOUT,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"import probe failed ({result.returncode}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    return ChildImports(
        demo=Path(payload["demo"]).resolve(),
        framework=Path(payload["framework"]).resolve(),
        version=payload["version"],
        executable=Path(payload["executable"]).resolve(),
    )


def run_console_smoke(argv: list[str], cwd: Path, home: Path) -> SmokeResult:
    """Drive a console command through startup → /help → /exit.

    Every wait is an ``expect`` on observable output, never a sleep, so the
    test is deterministic rather than timing-dependent.

    Args:
        argv: The command to launch — ``[python, "-m", "research_demo"]`` for
            the source checkout, or ``[".../bin/research-demo"]`` for the
            installed console script.
        cwd: Working directory for the child — keeps ``./.research_demo/``
            (project KB, permission workdir) out of the repo.
        home: Value for ``HOME``.

    Returns:
        The full ANSI-stripped transcript plus the process's exit status.
    """
    import pexpect

    log = io.StringIO()
    child = pexpect.spawn(
        argv[0],
        list(argv[1:]),
        cwd=str(cwd),
        env=child_env(home),
        encoding="utf-8",
        codec_errors="replace",
        timeout=STARTUP_TIMEOUT,
        dimensions=TERM_SIZE,
    )
    child.logfile_read = log
    steps: list[str] = []
    try:
        # 1. Startup: the welcome panel is drawn and the prompt appears.
        child.expect(r">>>", timeout=STARTUP_TIMEOUT)
        steps.append("prompt")

        # 2. /help renders the command table.
        child.sendline("/help")
        child.expect(r"Exit the application", timeout=STEP_TIMEOUT)
        steps.append("help")

        # 3. The prompt comes back, so the app is still interactive.
        child.expect(r">>>", timeout=STEP_TIMEOUT)
        steps.append("prompt-after-help")

        # 4. /exit shuts down cleanly.
        child.sendline("/exit")
        child.expect(r"Goodbye!", timeout=STEP_TIMEOUT)
        steps.append("goodbye")
        child.expect(pexpect.EOF, timeout=STEP_TIMEOUT)
        steps.append("eof")
    finally:
        child.close()

    return SmokeResult(
        transcript=_ANSI.sub("", log.getvalue()),
        exit_status=child.exitstatus,
        signal_status=child.signalstatus,
        argv=list(argv),
        steps=steps,
    )


def assert_clean_session(result: SmokeResult) -> None:
    """Shared assertions: the demo started, responded, and exited cleanly."""
    assert result.steps == [
        "prompt",
        "help",
        "prompt-after-help",
        "goodbye",
        "eof",
    ], f"console session did not complete: reached {result.steps} via {result.argv}"

    assert result.saw("Exit the application"), "the /help table was not rendered"
    assert result.saw("Goodbye!"), "the app did not say goodbye on /exit"

    for marker in ("Traceback (most recent call last)", "Unhandled exception"):
        assert marker not in result.transcript, (
            f"{marker!r} in console output:\n{_tail(result.transcript)}"
        )

    assert result.signal_status is None, (
        f"the console process died on signal {result.signal_status}"
    )
    assert result.exit_status == 0, (
        f"console exited with status {result.exit_status}:\n{_tail(result.transcript)}"
    )


def _tail(text: str, lines: int = 40) -> str:
    return "\n".join(text.splitlines()[-lines:])


def platform_skip_reason() -> str | None:
    """Why this platform cannot run a pty smoke, or None if it can."""
    if sys.platform.startswith("win"):
        return (
            "the console smoke needs a POSIX pty: Python's `pty` module and "
            "`pexpect.spawn` are POSIX-only, and prompt_toolkit's full-screen "
            "app cannot be driven through a pipe"
        )
    return None
