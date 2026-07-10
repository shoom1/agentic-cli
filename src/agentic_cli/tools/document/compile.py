"""Compile a LaTeX document to PDF with a host TeX engine.

``compile_document`` is a narrow, permission-gated tool: it runs ``latexmk``
(preferred) or ``pdflatex`` as a guarded subprocess — shell-escape is disabled
for the **pdflatex** path (``-no-shell-escape`` flag).  For ``latexmk``, the
engine relies on its default restricted mode; note that ``latexmk`` also reads
``.latexmkrc`` (arbitrary Perl) from the build directory and home directory, so
callers with an untrusted ``.tex``/build directory should prefer ``pdflatex``.
OS-sandbox confinement for the general case is deferred.

The tool runs with a wall-clock timeout and a scoped working dir, and returns a
structured result.  It does NOT execute arbitrary code; a ``report_writer``-style
agent uses it to turn an authored ``.tex`` into a PDF.

The subprocess receives only an allowlisted environment (``_ENV_PASSTHROUGH``),
not the full host environment, so host secrets aren't handed to the TeX process.
Delivery to ``output_pdf`` is a no-follow atomic write (temp file + ``os.replace``)
so an attacker-placed symlink at the destination can't redirect the write.

Provisioning is host-based: the engine must be on ``PATH`` (TeX Live / MacTeX).
If neither is found the tool returns a structured error with an install hint.

The subprocess call and the engine lookup sit behind module-level seams
(``_run``, ``_which``) so the logic is unit-tested offline without a real TeX
install.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from agentic_cli.tools.registry import ToolCategory, register_tool
from agentic_cli.workflow.permissions import Capability

_ENGINES = ("latexmk", "pdflatex")
_LOG_TAIL_LINES = 40

# Only these host env vars reach the TeX process. The tool must not hand the
# whole host environment (API keys, tokens) to a subprocess that — on the
# latexmk path — can execute arbitrary Perl from a .latexmkrc. PATH/HOME are
# needed for the engine binary and kpathsea; TEXINPUTS is set explicitly.
_ENV_PASSTHROUGH = (
    "PATH", "HOME", "TERM", "TMPDIR", "TEMP", "TMP",
    "LANG", "LC_ALL", "LC_CTYPE", "SOURCE_DATE_EPOCH",
)


def _build_env(assets_dir: str | None) -> dict[str, str]:
    """Minimal, allowlisted environment for the TeX subprocess."""
    env = {k: os.environ[k] for k in _ENV_PASSTHROUGH if k in os.environ}
    env.setdefault("PATH", os.defpath)
    if assets_dir:
        # Prepend assets_dir; the trailing empty entries let kpathsea append the
        # default search path. A caller-inherited TEXINPUTS is intentionally
        # dropped (not in the allowlist) so it can't redirect input resolution.
        env["TEXINPUTS"] = f"{assets_dir}{os.pathsep}{os.pathsep}"
    return env


def _deliver_no_follow(produced: Path, dest: Path) -> None:
    """Copy ``produced`` to ``dest`` without following a symlink at ``dest``.

    Writes to a private temp file in dest's directory, then atomically renames
    it over dest. ``os.replace`` swaps the destination *name*: if dest is a
    symlink the link itself is replaced (not written through), so an
    attacker-placed symlink can't redirect the write outside the intended path.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(dest.parent), prefix=f".{dest.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, open(produced, "rb") as src:
            shutil.copyfileobj(src, out)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp_path, dest)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _which(name: str) -> str | None:
    """Locate an executable on PATH (seam for tests)."""
    return shutil.which(name)


def _run(
    argv: list[str], *, cwd: str, env: dict[str, str], timeout: float
) -> subprocess.CompletedProcess:
    """Run a subprocess in its own process group so a timeout kills the whole
    tree (latexmk + its pdflatex grandchild), not just the direct child. Seam
    for tests. POSIX (macOS/Linux), which is what the framework targets."""
    proc = subprocess.Popen(
        argv, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.communicate()  # reap the killed group
        raise
    return subprocess.CompletedProcess(argv, proc.returncode, stdout=out, stderr=err)


def _detect_engine(engine: str | None) -> str | None:
    """Return the engine to use, or None if unavailable."""
    if engine is not None:
        return engine if (engine in _ENGINES and _which(engine)) else None
    for candidate in _ENGINES:
        if _which(candidate):
            return candidate
    return None


def _build_argv(engine: str, source: str) -> list[str]:
    """Compiler argv — never enables shell-escape."""
    if engine == "latexmk":
        return ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", source]
    return [
        "pdflatex", "-no-shell-escape", "-interaction=nonstopmode",
        "-halt-on-error", source,
    ]


def _parse_errors(log_text: str) -> list[str]:
    """Extract LaTeX error lines (those beginning with '!') from a log."""
    return [ln for ln in log_text.splitlines() if ln.startswith("!")]


@register_tool(
    category=ToolCategory.EXECUTION,
    capabilities=[
        Capability("document.compile", target_arg="source_path"),
        # The tool also reads assets_dir and writes output_pdf when those are
        # supplied; scope them explicitly (optional → not exercised when absent).
        Capability("filesystem.read", target_arg="assets_dir", optional=True),
        Capability("filesystem.write", target_arg="output_pdf", optional=True),
    ],
    description=(
        "Compile a LaTeX source file to PDF using a host TeX engine (latexmk or "
        "pdflatex). Shell-escape is disabled for pdflatex (-no-shell-escape); "
        "latexmk uses its default restricted mode but also reads .latexmkrc from "
        "the build/home directory. Returns the PDF path plus any compiler errors. "
        "Requires TeX Live/MacTeX on PATH."
    ),
)
def compile_document(
    source_path: str,
    output_pdf: str | None = None,
    assets_dir: str | None = None,
    engine: str | None = None,
    timeout_s: int = 120,
) -> dict[str, Any]:
    """Compile a LaTeX file to PDF (guarded subprocess; host TeX engine).

    Args:
        source_path: Path to the .tex file to compile.
        output_pdf: If set, the produced PDF is copied here (parents created);
            build intermediates stay in the source's directory.
        assets_dir: Directory prepended to TEXINPUTS so figures/resources resolve
            by bare name (e.g. an artifacts dir).
        engine: Force an engine ("latexmk"/"pdflatex"); default auto-detects
            (latexmk preferred).
        timeout_s: Wall-clock timeout; the process is killed on expiry.

    Returns:
        dict with success, pdf_path, engine, log_tail, errors, duration_ms, and
        (on setup/timeout failure) error.
    """
    src = Path(source_path)
    if not src.is_file():
        return {
            "success": False, "error": f"Source not found: {source_path}",
            "pdf_path": None, "engine": None, "log_tail": "", "errors": [],
            "duration_ms": 0,
        }

    chosen = _detect_engine(engine)
    if chosen is None:
        looked = engine or "/".join(_ENGINES)
        return {
            "success": False,
            "error": (
                f"No LaTeX engine on PATH (looked for {looked}). "
                "Install TeX Live or MacTeX."
            ),
            "pdf_path": None, "engine": None, "log_tail": "", "errors": [],
            "duration_ms": 0,
        }

    work_dir = src.parent
    env = _build_env(assets_dir)

    argv = _build_argv(chosen, src.name)
    start = time.monotonic()
    try:
        proc = _run(argv, cwd=str(work_dir), env=env, timeout=float(timeout_s))
    except subprocess.TimeoutExpired:
        return {
            "success": False, "error": f"Compilation timed out after {timeout_s}s",
            "pdf_path": None, "engine": chosen, "log_tail": "", "errors": [],
            "duration_ms": int((time.monotonic() - start) * 1000),
        }
    except OSError as exc:
        return {
            "success": False, "error": f"Failed to run {chosen}: {exc}",
            "pdf_path": None, "engine": chosen, "log_tail": "", "errors": [],
            "duration_ms": int((time.monotonic() - start) * 1000),
        }
    duration_ms = int((time.monotonic() - start) * 1000)

    log_path = work_dir / (src.stem + ".log")
    try:
        log_text = log_path.read_text(errors="replace") if log_path.is_file() else (proc.stdout or "")
    except OSError:
        log_text = proc.stdout or ""
    log_tail = "\n".join(log_text.splitlines()[-_LOG_TAIL_LINES:])
    produced = work_dir / (src.stem + ".pdf")
    success = proc.returncode == 0 and produced.is_file()

    if not success:
        return {
            "success": False, "pdf_path": None, "engine": chosen,
            "log_tail": log_tail, "errors": _parse_errors(log_text),
            "duration_ms": duration_ms, "error": None,
        }

    final = produced
    if output_pdf:
        dest = Path(output_pdf)
        try:
            _deliver_no_follow(produced, dest)
        except OSError as exc:
            return {
                "success": False,
                "error": f"Failed to deliver PDF to {output_pdf}: {exc}",
                "pdf_path": str(produced), "engine": chosen,
                "log_tail": log_tail, "errors": [], "duration_ms": duration_ms,
            }
        final = dest

    return {
        "success": True, "pdf_path": str(final), "engine": chosen,
        "log_tail": log_tail, "errors": [], "duration_ms": duration_ms,
        "error": None,
    }
