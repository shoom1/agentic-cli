"""Turns on a real console, over a pty: Ctrl+C, a question, typing ahead,
``/new``, ``/resume``, ``/compact`` and the status bar's ``ctx``.

thinking-prompt 0.4's Ctrl+C cancels the turn's task, its dialogs are
``Dialog`` objects, drawn in a box or inline, and it refuses Enter while a
turn runs. What the user sees of each only shows on a real terminal, so these
run
``tests/cli/console_app.py`` (``BaseCLIApp`` over a scripted workflow, no
provider, no key) in a child process and type at it. Every wait is an
``expect`` on output, never a sleep.
"""

from __future__ import annotations

import io
import json
import os
import re
import select
import sys
from pathlib import Path

import pytest

from tests.examples.console_smoke import (
    STARTUP_TIMEOUT,
    STEP_TIMEOUT,
    TERM_SIZE,
    child_env,
    platform_skip_reason,
    strip_ansi,
)

_SKIP = platform_skip_reason()
pytestmark = pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")

CONSOLE_APP = Path(__file__).with_name("console_app.py")


class Console:
    """The child app on a pty, and what it printed.

    ``env`` is the child's whole environment, ``child_env(home)`` when
    omitted; a test that needs one more variable starts from ``child_env``.
    ``args`` are the app's command-line arguments; ``prompt`` False skips
    waiting for the prompt, for a start that never draws one. ``stderr``
    sends the child's stderr elsewhere than the screen: to that file, or,
    with ``"terminal"``, to a second terminal (``stderr_text``), so what
    reaches the screen through the UI is told apart from what was written
    straight to stderr.
    """

    def __init__(
        self, cwd: Path, home: Path, env: dict[str, str] | None = None,
        *, args: tuple[str, ...] = (), prompt: bool = True, stderr: Path | str | None = None,
    ) -> None:
        import pexpect

        self.cwd = cwd
        self._log = io.StringIO()
        self._stderr_file = stderr if isinstance(stderr, Path) else None
        self._stderr_terminal: tuple[int, int] | None = None
        self._stderr_read = ""
        command = [sys.executable, str(CONSOLE_APP), *args]
        if stderr is not None:
            if stderr == "terminal":
                self._stderr_terminal = os.openpty()
                stderr = os.ttyname(self._stderr_terminal[1])
            command = ["/bin/sh", "-c", 'exec "$@" 2>"$0"', str(stderr), *command]
        self.child = pexpect.spawn(
            command[0],
            command[1:],
            cwd=str(cwd),
            env=child_env(home) if env is None else env,
            encoding="utf-8",
            codec_errors="replace",
            timeout=STEP_TIMEOUT,
            dimensions=TERM_SIZE,
        )
        self.child.logfile_read = self._log
        if prompt:
            self.expect(">>>", timeout=STARTUP_TIMEOUT)

    def expect(self, text: str, timeout: float = STEP_TIMEOUT) -> None:
        """Wait for ``text`` (literally) in the output."""
        self.child.expect_exact(text, timeout=timeout)

    def send(self, keys: str) -> None:
        self.child.send(keys)

    def type_line(self, text: str) -> None:
        self.child.sendline(text)

    @property
    def transcript(self) -> str:
        """Everything printed so far, as a human reads it."""
        return strip_ansi(self._log.getvalue())

    def stderr_text(self) -> str:
        """What the child wrote to its stderr, when that is not the screen."""
        if self._stderr_file is not None:
            return self._stderr_file.read_text() if self._stderr_file.exists() else ""
        if self._stderr_terminal is not None:
            master = self._stderr_terminal[0]
            while select.select([master], [], [], 0)[0]:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                self._stderr_read += chunk.decode("utf-8", "replace")
            return strip_ansi(self._stderr_read)
        return ""

    def close(self) -> None:
        self.child.close(force=True)
        if self._stderr_terminal is not None:
            for fd in self._stderr_terminal:
                os.close(fd)
            self._stderr_terminal = None

    def exit(self) -> None:
        """``/exit``, and the process ends cleanly."""
        import pexpect

        self.type_line("/exit")
        self.expect("Goodbye!")
        self.child.expect(pexpect.EOF, timeout=STEP_TIMEOUT)
        self.child.close()
        assert "Traceback (most recent call last)" not in self.transcript
        assert "Traceback (most recent call last)" not in self.stderr_text()
        assert (self.child.exitstatus, self.child.signalstatus) == (0, None)


@pytest.fixture
def console(tmp_path):
    home, cwd = tmp_path / "home", tmp_path / "cwd"
    home.mkdir()
    cwd.mkdir()
    console = Console(cwd, home)
    yield console
    console.child.close(force=True)


def test_ctrl_c_mid_turn_says_so_once_and_the_next_turn_runs(console):
    console.type_line("slow")
    console.expect("Working on it")

    console.child.sendcontrol("c")
    console.expect("Operation cancelled...")
    console.type_line("hello")
    console.expect("Echo: hello")
    console.exit()

    transcript = console.transcript
    assert transcript.count("Operation cancelled...") == 1
    assert "Cancelled." not in transcript
    assert "Handler error" not in transcript


def test_ctrl_c_while_a_turn_waits_for_the_workflow_says_so_once(tmp_path):
    """The workflow failed to start; the next turn starts it again and waits,
    under "Waiting for initialization...", when the user presses Ctrl+C."""
    home, cwd = tmp_path / "home", tmp_path / "cwd"
    home.mkdir()
    cwd.mkdir()
    (cwd / "init-fails").touch()
    console = Console(cwd, home)  # waits for the prompt, after the failure
    try:
        assert "Initialization failed" in console.transcript
        console.type_line("hello")
        console.expect("Waiting for initialization...")

        console.child.sendcontrol("c")
        console.expect("Operation cancelled...")
        (cwd / "ready").touch()
        console.type_line("again")
        console.expect("Echo: again")
        console.exit()
    finally:
        console.child.close(force=True)

    transcript = console.transcript
    assert transcript.count("Operation cancelled...") == 1
    assert "Echo: hello" not in transcript


def test_a_free_text_question_is_answered(console):
    console.type_line("ask")
    console.expect("ask_clarification - Input Required")
    console.expect("What colour?")

    console.send("\r")  # edit the answer
    console.send("teal")
    console.expect("teal")
    console.send("\r")  # keep it
    console.send("\t")  # on to OK
    console.send("\r")  # press it
    console.expect("You said: teal")
    console.exit()


def test_a_free_text_question_is_answered_inline(tmp_path):
    """With ``dialog_placement`` inline (the user's environment says so), the
    question is rows under the prompt, and the same keys answer it."""
    home, cwd = tmp_path / "home", tmp_path / "cwd"
    home.mkdir()
    cwd.mkdir()
    env = {**child_env(home), "AGENTIC_DIALOG_PLACEMENT": "inline"}
    console = Console(cwd, home, env=env)
    try:
        console.type_line("ask")
        console.expect("ask_clarification - Input Required")
        console.expect("What colour?")
        console.expect("1. OK")  # an inline dialog's buttons are numbered rows

        console.send("\r")  # edit the answer
        console.send("teal")
        console.expect("teal")
        console.send("\r")  # keep it
        console.send("\t")  # on to OK
        console.send("\r")  # press it
        console.expect("You said: teal")
        console.exit()
    finally:
        console.child.close(force=True)

    assert "<    OK    >" not in console.transcript  # a box's button


def test_a_line_typed_during_a_turn_waits_in_the_prompt(console):
    """0.4 refuses Enter while a turn runs and says it is busy; the line stays
    in the prompt, to send once the turn is over."""
    console.type_line("hold")
    console.expect("Holding.")

    console.send("draft")
    console.send("\r")
    console.expect("Busy")
    (console.cwd / "release").touch()
    console.expect("Released.")
    console.send("\r")
    console.expect("Echo: draft")
    console.exit()

    transcript = console.transcript
    assert transcript.index("Released.") < transcript.index("Echo: draft")


def test_new_starts_a_conversation_and_the_next_turn_runs_in_it(console):
    console.type_line("which")
    console.expect("Session: ")
    console.type_line("/new")
    console.expect("listed in /sessions.")
    (new, previous), = re.findall(
        r"New conversation ([0-9a-f]{12})\. Previous: ([0-9a-f]{12}), listed in /sessions\.",
        console.transcript,
    )
    console.type_line("which")
    console.expect(f"Session: {new}")
    console.exit()

    sessions = re.findall(r"Session: ([0-9a-f]{12})", console.transcript)
    assert sessions[0] == previous != new
    assert sessions[-1] == new


def test_resume_picks_a_conversation_and_shows_where_it_left_off(console):
    """The picker lists the other conversation; Enter opens it, its last
    messages are shown again, and the next turn runs in it."""
    console.type_line("Plan the trip")
    console.expect("Echo: Plan the trip")
    console.type_line("/new")
    console.expect("New conversation ")
    console.type_line("/resume")
    console.expect("Resume a conversation")
    console.expect("· Plan the trip · ")
    console.send("\r")
    console.expect("Resumed: Plan the trip (")
    console.expect("Echo: Plan the trip")
    (resumed,) = re.findall(r"Resumed: Plan the trip \(([0-9a-f]{12}), 1 turn,", console.transcript)
    console.type_line("which")
    console.expect(f"Session: {resumed}")
    console.exit()

    (new,) = re.findall(r"New conversation ([0-9a-f]{12})\.", console.transcript)
    assert resumed != new


def test_resume_of_a_conversation_that_is_not_stored_exits_before_the_prompt(tmp_path):
    import pexpect

    (tmp_path / "home").mkdir()
    console = Console(tmp_path, tmp_path / "home", args=("--resume", "zzz"), prompt=False)
    try:
        # The console app keeps its conversations in memory: nothing to open instead.
        console.expect("No conversation 'zzz' found. "
                       "(session_store is memory: conversations end with the app)",
                       timeout=STARTUP_TIMEOUT)
        console.child.expect(pexpect.EOF, timeout=STEP_TIMEOUT)
        console.child.close()
    finally:
        console.child.close(force=True)

    assert console.child.exitstatus == 1
    assert ">>>" not in console.transcript
    assert "Traceback (most recent call last)" not in console.transcript


def _seed(cwd: Path) -> None:
    """A conversation stored by an earlier run (``console_app.py``'s seed.json)."""
    (cwd / "seed.json").write_text(json.dumps({
        "abc123def456": {"messages": [["user", "Plan the trip"], ["assistant", "Sure."]],
                         "hours": 2},
    }))


def test_continue_resumes_the_newest_once_the_prompt_is_up(tmp_path):
    (tmp_path / "home").mkdir()
    _seed(tmp_path)
    console = Console(tmp_path, tmp_path / "home", args=("--continue",))
    try:
        console.expect("Resumed: Plan the trip (abc123def456, 1 turn, last active 2h ago).")
        console.expect("Sure.")
        console.type_line("which")
        console.expect("Session: abc123def456")
        console.exit()
    finally:
        console.child.close(force=True)

    transcript = console.transcript
    assert transcript.index(">>>") < transcript.index("Resumed:")


def test_continue_with_nothing_stored_starts_a_new_conversation(tmp_path):
    (tmp_path / "home").mkdir()
    console = Console(tmp_path, tmp_path / "home", args=("--continue",))
    try:
        console.type_line("which")
        console.expect("Session: ")
        console.exit()
    finally:
        console.child.close(force=True)

    assert "No conversation to continue; starting a new one." in console.transcript


def test_compact_shows_the_summarizing_line_then_what_it_did(console):
    console.type_line("/compact")
    console.expect("Summarizing the earlier conversation…")
    (console.cwd / "summarized").touch()
    console.expect("Context summarized: 2 messages, 7.4k → 1.2k tokens")
    console.exit()

    transcript = console.transcript
    assert transcript.index("Summarizing the earlier conversation…") < transcript.index(
        "Context summarized:"
    )
    assert "Nothing to summarize yet." not in transcript


def test_ctrl_c_during_compact_says_so_once_and_the_next_turn_runs(console):
    console.type_line("/compact")
    console.expect("Summarizing the earlier conversation…")

    console.child.sendcontrol("c")
    console.expect("Operation cancelled...")
    console.type_line("hello")
    console.expect("Echo: hello")
    console.exit()

    transcript = console.transcript
    assert transcript.count("Operation cancelled...") == 1
    for never in ("Context summarized", "Could not summarize", "Nothing to summarize yet.",
                  "Handler error"):
        assert never not in transcript


#: What the console app's ``warn`` turn logs and warns.
WARNED = (
    "console_warning_on_the_loop",
    "console_warning_in_a_thread",
    "console_stdlib_warning",
    "console_python_warning",
)


def _warn_turn(tmp_path: Path, stderr: Path | str) -> Console:
    """The ``warn`` turn, then ``/exit``, with the child's stderr at ``stderr``."""
    home, cwd = tmp_path / "home", tmp_path / "cwd"
    home.mkdir()
    cwd.mkdir()
    console = Console(cwd, home, stderr=stderr)
    try:
        console.type_line("warn")
        console.expect("Warned.")
        console.exit()
    finally:
        console.stderr_text()  # read before the terminal is closed
        console.close()
    return console


def test_log_lines_and_warnings_during_a_turn_print_through_the_ui(tmp_path):
    """With stderr on the terminal, a warning logged mid-turn, from the loop
    or any thread, and a Python warning, print above the prompt as the UI's
    own output does: written straight to stderr, they landed on the prompt
    line and broke the screen. (stderr is a second terminal here, so a line
    written to it is told apart from one the UI drew.)"""
    console = _warn_turn(tmp_path, "terminal")

    assert [line for line in WARNED if line not in console.transcript] == []
    assert [line for line in WARNED if line in console.stderr_text()] == []


def test_with_stderr_redirected_log_lines_stay_in_the_file(tmp_path):
    """``app 2>run.log`` keeps its log in the file, and off the screen."""
    console = _warn_turn(tmp_path, tmp_path / "run.log")

    assert [line for line in WARNED if line not in console.stderr_text()] == []
    assert [line for line in WARNED if line in console.transcript] == []


def test_the_status_bar_shows_how_full_the_context_is(console):
    console.type_line("hello")
    console.expect("Echo: hello")
    console.expect("ctx 4k/16k (25%)")
    console.exit()
