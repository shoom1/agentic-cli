"""While the prompt runs, each log line is added to the session, on the event
loop, whichever thread logged it, so it prints above the prompt; written to
stderr, it landed wherever the cursor was and broke the screen. The pty test
``test_console_turns.py::test_log_lines_and_warnings_during_a_turn_print_through_the_ui``
checks the same on a real console."""

from __future__ import annotations

import asyncio
import threading

import pytest
import structlog
from rich.text import Text

from agentic_cli.cli import app as app_module
from agentic_cli.cli.app import _log_lines_above_prompt
from tests.cli.dialog_fakes import autospec_session


@pytest.fixture
def terminal_stderr(monkeypatch):
    """stderr is the terminal the prompt is drawn on, as in an interactive run."""
    monkeypatch.setattr(app_module, "_stderr_is_terminal", lambda: True, raising=False)


async def _settle() -> None:
    """Let the callbacks other threads scheduled on the loop run."""
    for _ in range(10):
        await asyncio.sleep(0)


async def test_a_log_line_from_any_thread_is_added_to_the_session_on_the_loop(
    logging_configured, terminal_stderr
):
    session = autospec_session()
    added: list[tuple[object, int]] = []
    session.add_rich.side_effect = lambda renderable: added.append(
        (renderable, threading.get_ident())
    )
    log = structlog.get_logger("t")

    with _log_lines_above_prompt(session):
        log.warning("on_the_loop")
        await asyncio.to_thread(log.warning, "in_a_worker_thread")
        # An executor's thread runs without the loop's context.
        thread = threading.Thread(target=log.warning, args=("in_a_plain_thread",))
        thread.start()
        thread.join()
        await _settle()

    assert all(isinstance(renderable, Text) for renderable, _ in added)
    shown = [renderable.plain for renderable, _ in added]
    assert len(shown) == 3
    for event, line in zip(("on_the_loop", "in_a_worker_thread", "in_a_plain_thread"), shown):
        assert event in line
    assert {thread_id for _, thread_id in added} == {threading.get_ident()}


async def test_once_the_prompt_ends_log_lines_go_to_stderr_again(
    logging_configured, terminal_stderr, capsys
):
    session = autospec_session()
    with _log_lines_above_prompt(session):
        pass

    structlog.get_logger("t").warning("after_the_prompt")
    await _settle()

    session.add_rich.assert_not_called()
    assert "after_the_prompt" in capsys.readouterr().err


async def test_a_line_logged_on_the_loop_keeps_its_place_among_the_session_s_output(
    logging_configured, terminal_stderr
):
    """A warning logged just before an error the session shows (a turn's
    rate limit, say) is shown before it."""
    session = autospec_session()

    with _log_lines_above_prompt(session):
        structlog.get_logger("t").warning("logged_first")
        session.add_error("shown second")
        await _settle()

    calls = [call[0] for call in session.mock_calls]
    assert calls == ["add_rich", "add_error"]
    assert "logged_first" in session.add_rich.call_args.args[0].plain


async def test_a_line_logged_while_one_is_shown_goes_to_stderr(
    logging_configured, terminal_stderr, capsys
):
    """Showing a line must not show another forever: one logged meanwhile
    goes to stderr."""
    session = autospec_session()
    session.add_rich.side_effect = lambda renderable: structlog.get_logger("t").warning(
        "logged_while_showing"
    )

    with _log_lines_above_prompt(session):
        structlog.get_logger("t").warning("shown")
        await _settle()

    assert session.add_rich.call_count == 1
    assert "logged_while_showing" in capsys.readouterr().err


async def test_with_stderr_redirected_log_lines_stay_on_it(logging_configured, monkeypatch, capsys):
    """``app 2>run.log`` keeps its log in the file: lines are drawn above the
    prompt only when stderr is the terminal."""
    monkeypatch.setattr(app_module, "_stderr_is_terminal", lambda: False, raising=False)
    session = autospec_session()

    with _log_lines_above_prompt(session):
        structlog.get_logger("t").warning("to_the_file")
        await _settle()

    session.add_rich.assert_not_called()
    assert "to_the_file" in capsys.readouterr().err
