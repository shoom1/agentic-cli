"""While the prompt runs, each log line is added to the session, on the event
loop, whichever thread logged it, so it prints above the prompt; written to
stderr, it landed wherever the cursor was and broke the screen. The pty test
``test_console_turns.py::test_log_lines_and_warnings_during_a_turn_print_through_the_ui``
checks the same on a real console."""

from __future__ import annotations

import asyncio
import threading

import structlog
from rich.text import Text

from agentic_cli.cli.app import _log_lines_above_prompt
from tests.cli.dialog_fakes import autospec_session


async def _settle() -> None:
    """Let the callbacks other threads scheduled on the loop run."""
    for _ in range(10):
        await asyncio.sleep(0)


async def test_a_log_line_from_any_thread_is_added_to_the_session_on_the_loop(
    logging_configured,
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


async def test_once_the_prompt_ends_log_lines_go_to_stderr_again(logging_configured, capsys):
    session = autospec_session()
    with _log_lines_above_prompt(session):
        pass

    structlog.get_logger("t").warning("after_the_prompt")
    await _settle()

    session.add_rich.assert_not_called()
    assert "after_the_prompt" in capsys.readouterr().err
