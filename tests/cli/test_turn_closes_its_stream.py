"""A turn closes the workflow's event stream in its own task (``aclosing``).

A manager's ``process`` is an async generator that sets the turn's
ContextVars (the active turn, the service registry, the settings) and resets
them from their tokens as it ends (``BaseWorkflowManager._workflow_context``).
A turn that stopped iterating early (an event handler that raised, Ctrl+C
while a handler waited) left the generator to the event loop's finalizer,
which closes it from another context: the reset failed, "<Token ...> was
created in a different Context", and the loop logged the ValueError. The
turn now closes the generator where it iterated it, so its ``finally`` runs
before the turn returns, in the context its values were set in.

A stream that fails as it closes during Ctrl+C (its own cleanup raising)
does not turn the cancel into a failed turn: the close error is logged at
debug, with its traceback, and the turn ends cancelled, "Operation
cancelled..." said once.
"""

from __future__ import annotations

import asyncio
import gc
from types import SimpleNamespace

import pytest

from agentic_cli.cli import message_processor as message_processor_module
from agentic_cli.cli.message_processor import CANCELLED_LINE, MessageProcessor
from agentic_cli.workflow.events import EventType, WorkflowEvent
from agentic_cli.workflow.sessions import (
    SessionRef,
    get_active_turn,
    reset_active_turn,
    set_active_turn,
)
from tests.cli.dialog_fakes import autospec_session, opening_boxes
from tests.event_replay import ReplayController

SETTINGS = SimpleNamespace(default_user="u", verbose_thinking=False, dialog_placement="box")


class Workflow:
    """Streams two events inside its turn's context, as a manager does.

    ``hang``: after the first event it waits until it is cancelled.
    ``close_fails``: its cleanup raises as it closes.
    """

    def __init__(self, *, hang: bool = False, close_fails: bool = False) -> None:
        self.closed = False
        self.hang, self.close_fails = hang, close_fails
        self.waiting = asyncio.Event()

    def set_input_callback(self, callback) -> None:
        pass

    def clear_input_callback(self) -> None:
        pass

    async def _stream(self):
        token = set_active_turn(SessionRef("app", "u", "s1"))
        try:
            yield WorkflowEvent.text("one")
            if self.hang:
                self.waiting.set()
                await asyncio.Event().wait()
            yield WorkflowEvent.text("two")
        finally:
            reset_active_turn(token)
            self.closed = True
            if self.close_fails:
                raise RuntimeError("cleanup failed")

    def process(self, message, user_id, session_id=None):
        return self._stream()


@pytest.fixture
async def loop_errors():
    """What the event loop's exception handler is given during the test."""
    loop = asyncio.get_running_loop()
    seen: list[dict] = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _, context: seen.append(context))
    yield seen
    loop.set_exception_handler(previous)


async def settle() -> None:
    """Let a generator dropped by its consumer be finalized, if it was."""
    gc.collect()
    for _ in range(5):
        await asyncio.sleep(0)


async def test_a_handler_that_raises_closes_the_stream_in_the_turn(monkeypatch, loop_errors):
    async def broken(*args) -> None:
        raise RuntimeError("a handler bug")

    monkeypatch.setattr(MessageProcessor, "_EVENT_DISPATCH", {EventType.TEXT: broken})
    workflow = Workflow()
    ui = opening_boxes(autospec_session())

    result = await MessageProcessor().process(
        "hi", ReplayController(workflow), ui, SETTINGS, session_id="s1"
    )

    assert not result.delivered
    assert workflow.closed  # before the turn returned
    await settle()
    assert loop_errors == []
    assert get_active_turn() is None


async def test_ctrl_c_while_a_handler_waits_closes_the_stream_in_the_turn(
    monkeypatch, loop_errors
):
    handling = asyncio.Event()

    async def waits(*args) -> None:
        handling.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(MessageProcessor, "_EVENT_DISPATCH", {EventType.TEXT: waits})
    workflow = Workflow()
    ui = opening_boxes(autospec_session())

    turn = asyncio.create_task(MessageProcessor().process(
        "hi", ReplayController(workflow), ui, SETTINGS, session_id="s1"
    ))
    await asyncio.wait_for(handling.wait(), timeout=2)
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    assert workflow.closed
    await settle()
    assert loop_errors == []


@pytest.fixture
def logged(monkeypatch) -> list[dict]:
    """Every log call of ``message_processor``, with its level (recorded on
    the module's logger: ``capture_logs()`` misses a logger cached earlier)."""
    events: list[dict] = []

    class Recorder:
        def __getattr__(self, level: str):
            def log(event: str, **fields) -> None:
                events.append({"event": event, "log_level": level, **fields})

            return log

    monkeypatch.setattr(message_processor_module, "logger", Recorder())
    return events


def said_once_and_no_error(ui) -> bool:
    lines = [c.args for c in ui.add_message.call_args_list if c.args[1:] == (CANCELLED_LINE,)]
    return len(lines) == 1 and ui.add_error.call_count == 0


async def test_a_close_that_fails_while_ctrl_c_stops_a_handler_still_cancels(
    monkeypatch, logged
):
    handling = asyncio.Event()

    async def waits(*args) -> None:
        handling.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(MessageProcessor, "_EVENT_DISPATCH", {EventType.TEXT: waits})
    workflow = Workflow(close_fails=True)
    ui = opening_boxes(autospec_session())

    turn = asyncio.create_task(MessageProcessor().process(
        "hi", ReplayController(workflow), ui, SETTINGS, session_id="s1"
    ))
    await asyncio.wait_for(handling.wait(), timeout=2)
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    assert workflow.closed
    assert said_once_and_no_error(ui)
    (failed,) = [e for e in logged if e["event"] == "turn_stream_close_failed"]
    assert failed["log_level"] == "debug" and failed["exc_info"] is True


async def test_a_stream_that_fails_as_ctrl_c_unwinds_it_still_cancels(logged):
    workflow = Workflow(hang=True, close_fails=True)
    ui = opening_boxes(autospec_session())

    turn = asyncio.create_task(MessageProcessor().process(
        "hi", ReplayController(workflow), ui, SETTINGS, session_id="s1"
    ))
    await asyncio.wait_for(workflow.waiting.wait(), timeout=2)
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    assert workflow.closed
    assert said_once_and_no_error(ui)
    assert [e["log_level"] for e in logged if e["event"] == "turn_stream_close_failed"] == ["debug"]


async def test_a_close_that_fails_without_a_cancel_is_still_the_turn_s_error(monkeypatch):
    async def broken(*args) -> None:
        raise RuntimeError("a handler bug")

    monkeypatch.setattr(MessageProcessor, "_EVENT_DISPATCH", {EventType.TEXT: broken})
    workflow = Workflow(close_fails=True)
    ui = opening_boxes(autospec_session())

    result = await MessageProcessor().process(
        "hi", ReplayController(workflow), ui, SETTINGS, session_id="s1"
    )

    assert not result.delivered
    assert [c.args[0] for c in ui.add_error.call_args_list] == ["Workflow error: cleanup failed"]
