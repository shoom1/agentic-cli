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
"""

from __future__ import annotations

import asyncio
import gc
from types import SimpleNamespace

import pytest

from agentic_cli.cli.message_processor import MessageProcessor
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
    """Streams two events inside its turn's context, as a manager does."""

    def __init__(self) -> None:
        self.closed = False

    def set_input_callback(self, callback) -> None:
        pass

    def clear_input_callback(self) -> None:
        pass

    async def _stream(self):
        token = set_active_turn(SessionRef("app", "u", "s1"))
        try:
            yield WorkflowEvent.text("one")
            yield WorkflowEvent.text("two")
        finally:
            reset_active_turn(token)
            self.closed = True

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
