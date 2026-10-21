"""The whole turn boundary is safe, not just the happy path.

Three defects:

1. Cancelling the caller of ``MessageProcessor`` left the child consumer task
   running: ``_run_turn``'s ``finally`` cleared the HITL callback and the turn
   state while the workflow generator was still being driven, so a tool could
   still be executing with no callback to answer it and no owner to await it.
2. ``EventType.ERROR`` had no handler. A backend that reported a failure as an
   event rendered nothing and the turn was still reported ``COMPLETED``, so a
   background-job resume recorded a failed delivery as delivered.
3. (See ``tests/workflow/test_turn_serialization.py`` for the HITL callback
   being context-local rather than manager-global.)

The session is specced from ``ThinkingPromptSession``: a hand-written one
defined ``input_dialog``, which no thinking-prompt release has, so the
free-text question these tests ask passed here and failed on every real
session.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agentic_cli.cli.message_processor import MessageProcessor, TurnStatus
from agentic_cli.workflow.events import EventType, WorkflowEvent

from tests.cli.dialog_fakes import autospec_session, edited_dialog, opening_boxes


def _ui() -> MagicMock:
    """A session each of whose ``start_thinking`` calls opens a new box
    (``ui.boxes``)."""
    return opening_boxes(autospec_session())


def _shown(method: MagicMock) -> list[str]:
    """What each call of a session method (``ui.add_error``, ...) showed."""
    return [c.args[0] for c in method.call_args_list]


class _Workflow:
    """Tracks whether the HITL callback is currently installed."""

    def __init__(self) -> None:
        self.callback = None

    def set_input_callback(self, cb):
        self.callback = cb
        return None

    def clear_input_callback(self, token=None) -> None:
        self.callback = None


def _controller(workflow):
    return SimpleNamespace(workflow=workflow, update_status_bar=lambda ui: None)


def _settings():
    return SimpleNamespace(verbose_thinking=False, default_user="u")


async def _run(processor, ui, source_factory, workflow=None):
    workflow = workflow or _Workflow()
    return await processor._run_turn(
        source_factory, _controller(workflow), ui, _settings(), None
    )


class TestCallerCancellation:
    """A cancelled caller must not leave the event stream running behind it."""

    async def test_child_consumer_is_cancelled_and_awaited(self):
        started = asyncio.Event()
        observed: dict = {"cancelled": False, "closed": False, "callback_at_cancel": "unset"}
        workflow = _Workflow()

        def _source(_wf):
            async def _gen():
                started.set()
                try:
                    await asyncio.sleep(30)
                    yield WorkflowEvent(type=EventType.TEXT, content="never")
                except asyncio.CancelledError:
                    observed["cancelled"] = True
                    observed["callback_at_cancel"] = workflow.callback
                    raise
                finally:
                    observed["closed"] = True

            return _gen()

        turn = asyncio.create_task(
            _run(MessageProcessor(), _ui(), _source, workflow)
        )
        await asyncio.wait_for(started.wait(), timeout=2)

        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn

        assert observed["cancelled"] is True, "the event stream outlived its caller"
        assert observed["closed"] is True, "the consumer was never awaited"
        assert observed["callback_at_cancel"] is not None, (
            "the HITL callback was cleared while a tool could still ask for input"
        )
        assert workflow.callback is None, "the callback was not cleared afterwards"

    async def test_cancellation_does_not_leave_a_pending_task(self):
        """Nothing is left for the loop to garbage-collect mid-flight."""
        started = asyncio.Event()

        def _source(_wf):
            async def _gen():
                started.set()
                await asyncio.sleep(30)
                yield WorkflowEvent(type=EventType.TEXT, content="never")

            return _gen()

        turn = asyncio.create_task(_run(MessageProcessor(), _ui(), _source))
        await asyncio.wait_for(started.wait(), timeout=2)
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn

        others = [
            t
            for t in asyncio.all_tasks()
            if t is not asyncio.current_task() and not t.done()
        ]
        assert others == [], f"orphaned tasks survived the cancelled turn: {others}"


def _hitl_ui() -> MagicMock:
    """A session whose question stays open (``ui.dialog_open`` is set once it
    is shown) until the turn is cancelled."""
    ui = _ui()
    ui.dialog_open = asyncio.Event()

    async def show_dialog(dialog) -> bool:
        ui.dialog_open.set()
        await asyncio.sleep(30)  # the user is still typing when we're cancelled
        return True  # pragma: no cover

    ui.show_dialog.side_effect = show_dialog
    return ui


class _HitlWorkflow(_Workflow):
    """Records the teardown order and drives the installed HITL callback."""

    def __init__(self, trace: list[str]) -> None:
        super().__init__()
        self._trace = trace

    def clear_input_callback(self, token=None) -> None:
        self._trace.append("callback-cleared")
        super().clear_input_callback(token)


class TestCancellationDuringHitl:
    """Cancelling while a HITL dialog is open must not strand a thinking box.

    The dialog's ``finally`` unconditionally opened a *replacement* events box
    — including while the turn was unwinding — so a cancelled HITL turn left a
    box on screen that nothing would ever finish.
    """

    def _turn(self):
        trace: list[str] = []
        ui = _hitl_ui()
        workflow = _HitlWorkflow(trace)

        def _source(wf):
            async def _gen():
                try:
                    await wf.callback(
                        SimpleNamespace(
                            request_id="r",
                            tool_name="ask_clarification",
                            prompt="which?",
                            input_type=None,
                            choices=None,
                            default=None,
                        )
                    )
                    yield WorkflowEvent(type=EventType.TEXT, content="never")
                finally:
                    trace.append("consumer-done")

            return _gen()

        return trace, ui, workflow, _source

    async def test_every_thinking_context_is_finished_exactly_once(self):
        trace, ui, workflow, source = self._turn()

        turn = asyncio.create_task(_run(MessageProcessor(), ui, source, workflow))
        await asyncio.wait_for(ui.dialog_open.wait(), timeout=2)

        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn

        assert ui.boxes, "no thinking box was ever opened"
        counts = [box.finish.call_count for box in ui.boxes]
        assert counts == [1] * len(ui.boxes), (
            f"thinking boxes were not finished exactly once: {counts}"
        )

    async def test_consumer_settles_before_the_callback_is_cleared(self):
        trace, ui, workflow, source = self._turn()

        turn = asyncio.create_task(_run(MessageProcessor(), ui, source, workflow))
        await asyncio.wait_for(ui.dialog_open.wait(), timeout=2)

        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn

        assert trace.index("consumer-done") < trace.index("callback-cleared")

    async def test_normal_hitl_turn_still_reopens_the_events_box(self, monkeypatch):
        """The replacement box is right on the *success* path — keep it."""
        ui = _ui()
        ui.show_dialog.return_value = True  # OK, with the answer typed
        monkeypatch.setattr(
            "agentic_cli.cli.message_processor.Dialog", edited_dialog({"answer": "answer"})
        )
        workflow = _HitlWorkflow([])

        def _source(wf):
            async def _gen():
                answer = await wf.callback(
                    SimpleNamespace(
                        request_id="r", tool_name="t", prompt="p",
                        input_type=None, choices=None, default=None,
                    )
                )
                yield WorkflowEvent(type=EventType.TEXT, content=answer)

            return _gen()

        result = await _run(MessageProcessor(), ui, _source, workflow)

        assert result.status is TurnStatus.COMPLETED
        assert "answer" in _shown(ui.add_response)
        assert len(ui.boxes) == 2, "the events box was not reopened after the dialog"
        assert [box.finish.call_count for box in ui.boxes] == [1, 1]

    async def test_cancel_outside_hitl_finishes_the_events_box(self):
        started = asyncio.Event()
        ui = _hitl_ui()
        workflow = _HitlWorkflow([])

        def _source(_wf):
            async def _gen():
                started.set()
                await asyncio.sleep(30)
                yield WorkflowEvent(type=EventType.TEXT, content="never")

            return _gen()

        turn = asyncio.create_task(_run(MessageProcessor(), ui, _source, workflow))
        await asyncio.wait_for(started.wait(), timeout=2)
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn

        assert [box.finish.call_count for box in ui.boxes] == [1]


class _EventSource:
    """Yields a fixed list of events, once."""

    def __init__(self, *events: WorkflowEvent) -> None:
        self._events = list(events)
        self.invocations = 0

    def __call__(self, _workflow):
        self.invocations += 1

        async def _gen():
            for event in self._events:
                yield event

        return _gen()


class TestErrorEvents:
    """An ERROR event is rendered, and a fatal one fails the turn."""

    async def test_non_recoverable_error_is_rendered(self):
        ui = _ui()
        await _run(
            MessageProcessor(),
            ui,
            _EventSource(WorkflowEvent.error("model refused the request")),
        )
        assert any("model refused the request" in e for e in _shown(ui.add_error))

    async def test_non_recoverable_error_fails_the_turn(self):
        result = await _run(
            MessageProcessor(),
            _ui(),
            _EventSource(WorkflowEvent.error("backend exploded")),
        )
        assert result.status is TurnStatus.FAILED
        assert result.delivered is False
        assert "backend exploded" in (result.error or "")

    async def test_error_after_output_is_reported_partial(self):
        result = await _run(
            MessageProcessor(),
            _ui(),
            _EventSource(
                WorkflowEvent(type=EventType.TEXT, content="half an answer"),
                WorkflowEvent.error("then it died"),
            ),
        )
        assert result.status is TurnStatus.FAILED
        assert result.partial is True

    async def test_recoverable_error_is_rendered_and_the_turn_completes(self):
        """A recoverable error is informational: the stream owns the outcome."""
        ui = _ui()
        result = await _run(
            MessageProcessor(),
            ui,
            _EventSource(
                WorkflowEvent.error("one tool retried", recoverable=True),
                WorkflowEvent(type=EventType.TEXT, content="done anyway"),
            ),
        )
        assert any("one tool retried" in w for w in _shown(ui.add_warning))
        assert result.status is TurnStatus.COMPLETED
        assert result.delivered is True

    async def test_first_fatal_error_is_the_reported_one(self):
        result = await _run(
            MessageProcessor(),
            _ui(),
            _EventSource(
                WorkflowEvent.error("first failure"),
                WorkflowEvent.error("cascade"),
            ),
        )
        assert result.error == "first failure"
