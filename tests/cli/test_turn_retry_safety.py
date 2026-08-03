"""The harness never replays a turn.

``_run_turn`` used to re-invoke the event-source factory after a 429, first
unconditionally and then "only before the first visible event". Both are wrong:
ADK's ``Runner`` appends the input to the session while setting up the
invocation — before any event is yielded — so a 429 on the very first model
call has already persisted the user message (or the resumed
``FunctionResponse``). Replaying duplicates the turn.

Retrying now happens only inside the provider client, which can prove nothing
was accepted (ADK ``HttpRetryOptions`` for transient 5xx). A surfaced rate limit
fails the turn with an explicit message.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from agentic_cli.cli.message_processor import (
    MessageProcessor,
    TurnResult,
    TurnStatus,
)
from agentic_cli.workflow.events import EventType, WorkflowEvent

from tests.event_replay import RecordingSession


class _RateLimited(Exception):
    """Shaped like a provider 429 for ``is_rate_limit_error``."""

    def __init__(self) -> None:
        super().__init__("429 RESOURCE_EXHAUSTED: rate limit exceeded")


class _UI(RecordingSession):
    """RecordingSession plus the dialog surface ``_run_turn`` may reach for."""

    def __init__(self) -> None:
        super().__init__()
        self.dialogs = 0

    async def yes_no_dialog(self, title: str = "", text: str = "") -> bool:
        self.dialogs += 1
        return True


def _controller():
    workflow = MagicMock()
    workflow.set_input_callback = MagicMock()
    workflow.clear_input_callback = MagicMock()
    return SimpleNamespace(workflow=workflow, update_status_bar=lambda ui: None)


def _settings():
    return SimpleNamespace(verbose_thinking=False, default_user="u")


def _tool_call_event() -> WorkflowEvent:
    return WorkflowEvent(
        type=EventType.TOOL_CALL,
        content="calling",
        metadata={"tool_name": "write_file", "tool_args": {}},
    )


async def _run(processor, ui, source_factory):
    return await processor._run_turn(
        source_factory, _controller(), ui, _settings(), None
    )


class _DurableSource:
    """An event source that persists its input before yielding anything.

    Mirrors ADK: ``Runner.run_async`` appends ``new_message`` to the session
    during invocation setup, so the append happens even when the first model
    call raises.
    """

    def __init__(self, *, events=(), raises=None) -> None:
        self.appended: list[str] = []
        self._events = list(events)
        self._raises = raises

    def __call__(self, workflow):
        self.appended.append("input")

        async def _gen():
            for event in self._events:
                yield event
            if self._raises is not None:
                raise self._raises

        return _gen()


class TestNoReplay:
    async def test_rate_limit_before_any_event_does_not_replay(self):
        """The 'nothing ran yet' boundary does not exist — the input is stored."""
        source = _DurableSource(raises=_RateLimited())
        ui = _UI()

        result = await _run(MessageProcessor(), ui, source)

        assert source.appended == ["input"], "the turn was replayed"
        assert ui.dialogs == 0, "the user was offered a retry that is not safe"
        assert result.status is TurnStatus.FAILED
        assert result.partial is False

    async def test_rate_limit_after_a_tool_ran_does_not_replay(self):
        source = _DurableSource(events=[_tool_call_event()], raises=_RateLimited())
        ui = _UI()

        result = await _run(MessageProcessor(), ui, source)

        assert source.appended == ["input"]
        assert result.status is TurnStatus.FAILED
        assert result.partial is True

    async def test_resume_source_is_invoked_exactly_once(self):
        """A resumed FunctionResponse is persisted too — never re-delivered."""
        source = _DurableSource(
            events=[WorkflowEvent(type=EventType.TEXT, content="partial")],
            raises=_RateLimited(),
        )
        ui = _UI()

        result = await _run(MessageProcessor(), ui, source)

        assert source.appended == ["input"], "the job result was delivered twice"
        assert result.delivered is False

    async def test_non_rate_limit_failure_also_runs_once(self):
        source = _DurableSource(raises=RuntimeError("boom"))
        ui = _UI()

        result = await _run(MessageProcessor(), ui, source)

        assert source.appended == ["input"]
        assert result.status is TurnStatus.FAILED
        assert "boom" in (result.error or "")


class TestRateLimitReporting:
    async def test_error_explains_the_turn_was_not_replayed(self):
        ui = _UI()
        result = await _run(MessageProcessor(), ui, _DurableSource(raises=_RateLimited()))

        errors = " ".join(str(e) for e in ui.errors())
        assert "Rate limited" in errors
        assert "not retried" in errors.lower()
        assert result.error and "429" in result.error

    async def test_no_retry_dialog_is_ever_shown(self):
        ui = _UI()
        await _run(MessageProcessor(), ui, _DurableSource(raises=_RateLimited()))
        assert ui.dialogs == 0


class TestTurnResultContract:
    async def test_success_is_delivered(self):
        source = _DurableSource(
            events=[WorkflowEvent(type=EventType.TEXT, content="hi")]
        )
        result = await _run(MessageProcessor(), _UI(), source)

        assert result == TurnResult(TurnStatus.COMPLETED)
        assert result.delivered is True
        assert source.appended == ["input"]

    async def test_failure_is_not_delivered(self):
        result = await _run(
            MessageProcessor(), _UI(), _DurableSource(raises=RuntimeError("boom"))
        )
        assert result.delivered is False
