"""Tests for source-level context trimming detection.

Covers:
- WorkflowEvent.context_trimmed() factory, with what a summary or a trim did
- WorkflowEvent.context_summarizing(), and the CLI's status line for it
- MessageProcessor dispatch for CONTEXT_TRIMMED
- ADK heuristic fallback in _handle_llm_usage
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from agentic_cli.workflow.events import WorkflowEvent, EventType


# === WorkflowEvent.context_trimmed factory ===


class TestContextTrimmedEvent:
    def test_creates_correct_event_type(self):
        event = WorkflowEvent.context_trimmed(
            messages_before=20, messages_after=10, source="langgraph"
        )
        assert event.type == EventType.CONTEXT_TRIMMED

    def test_metadata_fields(self):
        event = WorkflowEvent.context_trimmed(
            messages_before=20,
            messages_after=10,
            source="langgraph",
            agent="researcher",
        )
        assert event.metadata["messages_before"] == 20
        assert event.metadata["messages_after"] == 10
        assert event.metadata["messages_removed"] == 10
        assert event.metadata["source"] == "langgraph"
        assert event.metadata["agent"] == "researcher"

    def test_content_string(self):
        event = WorkflowEvent.context_trimmed(
            messages_before=20, messages_after=15, source="adk_token_heuristic"
        )
        assert "5 messages removed" in event.content
        assert "adk_token_heuristic" in event.content

    def test_agent_optional(self):
        event = WorkflowEvent.context_trimmed(
            messages_before=10, messages_after=5, source="langgraph"
        )
        assert "agent" not in event.metadata

    def test_zero_messages_removed(self):
        event = WorkflowEvent.context_trimmed(
            messages_before=10, messages_after=10, source="langgraph"
        )
        assert event.metadata["messages_removed"] == 0

    def test_what_a_summary_did(self):
        event = WorkflowEvent.context_trimmed(
            messages_before=11, messages_after=2, source="summarization", agent="solo",
            messages_summarized=10, tokens_before=152_000, tokens_after=9_000,
        )

        assert event.metadata["messages_summarized"] == 10
        assert event.metadata["tokens_before"] == 152_000
        assert event.metadata["tokens_after"] == 9_000
        assert "turns_dropped" not in event.metadata

    def test_what_a_trim_did(self):
        event = WorkflowEvent.context_trimmed(
            messages_before=12, messages_after=6, source="fit_guard", turns_dropped=3
        )

        assert event.metadata["turns_dropped"] == 3
        for key in ("messages_summarized", "tokens_before", "tokens_after"):
            assert key not in event.metadata

    def test_the_counts_are_keyword_only(self):
        with pytest.raises(TypeError):
            WorkflowEvent.context_trimmed(11, 2, "summarization", "solo", 10)


# === WorkflowEvent.context_summarizing ===


class TestContextSummarizingEvent:
    def test_the_line_the_user_reads(self):
        event = WorkflowEvent.context_summarizing("solo")

        assert event.type is EventType.CONTEXT_SUMMARIZING
        assert EventType.CONTEXT_SUMMARIZING.value == "context_summarizing"
        assert event.content == "Summarizing the earlier conversation…"
        assert event.metadata == {"agent": "solo"}

    def test_the_agent_is_optional(self):
        assert WorkflowEvent.context_summarizing().metadata == {}


class TestHandleContextSummarizing:
    async def test_it_is_the_status_line_and_prints_nothing(self):
        from agentic_cli.cli.message_processor import (
            MessageProcessor,
            _EventProcessingState,
        )
        from tests.cli.dialog_fakes import autospec_session

        MessageProcessor._EVENT_DISPATCH = None
        handler = MessageProcessor._get_event_dispatch()[EventType.CONTEXT_SUMMARIZING]
        state = _EventProcessingState()
        ui = autospec_session()

        await handler(MessageProcessor(), WorkflowEvent.context_summarizing("solo"),
                      state, ui, MagicMock(), MagicMock())

        assert state.status_line == "Summarizing the earlier conversation…"
        assert ui.method_calls == []

    async def test_the_line_ends_with_the_summary(self):
        # The answer's model call runs next, and reports nothing until it ends.
        from agentic_cli.cli.message_processor import (
            MessageProcessor,
            _EventProcessingState,
        )
        from tests.cli.dialog_fakes import autospec_session

        MessageProcessor._EVENT_DISPATCH = None
        dispatch = MessageProcessor._get_event_dispatch()
        state = _EventProcessingState()
        usage = WorkflowEvent.llm_usage(model="local:small", prompt_tokens=10, completion_tokens=5)
        usage.metadata["purpose"] = "summarization"

        for event in (WorkflowEvent.context_summarizing("solo"), usage):
            await dispatch[event.type](MessageProcessor(), event, state, autospec_session(),
                                       MagicMock(), MagicMock())

        assert state.status_line == "Processing..."

    async def test_other_usage_leaves_the_line_alone(self):
        from agentic_cli.cli.message_processor import (
            MessageProcessor,
            _EventProcessingState,
        )
        from tests.cli.dialog_fakes import autospec_session

        MessageProcessor._EVENT_DISPATCH = None
        state = _EventProcessingState(status_line="Calling: web_fetch")
        usage = WorkflowEvent.llm_usage(model="local:small", prompt_tokens=10, completion_tokens=5)

        await MessageProcessor._get_event_dispatch()[EventType.LLM_USAGE](
            MessageProcessor(), usage, state, autospec_session(), MagicMock(), MagicMock()
        )

        assert state.status_line == "Calling: web_fetch"

    async def test_a_turn_that_summarizes_shows_only_its_answer(self):
        from tests.test_message_processor_render import _render

        ui, _, _ = await _render([
            WorkflowEvent.context_summarizing("solo"),
            WorkflowEvent.text("The answer."),
        ])

        assert ui.responses() == ["The answer."]
        assert not ui.warnings() and not ui.of("message") and not ui.rich()


# === Dispatch table includes CONTEXT_TRIMMED ===


class TestDispatchTable:
    def test_context_trimmed_in_dispatch(self):
        from agentic_cli.cli.message_processor import MessageProcessor

        # Clear cached dispatch table so it rebuilds
        MessageProcessor._EVENT_DISPATCH = None
        dispatch = MessageProcessor._get_event_dispatch()
        assert EventType.CONTEXT_TRIMMED in dispatch

    def test_handler_is_handle_context_trimmed(self):
        from agentic_cli.cli.message_processor import MessageProcessor

        MessageProcessor._EVENT_DISPATCH = None
        dispatch = MessageProcessor._get_event_dispatch()
        handler = dispatch[EventType.CONTEXT_TRIMMED]
        assert handler.__name__ == "_handle_context_trimmed"


# === _handle_context_trimmed behavior ===


class TestHandleContextTrimmed:
    @pytest.fixture
    def setup(self):
        from agentic_cli.cli.message_processor import (
            MessageProcessor,
            _EventProcessingState,
        )
        from agentic_cli.cli.usage_tracker import UsageTracker

        processor = MessageProcessor()
        state = _EventProcessingState(usage_tracker=UsageTracker())
        ui = MagicMock()
        settings = MagicMock()
        workflow = MagicMock()
        return processor, state, ui, settings, workflow

    async def test_increments_trimmed_count(self, setup):
        processor, state, ui, settings, workflow = setup
        event = WorkflowEvent.context_trimmed(
            messages_before=20, messages_after=10, source="langgraph"
        )
        await processor._handle_context_trimmed(event, state, ui, settings, workflow)
        assert state.usage_tracker.context_trimmed_count == 1

    async def test_sets_invocation_flag(self, setup):
        processor, state, ui, settings, workflow = setup
        event = WorkflowEvent.context_trimmed(
            messages_before=20, messages_after=10, source="langgraph"
        )
        await processor._handle_context_trimmed(event, state, ui, settings, workflow)
        assert state._context_trimmed_this_invocation is True

    async def test_warns_user_with_message_count(self, setup):
        processor, state, ui, settings, workflow = setup
        event = WorkflowEvent.context_trimmed(
            messages_before=20, messages_after=10, source="langgraph"
        )
        await processor._handle_context_trimmed(event, state, ui, settings, workflow)
        ui.add_warning.assert_called_once()
        warning = ui.add_warning.call_args[0][0]
        assert "10 messages removed" in warning
        assert "langgraph" in warning


# === ADK heuristic fallback in _handle_llm_usage ===


class TestADKHeuristicFallback:
    @pytest.fixture
    def setup(self):
        from agentic_cli.cli.message_processor import (
            MessageProcessor,
            _EventProcessingState,
        )
        from agentic_cli.cli.usage_tracker import UsageTracker

        processor = MessageProcessor()
        tracker = UsageTracker()
        # Simulate a previous invocation
        tracker.record({"prompt_tokens": 10000, "completion_tokens": 100})
        state = _EventProcessingState(
            usage_tracker=tracker,
            workflow_controller=MagicMock(),
        )
        ui = MagicMock()
        settings = MagicMock()
        workflow = MagicMock()
        return processor, state, ui, settings, workflow

    async def test_adk_fallback_detects_drop(self, setup):
        """When no CONTEXT_TRIMMED event was received, prompt_tokens drop triggers fallback."""
        processor, state, ui, settings, workflow = setup
        event = WorkflowEvent.llm_usage(
            model="gemini-2.5-pro",
            prompt_tokens=5000,
            completion_tokens=200,
        )
        await processor._handle_llm_usage(event, state, ui, settings, workflow)
        assert state.usage_tracker.context_trimmed_count == 1
        ui.add_warning.assert_called_once()
        warning = ui.add_warning.call_args[0][0]
        assert "token_drop_heuristic" in warning

    async def test_no_double_count_with_context_trimmed(self, setup):
        """When CONTEXT_TRIMMED was already received, no fallback detection."""
        processor, state, ui, settings, workflow = setup
        # Simulate CONTEXT_TRIMMED already handled
        state._context_trimmed_this_invocation = True
        state.usage_tracker.context_trimmed_count = 1

        event = WorkflowEvent.llm_usage(
            model="gemini-2.5-pro",
            prompt_tokens=5000,
            completion_tokens=200,
        )
        await processor._handle_llm_usage(event, state, ui, settings, workflow)
        # Count should still be 1, not incremented again
        assert state.usage_tracker.context_trimmed_count == 1
        ui.add_warning.assert_not_called()

    async def test_no_fallback_on_increase(self, setup):
        """No fallback when prompt_tokens increases (normal growth)."""
        processor, state, ui, settings, workflow = setup
        event = WorkflowEvent.llm_usage(
            model="gemini-2.5-pro",
            prompt_tokens=15000,
            completion_tokens=200,
        )
        await processor._handle_llm_usage(event, state, ui, settings, workflow)
        assert state.usage_tracker.context_trimmed_count == 0
        ui.add_warning.assert_not_called()

    async def test_flag_reset_after_llm_usage(self, setup):
        """_context_trimmed_this_invocation resets after _handle_llm_usage."""
        processor, state, ui, settings, workflow = setup
        state._context_trimmed_this_invocation = True

        event = WorkflowEvent.llm_usage(
            model="gemini-2.5-pro",
            prompt_tokens=5000,
            completion_tokens=200,
        )
        await processor._handle_llm_usage(event, state, ui, settings, workflow)
        assert state._context_trimmed_this_invocation is False
