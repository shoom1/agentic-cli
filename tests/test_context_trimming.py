"""Tests for source-level context trimming detection.

Covers:
- WorkflowEvent.context_trimmed() factory
- LangGraphBuilder._trim_events side-channel
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


# === LangGraphBuilder._trim_events ===


class TestBuilderTrimEvents:
    def test_trim_events_initialized_empty(self):
        from agentic_cli.workflow.langgraph.graph_builder import LangGraphBuilder

        settings = MagicMock()
        builder = LangGraphBuilder(settings)
        assert builder._trim_events == []

    def test_trim_events_appendable(self):
        from agentic_cli.workflow.langgraph.graph_builder import LangGraphBuilder

        settings = MagicMock()
        builder = LangGraphBuilder(settings)
        builder._trim_events.append({
            "messages_before": 20,
            "messages_after": 10,
            "agent": "test_agent",
        })
        assert len(builder._trim_events) == 1
        assert builder._trim_events[0]["messages_before"] == 20

    def test_drain_pattern(self):
        """Verify the drain pattern used by the manager works correctly."""
        from agentic_cli.workflow.langgraph.graph_builder import LangGraphBuilder

        settings = MagicMock()
        builder = LangGraphBuilder(settings)
        builder._trim_events.append({"messages_before": 20, "messages_after": 10, "agent": "a"})
        builder._trim_events.append({"messages_before": 15, "messages_after": 8, "agent": "b"})

        drained = []
        while builder._trim_events:
            drained.append(builder._trim_events.pop(0))

        assert len(drained) == 2
        assert builder._trim_events == []
        assert drained[0]["agent"] == "a"
        assert drained[1]["agent"] == "b"


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
        state = _EventProcessingState()
        state._usage_tracker = UsageTracker()
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
        assert state._usage_tracker.context_trimmed_count == 1

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
        state = _EventProcessingState()
        tracker = UsageTracker()
        # Simulate a previous invocation
        tracker.record({"prompt_tokens": 10000, "completion_tokens": 100})
        state._usage_tracker = tracker
        state._workflow_controller = MagicMock()
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
        assert state._usage_tracker.context_trimmed_count == 1
        ui.add_warning.assert_called_once()
        warning = ui.add_warning.call_args[0][0]
        assert "adk_token_heuristic" in warning

    async def test_no_double_count_with_context_trimmed(self, setup):
        """When CONTEXT_TRIMMED was already received, no fallback detection."""
        processor, state, ui, settings, workflow = setup
        # Simulate CONTEXT_TRIMMED already handled
        state._context_trimmed_this_invocation = True
        state._usage_tracker.context_trimmed_count = 1

        event = WorkflowEvent.llm_usage(
            model="gemini-2.5-pro",
            prompt_tokens=5000,
            completion_tokens=200,
        )
        await processor._handle_llm_usage(event, state, ui, settings, workflow)
        # Count should still be 1, not incremented again
        assert state._usage_tracker.context_trimmed_count == 1
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
        assert state._usage_tracker.context_trimmed_count == 0
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
