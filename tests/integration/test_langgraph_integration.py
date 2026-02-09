"""Integration tests for LangGraphWorkflowManager.

Tests the event processing pipeline by mocking the compiled graph's
astream_events() to yield scripted LangGraph-style event dicts, then
verifying the WorkflowEvents produced by the process() method.

Mock approach:
    Patch _get_llm_for_model() and the compiled graph to control
    the events yielded by astream_events(version="v2").
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from agentic_cli.workflow.events import EventType, WorkflowEvent
from agentic_cli.workflow.config import AgentConfig

from tests.integration.conftest import collect_events
from tests.integration.helpers import (
    assert_event_order,
    assert_no_errors,
    find_events,
    find_tool_calls,
    find_tool_results,
)


# ---------------------------------------------------------------------------
# LangGraph mock helpers
# ---------------------------------------------------------------------------

def _make_stream_event(
    event_kind: str,
    name: str = "",
    data: dict | None = None,
) -> dict:
    """Create a LangGraph v2 stream event dict."""
    return {
        "event": event_kind,
        "name": name,
        "data": data or {},
    }


def _make_chunk(content, usage_metadata=None):
    """Create a mock LLM response chunk with content."""
    chunk = SimpleNamespace(content=content, usage_metadata=usage_metadata)
    return chunk


def _make_output(content, tool_calls=None, usage_metadata=None):
    """Create a mock LLM output (for on_chat_model_end)."""
    output = SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
        usage_metadata=usage_metadata,
    )
    return output


# ---------------------------------------------------------------------------
# Manager factory
# ---------------------------------------------------------------------------

def _create_langgraph_manager(settings, agent_configs):
    """Create a LangGraphWorkflowManager with mocked initialization.

    Avoids importing LangGraph dependencies by mocking the compiled graph
    and LLM creation.
    """
    # Import here since langgraph may not be installed
    pytest.importorskip("langgraph")

    from agentic_cli.workflow.langgraph.manager import LangGraphWorkflowManager

    manager = LangGraphWorkflowManager(
        agent_configs=agent_configs,
        settings=settings,
        model="gemini-2.0-flash",
        checkpointer=None,
    )
    manager._initialized = True
    manager._compiled_graph = MagicMock()
    manager._llm = MagicMock()
    return manager


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLangGraphSimpleTextResponse:
    """Mock astream_events yields text chunk + end → assert TEXT event."""

    async def test_simple_text_response(self, mock_settings, simple_agent_config):
        manager = _create_langgraph_manager(mock_settings, simple_agent_config)

        scripted_events = [
            _make_stream_event(
                "on_chat_model_stream",
                name="assistant",
                data={"chunk": _make_chunk("Hello from LangGraph!")},
            ),
            _make_stream_event(
                "on_chat_model_end",
                name="assistant",
                data={"output": _make_output("Hello from LangGraph!")},
            ),
        ]

        async def mock_astream_events(state, config, version):
            for event in scripted_events:
                yield event

        manager._compiled_graph.astream_events = mock_astream_events

        events = await collect_events(manager, "Hi")

        text_events = find_events(events, EventType.TEXT)
        assert len(text_events) == 1
        assert text_events[0].content == "Hello from LangGraph!"
        assert_no_errors(events)


class TestLangGraphToolCallFlow:
    """Mock astream_events yields tool events → assert TOOL_CALL + TOOL_RESULT."""

    async def test_tool_call_flow(self, mock_settings, simple_agent_config):
        manager = _create_langgraph_manager(mock_settings, simple_agent_config)

        scripted_events = [
            # Agent starts
            _make_stream_event(
                "on_chat_model_start",
                name="assistant",
            ),
            # Tool invoked
            _make_stream_event(
                "on_tool_start",
                name="save_plan",
                data={"input": {"content": "My plan"}},
            ),
            # Tool completes
            _make_stream_event(
                "on_tool_end",
                name="save_plan",
                data={"output": {"success": True, "message": "Plan saved"}},
            ),
            # Agent responds with text
            _make_stream_event(
                "on_chat_model_stream",
                name="assistant",
                data={"chunk": _make_chunk("Plan saved successfully.")},
            ),
            _make_stream_event(
                "on_chat_model_end",
                name="assistant",
                data={"output": _make_output("Plan saved successfully.")},
            ),
        ]

        async def mock_astream_events(state, config, version):
            for event in scripted_events:
                yield event

        manager._compiled_graph.astream_events = mock_astream_events

        events = await collect_events(manager, "Create a plan")

        # Tool call event exists
        tool_calls = find_tool_calls(events, "save_plan")
        assert len(tool_calls) == 1

        # Tool result event exists
        tool_results = find_tool_results(events, "save_plan")
        assert len(tool_results) == 1
        assert tool_results[0].metadata["success"] is True

        # Text response
        text_events = find_events(events, EventType.TEXT)
        assert len(text_events) == 1

        # Ordering
        assert_event_order(
            events,
            [EventType.TOOL_CALL, EventType.TOOL_RESULT, EventType.TEXT],
        )


class TestLangGraphMultiTurnWithTools:
    """Mock yields tool_calls on first turn, text on second → full loop."""

    async def test_multi_turn(self, mock_settings, simple_agent_config):
        manager = _create_langgraph_manager(mock_settings, simple_agent_config)

        scripted_events = [
            # First turn: agent calls two tools
            _make_stream_event(
                "on_tool_start",
                name="save_plan",
                data={"input": {"content": "Plan A"}},
            ),
            _make_stream_event(
                "on_tool_end",
                name="save_plan",
                data={"output": {"success": True}},
            ),
            _make_stream_event(
                "on_tool_start",
                name="get_plan",
                data={"input": {}},
            ),
            _make_stream_event(
                "on_tool_end",
                name="get_plan",
                data={"output": {"success": True, "content": "Plan A"}},
            ),
            # Second turn: text response
            _make_stream_event(
                "on_chat_model_stream",
                name="assistant",
                data={"chunk": _make_chunk("Here's the plan I saved and retrieved.")},
            ),
            _make_stream_event(
                "on_chat_model_end",
                name="assistant",
                data={"output": _make_output("Here's the plan I saved and retrieved.")},
            ),
        ]

        async def mock_astream_events(state, config, version):
            for event in scripted_events:
                yield event

        manager._compiled_graph.astream_events = mock_astream_events

        events = await collect_events(manager, "Save and retrieve plan")

        # Both tools called
        all_calls = find_tool_calls(events)
        assert len(all_calls) == 2

        tool_names = [c.metadata["tool_name"] for c in all_calls]
        assert "save_plan" in tool_names
        assert "get_plan" in tool_names

        # Text response at the end
        text_events = find_events(events, EventType.TEXT)
        assert len(text_events) == 1


class TestLangGraphThinkingBlocks:
    """Mock returns thinking + text content blocks → assert THINKING + TEXT events."""

    async def test_thinking_and_text_blocks(self, mock_settings, simple_agent_config):
        manager = _create_langgraph_manager(mock_settings, simple_agent_config)

        # Content as list of dicts with type info (Anthropic-style)
        thinking_content = [
            {"type": "thinking", "text": "Let me think about this..."},
            {"type": "text", "text": "Here is my answer."},
        ]

        scripted_events = [
            _make_stream_event(
                "on_chat_model_stream",
                name="assistant",
                data={"chunk": _make_chunk(thinking_content)},
            ),
            _make_stream_event(
                "on_chat_model_end",
                name="assistant",
                data={"output": _make_output(thinking_content)},
            ),
        ]

        async def mock_astream_events(state, config, version):
            for event in scripted_events:
                yield event

        manager._compiled_graph.astream_events = mock_astream_events

        events = await collect_events(manager, "Think about this")

        thinking_events = find_events(events, EventType.THINKING)
        assert len(thinking_events) == 1
        assert "think" in thinking_events[0].content.lower()

        text_events = find_events(events, EventType.TEXT)
        assert len(text_events) == 1
        assert "answer" in text_events[0].content.lower()

        assert_event_order(events, [EventType.THINKING, EventType.TEXT])


class TestLangGraphUsageMetadata:
    """Mock on_chat_model_end with usage_metadata → assert LLM_USAGE event."""

    async def test_usage_metadata(self, mock_settings, simple_agent_config):
        manager = _create_langgraph_manager(mock_settings, simple_agent_config)

        usage = {
            "input_tokens": 200,
            "output_tokens": 80,
            "total_tokens": 280,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": None,
        }

        scripted_events = [
            _make_stream_event(
                "on_chat_model_stream",
                name="assistant",
                data={"chunk": _make_chunk("Response text.")},
            ),
            _make_stream_event(
                "on_chat_model_end",
                name="assistant",
                data={
                    "output": _make_output(
                        content="Response text.",
                        usage_metadata=usage,
                    ),
                },
            ),
        ]

        async def mock_astream_events(state, config, version):
            for event in scripted_events:
                yield event

        manager._compiled_graph.astream_events = mock_astream_events

        events = await collect_events(manager, "Hello")

        usage_events = find_events(events, EventType.LLM_USAGE)
        assert len(usage_events) == 1
        assert usage_events[0].metadata["prompt_tokens"] == 200
        assert usage_events[0].metadata["completion_tokens"] == 80
        assert usage_events[0].metadata["cached_tokens"] == 50


class TestLangGraphEmptyStream:
    """Mock yields no content events → no crash, no content events."""

    async def test_empty_stream(self, mock_settings, simple_agent_config):
        manager = _create_langgraph_manager(mock_settings, simple_agent_config)

        scripted_events = [
            _make_stream_event("on_chat_model_start", name="assistant"),
            # No streaming chunks
        ]

        async def mock_astream_events(state, config, version):
            for event in scripted_events:
                yield event

        manager._compiled_graph.astream_events = mock_astream_events

        events = await collect_events(manager, "Hello")

        text_events = find_events(events, EventType.TEXT)
        assert len(text_events) == 0
        assert_no_errors(events)
