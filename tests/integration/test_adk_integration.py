"""Integration tests for GoogleADKWorkflowManager.

Tests the event processing pipeline by mocking Runner.run_async() to yield
scripted ADK-style event objects, then verifying the WorkflowEvents produced
by _process_adk_event / _process_part.

Mock approach:
    Patch self._runner.run_async() with an async generator that yields
    mock ADK events (content.parts[] with .text, .function_call,
    .function_response attributes).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_cli.workflow.adk_manager import GoogleADKWorkflowManager
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.events import EventType, WorkflowEvent

from tests.integration.conftest import collect_events
from tests.integration.helpers import (
    assert_event_order,
    assert_no_errors,
    assert_tool_called,
    find_events,
    find_tool_calls,
    find_tool_results,
)


# ---------------------------------------------------------------------------
# ADK mock helpers
# ---------------------------------------------------------------------------

def _make_text_part(text: str, thought: bool = False) -> SimpleNamespace:
    """Create a mock ADK text part."""
    return SimpleNamespace(
        text=text,
        thought=thought,
        function_call=None,
        function_response=None,
        code_execution_result=None,
        executable_code=None,
        file_data=None,
    )


def _make_function_call_part(name: str, args: dict | None = None) -> SimpleNamespace:
    """Create a mock ADK function_call part."""
    return SimpleNamespace(
        text=None,
        thought=False,
        function_call=SimpleNamespace(name=name, args=args or {}),
        function_response=None,
        code_execution_result=None,
        executable_code=None,
        file_data=None,
    )


def _make_function_response_part(
    name: str,
    response: dict,
) -> SimpleNamespace:
    """Create a mock ADK function_response part."""
    part = SimpleNamespace(
        text=None,
        thought=False,
        function_call=None,
        function_response=SimpleNamespace(name=name, response=response),
        code_execution_result=None,
        executable_code=None,
        file_data=None,
    )
    return part


def _make_adk_event(parts: list, usage_metadata=None) -> SimpleNamespace:
    """Create a mock ADK event wrapping parts."""
    return SimpleNamespace(
        content=SimpleNamespace(parts=parts),
        usage_metadata=usage_metadata,
    )


# ---------------------------------------------------------------------------
# Manager factory
# ---------------------------------------------------------------------------

def _create_manager(mock_settings, agent_configs):
    """Create a GoogleADKWorkflowManager with mocked initialization."""
    manager = GoogleADKWorkflowManager(
        agent_configs=agent_configs,
        settings=mock_settings,
        model="gemini-2.0-flash",
    )
    # Pre-set internals to skip real initialization
    manager._initialized = True
    manager._session_service = MagicMock()
    manager._root_agent = MagicMock()
    manager._runner = MagicMock()
    return manager


async def _mock_session():
    """Return a mock session from get_or_create_session."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestADKSimpleTextResponse:
    """Mock yields text part → assert TEXT event."""

    async def test_simple_text_response(self, mock_settings, simple_agent_config):
        manager = _create_manager(mock_settings, simple_agent_config)

        # Script: one event with a text part
        scripted_events = [
            _make_adk_event([_make_text_part("Hello, world!")]),
        ]

        async def mock_run_async(**kwargs):
            for event in scripted_events:
                yield event

        manager._runner.run_async = mock_run_async
        manager._session_service.get_session = AsyncMock(return_value=MagicMock())

        events = await collect_events(manager, "Hi")

        text_events = find_events(events, EventType.TEXT)
        assert len(text_events) == 1
        assert text_events[0].content == "Hello, world!"
        assert_no_errors(events)


class TestADKToolCallAndResult:
    """Mock yields function_call then function_response → assert TOOL_CALL + TOOL_RESULT."""

    async def test_tool_call_and_result(self, mock_settings, simple_agent_config):
        manager = _create_manager(mock_settings, simple_agent_config)

        scripted_events = [
            _make_adk_event([
                _make_function_call_part("save_plan", {"content": "My plan"}),
            ]),
            _make_adk_event([
                _make_function_response_part(
                    "save_plan",
                    {"success": True, "message": "Plan saved"},
                ),
            ]),
            _make_adk_event([
                _make_text_part("I've saved your plan."),
            ]),
        ]

        async def mock_run_async(**kwargs):
            for event in scripted_events:
                yield event

        manager._runner.run_async = mock_run_async
        manager._session_service.get_session = AsyncMock(return_value=MagicMock())

        events = await collect_events(manager, "Make a plan")

        # Verify tool call event
        assert_tool_called(events, "save_plan", args_contain={"content": "My plan"})

        # Verify tool result event
        results = find_tool_results(events, "save_plan")
        assert len(results) == 1
        assert results[0].metadata["success"] is True

        # Verify text response after tool
        text_events = find_events(events, EventType.TEXT)
        assert any("saved" in e.content.lower() for e in text_events)

        # Verify event order
        assert_event_order(
            events,
            [EventType.TOOL_CALL, EventType.TOOL_RESULT, EventType.TEXT],
        )


class TestADKMultiToolSequence:
    """Mock yields multiple function_call/response pairs → correct ordering."""

    async def test_multi_tool_sequence(self, mock_settings, simple_agent_config):
        manager = _create_manager(mock_settings, simple_agent_config)

        scripted_events = [
            # First tool call
            _make_adk_event([
                _make_function_call_part("save_plan", {"content": "Step 1"}),
            ]),
            _make_adk_event([
                _make_function_response_part(
                    "save_plan",
                    {"success": True, "message": "Plan saved"},
                ),
            ]),
            # Second tool call
            _make_adk_event([
                _make_function_call_part("get_plan", {}),
            ]),
            _make_adk_event([
                _make_function_response_part(
                    "get_plan",
                    {"success": True, "content": "Step 1"},
                ),
            ]),
            # Final text
            _make_adk_event([
                _make_text_part("Done."),
            ]),
        ]

        async def mock_run_async(**kwargs):
            for event in scripted_events:
                yield event

        manager._runner.run_async = mock_run_async
        manager._session_service.get_session = AsyncMock(return_value=MagicMock())

        events = await collect_events(manager, "Plan and verify")

        # Both tools were called
        assert_tool_called(events, "save_plan")
        assert_tool_called(events, "get_plan")

        # Correct ordering: call → result → call → result → text
        assert_event_order(
            events,
            [
                EventType.TOOL_CALL,
                EventType.TOOL_RESULT,
                EventType.TOOL_CALL,
                EventType.TOOL_RESULT,
                EventType.TEXT,
            ],
        )

        # Total tool calls
        all_calls = find_tool_calls(events)
        assert len(all_calls) == 2


class TestADKThinkingDetection:
    """Mock yields text with thought=True → assert THINKING event."""

    async def test_thinking_detection(self, mock_settings, simple_agent_config):
        manager = _create_manager(mock_settings, simple_agent_config)

        scripted_events = [
            _make_adk_event([
                _make_text_part("Let me reason about this...", thought=True),
            ]),
            _make_adk_event([
                _make_text_part("Here is my answer."),
            ]),
        ]

        async def mock_run_async(**kwargs):
            for event in scripted_events:
                yield event

        manager._runner.run_async = mock_run_async
        manager._session_service.get_session = AsyncMock(return_value=MagicMock())

        events = await collect_events(manager, "Think about this")

        thinking_events = find_events(events, EventType.THINKING)
        assert len(thinking_events) == 1
        assert "reason" in thinking_events[0].content.lower()

        text_events = find_events(events, EventType.TEXT)
        assert len(text_events) == 1
        assert "answer" in text_events[0].content.lower()

        assert_event_order(events, [EventType.THINKING, EventType.TEXT])


class TestADKErrorHandling:
    """Mock yields function_response with error → assert TOOL_RESULT with success=False."""

    async def test_tool_error(self, mock_settings, simple_agent_config):
        manager = _create_manager(mock_settings, simple_agent_config)

        scripted_events = [
            _make_adk_event([
                _make_function_call_part("save_plan", {"content": "bad"}),
            ]),
            _make_adk_event([
                _make_function_response_part(
                    "save_plan",
                    {"success": False, "error": "Disk full"},
                ),
            ]),
            _make_adk_event([
                _make_text_part("Sorry, I could not save the plan."),
            ]),
        ]

        async def mock_run_async(**kwargs):
            for event in scripted_events:
                yield event

        manager._runner.run_async = mock_run_async
        manager._session_service.get_session = AsyncMock(return_value=MagicMock())

        events = await collect_events(manager, "Save plan")

        # Tool was called
        assert_tool_called(events, "save_plan")

        # Result has success=False
        results = find_tool_results(events, "save_plan")
        assert len(results) == 1
        assert results[0].metadata["success"] is False

        # Verify the error summary contains the error message
        assert "Disk full" in results[0].content or "Failed" in results[0].content


class TestADKEmptyEvents:
    """Mock yields event with no content → no crash, no events."""

    async def test_empty_event(self, mock_settings, simple_agent_config):
        manager = _create_manager(mock_settings, simple_agent_config)

        scripted_events = [
            SimpleNamespace(content=None, usage_metadata=None),
            SimpleNamespace(
                content=SimpleNamespace(parts=[]),
                usage_metadata=None,
            ),
            _make_adk_event([_make_text_part("Final answer.")]),
        ]

        async def mock_run_async(**kwargs):
            for event in scripted_events:
                yield event

        manager._runner.run_async = mock_run_async
        manager._session_service.get_session = AsyncMock(return_value=MagicMock())

        events = await collect_events(manager, "Hello")

        text_events = find_events(events, EventType.TEXT)
        assert len(text_events) == 1
        assert text_events[0].content == "Final answer."
        assert_no_errors(events)


class TestADKUsageMetadata:
    """Mock yields event with usage_metadata → assert LLM_USAGE event."""

    async def test_usage_metadata(self, mock_settings, simple_agent_config):
        manager = _create_manager(mock_settings, simple_agent_config)

        usage = SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=50,
            total_token_count=150,
            thoughts_token_count=None,
            cached_content_token_count=None,
        )

        scripted_events = [
            _make_adk_event(
                [_make_text_part("Response with usage.")],
                usage_metadata=usage,
            ),
        ]

        async def mock_run_async(**kwargs):
            for event in scripted_events:
                yield event

        manager._runner.run_async = mock_run_async
        manager._session_service.get_session = AsyncMock(return_value=MagicMock())

        events = await collect_events(manager, "Hello")

        usage_events = find_events(events, EventType.LLM_USAGE)
        assert len(usage_events) == 1
        assert usage_events[0].metadata["prompt_tokens"] == 100
        assert usage_events[0].metadata["completion_tokens"] == 50
