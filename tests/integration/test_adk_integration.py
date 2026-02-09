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

from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager
from agentic_cli.workflow.adk.event_processor import (
    _is_rate_limit_error,
    _parse_retry_delay,
)
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


# ---------------------------------------------------------------------------
# Rate limit helper tests
# ---------------------------------------------------------------------------


class TestIsRateLimitError:
    """Tests for _is_rate_limit_error helper."""

    def test_rate_limit_error_by_code(self):
        """ClientError with code=429 is detected."""
        error = Exception("Too many requests")
        error.code = 429
        assert _is_rate_limit_error(error) is True

    def test_rate_limit_error_by_status_string(self):
        """Error containing RESOURCE_EXHAUSTED is detected."""
        error = Exception("RESOURCE_EXHAUSTED: quota exceeded")
        assert _is_rate_limit_error(error) is True

    def test_non_rate_limit_error_500(self):
        """Server error (500) is not a rate limit error."""
        error = Exception("Internal server error")
        error.code = 500
        assert _is_rate_limit_error(error) is False

    def test_non_rate_limit_generic(self):
        """Generic exception is not a rate limit error."""
        error = ValueError("something went wrong")
        assert _is_rate_limit_error(error) is False


class TestParseRetryDelay:
    """Tests for _parse_retry_delay helper."""

    def test_parse_delay_from_details(self):
        """Extracts delay from structured error.details."""
        error = Exception("rate limited")
        error.details = {
            "error": {
                "details": [
                    {"retryDelay": "41s"},
                ]
            }
        }
        assert _parse_retry_delay(error) == 41.0

    def test_parse_delay_from_message(self):
        """Extracts delay from error message string."""
        error = Exception("Please retry in 2.5s")
        assert _parse_retry_delay(error) == 2.5

    def test_parse_delay_none_for_generic(self):
        """Returns None for generic exception without delay info."""
        error = Exception("something failed")
        assert _parse_retry_delay(error) is None

    def test_parse_delay_from_details_float(self):
        """Handles float seconds in retryDelay field."""
        error = Exception("rate limited")
        error.details = {
            "error": {
                "details": [
                    {"retryDelay": "10.5s"},
                ]
            }
        }
        assert _parse_retry_delay(error) == 10.5


class TestTaskProgressAutoClean:
    """Tests for auto-clear when all tasks are done and plan-based progress."""

    def test_auto_clears_when_all_done(self, mock_settings):
        """When all tasks are completed, emit final event then clear store."""
        from agentic_cli.tools.task_tools import TaskStore

        store = TaskStore(mock_settings)
        store.replace_all([
            {"description": "Task 1", "status": "completed"},
            {"description": "Task 2", "status": "completed"},
        ])

        mgr = _create_manager(mock_settings, [AgentConfig(name="test", prompt="test")])
        mgr._task_store = store

        # First call: emits final snapshot, then clears
        event = mgr._emit_task_progress_event()
        assert event is not None
        assert event.type == EventType.TASK_PROGRESS
        assert "[x] Task 1" in event.content
        assert "[x] Task 2" in event.content
        assert event.metadata["progress"]["completed"] == 2
        assert store.is_empty()

        # Second call: store is empty, returns None
        assert mgr._emit_task_progress_event() is None

    def test_no_auto_clear_with_pending(self, mock_settings):
        """When tasks are not all done, store is NOT cleared."""
        from agentic_cli.tools.task_tools import TaskStore

        store = TaskStore(mock_settings)
        store.replace_all([
            {"description": "Done task", "status": "completed"},
            {"description": "Pending task", "status": "pending"},
        ])

        mgr = _create_manager(mock_settings, [AgentConfig(name="test", prompt="test")])
        mgr._task_store = store

        event = mgr._emit_task_progress_event()
        assert event is not None
        assert not store.is_empty()

    def test_plan_progress_after_save_plan(self, mock_settings):
        """PlanStore with checkboxes emits TASK_PROGRESS when no TaskStore."""
        from agentic_cli.tools.planning_tools import PlanStore

        plan_store = PlanStore()
        plan_store.save(
            "## Setup\n"
            "- [x] Install deps\n"
            "- [ ] Configure env\n"
            "\n"
            "## Build\n"
            "- [ ] Compile\n"
        )

        mgr = _create_manager(mock_settings, [AgentConfig(name="test", prompt="test")])
        mgr._task_store = None
        mgr._plan_store = plan_store

        event = mgr._emit_task_progress_event()
        assert event is not None
        assert event.type == EventType.TASK_PROGRESS
        assert "Setup:" in event.content
        assert "[x] Install deps" in event.content
        assert "[ ] Configure env" in event.content
        assert "Build:" in event.content
        assert "[ ] Compile" in event.content
        assert event.metadata["progress"]["total"] == 3
        assert event.metadata["progress"]["completed"] == 1
        assert event.metadata["progress"]["pending"] == 2


class TestMessageProcessorRateLimit:
    """Tests that MessageProcessor handles 429 errors with user prompt."""

    async def test_rate_limit_retry_on_user_accept(self):
        """When user accepts retry, processor waits and retries."""
        from agentic_cli.cli.message_processor import MessageProcessor

        processor = MessageProcessor()

        # Mock workflow controller
        workflow_controller = MagicMock()
        workflow_controller.ensure_initialized = AsyncMock(return_value=True)

        # First call raises 429, second call succeeds
        call_count = 0

        async def mock_process(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                error = Exception("RESOURCE_EXHAUSTED: retry in 5s")
                error.code = 429
                raise error
            # Second call: yield a text event
            yield WorkflowEvent.text("Success!", "session")

        mock_workflow = MagicMock()
        mock_workflow.process = mock_process
        workflow_controller.workflow = mock_workflow

        # Mock UI
        ui = MagicMock()
        ui.start_thinking = MagicMock()
        ui.finish_thinking = MagicMock()
        ui.add_response = MagicMock()
        ui.add_warning = MagicMock()
        ui.add_error = MagicMock()
        ui.add_rich = MagicMock()
        ui.yes_no_dialog = AsyncMock(return_value=True)  # User accepts retry

        # Mock settings
        settings = MagicMock()
        settings.default_user = "test-user"
        settings.log_activity = False
        settings.verbose_thinking = False

        with patch("agentic_cli.cli.message_processor.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await processor.process(
                message="test",
                workflow_controller=workflow_controller,
                ui=ui,
                settings=settings,
            )

        # Verify retry happened
        assert call_count == 2
        ui.yes_no_dialog.assert_called_once()
        mock_sleep.assert_called_once_with(5.0)
        ui.add_warning.assert_called_once()
        # Success path should have been reached
        ui.add_response.assert_called_once_with("Success!", markdown=True)
        ui.add_error.assert_not_called()

    async def test_rate_limit_cancel_on_user_decline(self):
        """When user declines retry, processor shows error and stops."""
        from agentic_cli.cli.message_processor import MessageProcessor

        processor = MessageProcessor()

        workflow_controller = MagicMock()
        workflow_controller.ensure_initialized = AsyncMock(return_value=True)

        async def mock_process(**kwargs):
            error = Exception("RESOURCE_EXHAUSTED: retry in 30s")
            error.code = 429
            raise error
            yield  # make it an async generator  # noqa: E501

        mock_workflow = MagicMock()
        mock_workflow.process = mock_process
        workflow_controller.workflow = mock_workflow

        ui = MagicMock()
        ui.start_thinking = MagicMock()
        ui.finish_thinking = MagicMock()
        ui.add_error = MagicMock()
        ui.add_rich = MagicMock()
        ui.yes_no_dialog = AsyncMock(return_value=False)  # User declines

        settings = MagicMock()
        settings.default_user = "test-user"
        settings.log_activity = False
        settings.verbose_thinking = False

        await processor.process(
            message="test",
            workflow_controller=workflow_controller,
            ui=ui,
            settings=settings,
        )

        ui.yes_no_dialog.assert_called_once()
        ui.add_error.assert_called_once()
        assert "Workflow error" in ui.add_error.call_args[0][0]

    async def test_non_rate_limit_error_not_retried(self):
        """Non-429 errors are not retried, shown as workflow error."""
        from agentic_cli.cli.message_processor import MessageProcessor

        processor = MessageProcessor()

        workflow_controller = MagicMock()
        workflow_controller.ensure_initialized = AsyncMock(return_value=True)

        async def mock_process(**kwargs):
            raise RuntimeError("Something broke")
            yield  # noqa: E501

        mock_workflow = MagicMock()
        mock_workflow.process = mock_process
        workflow_controller.workflow = mock_workflow

        ui = MagicMock()
        ui.start_thinking = MagicMock()
        ui.finish_thinking = MagicMock()
        ui.add_error = MagicMock()
        ui.add_rich = MagicMock()
        ui.yes_no_dialog = AsyncMock()

        settings = MagicMock()
        settings.default_user = "test-user"
        settings.log_activity = False
        settings.verbose_thinking = False

        await processor.process(
            message="test",
            workflow_controller=workflow_controller,
            ui=ui,
            settings=settings,
        )

        # Should NOT prompt user for retry
        ui.yes_no_dialog.assert_not_called()
        ui.add_error.assert_called_once()
        assert "Something broke" in ui.add_error.call_args[0][0]


class TestUserInputCallback:
    """Tests for the direct _user_input_callback path in request_user_input."""

    async def test_callback_invoked_when_set(self, mock_settings, simple_agent_config):
        """When _user_input_callback is set, request_user_input calls it directly."""
        from agentic_cli.workflow.events import UserInputRequest, InputType

        manager = _create_manager(mock_settings, simple_agent_config)

        captured_requests: list[UserInputRequest] = []

        async def fake_callback(request: UserInputRequest) -> str:
            captured_requests.append(request)
            return "user answer"

        manager._user_input_callback = fake_callback

        request = UserInputRequest(
            request_id="req-1",
            tool_name="ask_clarification",
            prompt="What color?",
            input_type=InputType.TEXT,
        )
        result = await manager.request_user_input(request)

        assert result == "user answer"
        assert len(captured_requests) == 1
        assert captured_requests[0].request_id == "req-1"
        # Future pattern should NOT have been used
        assert len(manager._pending_input) == 0

    async def test_future_pattern_without_callback(self, mock_settings, simple_agent_config):
        """Without callback, request_user_input uses the Future pattern."""
        import asyncio
        from agentic_cli.workflow.events import UserInputRequest, InputType

        manager = _create_manager(mock_settings, simple_agent_config)
        assert manager._user_input_callback is None

        request = UserInputRequest(
            request_id="req-2",
            tool_name="ask_clarification",
            prompt="What size?",
            input_type=InputType.TEXT,
        )

        # Start request_user_input in background — it will block on the Future
        task = asyncio.create_task(manager.request_user_input(request))

        # Let the event loop tick so the Future is registered
        await asyncio.sleep(0)

        assert "req-2" in manager._pending_input

        # Resolve via provide_user_input
        resolved = manager.provide_user_input("req-2", "large")
        assert resolved is True

        result = await task
        assert result == "large"
        assert len(manager._pending_input) == 0
