"""Tests for ADK Plugins (ConfirmationPlugin, LLMLoggingPlugin)."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_cli.workflow.adk.plugins import (
    ConfirmationPlugin, LLMLoggingPlugin, is_dangerous,
)
from agentic_cli.workflow.events import EventType, UserInputRequest, WorkflowEvent


def _make_mock_tool(name: str) -> MagicMock:
    """Create a mock ADK BaseTool with a given name."""
    tool = MagicMock()
    tool.name = name
    return tool


def _make_mock_tool_context() -> MagicMock:
    """Create a mock ToolContext."""
    return MagicMock()


class TestIsDangerous:
    """Tests for the is_dangerous helper."""

    def test_dangerous_tool_detected(self):
        """A tool with PermissionLevel.DANGEROUS should be detected."""
        from agentic_cli.tools.registry import ToolDefinition, PermissionLevel, ToolRegistry

        mock_registry = ToolRegistry()
        mock_registry.register(
            lambda: None,
            name="risky_tool",
            permission_level=PermissionLevel.DANGEROUS,
        )

        with patch("agentic_cli.workflow.confirmation.get_registry", return_value=mock_registry):
            result = is_dangerous("risky_tool")
            assert result is True

    def test_safe_tool_not_dangerous(self):
        """A tool with PermissionLevel.SAFE should not be dangerous."""
        from agentic_cli.tools.registry import ToolDefinition, PermissionLevel, ToolRegistry

        mock_registry = ToolRegistry()
        mock_registry.register(
            lambda: None,
            name="safe_tool",
            permission_level=PermissionLevel.SAFE,
        )

        with patch("agentic_cli.workflow.confirmation.get_registry", return_value=mock_registry):
            result = is_dangerous("safe_tool")
            assert result is False

    def test_unknown_tool_not_dangerous(self):
        """An unregistered tool should not be dangerous."""
        from agentic_cli.tools.registry import ToolRegistry

        mock_registry = ToolRegistry()

        with patch("agentic_cli.workflow.confirmation.get_registry", return_value=mock_registry):
            result = is_dangerous("unknown_tool")
            assert result is False


class TestConfirmationPlugin:
    """Tests for the ConfirmationPlugin."""

    @pytest.fixture
    def plugin(self):
        return ConfirmationPlugin()

    async def test_safe_tool_passes_through(self, plugin):
        """Safe tools should not be intercepted (returns None)."""
        tool = _make_mock_tool("safe_tool")
        tool_context = _make_mock_tool_context()

        with patch("agentic_cli.workflow.adk.plugins.is_dangerous", return_value=False):
            result = await plugin.before_tool_callback(
                tool=tool,
                tool_args={"query": "hello"},
                tool_context=tool_context,
            )

        assert result is None

    async def test_dangerous_tool_approved(self, plugin):
        """DANGEROUS tool should pass through when user approves."""
        tool = _make_mock_tool("dangerous_tool")
        tool_context = _make_mock_tool_context()

        mock_workflow = AsyncMock()
        mock_workflow.request_user_input = AsyncMock(return_value="yes")

        with (
            patch("agentic_cli.workflow.adk.plugins.is_dangerous", return_value=True),
            patch("agentic_cli.workflow.confirmation.get_service", return_value=mock_workflow),
        ):
            result = await plugin.before_tool_callback(
                tool=tool,
                tool_args={"path": "/tmp/file"},
                tool_context=tool_context,
            )

        assert result is None
        mock_workflow.request_user_input.assert_called_once()
        # Verify the request was a CONFIRM type
        request = mock_workflow.request_user_input.call_args[0][0]
        assert isinstance(request, UserInputRequest)
        assert request.tool_name == "dangerous_tool"
        assert request.input_type.value == "confirm"

    async def test_dangerous_tool_denied(self, plugin):
        """DANGEROUS tool should return error dict when user denies."""
        tool = _make_mock_tool("dangerous_tool")
        tool_context = _make_mock_tool_context()

        mock_workflow = AsyncMock()
        mock_workflow.request_user_input = AsyncMock(return_value="no")

        with (
            patch("agentic_cli.workflow.adk.plugins.is_dangerous", return_value=True),
            patch("agentic_cli.workflow.confirmation.get_service", return_value=mock_workflow),
        ):
            result = await plugin.before_tool_callback(
                tool=tool,
                tool_args={"cmd": "rm -rf /"},
                tool_context=tool_context,
            )

        assert result is not None
        assert result["success"] is False
        assert "denied" in result["error"].lower()
        assert "dangerous_tool" in result["error"]

    async def test_no_workflow_passes_through(self, plugin):
        """When no workflow is available, tool should proceed (returns None)."""
        tool = _make_mock_tool("dangerous_tool")
        tool_context = _make_mock_tool_context()

        with (
            patch("agentic_cli.workflow.adk.plugins.is_dangerous", return_value=True),
            patch("agentic_cli.workflow.confirmation.get_service", return_value=None),
        ):
            result = await plugin.before_tool_callback(
                tool=tool,
                tool_args={"cmd": "ls"},
                tool_context=tool_context,
            )

        assert result is None

    async def test_no_callback_passes_through(self, plugin):
        """When workflow has no callback, tool should proceed (returns None)."""
        tool = _make_mock_tool("dangerous_tool")
        tool_context = _make_mock_tool_context()

        mock_workflow = AsyncMock()
        mock_workflow.request_user_input = AsyncMock(
            side_effect=RuntimeError("No callback registered")
        )

        with (
            patch("agentic_cli.workflow.adk.plugins.is_dangerous", return_value=True),
            patch("agentic_cli.workflow.confirmation.get_service", return_value=mock_workflow),
        ):
            result = await plugin.before_tool_callback(
                tool=tool,
                tool_args={"cmd": "ls"},
                tool_context=tool_context,
            )

        assert result is None

    async def test_various_approval_responses(self, plugin):
        """Multiple response strings should be treated as approval."""
        tool = _make_mock_tool("dangerous_tool")
        tool_context = _make_mock_tool_context()

        for response in ["yes", "y", "YES", "  Yes  ", "approve", "true"]:
            mock_workflow = AsyncMock()
            mock_workflow.request_user_input = AsyncMock(return_value=response)

            with (
                patch("agentic_cli.workflow.adk.plugins.is_dangerous", return_value=True),
                patch("agentic_cli.workflow.confirmation.get_service", return_value=mock_workflow),
            ):
                result = await plugin.before_tool_callback(
                    tool=tool,
                    tool_args={},
                    tool_context=tool_context,
                )

            assert result is None, f"Expected approval for response {response!r}"

    async def test_arg_summary_truncation(self, plugin):
        """Tool args should be summarized and truncated in the prompt."""
        tool = _make_mock_tool("dangerous_tool")
        tool_context = _make_mock_tool_context()

        mock_workflow = AsyncMock()
        mock_workflow.request_user_input = AsyncMock(return_value="yes")

        long_args = {
            "arg1": "x" * 100,
            "arg2": "short",
            "arg3": "included",
            "arg4": "excluded_beyond_3",
        }

        with (
            patch("agentic_cli.workflow.adk.plugins.is_dangerous", return_value=True),
            patch("agentic_cli.workflow.confirmation.get_service", return_value=mock_workflow),
        ):
            result = await plugin.before_tool_callback(
                tool=tool,
                tool_args=long_args,
                tool_context=tool_context,
            )

        assert result is None
        request = mock_workflow.request_user_input.call_args[0][0]
        # arg4 should not appear (only first 3 args included)
        assert "arg4" not in request.prompt
        assert "arg1" in request.prompt

    def test_plugin_name(self, plugin):
        """Plugin should have the correct name."""
        assert plugin.name == "confirmation"


# -------------------------------------------------------------------------
# LLMLoggingPlugin tests
# -------------------------------------------------------------------------


def _make_mock_callback_context(
    invocation_id: str = "inv-123",
    agent_name: str = "test_agent",
) -> MagicMock:
    """Create a mock CallbackContext with invocation_id and agent_name."""
    ctx = MagicMock()
    ctx.invocation_id = invocation_id
    ctx.agent_name = agent_name
    return ctx


def _make_mock_llm_request(
    model: str = "gemini-2.0-flash",
    contents: list | None = None,
    tools_dict: dict | None = None,
    config: MagicMock | None = None,
) -> MagicMock:
    """Create a mock LlmRequest."""
    req = MagicMock()
    req.model = model
    req.contents = contents
    req.tools_dict = tools_dict
    if config is None:
        req.config = None
    else:
        req.config = config
    return req


def _make_mock_llm_response(
    text: str = "Hello world",
    finish_reason: str | None = "STOP",
    model_version: str | None = "gemini-2.0-flash-001",
    error_code: str | None = None,
    error_message: str | None = None,
    usage_metadata: MagicMock | None = None,
) -> MagicMock:
    """Create a mock LlmResponse."""
    resp = MagicMock()

    # Build content with parts
    part = MagicMock()
    part.text = text
    part.thought = False
    part.function_call = None
    part.function_response = None
    part.executable_code = None
    part.code_execution_result = None

    content = MagicMock()
    content.parts = [part]
    resp.content = content

    resp.finish_reason = finish_reason
    resp.model_version = model_version
    resp.error_code = error_code
    resp.error_message = error_message
    resp.usage_metadata = usage_metadata

    return resp


class TestLLMLoggingPlugin:
    """Tests for the LLMLoggingPlugin."""

    @pytest.fixture
    def plugin(self, tmp_path, monkeypatch):
        """Create a plugin with log directory in tmp_path."""
        monkeypatch.chdir(tmp_path)
        return LLMLoggingPlugin(
            model_name="test-model",
            app_name="test_app",
        )

    def test_plugin_name(self, plugin):
        """Plugin should have the correct name."""
        assert plugin.name == "llm_logging"

    def test_log_file_created(self, plugin, tmp_path):
        """Log directory and file path should be set up."""
        expected_dir = tmp_path / ".test_app" / "logs"
        assert expected_dir.exists()
        assert plugin.get_log_file_path() == expected_dir / "llm_events.jsonl"

    async def test_before_model_captures_request(self, plugin):
        """before_model_callback should capture request data and buffer an event."""
        ctx = _make_mock_callback_context(invocation_id="inv-001")
        req = _make_mock_llm_request(model="gemini-2.0-flash")

        result = await plugin.before_model_callback(
            callback_context=ctx,
            llm_request=req,
        )

        assert result is None  # Should not short-circuit

        events = plugin.get_events()
        assert len(events) == 1
        assert events[0].type == EventType.LLM_REQUEST
        assert events[0].metadata["model"] == "gemini-2.0-flash"
        assert events[0].metadata["invocation_id"] == "inv-001"

    async def test_after_model_captures_response(self, plugin):
        """after_model_callback should capture response data and buffer events."""
        ctx = _make_mock_callback_context(invocation_id="inv-002", agent_name="coordinator")
        resp = _make_mock_llm_response(text="Test response")

        result = await plugin.after_model_callback(
            callback_context=ctx,
            llm_response=resp,
        )

        assert result is None  # Should not short-circuit

        events = plugin.get_events()
        assert len(events) == 1  # response event only (no usage)
        assert events[0].type == EventType.LLM_RESPONSE
        assert events[0].metadata["model"] == "test-model"
        assert events[0].metadata["author"] == "coordinator"
        assert events[0].metadata["invocation_id"] == "inv-002"

    async def test_after_model_captures_usage(self, plugin):
        """Usage metadata should produce an LLM_USAGE event."""
        ctx = _make_mock_callback_context(invocation_id="inv-003")

        usage = MagicMock()
        usage.prompt_token_count = 100
        usage.candidates_token_count = 50
        usage.total_token_count = 150
        usage.thoughts_token_count = None
        usage.cached_content_token_count = 10

        resp = _make_mock_llm_response(usage_metadata=usage)

        await plugin.after_model_callback(
            callback_context=ctx,
            llm_response=resp,
        )

        events = plugin.get_events()
        assert len(events) == 2  # response + usage
        usage_event = events[1]
        assert usage_event.type == EventType.LLM_USAGE
        assert usage_event.metadata["prompt_tokens"] == 100
        assert usage_event.metadata["completion_tokens"] == 50
        assert usage_event.metadata["total_tokens"] == 150
        assert usage_event.metadata["cached_tokens"] == 10

    async def test_latency_tracking(self, plugin):
        """Latency should be calculated between before and after callbacks."""
        ctx = _make_mock_callback_context(invocation_id="inv-lat")
        req = _make_mock_llm_request()

        usage = MagicMock()
        usage.prompt_token_count = 10
        usage.candidates_token_count = 5
        usage.total_token_count = 15
        usage.thoughts_token_count = None
        usage.cached_content_token_count = None

        resp = _make_mock_llm_response(usage_metadata=usage)

        await plugin.before_model_callback(
            callback_context=ctx,
            llm_request=req,
        )
        await plugin.after_model_callback(
            callback_context=ctx,
            llm_response=resp,
        )

        events = plugin.get_events()
        # request + response + usage = 3 events
        assert len(events) == 3
        usage_event = events[2]
        assert usage_event.type == EventType.LLM_USAGE
        assert usage_event.metadata["latency_ms"] is not None
        assert usage_event.metadata["latency_ms"] >= 0

    async def test_drain_events_clears_buffer(self, plugin):
        """drain_events() should return events and clear the buffer."""
        ctx = _make_mock_callback_context()
        req = _make_mock_llm_request()

        await plugin.before_model_callback(
            callback_context=ctx,
            llm_request=req,
        )

        drained = plugin.drain_events()
        assert len(drained) == 1

        # Buffer should be empty now
        assert plugin.get_events() == []
        assert plugin.drain_events() == []

    def test_clear_resets_state(self, plugin):
        """clear() should reset events and timestamps."""
        plugin._events = [MagicMock()]
        plugin._request_timestamps = {"inv-1": 12345.0}

        plugin.clear()

        assert plugin._events == []
        assert plugin._request_timestamps == {}

    async def test_max_events_enforced(self, tmp_path, monkeypatch):
        """Buffer should not exceed max_events."""
        monkeypatch.chdir(tmp_path)
        plugin = LLMLoggingPlugin(
            model_name="test-model",
            app_name="test_app",
            max_events=3,
        )

        for i in range(5):
            ctx = _make_mock_callback_context(invocation_id=f"inv-{i}")
            req = _make_mock_llm_request()
            await plugin.before_model_callback(
                callback_context=ctx,
                llm_request=req,
            )

        events = plugin.get_events()
        assert len(events) == 3
        # Should have the last 3 events (inv-2, inv-3, inv-4)
        assert events[0].metadata["invocation_id"] == "inv-2"
        assert events[2].metadata["invocation_id"] == "inv-4"

    async def test_writes_to_log_file(self, plugin):
        """Events should be written to the JSONL log file."""
        ctx = _make_mock_callback_context(invocation_id="inv-log")
        req = _make_mock_llm_request(model="gemini-2.0-flash")

        await plugin.before_model_callback(
            callback_context=ctx,
            llm_request=req,
        )

        log_path = plugin.get_log_file_path()
        assert log_path.exists()

        with open(log_path) as f:
            lines = f.readlines()
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["type"] == "llm_request"
        assert record["metadata"]["model"] == "gemini-2.0-flash"

    async def test_before_model_extracts_tools(self, plugin):
        """Tool names should be extracted from llm_request.tools_dict."""
        ctx = _make_mock_callback_context()
        req = _make_mock_llm_request(
            tools_dict={"search_web": MagicMock(), "read_file": MagicMock()},
        )

        await plugin.before_model_callback(
            callback_context=ctx,
            llm_request=req,
        )

        events = plugin.get_events()
        assert events[0].metadata["tools"] == ["search_web", "read_file"]

    async def test_before_model_extracts_system_instruction_string(self, plugin):
        """String system instructions should be captured."""
        ctx = _make_mock_callback_context()
        config = MagicMock()
        config.system_instruction = "You are a helpful assistant."
        config.temperature = None
        config.top_p = None
        config.top_k = None
        config.max_output_tokens = None
        config.stop_sequences = None
        req = _make_mock_llm_request(config=config)

        await plugin.before_model_callback(
            callback_context=ctx,
            llm_request=req,
        )

        events = plugin.get_events()
        assert events[0].metadata["system_instruction"] == "You are a helpful assistant."

    async def test_before_model_extracts_config(self, plugin):
        """Generation config fields should be captured."""
        ctx = _make_mock_callback_context()
        config = MagicMock()
        config.system_instruction = None
        config.temperature = 0.7
        config.top_p = 0.9
        config.top_k = None
        config.max_output_tokens = 2048
        config.stop_sequences = None
        req = _make_mock_llm_request(config=config)

        await plugin.before_model_callback(
            callback_context=ctx,
            llm_request=req,
        )

        events = plugin.get_events()
        gen_config = events[0].metadata["config"]
        assert gen_config["temperature"] == 0.7
        assert gen_config["top_p"] == 0.9
        assert gen_config["max_output_tokens"] == 2048

    async def test_after_model_captures_error(self, plugin):
        """Error info from the response should be captured."""
        ctx = _make_mock_callback_context()
        resp = _make_mock_llm_response(
            error_code="RATE_LIMIT",
            error_message="Too many requests",
        )

        await plugin.after_model_callback(
            callback_context=ctx,
            llm_response=resp,
        )

        events = plugin.get_events()
        assert events[0].metadata["error_code"] == "RATE_LIMIT"
        assert events[0].metadata["error_message"] == "Too many requests"

    async def test_serialize_parts_function_call(self, plugin):
        """Function call parts should be serialized correctly."""
        fc = MagicMock()
        fc.name = "search_web"
        fc.args = {"query": "test"}

        part = MagicMock()
        part.text = None
        part.thought = False
        part.function_call = fc
        part.function_response = None
        part.executable_code = None
        part.code_execution_result = None

        result = plugin._serialize_parts([part])
        assert len(result) == 1
        assert result[0]["function_call"]["name"] == "search_web"
        assert result[0]["function_call"]["args"] == {"query": "test"}

    async def test_serialize_parts_thinking(self, plugin):
        """Thought parts should include thought=True marker."""
        part = MagicMock()
        part.text = "Let me think..."
        part.thought = True
        part.function_call = None
        part.function_response = None
        part.executable_code = None
        part.code_execution_result = None

        result = plugin._serialize_parts([part])
        assert len(result) == 1
        assert result[0]["text"] == "Let me think..."
        assert result[0]["thought"] is True

    async def test_before_model_no_invocation_id(self, plugin):
        """Plugin should handle missing invocation_id gracefully."""
        ctx = MagicMock(spec=[])  # No attributes at all
        req = _make_mock_llm_request()

        await plugin.before_model_callback(
            callback_context=ctx,
            llm_request=req,
        )

        events = plugin.get_events()
        assert len(events) == 1
        # invocation_id should not be in metadata (None filtered out)
        assert "invocation_id" not in events[0].metadata

    async def test_serialize_contents(self, plugin):
        """Contents with roles and parts should be serialized."""
        part = MagicMock()
        part.text = "Hello"
        part.thought = False
        part.function_call = None
        part.function_response = None
        part.executable_code = None
        part.code_execution_result = None

        content = MagicMock()
        content.role = "user"
        content.parts = [part]

        result = plugin._serialize_contents([content])
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["parts"][0]["text"] == "Hello"
