"""Tests for ADK ConfirmationPlugin."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_cli.workflow.adk.plugins import (
    ConfirmationPlugin, is_dangerous, reset_dangerous_cache,
)
from agentic_cli.workflow.events import UserInputRequest


@pytest.fixture(autouse=True)
def _clear_dangerous_cache():
    """Reset the module-level dangerous cache between tests."""
    reset_dangerous_cache()
    yield
    reset_dangerous_cache()


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

        with patch("agentic_cli.workflow.adk.plugins.get_registry", return_value=mock_registry):
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

        with patch("agentic_cli.workflow.adk.plugins.get_registry", return_value=mock_registry):
            result = is_dangerous("safe_tool")
            assert result is False

    def test_unknown_tool_not_dangerous(self):
        """An unregistered tool should not be dangerous."""
        from agentic_cli.tools.registry import ToolRegistry

        mock_registry = ToolRegistry()

        with patch("agentic_cli.workflow.adk.plugins.get_registry", return_value=mock_registry):
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
            patch("agentic_cli.workflow.adk.plugins.get_context_workflow", return_value=mock_workflow),
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
            patch("agentic_cli.workflow.adk.plugins.get_context_workflow", return_value=mock_workflow),
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
            patch("agentic_cli.workflow.adk.plugins.get_context_workflow", return_value=None),
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
            patch("agentic_cli.workflow.adk.plugins.get_context_workflow", return_value=mock_workflow),
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
                patch("agentic_cli.workflow.adk.plugins.get_context_workflow", return_value=mock_workflow),
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
            patch("agentic_cli.workflow.adk.plugins.get_context_workflow", return_value=mock_workflow),
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
