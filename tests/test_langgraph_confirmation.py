"""Tests for LangGraph confirmation wrapper."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_cli.workflow.langgraph.graph_builder import LangGraphBuilder
from agentic_cli.workflow.events import UserInputRequest


@pytest.fixture(autouse=True)
def clear_dangerous_cache():
    """Reset the module-level dangerous cache between tests."""
    import agentic_cli.workflow.adk.plugins as plugins_mod
    plugins_mod._dangerous_cache = None
    yield
    plugins_mod._dangerous_cache = None


def _make_sync_tool(name: str = "my_tool"):
    """Create a simple sync tool function."""
    def tool_fn(x: str) -> dict:
        """A test tool."""
        return {"success": True, "result": x}
    tool_fn.__name__ = name
    return tool_fn


def _make_async_tool(name: str = "my_async_tool"):
    """Create a simple async tool function."""
    async def tool_fn(x: str) -> dict:
        """An async test tool."""
        return {"success": True, "result": x}
    tool_fn.__name__ = name
    return tool_fn


class TestWrapForConfirmation:
    """Tests for LangGraphBuilder._wrap_for_confirmation."""

    def test_safe_tool_passes_through(self):
        """Safe tools should be returned unwrapped."""
        tool = _make_sync_tool("safe_tool")

        with patch("agentic_cli.workflow.langgraph.graph_builder.is_dangerous", return_value=False):
            result = LangGraphBuilder._wrap_for_confirmation(tool)

        assert result is tool  # Same object, not wrapped

    def test_dangerous_tool_is_wrapped(self):
        """DANGEROUS tools should be wrapped (different function object)."""
        tool = _make_sync_tool("dangerous_tool")

        with patch("agentic_cli.workflow.langgraph.graph_builder.is_dangerous", return_value=True):
            result = LangGraphBuilder._wrap_for_confirmation(tool)

        assert result is not tool
        assert result.__name__ == "dangerous_tool"  # Name preserved

    async def test_dangerous_sync_tool_approved(self):
        """Approved DANGEROUS sync tool should execute and return result."""
        tool = _make_sync_tool("dangerous_tool")

        mock_workflow = AsyncMock()
        mock_workflow.request_user_input = AsyncMock(return_value="yes")

        with (
            patch("agentic_cli.workflow.langgraph.graph_builder.is_dangerous", return_value=True),
            patch("agentic_cli.workflow.langgraph.graph_builder.get_context_workflow", return_value=mock_workflow),
        ):
            wrapped = LangGraphBuilder._wrap_for_confirmation(tool)

        # Need to patch get_context_workflow at call time too
        with patch("agentic_cli.workflow.langgraph.graph_builder.get_context_workflow", return_value=mock_workflow):
            result = await wrapped(x="hello")

        assert result == {"success": True, "result": "hello"}
        mock_workflow.request_user_input.assert_called_once()

    async def test_dangerous_async_tool_approved(self):
        """Approved DANGEROUS async tool should execute and return result."""
        tool = _make_async_tool("dangerous_async")

        mock_workflow = AsyncMock()
        mock_workflow.request_user_input = AsyncMock(return_value="y")

        with (
            patch("agentic_cli.workflow.langgraph.graph_builder.is_dangerous", return_value=True),
            patch("agentic_cli.workflow.langgraph.graph_builder.get_context_workflow", return_value=mock_workflow),
        ):
            wrapped = LangGraphBuilder._wrap_for_confirmation(tool)

        with patch("agentic_cli.workflow.langgraph.graph_builder.get_context_workflow", return_value=mock_workflow):
            result = await wrapped(x="world")

        assert result == {"success": True, "result": "world"}

    async def test_dangerous_tool_denied(self):
        """Denied DANGEROUS tool should return error dict without executing."""
        call_count = 0
        def tool_fn(x: str) -> dict:
            """A test tool."""
            nonlocal call_count
            call_count += 1
            return {"success": True}
        tool_fn.__name__ = "dangerous_tool"

        mock_workflow = AsyncMock()
        mock_workflow.request_user_input = AsyncMock(return_value="no")

        with (
            patch("agentic_cli.workflow.langgraph.graph_builder.is_dangerous", return_value=True),
            patch("agentic_cli.workflow.langgraph.graph_builder.get_context_workflow", return_value=mock_workflow),
        ):
            wrapped = LangGraphBuilder._wrap_for_confirmation(tool_fn)

        with patch("agentic_cli.workflow.langgraph.graph_builder.get_context_workflow", return_value=mock_workflow):
            result = await wrapped(x="test")

        assert result["success"] is False
        assert "denied" in result["error"].lower()
        assert call_count == 0  # Original never called

    async def test_no_workflow_allows_execution(self):
        """When no workflow is available, tool should execute normally."""
        tool = _make_sync_tool("dangerous_tool")

        with (
            patch("agentic_cli.workflow.langgraph.graph_builder.is_dangerous", return_value=True),
            patch("agentic_cli.workflow.langgraph.graph_builder.get_context_workflow", return_value=None),
        ):
            wrapped = LangGraphBuilder._wrap_for_confirmation(tool)

        with patch("agentic_cli.workflow.langgraph.graph_builder.get_context_workflow", return_value=None):
            result = await wrapped(x="test")

        assert result == {"success": True, "result": "test"}

    async def test_no_callback_allows_execution(self):
        """When workflow has no callback, tool should execute normally."""
        tool = _make_sync_tool("dangerous_tool")

        mock_workflow = AsyncMock()
        mock_workflow.request_user_input = AsyncMock(
            side_effect=RuntimeError("No callback")
        )

        with (
            patch("agentic_cli.workflow.langgraph.graph_builder.is_dangerous", return_value=True),
            patch("agentic_cli.workflow.langgraph.graph_builder.get_context_workflow", return_value=mock_workflow),
        ):
            wrapped = LangGraphBuilder._wrap_for_confirmation(tool)

        with patch("agentic_cli.workflow.langgraph.graph_builder.get_context_workflow", return_value=mock_workflow):
            result = await wrapped(x="test")

        assert result == {"success": True, "result": "test"}

    def test_preserves_requires_attribute(self):
        """Wrapped function should preserve the 'requires' attribute."""
        tool = _make_sync_tool("dangerous_tool")
        tool.requires = ["sandbox_manager"]

        with patch("agentic_cli.workflow.langgraph.graph_builder.is_dangerous", return_value=True):
            wrapped = LangGraphBuilder._wrap_for_confirmation(tool)

        assert hasattr(wrapped, "requires")
        assert wrapped.requires == ["sandbox_manager"]

    def test_preserves_docstring(self):
        """Wrapped function should preserve the docstring."""
        tool = _make_sync_tool("dangerous_tool")

        with patch("agentic_cli.workflow.langgraph.graph_builder.is_dangerous", return_value=True):
            wrapped = LangGraphBuilder._wrap_for_confirmation(tool)

        assert wrapped.__doc__ == tool.__doc__

    def test_preserves_annotations(self):
        """Wrapped function should preserve type annotations."""
        tool = _make_sync_tool("dangerous_tool")

        with patch("agentic_cli.workflow.langgraph.graph_builder.is_dangerous", return_value=True):
            wrapped = LangGraphBuilder._wrap_for_confirmation(tool)

        assert wrapped.__annotations__ == tool.__annotations__

    async def test_prompt_includes_tool_name_and_args(self):
        """The confirmation prompt should include tool name and arg summary."""
        tool = _make_sync_tool("dangerous_tool")

        mock_workflow = AsyncMock()
        mock_workflow.request_user_input = AsyncMock(return_value="yes")

        with (
            patch("agentic_cli.workflow.langgraph.graph_builder.is_dangerous", return_value=True),
            patch("agentic_cli.workflow.langgraph.graph_builder.get_context_workflow", return_value=mock_workflow),
        ):
            wrapped = LangGraphBuilder._wrap_for_confirmation(tool)

        with patch("agentic_cli.workflow.langgraph.graph_builder.get_context_workflow", return_value=mock_workflow):
            await wrapped(x="hello")

        request = mock_workflow.request_user_input.call_args[0][0]
        assert isinstance(request, UserInputRequest)
        assert "dangerous_tool" in request.prompt
        assert request.tool_name == "dangerous_tool"
