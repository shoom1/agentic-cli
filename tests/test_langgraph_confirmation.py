"""Tests for LangGraph confirmation wrapper."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_cli.workflow.langgraph.graph_builder import LangGraphBuilder


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


# Patch targets — both imported into graph_builder from plugins
_IS_DANGEROUS = "agentic_cli.workflow.langgraph.graph_builder.is_dangerous"
_CONFIRM = "agentic_cli.workflow.langgraph.graph_builder.request_tool_confirmation"


class TestWrapForConfirmation:
    """Tests for LangGraphBuilder._wrap_for_confirmation."""

    def test_safe_tool_passes_through(self):
        """Safe tools should be returned unwrapped."""
        tool = _make_sync_tool("safe_tool")
        with patch(_IS_DANGEROUS, return_value=False):
            result = LangGraphBuilder._wrap_for_confirmation(tool)
        assert result is tool

    def test_dangerous_tool_is_wrapped(self):
        """DANGEROUS tools should be wrapped (different function object)."""
        tool = _make_sync_tool("dangerous_tool")
        with patch(_IS_DANGEROUS, return_value=True):
            result = LangGraphBuilder._wrap_for_confirmation(tool)
        assert result is not tool
        assert result.__name__ == "dangerous_tool"

    async def test_dangerous_sync_tool_approved(self):
        """Approved DANGEROUS sync tool should execute and return result."""
        tool = _make_sync_tool("dangerous_tool")
        with patch(_IS_DANGEROUS, return_value=True):
            wrapped = LangGraphBuilder._wrap_for_confirmation(tool)

        with patch(_CONFIRM, return_value=True):
            result = await wrapped(x="hello")
        assert result == {"success": True, "result": "hello"}

    async def test_dangerous_async_tool_approved(self):
        """Approved DANGEROUS async tool should execute and return result."""
        tool = _make_async_tool("dangerous_async")
        with patch(_IS_DANGEROUS, return_value=True):
            wrapped = LangGraphBuilder._wrap_for_confirmation(tool)

        with patch(_CONFIRM, return_value=True):
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

        with patch(_IS_DANGEROUS, return_value=True):
            wrapped = LangGraphBuilder._wrap_for_confirmation(tool_fn)

        with patch(_CONFIRM, return_value=False):
            result = await wrapped(x="test")

        assert result["success"] is False
        assert "denied" in result["error"].lower()
        assert call_count == 0

    async def test_no_workflow_allows_execution(self):
        """When request_tool_confirmation returns None, tool executes."""
        tool = _make_sync_tool("dangerous_tool")
        with patch(_IS_DANGEROUS, return_value=True):
            wrapped = LangGraphBuilder._wrap_for_confirmation(tool)

        with patch(_CONFIRM, return_value=None):
            result = await wrapped(x="test")
        assert result == {"success": True, "result": "test"}

    async def test_no_callback_allows_execution(self):
        """When request_tool_confirmation returns None (no callback), tool executes."""
        tool = _make_sync_tool("dangerous_tool")
        with patch(_IS_DANGEROUS, return_value=True):
            wrapped = LangGraphBuilder._wrap_for_confirmation(tool)

        with patch(_CONFIRM, return_value=None):
            result = await wrapped(x="test")
        assert result == {"success": True, "result": "test"}

    def test_preserves_context_guard_attribute(self):
        """Wrapped function should preserve the '_context_guard' attribute."""
        tool = _make_sync_tool("dangerous_tool")
        tool._context_guard = "test_guard"
        with patch(_IS_DANGEROUS, return_value=True):
            wrapped = LangGraphBuilder._wrap_for_confirmation(tool)
        assert wrapped._context_guard == "test_guard"

    def test_preserves_docstring(self):
        """Wrapped function should preserve the docstring."""
        tool = _make_sync_tool("dangerous_tool")
        with patch(_IS_DANGEROUS, return_value=True):
            wrapped = LangGraphBuilder._wrap_for_confirmation(tool)
        assert wrapped.__doc__ == tool.__doc__

    def test_preserves_annotations(self):
        """Wrapped function should preserve type annotations."""
        tool = _make_sync_tool("dangerous_tool")
        with patch(_IS_DANGEROUS, return_value=True):
            wrapped = LangGraphBuilder._wrap_for_confirmation(tool)
        assert wrapped.__annotations__ == tool.__annotations__

    async def test_confirmation_called_with_tool_args(self):
        """request_tool_confirmation should receive tool name and kwargs."""
        tool = _make_sync_tool("dangerous_tool")
        with patch(_IS_DANGEROUS, return_value=True):
            wrapped = LangGraphBuilder._wrap_for_confirmation(tool)

        mock_confirm = AsyncMock(return_value=True)
        with patch(_CONFIRM, mock_confirm):
            await wrapped(x="hello")

        mock_confirm.assert_called_once_with("dangerous_tool", {"x": "hello"})
