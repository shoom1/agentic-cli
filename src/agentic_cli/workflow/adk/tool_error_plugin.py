"""ADK plugin: a tool that raises fails its call, not the turn.

ADK re-raises an exception from a tool unless an ``on_tool_error_callback``
returns a response for it, which ends the whole turn. Framework tools return
``{"success": False, ...}`` on failure; this plugin gives anything that still
raises (a bug, or an application's own tool) the same shape, so the model
reads the error and can recover. ``asyncio.CancelledError`` is not an
``Exception`` and never reaches the callback, so cancelling a turn still
cancels it.

A call to a tool name the agent does not have is the model's own mistake. ADK
resolves the name to a stand-in (``is_unknown_tool``), lets the before-tool
callbacks see it (the permission gate lets it through: it runs nothing), and
then reports its lookup ``ValueError`` here. The model reads ADK's first two
lines, the agent's tools by name (``tool_errors.unknown_tool_result``), and
the log says so at info, not as a warning over the prompt.

The results come from ``workflow/tool_errors.py``, which the Deep Agents
backend's middleware uses too, so both answer the model in the same words.
LangGraph needs no equivalent: its ``ToolNode`` runs with
``handle_tool_errors=True``.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools import BaseTool

from agentic_cli.logging import Loggers
from agentic_cli.workflow.tool_errors import tool_error_result, unknown_tool_result

if TYPE_CHECKING:
    from google.adk.tools.tool_context import ToolContext

logger = Loggers.workflow()

#: The description of ADK's stand-in for an unknown tool name (contract-tested).
_NOT_FOUND = "Tool not found"
#: How the second line of ADK's lookup error starts: the agent's tools follow.
_AVAILABLE = "Available tools:"


def is_unknown_tool(tool: Any) -> bool:
    """Whether ``tool`` is ADK's stand-in for a tool name the agent does not have.

    ADK resolves such a call to a bare ``BaseTool`` named after it and
    described "Tool not found", and runs the before-tool callbacks on it
    before it reports the lookup's ``ValueError`` to the tool-error
    callbacks (contract-tested). A bare ``BaseTool`` runs nothing: its
    ``run_async`` raises. Only the exact class counts, never a subclass.
    """
    return type(tool) is BaseTool and getattr(tool, "description", None) == _NOT_FOUND


def _available_tools(error: BaseException) -> list[str] | None:
    """The tool names ADK's lookup error lists, or None when it lists none."""
    lines = str(error).splitlines()
    if len(lines) < 2 or not lines[1].startswith(_AVAILABLE):
        return None
    return [name.strip() for name in lines[1][len(_AVAILABLE):].split(",") if name.strip()]


class ToolErrorPlugin(BasePlugin):
    """Return an error result for a tool call that raised."""

    def __init__(self) -> None:
        super().__init__(name="tool_error")

    async def on_tool_error_callback(
        self,
        *,
        tool: "BaseTool",
        tool_args: dict[str, Any],
        tool_context: "ToolContext | None",
        error: Exception,
    ) -> dict:
        if is_unknown_tool(tool) and isinstance(error, ValueError):
            available = _available_tools(error)
            if available is not None:
                logger.info(
                    "tool_not_found",
                    tool=tool.name,
                    agent=getattr(tool_context, "agent_name", None),
                )
                return unknown_tool_result(tool.name, available)
        return tool_error_result(tool.name, error)
