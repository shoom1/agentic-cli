"""ADK plugin: a tool that raises fails its call, not the turn.

ADK re-raises an exception from a tool (and a call to a tool name that does not
exist) unless an ``on_tool_error_callback`` returns a response for it, which
ends the whole turn. Framework tools return ``{"success": False, ...}`` on
failure; this plugin gives anything that still raises (a bug, or an
application's own tool) the same shape, so the model reads the error and can
recover. ``asyncio.CancelledError`` is not an ``Exception`` and never reaches
the callback, so cancelling a turn still cancels it.

LangGraph needs no equivalent: its ``ToolNode`` runs with
``handle_tool_errors=True``.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from google.adk.plugins.base_plugin import BasePlugin

from agentic_cli.constants import truncate
from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from google.adk.tools import BaseTool
    from google.adk.tools.tool_context import ToolContext

logger = Loggers.workflow()

_MAX_MESSAGE = 500


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
        logger.warning(
            "tool_raised",
            tool=tool.name,
            error_type=type(error).__name__,
            error=truncate(str(error), _MAX_MESSAGE),
            exc_info=error,
        )
        return {
            "success": False,
            "error": truncate(
                f"{tool.name} failed: {type(error).__name__}: {error}", _MAX_MESSAGE
            ),
        }
