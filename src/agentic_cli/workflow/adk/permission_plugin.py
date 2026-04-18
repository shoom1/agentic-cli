"""ADK plugin that gates tool calls via PermissionEngine.

Adapter check order (mirrors LangGraph wrapper for consistency):
1. EXEMPT tool → allow, no engine call.
2. Tool has no capability declaration → deny (author error, loud).
3. Engine absent from service registry → allow (test/dev fallback).
4. Otherwise call engine.check() and return None on allow, error dict on deny.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from google.adk.plugins.base_plugin import BasePlugin

from agentic_cli.logging import Loggers
from agentic_cli.tools.registry import get_registry
from agentic_cli.workflow.permissions.capabilities import _CapabilityExempt
from agentic_cli.workflow.service_registry import PERMISSION_ENGINE, get_service

if TYPE_CHECKING:
    from google.adk.tools import BaseTool
    from google.adk.tools.tool_context import ToolContext

logger = Loggers.workflow()


class PermissionPlugin(BasePlugin):
    """ADK plugin: gates every tool call through :class:`PermissionEngine`."""

    def __init__(self) -> None:
        super().__init__(name="permission")

    async def before_tool_callback(
        self,
        *,
        tool: "BaseTool",
        tool_args: dict[str, Any],
        tool_context: "ToolContext | None",
    ) -> dict | None:
        defn = get_registry().get(tool.name)
        caps = defn.capabilities if defn else None

        if isinstance(caps, _CapabilityExempt):
            return None
        if not caps:
            logger.warning("permission_undeclared", tool=tool.name)
            return {
                "success": False,
                "error": "Permission denied: tool has no capability declaration",
            }

        engine = get_service(PERMISSION_ENGINE)
        if engine is None:
            return None  # test/dev fallback

        result = await engine.check(tool.name, caps, tool_args)
        if result.allowed:
            return None
        return {"success": False, "error": f"Permission denied: {result.reason}"}
