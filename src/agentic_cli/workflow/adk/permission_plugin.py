"""ADK plugin that gates tool calls via PermissionEngine.

Adapter check order (mirrors LangGraph wrapper for consistency):
1. EXEMPT tool → allow, no engine call.
2. Unregistered MCP toolset tool → gate through the engine under a synthetic
   ``mcp`` capability (no rule → ASK).
3. Tool has no capability declaration → deny (author error, loud).
4. Engine absent from service registry → allow (test/dev fallback).
5. Otherwise call engine.check() and return None on allow, error dict on deny.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from google.adk.plugins.base_plugin import BasePlugin

from agentic_cli.logging import Loggers
from agentic_cli.tools.registry import ToolCategory, get_registry, register_tool
from agentic_cli.workflow.permissions import EXEMPT
from agentic_cli.workflow.permissions.capabilities import Capability, _CapabilityExempt
from agentic_cli.workflow.service_registry import PERMISSION_ENGINE, get_service

if TYPE_CHECKING:
    from google.adk.tools import BaseTool
    from google.adk.tools.tool_context import ToolContext

logger = Loggers.workflow()


# ADK auto-injects ``transfer_to_agent`` into coordinator agents that declare
# ``sub_agents``. It's an internal routing primitive, not an external side
# effect, so register it with EXEMPT so the plugin lets it through.
try:
    from google.adk.tools.transfer_to_agent_tool import transfer_to_agent

    register_tool(capabilities=EXEMPT, category=ToolCategory.PLANNING)(transfer_to_agent)
except ImportError:
    pass


def _is_mcp_tool(tool: "BaseTool") -> bool:
    """True if ``tool`` is an ADK MCP toolset tool (not in our registry)."""
    try:
        # McpTool is the base class; MCPTool is a deprecated subclass, so an
        # isinstance check against McpTool catches both.
        from google.adk.tools.mcp_tool import McpTool

        if isinstance(tool, McpTool):
            return True
    except Exception:
        pass
    return type(tool).__name__ in {"MCPTool", "McpTool"}


# Synthetic capability for MCP tools; target is the MCP tool name. With no
# matching rule the engine asks the user (default ASK). Allow/deny rules with
# capability ``mcp`` and a tool-name glob target govern MCP access.
_MCP_TARGET_ARG = "__mcp_target__"


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
            # MCP toolset tools aren't registered; gate them through the engine
            # under a synthetic 'mcp' capability (no rule → ASK).
            if _is_mcp_tool(tool):
                return await self._check_mcp(tool)
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

    async def _check_mcp(self, tool: "BaseTool") -> dict | None:
        """Gate an MCP tool through the engine under a synthetic capability."""
        engine = get_service(PERMISSION_ENGINE)
        if engine is None:
            return None  # test/dev fallback
        caps = [Capability("mcp", target_arg=_MCP_TARGET_ARG)]
        result = await engine.check(tool.name, caps, {_MCP_TARGET_ARG: tool.name})
        if result.allowed:
            return None
        return {"success": False, "error": f"Permission denied: {result.reason}"}
