"""ADK plugin that gates tool calls via PermissionEngine.

Tool identity is *object* identity, resolved through the registry's identity
binding. It is never ``tool.name`` (ADK derives that from the callable, so an
unregistered function named after a registered tool would inherit its
capabilities — an EXEMPT one would be waved straight through), never the tool's
class name, and never equality.

Adapter check order (mirrors LangGraph wrapper for consistency):
1. Resolve the definition bound to this exact object. Framework-issued tools —
   registered callables, service-bound factory variants, and the native ADK
   tool objects the framework constructs (skill tools) — are bound at their
   construction site.
2. Failing that, unwrap ``.func`` only for ADK's own function-tool types
   (:data:`_TRUSTED_FUNCTION_TOOL_TYPES`), whose contract is to invoke exactly
   that callable, and resolve the callable by identity.
3. EXEMPT tool → allow, no engine call.
4. A genuine ``McpTool`` instance (isinstance, not class name) → gate under a
   synthetic ``mcp`` capability (no rule → ASK); its tools are created inside
   ADK when the server connects, so they cannot be bound in advance.
5. Anything still unresolved → deny. That includes an unregistered callable and
   any tool object the framework did not issue, whatever it calls itself.
6. No capability declaration → deny (author error, loud).
7. Engine absent from service registry → fail closed (deny) when permissions
   are enabled; allow only when permissions are disabled.
8. Otherwise call engine.check() and return None on allow, error dict on deny.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools import FunctionTool, LongRunningFunctionTool

from agentic_cli.logging import Loggers
from agentic_cli.tools.registry import (
    ToolCategory,
    identify_tool,
    register_tool,
)
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


def _native_transfer_tool_type() -> type | None:
    """ADK's exact ``TransferToAgentTool`` class, or None if unavailable."""
    try:
        from google.adk.tools.transfer_to_agent_tool import TransferToAgentTool
    except ImportError:  # pragma: no cover - ADK always ships it today
        return None
    return TransferToAgentTool


# ADK's own function-tool types: their documented contract is to call exactly
# ``self.func``, so the callable's identity is the tool's identity. Matched by
# exact type — a subclass may override ``run_async`` and run something else
# while still advertising a genuine ``func``.
#
# ``TransferToAgentTool`` is in the list for the same reason and on the same
# terms. ADK auto-injects it into any agent with ``sub_agents``, and it is a
# ``FunctionTool`` *subclass*, so an exact-type check on ``FunctionTool`` alone
# denied the built-in routing tool as unregistered — delegation could not work
# at all, even with permissions disabled. It is safe to add because ADK
# constructs it as ``super().__init__(func=transfer_to_agent)`` and overrides
# only ``_get_declaration`` (to add the agent-name enum), never ``run_async``:
# what it invokes is still exactly ``self.func``. Listing the exact class keeps
# every other ``FunctionTool`` subclass denied.
_TRUSTED_FUNCTION_TOOL_TYPES = tuple(
    t
    for t in (FunctionTool, LongRunningFunctionTool, _native_transfer_tool_type())
    if t is not None
)


def _trusted_wrapped_callable(tool: "BaseTool") -> Any | None:
    """The callable an ADK function tool will actually invoke, or None.

    Only exact trusted types are unwrapped: ``.func`` on anything else is just
    an attribute, and an attribute is not evidence of what the tool does.
    """
    if type(tool) not in _TRUSTED_FUNCTION_TOOL_TYPES:
        return None
    return getattr(tool, "func", None)


def _is_mcp_tool(tool: "BaseTool") -> bool:
    """True if ``tool`` really is an ADK MCP toolset tool.

    ``isinstance`` against the class ADK ships — never the class *name*, which
    any application can choose. ``McpTool`` is the base class and ``MCPTool`` a
    deprecated subclass, so one check covers both. If the MCP extra is not
    installed nothing can be an MCP tool, and the caller denies.
    """
    try:
        from google.adk.tools.mcp_tool import McpTool
    except Exception:
        return False
    return isinstance(tool, McpTool)


# Synthetic capability for MCP tools; target is the MCP tool name. With no
# matching rule the engine asks the user (default ASK). Allow/deny rules with
# capability ``mcp`` and a tool-name glob target govern MCP access.
_MCP_TARGET_ARG = "__mcp_target__"


class _Sentinel:
    """Resolution outcome that is not a ``ToolDefinition``."""

    __slots__ = ("_label",)

    def __init__(self, label: str) -> None:
        self._label = label

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{self._label}>"


# The tool carries no registry identity: deny.
_UNVERIFIED = _Sentinel("unverified-tool")
# A genuine MCP tool: gate under the synthetic ``mcp`` capability.
_MCP = _Sentinel("mcp-tool")


def _no_engine_result(tool_name: str) -> dict | None:
    """Return value when the permission engine is absent from the registry.

    Fail closed: if permissions are enabled but no engine is wired, deny the
    call. In production ``base_manager`` always constructs the engine, so a
    missing engine with permissions on is a misconfiguration — allowing would
    silently bypass all gating. Only permissions-disabled allows (None).
    """
    from agentic_cli.config import get_settings

    if get_settings().permissions_enabled:
        logger.warning("permission_engine_missing", tool=tool_name)
        return {
            "success": False,
            "error": "Permission denied: permission engine unavailable",
        }
    return None


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
        defn = self._resolve_definition(tool)

        if defn is _UNVERIFIED:
            # Nothing the framework issued. Its name (or its class name) may
            # match a registered tool; that proves nothing.
            logger.warning("permission_unregistered_tool", tool=tool.name)
            return {
                "success": False,
                "error": (
                    "Permission denied: tool is not registered "
                    "(register it with @register_tool to declare capabilities)"
                ),
            }

        if defn is _MCP:
            # MCP tools are created inside ADK when the server connects, so
            # they carry no binding; gate them under a synthetic capability.
            return await self._check_mcp(tool)

        caps = defn.capabilities

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
            return _no_engine_result(defn.name)

        result = await engine.check(defn.name, caps, tool_args)
        if result.allowed:
            return None
        return {"success": False, "error": f"Permission denied: {result.reason}"}

    @staticmethod
    def _resolve_definition(tool: "BaseTool"):
        """Resolve what this exact tool object is authorised to do.

        Returns the bound ``ToolDefinition``; ``_MCP`` for a genuine MCP tool
        (gated under a synthetic capability); or ``_UNVERIFIED`` for anything
        the framework did not issue, which the caller denies. There is no
        name-based path: a name is chosen by whoever built the tool.
        """
        defn = identify_tool(tool)
        if defn is not None:
            return defn

        func = _trusted_wrapped_callable(tool)
        if func is not None:
            return identify_tool(func) or _UNVERIFIED

        if _is_mcp_tool(tool):
            return _MCP
        return _UNVERIFIED

    async def _check_mcp(self, tool: "BaseTool") -> dict | None:
        """Gate an MCP tool through the engine under a synthetic capability."""
        engine = get_service(PERMISSION_ENGINE)
        if engine is None:
            return _no_engine_result(tool.name)
        caps = [Capability("mcp", target_arg=_MCP_TARGET_ARG)]
        result = await engine.check(tool.name, caps, {_MCP_TARGET_ARG: tool.name})
        if result.allowed:
            return None
        return {"success": False, "error": f"Permission denied: {result.reason}"}
