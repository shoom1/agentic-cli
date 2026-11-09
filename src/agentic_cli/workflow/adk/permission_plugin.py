"""ADK plugin that gates tool calls via PermissionEngine.

Tool identity is *object* identity, resolved through the registry's identity
binding. It is never ``tool.name`` (ADK derives that from the callable, so an
unregistered function named after a registered tool would inherit its
capabilities — an EXEMPT one would be waved straight through), never the tool's
class name, and never equality.

Adapter check order (mirrors LangGraph wrapper for consistency):
0. ADK's stand-in for a tool name the agent does not have (a bare
   ``BaseTool``, ``tool_error_plugin.is_unknown_tool``) passes: it runs
   nothing, and ``ToolErrorPlugin`` answers ADK's lookup error with ADK's
   own text, as Deep Agents' gate does.
1. Resolve the definition bound to this exact object. Framework-issued tools —
   registered callables, service-bound factory variants, and the native ADK
   tool objects the framework constructs (skill tools, and the delegation tool
   ADK builds for each sub-agent the framework built, ``adk/delegation.py``) —
   are bound at their construction site.
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

import contextlib
from typing import Any, TYPE_CHECKING

from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools import FunctionTool, LongRunningFunctionTool

from agentic_cli.logging import Loggers
from agentic_cli.tools.registry import (
    ToolCategory,
    ToolDefinition,
    identify_tool,
    register_tool,
)
from agentic_cli.workflow.adk.tool_error_plugin import is_unknown_tool
from agentic_cli.workflow.outside_content import OUTSIDE_CONTENT_KEY, brings_outside_content
from agentic_cli.workflow.permissions import EXEMPT
from agentic_cli.workflow.permissions.capabilities import Capability, _CapabilityExempt
from agentic_cli.workflow.service_registry import PERMISSION_ENGINE, get_service
from agentic_cli.workflow.sessions import get_active_turn

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.tools import BaseTool
    from google.adk.tools.tool_context import ToolContext

    from agentic_cli.workflow.interrupts import OpenCalls
    from agentic_cli.workflow.outside_content import OutsideContentMarks

logger = Loggers.workflow()


# ADK auto-injects ``transfer_to_agent`` into an agent that can transfer to
# another: a native ADK YAML app's (agents built from configs delegate
# instead, ``adk/delegation.py``). It's an internal routing primitive, not an
# external side effect, so register it with EXEMPT so the plugin lets it
# through.
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
# terms. ADK auto-injects it into an agent that can transfer (a native ADK
# YAML app's), and it is a ``FunctionTool`` *subclass*, so an exact-type check
# on ``FunctionTool`` alone denied the built-in routing tool as unregistered —
# delegation could not work at all, even with permissions disabled. It is
# safe to add because ADK
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


def tool_definition(tool: "BaseTool") -> ToolDefinition | None:
    """The registry's definition of this exact tool object, or None.

    Bound to the object itself, or to the callable one of ADK's own
    function-tool types runs (``_TRUSTED_FUNCTION_TOOL_TYPES``); never found
    by a name. None for anything else: an MCP tool, or a tool the framework
    did not issue. How the permission gate (steps 1 and 2) and the tool
    result cap identify a tool.
    """
    defn = identify_tool(tool)
    if defn is not None:
        return defn
    func = _trusted_wrapped_callable(tool)
    return identify_tool(func) if func is not None else None


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


def _model_side_tool(llm_request: Any, callback_context: Any) -> str | None:
    """What brings outside content into the conversation with no tool call of
    ours: the first model-side tool in the request, else the agent's code
    executor, else None.

    A request tool that is more than function declarations (``google_search``,
    ``url_context``, ``retrieval``, ``enterprise_web_search``,
    ``code_execution``, and whatever ADK's genai adds) is the model's own: the
    field that is set names it. An agent's code executor runs the model's code
    blocks and puts their output in the conversation, which nothing in the
    request shows; it is read through ADK's public
    ``get_invocation_context()``.
    """
    config = getattr(llm_request, "config", None)
    for tool in getattr(config, "tools", None) or ():
        for field in getattr(type(tool), "model_fields", ()):
            if field != "function_declarations" and getattr(tool, field, None) is not None:
                return field
    invocation_context = getattr(callback_context, "get_invocation_context", None)
    invocation = invocation_context() if invocation_context is not None else None
    if getattr(getattr(invocation, "agent", None), "code_executor", None) is not None:
        return "code_executor"
    return None


class PermissionPlugin(BasePlugin):
    """ADK plugin: gates every tool call through :class:`PermissionEngine`.

    It also marks the conversation when it allows a call that brings outside
    content (``workflow/outside_content.py``), in ``marks`` and in the
    session's state, and when a model request carries a model-side tool or the
    agent has a code executor (``before_model_callback``): those never reach
    the tool callback.

    Args:
        open_calls: The manager's record of its calls, marked while a call
            waits at its prompt: a cancel there means it was not run.
        marks: The manager's record of each conversation's mark.
        root: The root agent's name, whose session state carries the mark.
    """

    def __init__(
        self,
        open_calls: "OpenCalls | None" = None,
        *,
        marks: "OutsideContentMarks | None" = None,
        root: str | None = None,
    ) -> None:
        super().__init__(name="permission")
        self._open_calls = open_calls
        self._marks = marks
        self._root = root

    def _asking(self, tool_context: "ToolContext | None"):
        call_id = getattr(tool_context, "function_call_id", None)
        if self._open_calls is None:
            return contextlib.nullcontext()
        return self._open_calls.asking(call_id)

    async def before_tool_callback(
        self,
        *,
        tool: "BaseTool",
        tool_args: dict[str, Any],
        tool_context: "ToolContext | None",
    ) -> dict | None:
        if is_unknown_tool(tool):
            # ADK's stand-in for a name the agent has no tool for: it runs
            # nothing, and ToolErrorPlugin answers ADK's lookup error next.
            return None
        defn = self._resolve_definition(tool)
        denial = await self._gate(tool, defn, tool_args, tool_context)
        if denial is None:
            self._note_outside_content(tool, defn, tool_args, tool_context)
        return denial

    def _note_outside_content(
        self,
        tool: "BaseTool",
        defn: Any,
        tool_args: dict[str, Any],
        tool_context: "ToolContext | None",
    ) -> None:
        """Mark the conversation when an allowed call brings outside content,
        in the marks and in the session's state (it rides on the call's event)."""
        if self._marks is None:
            return
        if defn is _MCP:
            name = tool.name
        elif isinstance(defn, ToolDefinition) and brings_outside_content(defn, tool_args):
            name = defn.name
        else:
            return
        ref = get_active_turn()
        if ref is None or not self._marks.mark(ref, name):
            return
        logger.debug("conversation_marked_outside_content", tool=name, session_id=ref.session_id)
        if tool_context is not None:
            tool_context.state[OUTSIDE_CONTENT_KEY] = name

    async def before_model_callback(
        self, *, callback_context: "CallbackContext", llm_request: Any
    ) -> None:
        """Mark the conversation when the request brings a model-side tool, and
        write its mark into the root's session state when a sub-agent's call
        set it (a sub-agent's state may be its own).

        A model-side tool (Google Search, URL context, Vertex AI Search, a code
        executor) puts search results, pages or program output into the
        conversation without a tool call, so ``before_tool_callback`` never
        sees it. ADK has built the request's tools by now (each tool's
        ``process_llm_request`` runs before the model callbacks), so the
        request names them; an agent's own code executor is read from the agent.
        """
        if self._marks is None:
            return None
        ref = get_active_turn()
        name = _model_side_tool(llm_request, callback_context)
        if name is not None and ref is not None and self._marks.mark(ref, name):
            logger.debug(
                "conversation_marked_outside_content", tool=name, session_id=ref.session_id
            )
        if self._root is None or callback_context.agent_name != self._root:
            return None
        mark = self._marks.get(ref)
        if mark and callback_context.state.get(OUTSIDE_CONTENT_KEY) != mark:
            callback_context.state[OUTSIDE_CONTENT_KEY] = mark
        return None

    async def _gate(
        self,
        tool: "BaseTool",
        defn: Any,
        tool_args: dict[str, Any],
        tool_context: "ToolContext | None",
    ) -> dict | None:
        """The permission decision: None to let the call run, else its error result."""
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
            with self._asking(tool_context):
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

        with self._asking(tool_context):
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
        defn = tool_definition(tool)
        if defn is not None:
            return defn
        if _trusted_wrapped_callable(tool) is None and _is_mcp_tool(tool):
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
