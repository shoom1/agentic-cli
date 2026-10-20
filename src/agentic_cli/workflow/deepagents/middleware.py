"""Middleware every Deep Agents agent runs with.

Each agent is its own ``create_agent`` graph with its own stack, so each stack
gets these; sub-agents are gated as strictly as the root.

- :class:`PermissionMiddleware` gates every tool call through the permission
  engine, identifying the tool by the object the registry issued.
- :class:`ToolCallErrorMiddleware`, inside the gate, gives a tool that raised
  the error result ADK's ``ToolErrorPlugin`` gives, so the turn goes on.

A gate or error result is the call's ``ToolMessage`` with ``status="error"``
and the result dict as JSON content: the model reads it and carries on.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolCall, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

from agentic_cli.logging import Loggers
from agentic_cli.tools._core.delegation import TASK_TOOL
from agentic_cli.tools.registry import ToolDefinition, get_registry, identify_tool
from agentic_cli.workflow.permissions.capabilities import _CapabilityExempt
from agentic_cli.workflow.service_registry import PERMISSION_ENGINE, get_service
from agentic_cli.workflow.tool_errors import tool_error_result, unknown_tool_result

logger = Loggers.workflow()

_ToolResult = ToolMessage | Command


def error_tool_message(result: dict, call: ToolCall) -> ToolMessage:
    """``result`` as the error ``ToolMessage`` answering ``call``."""
    return ToolMessage(
        content=json.dumps(result),
        tool_call_id=call["id"],
        name=call["name"],
        status="error",
    )


def _denied(reason: str, call: ToolCall) -> ToolMessage:
    return error_tool_message(
        {"success": False, "error": f"Permission denied: {reason}"}, call
    )


class PermissionMiddleware(AgentMiddleware):
    """Gate every tool call of one agent through the permission engine.

    The tool is identified by object identity, never by its name, its class
    name or equality, in this order:

    1. ``bound(tool)``: objects the framework built and bound (``task``, the
       skill tools, the MCP tools).
    2. For a tool whose type is exactly ``StructuredTool``, the callable it
       will run: ``.coroutine`` if it has one (the async path runs it), else
       ``.func``. That is what a ``StructuredTool`` is contracted to run; a
       subclass may run anything while advertising a genuine ``.func``.
    3. EXEMPT runs without an engine call.
    4. Unresolved, or no capability declaration: denied.
    5. No engine in the service registry: denied with permissions on; with
       permissions off the tool runs, as on the other backends.
    6. ``engine.check(name, capabilities, args)`` under the registered name.
       An ASK prompt reaches the consumer through the context-local input
       callback, which LangGraph copies into every tool task, sub-agents'
       included.

    A call to a tool the agent does not have gets ADK's "not found" result
    without an engine call. The sync tool path fails closed: turns run
    through ``astream``, so nothing legitimate takes it.

    Args:
        agent_name: The agent whose stack this is (for logs).
        bound: Resolves a tool object the framework bound. Defaults to the
            framework registry's identity map.
    """

    def __init__(
        self,
        agent_name: str,
        *,
        bound: Callable[[object], ToolDefinition | None] = identify_tool,
    ) -> None:
        super().__init__()
        self.agent_name = agent_name
        self._bound = bound

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[_ToolResult]],
    ) -> _ToolResult:
        call = request.tool_call
        if request.tool is None:
            available = [tool.name for tool in getattr(request.runtime, "tools", None) or ()]
            logger.warning("tool_not_found", tool=call["name"], agent=self.agent_name)
            return error_tool_message(unknown_tool_result(call["name"], available), call)
        denial = await self._check(request.tool, call)
        if denial is not None:
            return denial
        return await handler(request)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], _ToolResult],
    ) -> _ToolResult:
        logger.warning(
            "permission_sync_tool_path", tool=request.tool_call["name"], agent=self.agent_name
        )
        return _denied("synchronous tool path is not supported", request.tool_call)

    def _resolve(self, tool: BaseTool) -> ToolDefinition | None:
        """What this exact tool object is, or None (steps 1 and 2)."""
        definition = self._bound(tool)
        if definition is not None:
            return definition
        if type(tool) is StructuredTool:
            runs = tool.coroutine if tool.coroutine is not None else tool.func
            return self._bound(runs) if runs is not None else None
        return None

    async def _check(self, tool: BaseTool, call: ToolCall) -> ToolMessage | None:
        """None when the call may run, else its denial (steps 3 to 6)."""
        definition = self._resolve(tool)
        if definition is None:
            logger.warning(
                "permission_unregistered_tool", tool=call["name"], agent=self.agent_name
            )
            return _denied(
                "tool is not registered "
                "(register it with @register_tool to declare capabilities)",
                call,
            )

        capabilities = definition.capabilities
        if isinstance(capabilities, _CapabilityExempt):
            return None
        if not capabilities:
            logger.warning(
                "permission_undeclared", tool=definition.name, agent=self.agent_name
            )
            return _denied("tool has no capability declaration", call)

        engine = get_service(PERMISSION_ENGINE)
        if engine is None:
            from agentic_cli.config import get_settings

            if not get_settings().permissions_enabled:
                return None
            # Permissions on but no engine wired is a misconfiguration (the
            # base manager always builds one): deny rather than run ungated.
            logger.warning(
                "permission_engine_missing", tool=definition.name, agent=self.agent_name
            )
            return _denied("permission engine unavailable", call)

        result = await engine.check(definition.name, capabilities, call["args"])
        if result.allowed:
            return None
        return _denied(result.reason, call)


def _error_message(call: ToolCall, error: Exception) -> ToolMessage:
    return error_tool_message(tool_error_result(call["name"], error), call)


def _is_delegation(tool: BaseTool | None) -> bool:
    """Whether ``tool`` is the framework's ``task`` tool (by identity)."""
    declaration = get_registry().get(TASK_TOOL)
    return declaration is not None and identify_tool(tool) is declaration


class ToolCallErrorMiddleware(AgentMiddleware):
    """A tool that raises fails its call, not the turn.

    The exception becomes ADK's error result (``tool_error_result``: the tool,
    the exception type and message, cut to 500 characters, logged as
    ``tool_raised``). Only ``Exception`` is caught: cancelling (Ctrl+C) still
    cancels, and LangGraph's control flow (``GraphBubbleUp``: interrupts,
    parent commands) passes through.

    A failed delegation is not a tool error. The sub-agent's own stack already
    turned its tools' exceptions into results, so what escapes ``task`` is the
    sub-agent's model or run failing; it ends the turn, as a model failure at
    the root does.
    """

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[_ToolResult]],
    ) -> _ToolResult:
        try:
            return await handler(request)
        except GraphBubbleUp:
            raise
        except Exception as error:
            if _is_delegation(request.tool):
                raise
            return _error_message(request.tool_call, error)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], _ToolResult],
    ) -> _ToolResult:
        try:
            return handler(request)
        except GraphBubbleUp:
            raise
        except Exception as error:
            if _is_delegation(request.tool):
                raise
            return _error_message(request.tool_call, error)
