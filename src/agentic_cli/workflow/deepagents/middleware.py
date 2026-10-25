"""Middleware every Deep Agents agent runs with.

Each agent is its own ``create_agent`` graph with its own stack, so each stack
gets these; sub-agents are gated as strictly as the root.

- :class:`PermissionMiddleware` gates every tool call through the permission
  engine, identifying the tool by the object the registry issued.
- :class:`ToolResultCapMiddleware`, inside the gate, cuts a tool result too
  big for the context, as ADK's ``ToolResultCapPlugin`` does.
- :class:`ToolCallErrorMiddleware`, inside the cap, gives a tool that raised
  the error result ADK's ``ToolErrorPlugin`` gives, so the turn goes on.
- :class:`ActiveToolCallMiddleware`, innermost, runs each tool as the active
  tool call, which links a job the tool starts to its call.
- :class:`ForeignReasoningMiddleware`, on the model side, sends no model
  another provider's reasoning.
- :class:`ContextOverflowMiddleware`, on the model side inside the
  summarization, gives it every provider's overflow as LangChain's
  ``ContextOverflowError``, so it summarizes and retries.

A gate or error result is the call's ``ToolMessage`` with ``status="error"``
and the result dict as JSON content: the model reads it and carries on.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.errors import GraphBubbleUp
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langgraph.types import Command

from agentic_cli.logging import Loggers
from agentic_cli.models import ProviderKind
from agentic_cli.paths import resolve_path
from agentic_cli.tools._core.delegation import TASK_TOOL
from agentic_cli.tools.registry import ToolDefinition, get_registry, identify_tool
from agentic_cli.workflow.context import is_context_overflow
from agentic_cli.workflow.interrupts import MARKER, OpenCalls
from agentic_cli.workflow.langchain_models import without_foreign_reasoning
from agentic_cli.workflow.permissions.capabilities import _CapabilityExempt
from agentic_cli.workflow.service_registry import PERMISSION_ENGINE, get_service
from agentic_cli.workflow.sessions import (
    ToolCallRef,
    get_active_turn,
    reset_active_tool_call,
    set_active_tool_call,
)
from agentic_cli.workflow.tool_errors import tool_error_result, unknown_tool_result
from agentic_cli.workflow.tool_output import cap_tool_result, is_saved_output

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
       skill tools, the MCP tools). The MCP tools are bound in their
       manager's own map, which the manager chains before the registry's.
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
            framework registry's identity map; a Deep Agents manager passes
            its MCP tools' map chained before it.
        open_calls: The manager's record of its calls, marked while a call
            waits at its prompt: a cancel there means it was not run.
    """

    def __init__(
        self,
        agent_name: str,
        *,
        bound: Callable[[object], ToolDefinition | None] = identify_tool,
        open_calls: OpenCalls | None = None,
    ) -> None:
        super().__init__()
        self.agent_name = agent_name
        self._bound = bound
        self._open_calls = open_calls

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
        return _resolve_tool(tool, self._bound)

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

        if self._open_calls is None:
            result = await engine.check(definition.name, capabilities, call["args"])
        else:
            with self._open_calls.asking(call["id"]):
                result = await engine.check(definition.name, capabilities, call["args"])
        if result.allowed:
            return None
        return _denied(result.reason, call)


def _resolve_tool(
    tool: BaseTool | None, bound: Callable[[object], ToolDefinition | None]
) -> ToolDefinition | None:
    """What this exact tool object is, by identity, or None.

    ``bound(tool)`` first: the objects the framework built and bound. Then,
    for a tool whose type is exactly ``StructuredTool``, the callable it will
    run (``.coroutine`` if it has one, else ``.func``).
    """
    if tool is None:
        return None
    definition = bound(tool)
    if definition is not None:
        return definition
    if type(tool) is StructuredTool:
        runs = tool.coroutine if tool.coroutine is not None else tool.func
        return bound(runs) if runs is not None else None
    return None


class ToolResultCapMiddleware(AgentMiddleware):
    """A tool result too big for the context is cut.

    Every result passes ``workflow/tool_output.py::cap_tool_result``: one over
    ``max_chars`` reaches the model cut, with a note pointing at its full
    output in ``folder``. The middleware sits inside the permission gate,
    whose denials are short, and outside the tool-error middleware, so an
    error result is capped too.

    What it reads of a result:

    - A ``ToolMessage`` whose content is a string: a JSON object (a framework
      tool's dict) is cut as a dict and serialized again; other text is cut
      as a string.
    - Content that is a list: its text blocks are joined into one, cut as a
      string, which stands where the first was, with its other keys; other
      blocks (images) are kept in their order, so one that sat between two
      text blocks now follows the text.
    - A ``Command`` (the state tools, ``task``): each ``ToolMessage`` in its
      ``update["messages"]``.

    A call of the framework's ``read_file`` (by identity, as the gate
    resolves it) on a saved output is cut without saving it again.

    Args:
        max_chars: ``tool_result_max_chars``.
        folder: Where full outputs are saved (``tool_output_dir``).
    """

    def __init__(self, max_chars: int, folder: Path) -> None:
        super().__init__()
        self.max_chars = max_chars
        self.folder = folder

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[_ToolResult]],
    ) -> _ToolResult:
        return self._capped(request, await handler(request))

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], _ToolResult],
    ) -> _ToolResult:
        return self._capped(request, handler(request))

    def _capped(self, request: ToolCallRequest, result: _ToolResult) -> _ToolResult:
        name = request.tool_call["name"]
        try:
            source = self._saved_output_read(request)
            if isinstance(result, ToolMessage):
                return self._message(result, name, source)
            if isinstance(result, Command):
                return self._command(result, name, source)
        except Exception as e:
            # The cap never raises by design; a bug in it must not end the
            # turn. The result goes uncut, and the traceback goes to the
            # debug log, not over the UI.
            logger.warning(
                "tool_result_cap_failed", tool=name, error_type=type(e).__name__, error=str(e)
            )
            logger.debug("tool_result_cap_failed", tool=name, exc_info=e)
        return result

    def _saved_output_read(self, request: ToolCallRequest) -> Path | None:
        """The saved output this call reads, when it is the framework's ``read_file``."""
        read_file = get_registry().get("read_file")
        if read_file is None or _resolve_tool(request.tool, identify_tool) is not read_file:
            return None
        try:
            path = resolve_path(request.tool_call["args"]["path"])
        except (KeyError, TypeError, ValueError):
            return None
        return path if is_saved_output(path, self.folder) else None

    def _command(self, command: Command, name: str, source: Path | None) -> Command:
        update = command.update
        if not isinstance(update, dict) or not isinstance(update.get("messages"), list):
            return command
        messages = [
            self._message(m, name, source) if isinstance(m, ToolMessage) else m
            for m in update["messages"]
        ]
        if all(new is old for new, old in zip(messages, update["messages"])):
            return command
        return dataclasses.replace(command, update={**update, "messages": messages})

    def _message(self, message: ToolMessage, name: str, source: Path | None) -> ToolMessage:
        content = message.content
        if isinstance(content, str):
            capped: Any = self._text(content, name, source)
        elif isinstance(content, list):
            capped = self._blocks(content, name, source)
        else:
            return message
        if capped is content:
            return message
        return message.model_copy(update={"content": capped})

    def _text(self, text: str, name: str, source: Path | None) -> str:
        if len(text) <= self.max_chars:
            return text
        result = _json_object(text)
        if result is None:
            return self._cap(text, name, source)
        # Sent as the cap measured it, cut or not: the tool's own JSON may be
        # several times longer (\uXXXX escapes, indentation).
        return json.dumps(self._cap(result, name, source), ensure_ascii=False)

    def _blocks(self, blocks: list, name: str, source: Path | None) -> list:
        texts = [text for text in map(_block_text, blocks) if text is not None]
        if not texts:
            return blocks
        joined = "\n".join(texts)
        capped = self._cap(joined, name, source)
        if capped is joined:
            return blocks
        # One text block, where the first was and with its other keys (an id,
        # cache_control); the other blocks as they were.
        kept: list = []
        placed = False
        for block in blocks:
            if _block_text(block) is None:
                kept.append(block)
            elif not placed:
                first = block if isinstance(block, dict) else {"type": "text"}
                kept.append({**first, "text": capped})
                placed = True
        return kept

    def _cap(self, result: Any, name: str, source: Path | None) -> Any:
        return cap_tool_result(
            result, tool_name=name, max_chars=self.max_chars, folder=self.folder, source=source
        )


def _json_object(text: str) -> dict | None:
    """The dict ``text`` holds as JSON, or None."""
    if not text.lstrip().startswith("{"):
        return None
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _block_text(block: Any) -> str | None:
    """A content block's text, or None when it is not a text block."""
    if isinstance(block, str):
        return block
    if isinstance(block, dict) and block.get("type") == "text":
        text = block.get("text")
        return text if isinstance(text, str) else None
    return None


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


class InterruptedCallsMiddleware(AgentMiddleware):
    """Every call gets one result, the ones a stopped turn left open too.

    It takes the place of Deep Agents' ``PatchToolCallsMiddleware``, first in
    every stack: before each run it answers a call no ``ToolMessage``
    answers, as ``workflow/interrupts.py`` says. A call that finished before
    the cancel keeps its result: LangGraph drops a stopped step's writes when
    new input arrives, so that result is kept here as it passes (outermost,
    after the gate, the cap and the error handling). A call that waited at
    its permission prompt was not run; one whose job was submitted runs on;
    any other was interrupted, by the user when this manager saw the cancel.
    On the main agent, a turn the user stopped before any reply leaves
    ``MARKER`` as its reply, so the next turn answers only the new message.

    Args:
        open_calls: The manager's record of its calls.
        root: Whether this is the main agent's stack.
    """

    def __init__(self, open_calls: OpenCalls, *, root: bool) -> None:
        super().__init__()
        self._open = open_calls
        self._root = root

    def before_agent(self, state: Any, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        messages: list[AnyMessage] = state["messages"]
        if not messages:
            return None
        by_user = self._root and self._open.take_cancel(get_active_turn())
        answered = {message.tool_call_id for message in messages if message.type == "tool"}
        patched: list[AnyMessage] = []
        changed = False
        for message in messages:
            patched.append(message)
            if not isinstance(message, AIMessage):
                continue
            for call in (*message.tool_calls, *message.invalid_tool_calls):
                call_id = call["id"]
                if call_id is None or call_id in answered:
                    continue
                patched.append(self._answer(call, by_user=by_user))
                changed = True
        if (
            by_user and not changed and len(patched) >= 2
            and isinstance(patched[-1], HumanMessage) and isinstance(patched[-2], HumanMessage)
        ):
            patched.insert(-1, AIMessage(content=MARKER))
            changed = True
        if not changed:
            return None
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *patched]}

    def _answer(self, call: Any, *, by_user: bool) -> ToolMessage:
        name = call["name"] or "unknown"
        if call.get("type") == "invalid_tool_call":
            return ToolMessage(
                content=f"Tool call {name} with id {call['id']} could not be executed - "
                "arguments were malformed or truncated.",
                name=name, tool_call_id=call["id"], status="error",
            )
        answer = self._open.answer(call["id"], by_user=by_user)
        if isinstance(answer, ToolMessage):
            return answer
        return ToolMessage(
            content=json.dumps(answer), name=name, tool_call_id=call["id"],
            status="success" if answer.get("success") else "error",
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[_ToolResult]],
    ) -> _ToolResult:
        result = await handler(request)
        self._record(request.tool_call, result)
        return result

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], _ToolResult],
    ) -> _ToolResult:
        result = handler(request)
        self._record(request.tool_call, result)
        return result

    def _record(self, call: ToolCall, result: _ToolResult) -> None:
        if isinstance(result, Command):
            update = result.update if isinstance(result.update, dict) else {}
            result = next((
                m for m in update.get("messages", ())
                if isinstance(m, ToolMessage) and m.tool_call_id == call["id"]
            ), None)
        if isinstance(result, ToolMessage):
            self._open.finished(call["id"], result)


class ActiveToolCallMiddleware(AgentMiddleware):
    """Each tool runs with its own call as the active tool call.

    The call's ``(id, name)`` is set around the tool and reset afterwards,
    however it ends (``workflow.sessions.set_active_tool_call``).
    ``JobManager.submit`` reads it, so a job a long-running tool starts is
    linked to that call even when the tool, written for ADK, reads its call
    id from a ``tool_context`` that is None here. Each call runs in its own
    task, so parallel calls never see each other's; a sub-agent's call runs
    inside its coordinator's ``task`` call and restores it when done.
    """

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[_ToolResult]],
    ) -> _ToolResult:
        token = set_active_tool_call(_call_ref(request.tool_call))
        try:
            return await handler(request)
        finally:
            reset_active_tool_call(token)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], _ToolResult],
    ) -> _ToolResult:
        token = set_active_tool_call(_call_ref(request.tool_call))
        try:
            return handler(request)
        finally:
            reset_active_tool_call(token)


def _call_ref(call: ToolCall) -> ToolCallRef:
    return ToolCallRef(call_id=call["id"], name=call["name"])


class ContextOverflowMiddleware(AgentMiddleware):
    """A provider's "too long" reaches Deep Agents' summarization as one.

    Deep Agents' summarization falls back to summarizing and retrying once
    when a model call raises a context overflow it recognizes: LangChain's
    ``ContextOverflowError``, or a 400, 413 or 422 ``status_code`` with one of
    its phrases. Some providers say it otherwise: Gemini's error carries its
    status in ``code`` and words it its own way, and Claude's request-size
    limit is a 413 that names no context. This middleware, between the
    summarization and the model, re-raises every overflow
    ``workflow/context.py`` recognizes as ``ContextOverflowError``, from the
    original; anything else, a ``ContextOverflowError`` included, passes
    unchanged.
    """

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> Any:
        try:
            return await handler(request)
        except ContextOverflowError:
            raise
        except Exception as error:
            if is_context_overflow(error):
                raise ContextOverflowError(str(error)) from error
            raise

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> Any:
        try:
            return handler(request)
        except ContextOverflowError:
            raise
        except Exception as error:
            if is_context_overflow(error):
                raise ContextOverflowError(str(error)) from error
            raise


class ForeignReasoningMiddleware(AgentMiddleware):
    """No model is sent another provider's, or another agent's, reasoning.

    A model switch that keeps the session sends the new model AI messages
    another provider wrote, whose reasoning it would refuse (Claude rejects a
    thinking block it did not sign) or warn about. A forked sub-agent
    (``context="inherit"``) is sent its coordinator's messages, whose
    reasoning another model may refuse too, and which ADK leaves out of a
    sub-agent's view as well. Each request goes out through
    ``without_foreign_reasoning``; the state, and so the checkpoint, keeps
    every message as it was, and an agent's own reasoning goes back to it.

    Args:
        kind: The provider of the agent's model.
        agent: The agent's name: AI messages named after another agent are
            sent without their reasoning. None leaves only other providers'
            reasoning out.
    """

    def __init__(self, kind: ProviderKind, agent: str | None = None) -> None:
        super().__init__()
        self.kind = kind
        self.agent = agent

    def _own(self, request: ModelRequest) -> ModelRequest:
        return request.override(
            messages=without_foreign_reasoning(request.messages, self.kind, agent=self.agent)
        )

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> Any:
        return await handler(self._own(request))

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> Any:
        return handler(self._own(request))
