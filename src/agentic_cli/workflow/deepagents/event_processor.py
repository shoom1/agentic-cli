"""A Deep Agents turn's stream as ``WorkflowEvent``s.

A turn streams ``astream(input, config, stream_mode=["updates", "tasks"],
subgraphs=True)``. Each item is ``(namespace, mode, payload)``; ``()`` is the
root agent, and a sub-agent's items arrive under ``("tools:<task-id>",)``,
the namespace of the coordinator's tools task that runs its ``task`` call.
The ``tasks`` mode announces that task with its tool call, whose
``subagent_type`` names the agent; deeper nesting resolves the same way, one
namespace segment at a time.

Each finished node output becomes events, as ADK reports them: one per
finished message, no tokens. Only the agents' own nodes speak:

- a ``model`` update's ``AIMessage``: ``THINKING`` (its reasoning blocks) and
  ``TEXT`` (its text blocks) in order, one ``TOOL_CALL`` per tool call, and
  ``LLM_USAGE`` on that agent's own model;
- a ``tools`` update's ``ToolMessage``s, including those a ``Command`` carried
  (the state tools, ``task``): ``TOOL_RESULT``, a JSON dict parsed back; and
  its ``tasks``, when they changed: ``TASK_PROGRESS``, in the same step.

Middleware hooks' updates are not output (``PatchToolCallsMiddleware`` re-adds
the whole history when it patches it), and a model call inside a tool
(``generate_simple``) is no node at all, so it cannot leak into the answer.
Every event carries ``metadata["agent"]``.

What happens inside a node and is no node output (a summary call and the
summary it made, and with ``raw_llm_logging`` every model call) reaches
``RunCallback`` instead, the turn's LangChain callback handler.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage

from agentic_cli.constants import truncate
from agentic_cli.tools._core.delegation import TASK_TOOL
from agentic_cli.tools._core.tasks import task_progress_data
from agentic_cli.workflow.context import (
    OVERFLOW_ERROR_CODE,
    is_context_overflow,
    overflow_message,
)
from agentic_cli.workflow.deepagents.summarization import CONTEXT_SUMMARIZED
from agentic_cli.workflow.events import WorkflowEvent

if TYPE_CHECKING:
    from langchain_core.outputs import LLMResult

    from agentic_cli.models import ResolvedModel
    from agentic_cli.workflow.llm_log import LLMLogWriter

#: The tag LangChain's summarization puts on its summary call's run metadata.
_SUMMARY_SOURCE = "summarization"

#: Why a model call was made (a raw log record's ``purpose``): an agent's own
#: step, a summary of its history, or a call one of its tools made.
PURPOSE_AGENT = "agent"
PURPOSE_SUMMARY = _SUMMARY_SOURCE
PURPOSE_TOOL = "tool"

#: Longest error text a raw log record carries.
_MAX_ERROR = 500

#: LangChain's message types as the roles a provider knows.
_ROLES = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}

Namespace = tuple[str, ...]


class DeepAgentsEventProcessor:
    """Turns one turn's stream items into ``WorkflowEvent``s.

    Args:
        root: The root agent's name: the agent of namespace ``()``, and of any
            namespace the stream never announced.
        agent_models: Each agent's resolved model, by name; ``LLM_USAGE`` is
            reported against the agent's own.
        session_id: Put on ``TEXT`` and ``THINKING``, as the other backends do.
    """

    def __init__(
        self,
        *,
        root: str,
        agent_models: Mapping[str, "ResolvedModel"],
        session_id: str | None = None,
    ) -> None:
        self._root = root
        self._agent_models = agent_models
        self._session_id = session_id
        # Namespace -> the agent whose graph runs there.
        self._agents: dict[Namespace, str] = {(): root}
        # Tool call id -> tool name: a ToolMessage a Command carried may have
        # no name of its own.
        self._call_names: dict[str, str] = {}
        self._reported_tasks: list[dict] | None = None
        self._failed_agent: str | None = None

    def agent(self, namespace: Iterable[str]) -> str:
        """The agent whose graph runs at ``namespace``."""
        return self._agents.get(tuple(namespace), self._root)

    def process(
        self, namespace: Iterable[str], mode: str, payload: Any
    ) -> list[WorkflowEvent]:
        """The events of one stream item (none for most)."""
        namespace = tuple(namespace)
        if mode == "tasks":
            self._track(namespace, payload)
            return []
        if mode != "updates" or not isinstance(payload, dict):
            return []
        agent = self.agent(namespace)
        events: list[WorkflowEvent] = []
        for node, update in payload.items():
            for values in update if isinstance(update, list) else [update]:
                if not isinstance(values, dict):
                    continue
                if node == "model":
                    events += self._model_output(agent, values)
                elif node == "tools":
                    events += self._tools_output(agent, values)
        return events

    def turn_failed(self, error: BaseException) -> WorkflowEvent:
        """The ``ERROR`` that ends a turn whose stream raised ``error``.

        Attributed to the agent whose task failed first, the innermost: a
        sub-agent's failure fails its coordinator's ``task`` call after it.
        A context overflow (a provider's, or Deep Agents' own when a summary
        still does not fit) is told in one sentence naming that agent's
        model, with ``error_code`` ``CONTEXT_OVERFLOW``, as on ADK.
        """
        agent = self._failed_agent or self._root
        if is_context_overflow(error):
            model = self._agent_models.get(agent) or self._agent_models.get(self._root)
            event = WorkflowEvent.error(
                overflow_message(str(model.ref) if model is not None else "the model"),
                error_code=OVERFLOW_ERROR_CODE,
            )
        else:
            event = WorkflowEvent.error(
                str(error) or type(error).__name__, error_code=type(error).__name__
            )
        return self._attributed(event, agent)

    # -- the tasks mode --------------------------------------------------------

    def _track(self, namespace: Namespace, payload: Any) -> None:
        """Learn which agent runs where, and which task failed first."""
        if not isinstance(payload, dict):
            return
        if "input" in payload:  # a task starts
            delegates = payload.get("name") == "tools"
            subagent = _delegated_to(payload) if delegates else None
            if subagent is not None:
                self._agents[(*namespace, f"tools:{payload['id']}")] = subagent
        elif payload.get("error") is not None and self._failed_agent is None:
            self._failed_agent = self.agent(namespace)

    # -- node outputs ----------------------------------------------------------

    def _model_output(self, agent: str, values: dict) -> list[WorkflowEvent]:
        events: list[WorkflowEvent] = []
        for message in values.get("messages") or ():
            if not isinstance(message, AIMessage):
                continue
            for block in message.content_blocks:
                if block.get("type") == "reasoning" and block.get("reasoning"):
                    events.append(
                        WorkflowEvent.thinking(block["reasoning"], self._session_id)
                    )
                elif block.get("type") == "text" and block.get("text"):
                    events.append(WorkflowEvent.text(block["text"], self._session_id))
            for call in message.tool_calls:
                self._call_names[call["id"]] = call["name"]
                event = WorkflowEvent.tool_call(call["name"], call["args"])
                event.metadata["tool_call_id"] = call["id"]
                events.append(event)
            if message.usage_metadata:
                events.append(self._usage(agent, message.usage_metadata))
        return [self._attributed(event, agent) for event in events]

    def _tools_output(self, agent: str, values: dict) -> list[WorkflowEvent]:
        events = [
            self._tool_result(message)
            for message in values.get("messages") or ()
            if isinstance(message, ToolMessage)
        ]
        if "tasks" in values:
            events += self._progress(values["tasks"])
        return [self._attributed(event, agent) for event in events]

    def _tool_result(self, message: ToolMessage) -> WorkflowEvent:
        name = message.name or self._call_names.get(message.tool_call_id) or "unknown"
        result: Any = message.content
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                result = parsed
        success = message.status != "error"
        error: str | None = None
        if isinstance(result, dict):
            success = success and result.get("success") is not False
            if not success and isinstance(result.get("error"), str):
                error = result["error"]
        elif not success:
            error = message.text
        event = WorkflowEvent.tool_result(name, result, success=success, error=error)
        event.metadata["tool_call_id"] = message.tool_call_id
        return event

    def _progress(self, tasks: Any) -> list[WorkflowEvent]:
        """``TASK_PROGRESS`` for a changed task list.

        A sub-agent's list flows back to its coordinator with its result;
        reporting it again would say nothing new.
        """
        if not isinstance(tasks, list) or tasks == self._reported_tasks:
            return []
        self._reported_tasks = tasks
        progress = task_progress_data(tasks)
        if progress is None:
            return []
        return [
            WorkflowEvent.task_progress(
                display=progress["display"],
                progress=progress["progress"],
                current_task_id=progress["current_task_id"],
                current_task_description=progress["current_task_description"],
            )
        ]

    def _usage(self, agent: str, usage: Mapping[str, Any]) -> WorkflowEvent:
        model = self._agent_models.get(agent) or self._agent_models.get(self._root)
        event = usage_event(model, usage)
        if agent != self._root:
            # A sub-agent's prompt is its brief, not the conversation's context.
            event.metadata["purpose"] = "subagent"
        return event

    @staticmethod
    def _attributed(event: WorkflowEvent, agent: str) -> WorkflowEvent:
        event.metadata["agent"] = agent
        return event


def usage_event(model: "ResolvedModel | None", usage: Mapping[str, Any]) -> WorkflowEvent:
    """``LLM_USAGE`` for a call on ``model`` that reported ``usage``.

    ``usage`` is LangChain's ``usage_metadata``: cache reads and writes are its
    ``input_token_details``, thinking its ``output_token_details.reasoning``.
    """
    inputs = usage.get("input_token_details") or {}
    outputs = usage.get("output_token_details") or {}
    # When Anthropic breaks cache writes down by TTL, langchain-anthropic
    # reports them under these keys and sets "cache_creation" to 0.
    by_ttl = sum(
        inputs.get(key) or 0
        for key in ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")
    )
    return WorkflowEvent.llm_usage(
        model=str(model.ref) if model is not None else "unknown",
        prompt_tokens=usage.get("input_tokens"),
        completion_tokens=usage.get("output_tokens"),
        total_tokens=usage.get("total_tokens"),
        thinking_tokens=outputs.get("reasoning"),
        cached_tokens=inputs.get("cache_read"),
        cache_creation_tokens=by_ttl or inputs.get("cache_creation"),
    )


def _delegated_to(payload: dict) -> str | None:
    """The sub-agent a tools task runs, when it runs one ``task`` call.

    Each tool call is its own tools task (``Send``), so its input is a list
    holding that one call.
    """
    calls = payload.get("input")
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
        return None
    call = calls[0]
    if call.get("name") != TASK_TOOL:
        return None
    subagent = (call.get("args") or {}).get("subagent_type")
    return subagent if isinstance(subagent, str) else None


@dataclass
class _ModelCall:
    """One model call in flight: whose, why, on what, since when."""

    agent: str
    purpose: str
    model: str
    started: float

    def latency_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000


class RunCallback(AsyncCallbackHandler):
    """Collects one turn's events that the stream cannot carry.

    Attached to the turn's run config, LangChain hands it to every model call
    in the run, sub-agents' and tools' included. The manager drains it at each
    stream item, before that item's own events: what it collected happened
    inside the node whose output the item is.

    - A summary call (run metadata ``lc_source="summarization"``) reports its
      ``LLM_USAGE``, with ``purpose="summarization"``, on the model of the
      agent it summarized for.
    - A summary applied (``CONTEXT_SUMMARIZED``) is ``CONTEXT_TRIMMED``, with
      ``source="summarization"``.
    - With ``llm_log`` (``raw_llm_logging``), every model call is an
      ``LLM_REQUEST`` and an ``LLM_RESPONSE``, also written to the log. Each
      names its ``agent`` (``lc_agent_name``: for a call a tool made, the
      agent whose tool it is) and its ``purpose``: ``"agent"``,
      ``"summarization"`` or ``"tool"``. It records what LangChain sends, not
      the HTTP body, and adds no ``LLM_USAGE``: the stream reports usage.

    Args:
        root: The root agent, for a call no agent is named on.
        agent_models: Each agent's resolved model, by name.
        llm_log: Where raw LLM traffic is written; None logs none.
    """

    def __init__(
        self,
        *,
        root: str,
        agent_models: Mapping[str, "ResolvedModel"],
        llm_log: "LLMLogWriter | None" = None,
    ) -> None:
        super().__init__()
        self._root = root
        self._agent_models = agent_models
        self._llm_log = llm_log
        # A sync tool's model call reports from LangChain's executor thread.
        self._events: deque[WorkflowEvent] = deque()
        # The calls in flight that are reported when they end, by run id:
        # summary calls, and every call when logging.
        self._calls: dict[UUID, _ModelCall] = {}

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        metadata = metadata or {}
        purpose = _purpose(metadata)
        if self._llm_log is None and purpose != PURPOSE_SUMMARY:
            return
        # create_agent names its runs after the agent, and a tool's runs
        # inherit the name of the agent whose tool it is.
        agent = metadata.get("lc_agent_name") or self._root
        params = kwargs.get("invocation_params") or {}
        call = _ModelCall(
            agent=agent,
            purpose=purpose,
            model=_model_id(params) or self._model_ref(agent),
            started=time.perf_counter(),
        )
        self._calls[run_id] = call
        if self._llm_log is not None:
            sent = messages[0] if messages else []
            self._log(_request_event(call, sent, params), run_id)

    async def on_llm_end(self, response: "LLMResult", *, run_id: UUID, **kwargs: Any) -> None:
        call = self._calls.pop(run_id, None)
        if call is None:
            return
        if self._llm_log is not None:
            self._log(_response_event(call, response), run_id)
        if call.purpose == PURPOSE_SUMMARY:
            event = usage_event(self._agent_models.get(call.agent), _usage_of(response))
            event.metadata.update(purpose=PURPOSE_SUMMARY, agent=call.agent)
            self._emit(event)

    async def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        call = self._calls.pop(run_id, None)
        if call is None or self._llm_log is None:
            return
        event = WorkflowEvent.llm_response(
            model=call.model,
            error_code=type(error).__name__,
            error_message=truncate(str(error), _MAX_ERROR),
            author=call.agent,
        )
        event.metadata["latency_ms"] = call.latency_ms()
        self._log(_attributed_call(event, call), run_id)

    async def on_custom_event(
        self, name: str, data: Any, *, run_id: UUID, **kwargs: Any
    ) -> None:
        if name != CONTEXT_SUMMARIZED or not isinstance(data, dict):
            return
        agent = data.get("agent") or self._root
        event = WorkflowEvent.context_trimmed(
            messages_before=data["messages_before"],
            messages_after=data["messages_after"],
            source=_SUMMARY_SOURCE,
            agent=agent,
        )
        event.metadata["messages_summarized"] = data["messages_summarized"]
        self._emit(event)

    def _model_ref(self, agent: str) -> str:
        model = self._agent_models.get(agent)
        return str(model.ref) if model is not None else "unknown"

    def _log(self, event: WorkflowEvent, run_id: UUID) -> None:
        """Stream a raw record and write it (a failed write is skipped)."""
        event.metadata["run_id"] = str(run_id)
        self._emit(event)
        self._llm_log.write(event)

    def _emit(self, event: WorkflowEvent) -> None:
        self._events.append(event)

    def drain(self) -> list[WorkflowEvent]:
        """The events collected since the last drain, oldest first."""
        events: list[WorkflowEvent] = []
        while self._events:
            events.append(self._events.popleft())
        return events


def _purpose(metadata: Mapping[str, Any]) -> str:
    """Why a model call was made, from its run metadata.

    A summary call carries ``lc_source``; a call a tool made runs under the
    ``tools`` node; every other call is an agent's step (its ``model`` node).
    """
    if metadata.get("lc_source") == _SUMMARY_SOURCE:
        return PURPOSE_SUMMARY
    if metadata.get("langgraph_node") == "tools":
        return PURPOSE_TOOL
    return PURPOSE_AGENT


def _model_id(params: Mapping[str, Any]) -> str | None:
    """The model a chat model sends a call to, from its invocation parameters."""
    model = params.get("model") or params.get("model_name")
    return str(model) if model else None


def _request_event(
    call: _ModelCall, messages: list[BaseMessage], params: Mapping[str, Any]
) -> WorkflowEvent:
    """``LLM_REQUEST`` for a call sending ``messages`` with ``params``.

    The system prompt is ``system_instruction``, the tools are their names,
    and ``config`` is every other parameter sent (the cache marks among
    them: Claude's ``cache_control``).
    """
    system = [m.text for m in messages if isinstance(m, SystemMessage)]
    event = WorkflowEvent.llm_request(
        model=call.model,
        messages=[
            _message_record(message)
            for message in messages
            if not isinstance(message, SystemMessage)
        ],
        tools=_tool_names(params.get("tools")),
        system_instruction="\n".join(system) if system else None,
        config={key: value for key, value in params.items() if key != "tools"},
    )
    return _attributed_call(event, call)


def _response_event(call: _ModelCall, response: "LLMResult") -> WorkflowEvent:
    """``LLM_RESPONSE`` for a call that answered ``response``."""
    generation = next((g for gens in response.generations for g in gens), None)
    message = getattr(generation, "message", None)
    info = {**(getattr(generation, "generation_info", None) or {})}
    if isinstance(message, AIMessage):
        info = {**message.response_metadata, **info}
    finish = info.get("finish_reason") or info.get("stop_reason") or info.get("status")
    answered = isinstance(message, AIMessage)
    event = WorkflowEvent.llm_response(
        model=call.model,
        content=message.text if answered else None,
        finish_reason=str(finish) if finish else None,
        model_version=info.get("model_name") or info.get("model") or None,
        author=call.agent,
        raw_parts=list(message.content_blocks) if answered else None,
    )
    event.metadata["latency_ms"] = call.latency_ms()
    if answered and message.usage_metadata:
        event.metadata["usage"] = dict(message.usage_metadata)
    return _attributed_call(event, call)


def _attributed_call(event: WorkflowEvent, call: _ModelCall) -> WorkflowEvent:
    event.metadata.update(agent=call.agent, purpose=call.purpose)
    return event


def _message_record(message: BaseMessage) -> dict[str, Any]:
    """One message sent, as a raw log record holds it."""
    record: dict[str, Any] = {
        "role": _ROLES.get(message.type, message.type),
        "content": message.content,
    }
    if isinstance(message, AIMessage) and message.tool_calls:
        record["tool_calls"] = [
            {"name": call["name"], "args": call["args"], "id": call["id"]}
            for call in message.tool_calls
        ]
    if isinstance(message, ToolMessage):
        record["tool_call_id"] = message.tool_call_id
        if message.status == "error":
            record["status"] = "error"
    if message.name:
        record["name"] = message.name
    return record


def _tool_names(tools: Any) -> list[str] | None:
    """The names of the tools a call offers, in either provider format.

    Anthropic's schemas name themselves (``{"name": ...}``); OpenAI's, and
    the Gemini adapter's, name their function (``{"function": {"name": ...}}``).
    """
    if not tools:
        return None
    names = []
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        name = tool.get("name") or (tool.get("function") or {}).get("name")
        if name:
            names.append(str(name))
    return names


def _usage_of(response: "LLMResult") -> Mapping[str, Any]:
    """The usage a chat model reported for its (first) generation, or none."""
    for generations in response.generations:
        for generation in generations:
            usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
            if usage:
                return usage
    return {}
