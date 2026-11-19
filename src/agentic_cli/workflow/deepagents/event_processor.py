"""A Deep Agents turn's stream as ``WorkflowEvent``s, and a sub-agent run's.

A turn streams the root agent's graph: ``astream(input, config,
stream_mode=["updates"], subgraphs=True)``, each item ``(namespace, mode,
payload)``. A sub-agent's graph never streams there: its run is the
framework's ``task`` call (``agents/delegate.py``), which runs the
sub-agent's graph apart from its caller's stream
(``DeepAgentsWorkflowManager._run_agent``), through a processor of its own,
made for that run (``run=``), whose events reach the turn's live events.
Only the root namespace's items are the agent's: a graph one of its tools
runs under the config it inherits streams in a namespace of its own, and
none of its items is an event.

Each finished node output becomes events, as ADK reports them: one per
finished message, no tokens. Only the agent's own nodes speak:

- a ``model`` update's ``AIMessage``: ``THINKING`` (its reasoning blocks) and
  ``TEXT`` (its text blocks) in order, one ``TOOL_CALL`` per tool call, and
  ``LLM_USAGE`` on that agent's own model, with its context numbers
  (``context_trigger``, ``window_known``). Only the root replies to the user:
  a sub-agent run's ``TEXT`` is marked ``metadata["subagent"]`` (activity, not
  the reply) and its usage ``purpose="subagent"``, an inheriting one's
  included (its prompt holds the conversation, but it is not the main
  agent's context);
- a ``tools`` update's ``ToolMessage``s, including those a ``Command`` carried
  (the state tools, ``task``): ``TOOL_RESULT``, a JSON dict parsed back, a
  ``task`` call's shown from its envelope (``agents/runs.py::show_as_delegation``);
  and, for the root, its ``tasks``, when they changed: ``TASK_PROGRESS``, in
  the same step (a sub-agent's list is its own, kept with its run's record:
  the root reports only its own).

Middleware hooks' updates are not output (``PatchToolCallsMiddleware`` re-adds
the whole history when it patches it), and a model call inside a tool
(``generate_simple``) is no node at all, so it cannot leak into the answer.
Every event carries ``metadata["agent"]`` and its run (``WorkflowEvent.run``):
the root's own (``root_run``), or the sub-agent run the processor was made
for; a turn's own ``ERROR`` names none. The ``AGENT_RUN`` events are the
framework's (``Delegations``), not the stream's.

What happens inside a node and is no node output (a summary call and the
summary it made, and with ``raw_llm_logging`` every model call) reaches
``RunCallback`` instead, the LangChain callback handler of the turn's run
and of each sub-agent run, which hands it to the turn's ``LiveEvents`` as it
happens: a summary that starts is reported while the model node still waits
on it.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage

from agentic_cli.constants import truncate
from agentic_cli.tools._core.delegation import TASK_TOOL
from agentic_cli.tools._core.tasks import task_progress_data
from agentic_cli.workflow.agents.envelope import read_envelope
from agentic_cli.workflow.agents.runs import show_as_delegation
from agentic_cli.workflow.context import (
    OVERFLOW_ERROR_CODE,
    ContextBudget,
    is_context_overflow,
    overflow_message,
    usage_fields,
)
from agentic_cli.workflow.deepagents.middleware import answer_latency, mark_latency
from agentic_cli.workflow.deepagents.summarization import CONTEXT_SUMMARIZED
from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.sessions import RunRef, root_run
from agentic_cli.workflow.usage import UsageLedgers, langchain_tokens, llm_usage_event

if TYPE_CHECKING:
    from langchain_core.outputs import LLMResult

    from agentic_cli.models import ResolvedModel
    from agentic_cli.workflow.live_events import LiveEvents
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

class DeepAgentsEventProcessor:
    """Turns one agent's stream items into ``WorkflowEvent``s: the root's turn, or a sub-agent run.

    Args:
        root: The root agent's name: the agent of every event, unless the
            processor is made for a sub-agent run (``run``).
        agent_models: Each agent's resolved model, by name; ``LLM_USAGE`` is
            reported against the agent's own.
        session_id: Put on ``TEXT`` and ``THINKING``, as the other backends do.
        budgets: Each agent's context numbers, by name, which its
            ``LLM_USAGE`` carries; none without.
        ledgers: The manager's usage ledgers: each call's usage is recorded
            in its conversation's as it is made, before ``on_event``; its own
            when not given.
        run: The sub-agent run whose graph streams here: every event is
            that run's and its agent's, its text is activity
            (``metadata["subagent"]``) and its usage ``purpose="subagent"``.
            None: the root's turn.
    """

    def __init__(
        self,
        *,
        root: str,
        agent_models: Mapping[str, "ResolvedModel"],
        session_id: str | None = None,
        budgets: Mapping[str, ContextBudget] | None = None,
        ledgers: UsageLedgers | None = None,
        run: RunRef | None = None,
    ) -> None:
        self._root = root
        self._agent_models = agent_models
        self._session_id = session_id
        self._budgets: Mapping[str, ContextBudget] = budgets or {}
        self._run = run if run is not None else root_run(root)
        self._agent = self._run.agent
        # Call id -> tool name: a ToolMessage a Command carried may have no
        # name of its own.
        self._call_names: dict[str, str] = {}
        self._reported_tasks: list[dict] | None = None
        self._ledgers = ledgers if ledgers is not None else UsageLedgers()

    @property
    def run(self) -> RunRef:
        """The run every event this processor makes is: the root's own, or a sub-agent run."""
        return self._run

    def process(
        self, namespace: Iterable[str], mode: str, payload: Any
    ) -> list[WorkflowEvent]:
        """The events of one stream item (none for most).

        ``namespace`` is the item's place in the stream. Only the agent's own
        graph's items (the root namespace, ``()``) are its events: a sub-agent
        runs apart, on a stream of its own, so an item in a namespace of its
        own is a graph one of its tools ran under the config it inherited,
        whose model text is no agent's reply.
        """
        if tuple(namespace) or mode != "updates" or not isinstance(payload, dict):
            return []
        events: list[WorkflowEvent] = []
        for node, update in payload.items():
            for values in update if isinstance(update, list) else [update]:
                if not isinstance(values, dict):
                    continue
                if node == "model":
                    events += self._model_output(values)
                elif node == "tools":
                    events += self._tools_output(values)
        for event in events:
            # Recorded as it is made: an app's on_event cannot drop a row.
            self._ledgers.record(event)
        return events

    def turn_failed(self, error: BaseException) -> WorkflowEvent:
        """The one ``ERROR`` that ends a turn whose stream raised ``error``.

        ``WorkflowEvent.turn_failed``'s (the base ``process()``'s contract),
        the agent's. A context overflow (a provider's, or Deep Agents' own
        when a summary still does not fit) is told in one sentence naming the
        agent's model, with ``error_code`` ``CONTEXT_OVERFLOW``, as on ADK.
        """
        if is_context_overflow(error):
            model = self._agent_models.get(self._agent) or self._agent_models.get(self._root)
            event = WorkflowEvent.turn_failed(
                error,
                overflow_message(str(model.ref) if model is not None else "the model"),
                error_code=OVERFLOW_ERROR_CODE,
            )
        else:
            event = WorkflowEvent.turn_failed(error)
        event.metadata["agent"] = self._agent
        return event

    # -- node outputs ----------------------------------------------------------

    def _model_output(self, values: dict) -> list[WorkflowEvent]:
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
                    text = WorkflowEvent.text(block["text"], self._session_id)
                    if not self._run.is_root:
                        # Only the root replies to the user: a sub-agent's
                        # text is its work, read by its parent as the
                        # result of its ``task`` call.
                        text.metadata["subagent"] = True
                    events.append(text)
            for call in message.tool_calls:
                self._call_names[call["id"]] = call["name"]
                event = WorkflowEvent.tool_call(call["name"], call["args"])
                event.metadata["tool_call_id"] = call["id"]
                events.append(event)
            if message.usage_metadata:
                events.append(self._usage(message.usage_metadata, answer_latency(message)))
        return [self._attributed(event) for event in events]

    def _tools_output(self, values: dict) -> list[WorkflowEvent]:
        events: list[WorkflowEvent] = []
        for message in values.get("messages") or ():
            if isinstance(message, ToolMessage):
                events.append(self._tool_result(message))
        if "tasks" in values and self._run.is_root:
            events += self._progress(values["tasks"])
        return [self._attributed(event) for event in events]

    def _tool_result(self, message: ToolMessage) -> WorkflowEvent:
        """``TOOL_RESULT`` for ``message``; a delegation's shown from its envelope."""
        name = message.name or self._call_names.get(message.tool_call_id) or "unknown"
        result, success, error = _result_of(message)
        event = WorkflowEvent.tool_result(name, result, success=success, error=error)
        event.metadata["tool_call_id"] = message.tool_call_id
        found = read_envelope(result) if name == TASK_TOOL else None
        if found is not None:
            show_as_delegation(event, found)
        return event

    def _progress(self, tasks: Any) -> list[WorkflowEvent]:
        """``TASK_PROGRESS`` for a changed task list (the root's)."""
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

    def _usage(self, usage: Mapping[str, Any], latency_ms: float | None) -> WorkflowEvent:
        model = self._agent_models.get(self._agent) or self._agent_models.get(self._root)
        event = usage_event(model, usage, latency_ms=latency_ms)
        event.metadata.update(usage_fields(self._budgets.get(self._agent)))
        if not self._run.is_root:
            # A sub-agent's prompt is its brief, not the conversation's context.
            event.metadata["purpose"] = "subagent"
        return event

    def _attributed(self, event: WorkflowEvent) -> WorkflowEvent:
        event.metadata["agent"] = self._agent
        event.run = self._run
        return event


def usage_event(
    model: "ResolvedModel | None",
    usage: Mapping[str, Any],
    *,
    latency_ms: float | None = None,
) -> WorkflowEvent:
    """``LLM_USAGE`` for a call on ``model`` that reported ``usage``.

    ``usage`` is LangChain's ``usage_metadata``, read as every partner
    package writes it (``workflow/usage.py``'s ``langchain_tokens``): cache
    reads and writes are its ``input_token_details``, thinking its
    ``output_token_details.reasoning``. ``latency_ms`` is how long the call
    took, when it was measured.
    """
    return llm_usage_event(
        str(model.ref) if model is not None else "unknown",
        langchain_tokens(usage),
        total_tokens=usage.get("total_tokens"),
        latency_ms=latency_ms,
    )


def _result_of(message: ToolMessage) -> tuple[Any, bool, str | None]:
    """A tool message's result (a JSON dict parsed back), whether it succeeded, and its error.

    It failed when it is an error message or a dict that says ``success``
    false; its error is the dict's ``error``, else an error message's text.
    """
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
    return result, success, error


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
    """Reports the events of a turn's run, or a sub-agent run's, that the stream cannot carry.

    Attached to the run's config (the turn's root agent's, and each
    sub-agent run's own: ``DeepAgentsWorkflowManager._run_agent``), LangChain
    hands it to every model call in the run, its tools' included. Each event
    goes out through ``live``, the turn's, as it happens (the manager merges
    it into the turn's stream), so it shows while the node it happened in
    still runs.

    - Every model call is timed, from its start to its end, and its answer
      marked with the milliseconds (``middleware.mark_latency``:
      ``response_metadata["agentic_cli"]["latency_ms"]``, beside any other
      mark), before any middleware sees it: the event processor gives the
      step's ``LLM_USAGE`` its ``latency_ms`` from it. No adapter sends it.
    - A summary call (run metadata ``lc_source="summarization"``) reports
      ``CONTEXT_SUMMARIZING`` as it starts, then its ``LLM_USAGE``, with
      ``purpose="summarization"``, on the model of the agent it summarized
      for, with that agent's context numbers and the call's latency.
    - A summary applied (``CONTEXT_SUMMARIZED``) is ``CONTEXT_TRIMMED``, with
      ``source="summarization"``, the messages it stands for and the
      conversation's tokens before and after.
    - With ``llm_log`` (``raw_llm_logging``), every model call is an
      ``LLM_REQUEST`` and an ``LLM_RESPONSE``, also written to the log. Each
      names its ``agent`` (``lc_agent_name``: for a call a tool made, the
      agent whose tool it is) and its ``purpose``: ``"agent"``,
      ``"summarization"`` or ``"tool"``; a sub-agent's call, its run too (the
      active run ``LiveEvents.emit`` gives the event, which the log writes as
      ``run``). It records what LangChain sends, not the HTTP body, and adds
      no ``LLM_USAGE``: the stream reports usage.

    Args:
        root: The root agent, for a call no agent is named on.
        agent_models: Each agent's resolved model, by name.
        llm_log: Where raw LLM traffic is written; None logs none.
        live: The turn's live events, which every event goes out through
            (and which give a failed summary's warning once a turn).
        budgets: Each agent's context numbers, by name, which a summary's
            ``LLM_USAGE`` carries; none without.
        ledgers: The manager's usage ledgers: a summary's usage is recorded
            in its conversation's before it goes out.
    """

    def __init__(
        self,
        *,
        root: str,
        agent_models: Mapping[str, "ResolvedModel"],
        llm_log: "LLMLogWriter | None" = None,
        live: "LiveEvents",
        budgets: Mapping[str, ContextBudget] | None = None,
        ledgers: UsageLedgers | None = None,
    ) -> None:
        super().__init__()
        self._ledgers = ledgers
        self._root = root
        self._agent_models = agent_models
        self._budgets: Mapping[str, ContextBudget] = budgets or {}
        self._llm_log = llm_log
        self._live = live
        # Every call in flight, by run id: each is timed, and a summary's,
        # and every call when logging, reported as it ends.
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
        if purpose == PURPOSE_SUMMARY:
            self._emit(WorkflowEvent.context_summarizing(agent))
        if self._llm_log is not None:
            sent = messages[0] if messages else []
            self._log(_request_event(call, sent, params), run_id)

    async def on_llm_end(self, response: "LLMResult", *, run_id: UUID, **kwargs: Any) -> None:
        call = self._calls.pop(run_id, None)
        if call is None:
            return
        latency = call.latency_ms()
        for generation in (g for generations in response.generations for g in generations):
            message = getattr(generation, "message", None)
            if isinstance(message, AIMessage):
                mark_latency(message, latency)
        if self._llm_log is not None:
            self._log(_response_event(call, response, latency), run_id)
        if call.purpose == PURPOSE_SUMMARY:
            event = usage_event(
                self._agent_models.get(call.agent), _usage_of(response), latency_ms=latency
            )
            event.metadata.update(
                purpose=PURPOSE_SUMMARY, agent=call.agent,
                **usage_fields(self._budgets.get(call.agent)),
            )
            if self._ledgers is not None:
                self._ledgers.record(event)
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
        from agentic_cli.workflow.context import SUMMARY_FAILED, SUMMARY_FAILED_CODE
        from agentic_cli.workflow.deepagents.summarization import SUMMARY_NOT_WRITTEN

        if not isinstance(data, dict):
            return
        agent = data.get("agent") or self._root
        if name == SUMMARY_NOT_WRITTEN:
            # Once a turn, however often its calls, and its runs', try again.
            if self._live.first(SUMMARY_FAILED_CODE):
                text = SUMMARY_FAILED if agent == self._root else f"{agent}: {SUMMARY_FAILED}"
                self._emit(WorkflowEvent.error(
                    text, error_code=SUMMARY_FAILED_CODE, recoverable=True
                ))
            return
        if name != CONTEXT_SUMMARIZED:
            return
        self._emit(WorkflowEvent.context_trimmed(
            messages_before=data["messages_before"],
            messages_after=data["messages_after"],
            source=_SUMMARY_SOURCE,
            agent=agent,
            messages_summarized=data["messages_summarized"],
            tokens_before=data.get("tokens_before"),
            tokens_after=data.get("tokens_after"),
        ))

    def _model_ref(self, agent: str) -> str:
        model = self._agent_models.get(agent)
        return str(model.ref) if model is not None else "unknown"

    def _log(self, event: WorkflowEvent, run_id: UUID) -> None:
        """Stream a raw record and write it (a failed write is skipped)."""
        event.metadata["run_id"] = str(run_id)
        self._emit(event)
        self._llm_log.write(event)

    def _emit(self, event: WorkflowEvent) -> None:
        self._live.emit(event)


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


def _response_event(
    call: _ModelCall, response: "LLMResult", latency_ms: float
) -> WorkflowEvent:
    """``LLM_RESPONSE`` for a call that answered ``response`` after ``latency_ms``."""
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
    event.metadata["latency_ms"] = latency_ms
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
