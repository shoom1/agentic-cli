"""ADK event processing: converts ADK events into WorkflowEvents.

Extracted from GoogleADKWorkflowManager to isolate event conversion
logic that runs on every message in the hot path.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from typing import AsyncGenerator, Any, Callable

from google.genai import types

from agentic_cli.tools._core.tasks import task_progress_data
from agentic_cli.workflow.adk.branches import branch_run
from agentic_cli.workflow.adk.call_clock import call_latency
from agentic_cli.workflow.adk.interrupted_calls_plugin import ABORTED
from agentic_cli.workflow.adk.subagent_outcome_plugin import OUTCOME_KEY
from agentic_cli.workflow.agents.outcome import (
    CUT,
    REFUSAL_ERROR,
    REFUSED,
    cut_off_warning,
)
from agentic_cli.workflow.agents.runs import (
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_INTERRUPTED,
    RUN_REFUSED,
    RUN_STARTED,
    SUBAGENT_ERROR_PREFIX,
    AgentRuns,
    result_head,
    run_title,
    show_as_delegation,
)
from agentic_cli.workflow.context import OVERFLOW_ERROR_CODE, ContextBudget, usage_fields
from agentic_cli.workflow.events import WorkflowEvent, EventType
from agentic_cli.workflow.sessions import RunRef, get_active_turn, root_run
from agentic_cli.workflow.usage import UsageLedgers, adk_usage_event
from agentic_cli.logging import Loggers

logger = Loggers.workflow()

#: The error codes of a reply that is no answer: a Gemini candidate's finish
#: reason (any but STOP) or a prompt's block reason, which ADK records on a
#: response with no content (``LlmResponse.create``), and the code ADK gives a
#: STOP reply with no parts, or only thoughts or blank text
#: (``apply_empty_response_policy``). A failed call's error event carries an
#: API status or an exception's class name instead.
NO_ANSWER_CODES = (
    frozenset({reason.value for reason in types.FinishReason} - {types.FinishReason.STOP.value})
    | frozenset(reason.value for reason in types.BlockedReason)
    | {"MODEL_RETURNED_NO_CONTENT"}
)

#: Finish reasons that are no reason: the reply ended as it should.
_NORMAL_FINISH = frozenset(
    {types.FinishReason.STOP.value, types.FinishReason.FINISH_REASON_UNSPECIFIED.value}
)


class ADKEventProcessor:
    """Converts ADK events into framework-agnostic WorkflowEvents.

    Owns an optional on_event callback for event transformation/filtering.

    Only the agents that reply to the user write the reply
    (``reply_agents``): the root, and every chat-mode agent it can transfer
    to, which takes the conversation (a native ADK YAML app's). An event
    another agent wrote (a sub-agent its parent delegated to,
    ``workflow/adk/delegation.py``) is its work: its text is marked
    ``metadata["agent"]``/``["subagent"]`` (the CLI shows it as activity, not
    as the reply), and its call's usage is ``purpose="subagent"``, since that
    prompt is the sub-agent's brief, not the conversation's context. A
    sub-agent's failed model call does not end the turn on ADK: ADK's
    delegation tool hands its parent "Error running sub-agent: …" and the
    parent carries on. ADK records the provider's reason on the sub-agent's
    error event, which becomes a recoverable ``ERROR``, "<agent> failed:
    <reason>", so the user is told why. With neither a root nor reply agents,
    nothing is marked.

    A call's ``LLM_USAGE`` is credited to the agent that made it, the event's
    author: ``metadata["agent"]``, that agent's own model (``models``; the
    default model for an agent the framework did not build) and its context
    numbers (``context_trigger``, ``window_known``, from ``budgets``), which
    the CLI shows the main agent's prompt against. Its counts are normalized
    by that model's provider kind (``kinds``; ``workflow/usage.py``'s
    ``adk_tokens``: the thinking counts in the output, a Claude cache write
    in ``cache_creation_tokens``; an agent ``kinds`` does not hold runs on a
    model the manager did not resolve, a native ADK YAML app's own, and is
    read as an unknown kind's), and it is recorded in the conversation's
    ledger (``ledgers``) as it is made, before ``on_event``, with how long
    the call took (``latency_ms``: the mark ``CallLatencyPlugin`` puts on
    the answer, ``workflow/adk/call_clock.py``). A sub-agent run's end
    carries what the run used, by model (``metadata["usage"]``).

    A tool's result is a failed ``TOOL_RESULT`` when it says so (``success``
    false, an ``error`` value) or is an MCP tool's error result (``isError``),
    its error in ``metadata["error"]``, as on Deep Agents.

    A tool that wrote the session's task list (``save_tasks``: ``tasks`` in
    the event's state delta) is followed by ``TASK_PROGRESS``, after its
    result, when the list differs from the one this turn last reported
    (``start_turn`` forgets it), as on Deep Agents: a call that changed
    nothing reports nothing, and a list whose tasks are all done stays in the
    session for ``get_tasks``.

    A conversation the model of an agent that replies no longer takes ends
    its turn with an error response from the overflow handler
    (``workflow/adk/overflow_plugin.py``), ``error_code="CONTEXT_OVERFLOW"``:
    it becomes the turn's ``ERROR``, with the handler's clear text. A reply of
    such an agent's that is no answer (nothing but thoughts or blank text,
    and an error code of ``NO_ANSWER_CODES``: a safety stop, a blocked prompt,
    a malformed function call, every token spent thinking, nothing at all; or,
    with no error code, a finish reason other than STOP: a Claude refusal, a
    reply cut at its token limit) ends the turn too, raising nothing: it
    becomes the turn's ``ERROR``, "<model> returned no answer", its code
    the error code or finish reason. Both are the turn's one ``ERROR``, with
    a failed turn's fields (``WorkflowEvent.turn_failed``: ``rate_limited``
    false, the code as ``error_type``). No other error code of the main agent
    is mapped: ADK records one for an exception that then leaves the run,
    which the manager reports as the turn's one ``ERROR``.

    Every event an agent wrote names its run (``WorkflowEvent.run``): the run
    of the delegation its branch names (``workflow/adk/branches.py``), else
    the agent's own at depth 0 (``root_run`` for the root; an agent the root
    transfers to is ``RunRef(<name>, <name>, None, None, 0)``); the user's
    message names none, nor does the turn's own ``ERROR``. A tool's call and
    result name the call (``tool_call_id``), as on Deep Agents. A call that
    delegates to one of the calling agent's own single-turn sub-agents
    (``delegates``, by the caller, the event's author: ADK names the tool
    after the sub-agent) is followed by the ``AGENT_RUN`` that starts the
    sub-agent's run, and its result by the one that ends it: ``failed`` for
    ADK's answer for a sub-agent that failed (``Error running sub-agent:
    …``) or an error result (``success`` false: the tool raised), with its
    error, ``interrupted`` for the seal of a stopped turn, ``refused`` when
    ``SubAgentOutcomePlugin`` marked the run's last answer refused, else
    ``completed`` (``partial``, and followed by one warning,
    ``cut_off_warning``, when it marked the answer cut); the result itself is
    shown as on Deep Agents (``agents/runs.py::show_as_delegation``: the
    answer's first line, a failed one as a failed result). Any other call is
    an ordinary tool call, with no run: a name the caller has no such
    sub-agent for runs none (ADK answers an unknown tool). Each ``AGENT_RUN``
    is recorded in ``runs`` as it is made, before ``on_event``.

    Args:
        model: The default model, which usage names for an agent ``models``
            does not hold.
        on_event: Optional hook to transform/filter events before yielding.
        root: The root agent's name: the one agent that replies, when
            ``reply_agents`` is not given.
        models: Each agent's model reference, by name.
        budgets: Each agent's context numbers, by name.
        reply_agents: The agents that reply to the user, the root among them
            (``workflow/adk/build.py::reply_agents``); None for the root alone.
        runs: The manager's sub-agent runs, which this processor names and
            records; its own when not given.
        delegates: Each agent's single-turn sub-agents, which it delegates
            to through a tool named after each, by the agent's name
            (``build.delegated_agents``); a collection of names alone is the
            root's.
        kinds: Each agent's model's provider kind (``ProviderKind``'s value), by name.
        ledgers: The manager's usage ledgers, which this processor records
            each call's usage in; its own when not given.

    The manager builds one at each initialization, from the app it built.
    """

    def __init__(
        self,
        model: str,
        on_event: Callable[[WorkflowEvent], WorkflowEvent | None] | None = None,
        root: str | None = None,
        models: Mapping[str, str] | None = None,
        budgets: Mapping[str, ContextBudget] | None = None,
        reply_agents: Collection[str] | None = None,
        runs: AgentRuns | None = None,
        delegates: Mapping[str, Collection[str]] | Collection[str] = (),
        kinds: Mapping[str, str] | None = None,
        ledgers: UsageLedgers | None = None,
    ) -> None:
        self._model = model
        self._on_event = on_event
        self.root = root
        self.reply_agents: Collection[str] | None = (
            None if reply_agents is None else frozenset(reply_agents)
        )
        self.models: Mapping[str, str] = models or {}
        self.budgets: Mapping[str, ContextBudget] = budgets or {}
        # The task list this turn last reported, None before any.
        self._reported_tasks: list | None = None
        self.runs = runs if runs is not None else AgentRuns()
        if isinstance(delegates, Mapping):
            by_agent = {name: frozenset(own) for name, own in delegates.items()}
        else:
            by_agent = {root: frozenset(delegates)} if root is not None and delegates else {}
        self.delegates: dict[str, frozenset[str]] = by_agent
        # A sub-agent run's last answer's outcome (SubAgentOutcomePlugin's
        # mark), by run id, until its delegation's result ends the run.
        self._outcomes: dict[str, str] = {}
        self.kinds: Mapping[str, str] = kinds or {}
        self.ledgers = ledgers if ledgers is not None else UsageLedgers()

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        self._model = value

    @property
    def on_event(self) -> Callable[[WorkflowEvent], WorkflowEvent | None] | None:
        return self._on_event

    @on_event.setter
    def on_event(self, value: Callable[[WorkflowEvent], WorkflowEvent | None] | None) -> None:
        self._on_event = value

    def start_turn(self) -> None:
        """Forget the task list reported (a turn reports the first one it
        sees), and the outcomes of runs a stopped turn left open."""
        self._reported_tasks = None
        self._outcomes.clear()

    async def process_event(
        self,
        adk_event: Any,
        session_id: str,
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Process a single ADK event and yield WorkflowEvents.

        Args:
            adk_event: ADK event containing content parts.
            session_id: Current session ID.

        Yields:
            WorkflowEvent objects for each processed part.
        """
        subagent = self._subagent(adk_event)
        run = self._run_ref(adk_event)
        outcome = (getattr(adk_event, "custom_metadata", None) or {}).get(OUTCOME_KEY)
        if outcome and run is not None and run.run is not None:
            self._outcomes[run.run] = outcome

        error_code = getattr(adk_event, "error_code", None)
        if subagent is not None and error_code:
            # The root's own failure leaves the run, which the manager
            # reports as the turn's ERROR; this one never does.
            error = WorkflowEvent.error(
                f"{subagent} failed: {adk_event.error_message or error_code}",
                error_code=error_code,
                recoverable=True,
            )
            error.metadata["agent"] = subagent
            error.run = run
            error = self._apply_hook(error)
            if error:
                yield error
        elif error_code == OVERFLOW_ERROR_CODE:
            # The turn's one ERROR (the base process()'s contract), though
            # no exception ended it: the overflow handler answered the call.
            error = self._apply_hook(WorkflowEvent.turn_failed(
                message=adk_event.error_message or error_code,
                error_code=OVERFLOW_ERROR_CODE,
            ))
            if error:
                yield error
        elif subagent is None and (code := _no_answer(adk_event)) is not None:
            model = self.models.get(self._agent(adk_event), self._model)
            message = adk_event.error_message
            # ADK's own message for an empty reply says no more than the code.
            said = f": {message}" if message and code != "MODEL_RETURNED_NO_CONTENT" else ""
            error = self._apply_hook(WorkflowEvent.turn_failed(
                message=f"{model} returned no answer{said}", error_code=code
            ))
            if error:
                yield error

        # Extract usage_metadata if available
        if hasattr(adk_event, "usage_metadata") and adk_event.usage_metadata:
            usage = adk_event.usage_metadata
            invocation_id = getattr(adk_event, "invocation_id", None)
            agent = self._agent(adk_event)

            usage_event = adk_usage_event(
                self.models.get(agent, self._model),
                usage,
                self.kinds.get(agent),
                invocation_id=invocation_id,
                latency_ms=call_latency(adk_event),
            )
            if agent is not None:
                usage_event.metadata["agent"] = agent
            usage_event.metadata.update(usage_fields(self.budgets.get(agent)))
            if subagent is not None:
                usage_event.metadata["purpose"] = "subagent"
            usage_event.run = run
            # Recorded before the hook: an app's on_event cannot drop a row.
            self.ledgers.record(usage_event)
            usage_event = self._apply_hook(usage_event)
            if usage_event:
                yield usage_event

        parts = adk_event.content.parts if adk_event.content else None
        for part in parts or ():
            # Made before the part's own event, which a delegation's end shows.
            run_event = self._delegation_run(part, run, sealed=error_code == ABORTED)
            workflow_event = self.process_part(part, session_id)
            if workflow_event:
                workflow_event.run = run
                if subagent is not None and workflow_event.type is EventType.TEXT:
                    workflow_event.metadata.update(agent=subagent, subagent=True)
                if run_event is not None and workflow_event.type is EventType.TOOL_RESULT:
                    show_as_delegation(
                        workflow_event, _result_text(part.function_response.response), run_event
                    )
                workflow_event = self._apply_hook(workflow_event)
                if workflow_event:
                    yield workflow_event
            if run_event is not None:
                # A run cut short is followed by its one warning.
                for event in (run_event, cut_off_warning(run_event)):
                    event = self._apply_hook(event) if event is not None else None
                    if event:
                        yield event

        progress = self._progress(adk_event)
        if progress is not None:
            progress.run = run
            progress = self._apply_hook(progress)
            if progress:
                yield progress

    def _progress(self, adk_event: Any) -> WorkflowEvent | None:
        """``TASK_PROGRESS`` for a task list ``adk_event`` wrote, when it changed.

        A sub-agent's list is the session's too: the coordinator reads it,
        and reporting it again would say nothing new.
        """
        delta = getattr(getattr(adk_event, "actions", None), "state_delta", None) or {}
        tasks = delta.get("tasks")
        if not isinstance(tasks, list) or tasks == self._reported_tasks:
            return None
        self._reported_tasks = tasks
        progress = task_progress_data(tasks)
        if progress is None:
            return None
        return WorkflowEvent.task_progress(
            display=progress["display"],
            progress=progress["progress"],
            current_task_id=progress["current_task_id"],
            current_task_description=progress["current_task_description"],
        )

    def _subagent(self, adk_event: Any) -> str | None:
        """The sub-agent that wrote ``adk_event``, or None (an agent that
        replies, the user)."""
        agent = self._agent(adk_event)
        replying = self.reply_agents
        if replying is None:
            replying = () if self.root is None else (self.root,)
        if agent is None or not replying or agent in replying:
            return None
        return agent

    @staticmethod
    def _agent(adk_event: Any) -> str | None:
        """The agent that wrote ``adk_event``, or None (the user's own)."""
        author = getattr(adk_event, "author", None)
        return author if author and author != "user" else None

    def _run_ref(self, adk_event: Any) -> RunRef | None:
        """The run that wrote ``adk_event``: its delegation branch's, else the
        agent's own at depth 0; None for the user's own."""
        agent = self._agent(adk_event)
        if agent is None:
            return None
        run = branch_run(self.runs, get_active_turn(), getattr(adk_event, "branch", None))
        if run is not None:
            return run
        if self.root is None or agent == self.root:
            return root_run(agent)
        return RunRef(agent=agent, instance=agent, run=None, parent_run=None, depth=0)

    def _run_usage(self, run: RunRef) -> dict[str, dict[str, int]]:
        """What ``run`` used, by model, from its conversation's ledger."""
        return self.ledgers.ledger(get_active_turn()).run_usage(run.run or "")

    def _delegation_run(
        self, part: Any, parent: RunRef | None, *, sealed: bool
    ) -> WorkflowEvent | None:
        """The ``AGENT_RUN`` a delegation ``part`` starts or ends, recorded; else None.

        A delegation is a call of one of ``parent``'s agent's own single-turn
        sub-agents (``delegates``); any other part is none.

        Args:
            part: A part of an event of ``parent``'s.
            parent: The run that made the call (or got its result).
            sealed: Whether the event seals a stopped turn's calls (ADK's
                ``INVOCATION_ABORTED``).
        """
        call = getattr(part, "function_call", None)
        response = getattr(part, "function_response", None)
        own = self.delegates.get(parent.agent, frozenset()) if parent is not None else frozenset()
        if call is not None and call.name in own and call.id:
            child = self.runs.run_for(get_active_turn(), parent, call.id, call.name)
            request = (call.args or {}).get("request")
            event = WorkflowEvent.agent_run(
                child, RUN_STARTED, call_id=call.id,
                title=run_title(request if isinstance(request, str) else ""),
            )
        elif response is not None and response.name in own and response.id:
            child = self.runs.run_for(get_active_turn(), parent, response.id, response.name)
            text = _result_text(response.response)
            usage = self._run_usage(child)
            outcome = self._outcomes.pop(child.run or "", None)
            succeeded, error = _outcome(response.response)
            if sealed:
                event = WorkflowEvent.agent_run(
                    child, RUN_INTERRUPTED, call_id=response.id, result=result_head(text),
                    usage=usage,
                )
            elif text.startswith(SUBAGENT_ERROR_PREFIX):
                event = WorkflowEvent.agent_run(
                    child, RUN_FAILED, call_id=response.id, result="",
                    error=text.removeprefix(SUBAGENT_ERROR_PREFIX), usage=usage,
                )
            elif not succeeded:
                event = WorkflowEvent.agent_run(
                    child, RUN_FAILED, call_id=response.id, result="", error=error or text,
                    usage=usage,
                )
            elif outcome == REFUSED:
                event = WorkflowEvent.agent_run(
                    child, RUN_REFUSED, call_id=response.id, result=result_head(text),
                    error=REFUSAL_ERROR, usage=usage,
                )
            else:
                event = WorkflowEvent.agent_run(
                    child, RUN_COMPLETED, call_id=response.id, result=result_head(text),
                    partial=outcome == CUT, usage=usage,
                )
        else:
            return None
        self.runs.observe(event)
        return event

    def process_part(self, part: Any, session_id: str) -> WorkflowEvent | None:
        """Process a single part from the agent response.

        Args:
            part: ADK response part.
            session_id: Current session ID.

        Returns:
            WorkflowEvent if the part produces one, None otherwise.
        """
        if part.text:
            if getattr(part, "thought", False):
                return WorkflowEvent.thinking(part.text, session_id)
            return WorkflowEvent.text(part.text, session_id)

        if part.function_call:
            logger.debug("tool_call", tool_name=part.function_call.name)
            return _with_call_id(
                WorkflowEvent.tool_call(
                    tool_name=part.function_call.name,
                    tool_args=dict(part.function_call.args) if part.function_call.args else None,
                ),
                getattr(part.function_call, "id", None),
            )

        # Handle function response (tool result)
        if hasattr(part, "function_response") and part.function_response:
            tool_name = part.function_response.name
            response = part.function_response.response
            success, error = _outcome(response)

            logger.debug("tool_result", tool_name=tool_name, success=success)

            event = WorkflowEvent.tool_result(
                tool_name=tool_name,
                result=response,
                success=success,
                error=error,
            )
            if _is_mcp_error(response):
                # Shown as any failed result is ("Failed: <error>").
                event.content = f"Failed: {error}"
            return _with_call_id(event, getattr(part.function_response, "id", None))

        if part.code_execution_result:
            return WorkflowEvent.code_execution(str(part.code_execution_result.outcome))

        if part.executable_code:
            return WorkflowEvent.executable_code(
                part.executable_code.code,
                str(part.executable_code.language),
            )

        if part.file_data:
            return WorkflowEvent.file_data(part.file_data.display_name or "unknown")

        return None

    def _apply_hook(self, event: WorkflowEvent) -> WorkflowEvent | None:
        """Apply optional event transformation hook."""
        if self._on_event:
            return self._on_event(event)
        return event


def _is_mcp_error(response: Any) -> bool:
    """Whether ``response`` is an MCP tool's error result.

    ADK's ``McpTool`` returns the server's ``CallToolResult`` as a dict
    (``model_dump(exclude_none=True, mode="json")``), ``isError`` true for a
    failure the server reports.
    """
    return isinstance(response, dict) and response.get("isError") is True


def _mcp_error_text(content: list[Any]) -> str:
    """An MCP error result's error, as ``langchain-mcp-adapters`` words it on
    Deep Agents: its text blocks, a line each, else what it holds instead."""
    texts = [
        str(block.get("text", "")) for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    if texts:
        return "\n".join(texts)
    if content:
        return (
            "MCP tool returned an error with no text content "
            f"({len(content)} non-text content block(s))."
        )
    return "MCP tool returned an error with empty content."


def _outcome(response: Any) -> tuple[bool, str | None]:
    """Whether a tool's result is a success, and its error when it is not.

    The framework's shape fails with an ``error`` value or ``success``
    false, and its error is that value. An MCP error result fails too, its
    error in the words Deep Agents reports (``_mcp_error_text``).
    """
    if _is_mcp_error(response):
        return False, _mcp_error_text(response.get("content") or [])
    if not isinstance(response, dict):
        return True, None
    # An error value, not just the key; an explicit "success" decides.
    success = not response.get("error")
    if "success" in response:
        success = response["success"]
    error = response.get("error")
    return success, error if not success and isinstance(error, str) else None


def _with_call_id(event: WorkflowEvent, call_id: Any) -> WorkflowEvent:
    """``event`` naming the model's id for its call (``tool_call_id``), as on Deep Agents."""
    if isinstance(call_id, str) and call_id:
        event.metadata["tool_call_id"] = call_id
    return event


def _result_text(response: Any) -> str:
    """A delegation's result as text: ADK keeps a text result under ``result``."""
    if isinstance(response, dict):
        result = response.get("result")
        return result if isinstance(result, str) else json.dumps(response, default=str)
    return "" if response is None else str(response)


def _no_answer(adk_event: Any) -> str | None:
    """Why a whole reply is no answer (its error code or finish reason), or None.

    A part counts unless it is a thought or blank text (a function call
    counts); a streamed fragment (``partial``) is never judged.
    """
    if getattr(adk_event, "partial", False):
        return None
    content = getattr(adk_event, "content", None)
    for part in (content.parts if content is not None else None) or []:
        if not part.thought and (part.text is None or part.text.strip()):
            return None
    error_code = getattr(adk_event, "error_code", None)
    if error_code is not None:
        code = getattr(error_code, "value", error_code)
        return code if code in NO_ANSWER_CODES else None
    finish = getattr(adk_event, "finish_reason", None)
    if finish is None:
        return None
    reason = getattr(finish, "value", finish)
    return None if reason in _NORMAL_FINISH else reason
