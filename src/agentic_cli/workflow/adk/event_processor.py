"""ADK event processing: converts ADK events into WorkflowEvents.

Extracted from GoogleADKWorkflowManager to isolate event conversion
logic that runs on every message in the hot path.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping
from typing import AsyncGenerator, Any, Callable

from google.genai import types

from agentic_cli.tools._core.delegation import TASK_TOOL
from agentic_cli.tools._core.tasks import task_progress_data
from agentic_cli.workflow.adk.call_clock import call_latency
from agentic_cli.workflow.adk.subagent_outcome_plugin import OUTCOME_KEY
from agentic_cli.workflow.agents.envelope import read_envelope
from agentic_cli.workflow.agents.runs import show_as_delegation
from agentic_cli.workflow.context import OVERFLOW_ERROR_CODE, ContextBudget, usage_fields
from agentic_cli.workflow.events import WorkflowEvent, EventType
from agentic_cli.workflow.sessions import RunRef, root_run
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
#: ADK's code for a reply with nothing in it: no parts, or only thoughts or
#: blank text (``apply_empty_response_policy``).
_NO_CONTENT = "MODEL_RETURNED_NO_CONTENT"
#: ADK's Claude adapter's words for a refusal or a reply cut at its token
#: limit with nothing in it, from google-adk 2.15 (``anthropic_llm``'s
#: ``_empty_response_error``): "Claude stopped with SAFETY ('refusal') and no
#: content.", the reason also its error code. Like ``_NO_CONTENT``'s, they say
#: no more than the code; an older ADK gives such a reply no error at all.
#: Its words for another stop say more (``model_context_window_exceeded``, as
#: ``MAX_TOKENS``: the context filled up), and are kept.
ADK_CLAUDE_NO_CONTENT = re.compile(
    r"Claude stopped with (?:SAFETY \('refusal'\)|MAX_TOKENS \('max_tokens'\)) and no content\."
)


class ADKEventProcessor:
    """Converts ADK events into framework-agnostic WorkflowEvents.

    Owns an optional on_event callback for event transformation/filtering.

    One processor reads the conversation's own runner (the manager's, one
    per initialization); one made for a sub-agent run (``run=``) reads that
    run's, on its child session (``GoogleADKWorkflowManager._run_agent``).

    Only the agents that reply to the user write the reply
    (``reply_agents``): the root, and every chat-mode agent it can transfer
    to, which takes the conversation (a native ADK YAML app's). An event
    another agent wrote is its work: its text is marked
    ``metadata["agent"]``/``["subagent"]`` (the CLI shows it as activity, not
    as the reply), and its call's usage is ``purpose="subagent"``, since that
    prompt is the sub-agent's brief, not the conversation's context. With
    neither a root nor reply agents, nothing is marked.

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
    the answer, ``workflow/adk/call_clock.py``).

    A tool's result is a failed ``TOOL_RESULT`` when it says so (``success``
    false, an ``error`` value) or is an MCP tool's error result (``isError``),
    its error in ``metadata["error"]``, as on Deep Agents. A ``task`` call's
    result is the JSON text of what the delegation returned
    (``adk/task_tool.py``), read back as its dict, as Deep Agents reads its
    ``ToolMessage``; an envelope is shown as every backend shows one
    (``agents/runs.py::show_as_delegation``).

    A tool that wrote the session's task list (``save_tasks``: ``tasks`` in
    the event's state delta) is followed by ``TASK_PROGRESS``, after its
    result, when the list differs from the one this turn last reported
    (``start_turn`` forgets it), as on Deep Agents: a call that changed
    nothing reports nothing, and a list whose tasks are all done stays in the
    session for ``get_tasks``. A sub-agent run's list is its own: a run's
    processor reports none, and the conversation's reports only its own.

    A conversation the model of an agent that replies no longer takes ends
    its turn with an error response from the overflow handler
    (``workflow/adk/overflow_plugin.py``), ``error_code="CONTEXT_OVERFLOW"``:
    it becomes the turn's ``ERROR``, with the handler's clear text. A reply of
    such an agent's that is no answer (nothing but thoughts or blank text,
    and an error code of ``NO_ANSWER_CODES``: a safety stop, a blocked prompt,
    a malformed function call, every token spent thinking, nothing at all; or,
    with no error code, a finish reason other than STOP: a Claude refusal, a
    reply cut at its token limit, which google-adk 2.15 gives an error code
    too) ends the turn too, raising nothing: it becomes the turn's ``ERROR``,
    "<model> returned no answer", its code the error code or finish reason,
    then the reply's error message, unless that is ADK's own words for an
    empty reply that say no more than the code (``_NO_CONTENT``'s, and
    ``ADK_CLAUDE_NO_CONTENT``: a Claude refusal's or a cut reply's). Both are
    the turn's one ``ERROR``, with a failed turn's fields
    (``WorkflowEvent.turn_failed``: ``rate_limited`` false, the code as
    ``error_type``). No other error code of the main agent is mapped: ADK
    records one for an exception that then leaves the run, which the manager
    reports as the turn's one ``ERROR``.

    Every event an agent wrote names its run (``WorkflowEvent.run``): the
    sub-agent run the processor was made for, else the agent's own at depth
    0 (``root_run`` for the root; an agent the root transfers to is
    ``RunRef(<name>, <name>, None, None, 0)``); the user's message names
    none, nor does the turn's own ``ERROR``. A tool's call and result name
    the call (``tool_call_id``), as on Deep Agents. The ``AGENT_RUN`` events
    are the framework's (``agents/delegate.py``), not this processor's.

    A sub-agent run's processor reports no ``ERROR``: it keeps what ends the
    run for its manager to end it with (``Delegations`` warns the user
    once). Its last answer with text (``answer``: the last of its replies
    that has any, one that also calls tools included, as Deep Agents' own
    ``task`` reads a run: an empty reply after it passed over), that
    answer's outcome mark
    (``outcome``: ``SubAgentOutcomePlugin``'s), and why it failed
    (``failure``: the overflow handler's answer, ``CONTEXT_OVERFLOW``, or a
    reply that is no answer but an empty one, which is no failure: the run
    ends with the answer it has).

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
        kinds: Each agent's model's provider kind (``ProviderKind``'s value), by name.
        ledgers: The manager's usage ledgers, which this processor records
            each call's usage in; its own when not given.
        run: The sub-agent run whose runner this processor reads; None for
            the conversation's own.

    The manager builds one at each initialization, from the app it built,
    and one for each sub-agent run.
    """

    def __init__(
        self,
        model: str,
        on_event: Callable[[WorkflowEvent], WorkflowEvent | None] | None = None,
        root: str | None = None,
        models: Mapping[str, str] | None = None,
        budgets: Mapping[str, ContextBudget] | None = None,
        reply_agents: Collection[str] | None = None,
        kinds: Mapping[str, str] | None = None,
        ledgers: UsageLedgers | None = None,
        *,
        run: RunRef | None = None,
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
        self.kinds: Mapping[str, str] = kinds or {}
        self.ledgers = ledgers if ledgers is not None else UsageLedgers()
        self.run = run
        #: A sub-agent run's last answer with text ("" before one).
        self.answer = ""
        #: That run's last answer's outcome mark, None for none.
        self.outcome: str | None = None
        #: Why that run failed: ``(error, code)``, None while it has not.
        self.failure: tuple[str, str] | None = None

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
        """Forget the task list reported: a turn reports the first one it sees."""
        self._reported_tasks = None

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
        error_code = getattr(adk_event, "error_code", None)
        if self.run is not None:
            self._keep_run_end(adk_event)
        elif subagent is None and error_code == OVERFLOW_ERROR_CODE:
            # The turn's one ERROR (the base process()'s contract), though
            # no exception ended it: the overflow handler answered the call.
            error = self._apply_hook(WorkflowEvent.turn_failed(
                message=adk_event.error_message or error_code,
                error_code=OVERFLOW_ERROR_CODE,
            ))
            if error:
                yield error
        elif subagent is None and (code := _no_answer(adk_event)) is not None:
            error = self._apply_hook(WorkflowEvent.turn_failed(
                message=self._no_answer_text(adk_event, code), error_code=code
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
            workflow_event = self.process_part(part, session_id)
            if workflow_event:
                workflow_event.run = run
                if subagent is not None and workflow_event.type is EventType.TEXT:
                    workflow_event.metadata.update(agent=subagent, subagent=True)
                workflow_event = self._apply_hook(workflow_event)
                if workflow_event:
                    yield workflow_event

        progress = self._progress(adk_event) if self.run is None else None
        if progress is not None:
            progress.run = run
            progress = self._apply_hook(progress)
            if progress:
                yield progress

    def _keep_run_end(self, adk_event: Any) -> None:
        """Keep what ``adk_event`` says of the run's end: its answer, its outcome, its failure."""
        if self._agent(adk_event) is None or getattr(adk_event, "partial", False):
            return
        outcome = (getattr(adk_event, "custom_metadata", None) or {}).get(OUTCOME_KEY)
        if outcome:
            self.outcome = outcome
        error_code = getattr(adk_event, "error_code", None)
        if error_code == OVERFLOW_ERROR_CODE:
            self.failure = (adk_event.error_message or error_code, OVERFLOW_ERROR_CODE)
            return
        code = _no_answer(adk_event)
        if code is not None and code != _NO_CONTENT:
            self.failure = (self._no_answer_text(adk_event, code), code)
            return
        content = getattr(adk_event, "content", None)
        parts = (content.parts if content is not None else None) or []
        text = "".join(part.text for part in parts if part.text and not part.thought)
        if text.strip():
            self.answer = text.rstrip()

    def _no_answer_text(self, adk_event: Any, code: str) -> str:
        """What a reply that is no answer is told as: ``<model> returned no answer[: <message>]``."""
        model = self.models.get(self._agent(adk_event), self._model)
        message = adk_event.error_message
        # ADK's own message for an empty reply says no more than the code.
        own = code == _NO_CONTENT or (message and ADK_CLAUDE_NO_CONTENT.fullmatch(message))
        said = f": {message}" if message and not own else ""
        return f"{model} returned no answer{said}"

    def _progress(self, adk_event: Any) -> WorkflowEvent | None:
        """``TASK_PROGRESS`` for a task list ``adk_event`` wrote, when it changed."""
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
        if agent is None:
            return None
        if self.run is not None:
            return agent
        replying = self.reply_agents
        if replying is None:
            replying = () if self.root is None else (self.root,)
        if not replying or agent in replying:
            return None
        return agent

    @staticmethod
    def _agent(adk_event: Any) -> str | None:
        """The agent that wrote ``adk_event``, or None (the user's own)."""
        author = getattr(adk_event, "author", None)
        return author if author and author != "user" else None

    def _run_ref(self, adk_event: Any) -> RunRef | None:
        """The run that wrote ``adk_event``: the processor's sub-agent run, else
        the agent's own at depth 0; None for the user's own."""
        agent = self._agent(adk_event)
        if agent is None:
            return None
        if self.run is not None:
            return self.run
        if self.root is None or agent == self.root:
            return root_run(agent)
        return RunRef(agent=agent, instance=agent, run=None, parent_run=None, depth=0)

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
            if tool_name == TASK_TOOL:
                response = _delegated(response)
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
            found = read_envelope(response) if tool_name == TASK_TOOL else None
            if found is not None:
                show_as_delegation(event, found)
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


def _delegated(response: Any) -> Any:
    """A ``task`` call's result as its delegation returned it: the dict whose JSON text
    ADK stores under ``result`` (``adk/task_tool.py``); anything else as it is."""
    if isinstance(response, dict) and set(response) == {"result"}:
        try:
            parsed = json.loads(response["result"])
        except (TypeError, ValueError):
            return response
        if isinstance(parsed, dict):
            return parsed
    return response


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
