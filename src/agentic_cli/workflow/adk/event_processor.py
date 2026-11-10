"""ADK event processing: converts ADK events into WorkflowEvents.

Extracted from GoogleADKWorkflowManager to isolate event conversion
logic that runs on every message in the hot path.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import AsyncGenerator, Any, Callable

from google.genai import types

from agentic_cli.tools._core.tasks import task_progress_data
from agentic_cli.workflow.context import OVERFLOW_ERROR_CODE, ContextBudget, usage_fields
from agentic_cli.workflow.events import WorkflowEvent, EventType
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

    Only the root agent replies to the user. An event another agent wrote (a
    sub-agent its parent delegated to, ``workflow/adk/delegation.py``) is its
    work: its text is marked ``metadata["agent"]``/``["subagent"]`` (the CLI
    shows it as activity, not as the reply), and its call's usage is
    ``purpose="subagent"``, since that prompt is the sub-agent's brief, not
    the conversation's context. A sub-agent's failed model call does not end
    the turn on ADK: ADK's delegation tool hands its parent "Error running
    sub-agent: …" and the parent carries on. ADK records the provider's reason
    on the sub-agent's error event, which becomes a recoverable ``ERROR``,
    "<agent> failed: <reason>", so the user is told why. With no root (a
    native ADK YAML app, whose agents transfer and do own the reply) nothing
    is marked.

    A call's ``LLM_USAGE`` is credited to the agent that made it, the event's
    author: ``metadata["agent"]``, that agent's own model (``models``; the
    default model for an agent the framework did not build) and its context
    numbers (``context_trigger``, ``window_known``, from ``budgets``), which
    the CLI shows the main agent's prompt against.

    A tool's result is a failed ``TOOL_RESULT`` when it says so (``success``
    false, an ``error`` value) or is an MCP tool's error result (``isError``),
    its error in ``metadata["error"]``, as on Deep Agents.

    A tool that wrote the session's task list (``save_tasks``: ``tasks`` in
    the event's state delta) is followed by ``TASK_PROGRESS``, after its
    result, when the list differs from the one this turn last reported
    (``start_turn`` forgets it), as on Deep Agents: a call that changed
    nothing reports nothing, and a list whose tasks are all done stays in the
    session for ``get_tasks``.

    A conversation the main agent's model no longer takes ends its turn with
    an error response from the overflow handler
    (``workflow/adk/overflow_plugin.py``), ``error_code="CONTEXT_OVERFLOW"``:
    it becomes the turn's ``ERROR``, with the handler's clear text. A reply of
    the main agent's that is no answer (nothing but thoughts or blank text,
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

    Args:
        model: The default model, which usage names for an agent ``models``
            does not hold.
        on_event: Optional hook to transform/filter events before yielding.
        root: The root agent's name, or None to mark nothing.
        models: Each agent's model reference, by name.
        budgets: Each agent's context numbers, by name.

    ``root``, ``models`` and ``budgets`` are attributes the manager sets at
    each initialization, when the agents are built.
    """

    def __init__(
        self,
        model: str,
        on_event: Callable[[WorkflowEvent], WorkflowEvent | None] | None = None,
        root: str | None = None,
        models: Mapping[str, str] | None = None,
        budgets: Mapping[str, ContextBudget] | None = None,
    ) -> None:
        self._model = model
        self._on_event = on_event
        self.root = root
        self.models: Mapping[str, str] = models or {}
        self.budgets: Mapping[str, ContextBudget] = budgets or {}
        # The task list this turn last reported, None before any.
        self._reported_tasks: list | None = None

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

            usage_event = WorkflowEvent.llm_usage(
                model=self.models.get(agent, self._model),
                prompt_tokens=getattr(usage, "prompt_token_count", None),
                completion_tokens=getattr(usage, "candidates_token_count", None),
                total_tokens=getattr(usage, "total_token_count", None),
                thinking_tokens=getattr(usage, "thoughts_token_count", None),
                cached_tokens=getattr(usage, "cached_content_token_count", None),
                invocation_id=invocation_id,
            )
            if agent is not None:
                usage_event.metadata["agent"] = agent
            usage_event.metadata.update(usage_fields(self.budgets.get(agent)))
            if subagent is not None:
                usage_event.metadata["purpose"] = "subagent"
            usage_event = self._apply_hook(usage_event)
            if usage_event:
                yield usage_event

        parts = adk_event.content.parts if adk_event.content else None
        for part in parts or ():
            workflow_event = self.process_part(part, session_id)
            if workflow_event:
                if subagent is not None and workflow_event.type is EventType.TEXT:
                    workflow_event.metadata.update(agent=subagent, subagent=True)
                workflow_event = self._apply_hook(workflow_event)
                if workflow_event:
                    yield workflow_event

        progress = self._progress(adk_event)
        if progress is not None:
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
        """The sub-agent that wrote ``adk_event``, or None (the root, the user)."""
        agent = self._agent(adk_event)
        if self.root is None or agent in (None, self.root):
            return None
        return agent

    @staticmethod
    def _agent(adk_event: Any) -> str | None:
        """The agent that wrote ``adk_event``, or None (the user's own)."""
        author = getattr(adk_event, "author", None)
        return author if author and author != "user" else None

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
            return WorkflowEvent.tool_call(
                tool_name=part.function_call.name,
                tool_args=dict(part.function_call.args) if part.function_call.args else None,
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
            return event

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
