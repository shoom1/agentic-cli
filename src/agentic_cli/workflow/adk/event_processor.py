"""ADK event processing: converts ADK events into WorkflowEvents.

Extracted from GoogleADKWorkflowManager to isolate event conversion
logic that runs on every message in the hot path.
"""

from __future__ import annotations

from typing import AsyncGenerator, Any, Callable

from agentic_cli.workflow.context import OVERFLOW_ERROR_CODE
from agentic_cli.workflow.events import WorkflowEvent, EventType
from agentic_cli.logging import Loggers

logger = Loggers.workflow()


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

    A conversation the main agent's model no longer takes ends its turn with
    an error response from the overflow handler
    (``workflow/adk/overflow_plugin.py``), ``error_code="CONTEXT_OVERFLOW"``:
    it becomes the turn's ``ERROR``, with the handler's clear text. No other
    error code of the main agent is mapped: ADK records one for an exception
    that then leaves the run, which reports it.

    Args:
        model: Model name used for usage metadata.
        on_event: Optional hook to transform/filter events before yielding.
        root: The root agent's name, or None to mark nothing.
    """

    def __init__(
        self,
        model: str,
        on_event: Callable[[WorkflowEvent], WorkflowEvent | None] | None = None,
        root: str | None = None,
    ) -> None:
        self._model = model
        self._on_event = on_event
        self.root = root

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
            # The root's own failure leaves the run, which reports it; this
            # one never does.
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
            error = self._apply_hook(WorkflowEvent.error(
                adk_event.error_message or error_code, error_code=OVERFLOW_ERROR_CODE
            ))
            if error:
                yield error

        # Extract usage_metadata if available
        if hasattr(adk_event, "usage_metadata") and adk_event.usage_metadata:
            usage = adk_event.usage_metadata
            invocation_id = getattr(adk_event, "invocation_id", None)

            usage_event = WorkflowEvent.llm_usage(
                model=self._model,
                prompt_tokens=getattr(usage, "prompt_token_count", None),
                completion_tokens=getattr(usage, "candidates_token_count", None),
                total_tokens=getattr(usage, "total_token_count", None),
                thinking_tokens=getattr(usage, "thoughts_token_count", None),
                cached_tokens=getattr(usage, "cached_content_token_count", None),
                invocation_id=invocation_id,
            )
            if subagent is not None:
                usage_event.metadata["purpose"] = "subagent"
            usage_event = self._apply_hook(usage_event)
            if usage_event:
                yield usage_event

        if not adk_event.content or not adk_event.content.parts:
            return

        for part in adk_event.content.parts:
            workflow_event = self.process_part(part, session_id)
            if workflow_event:
                if subagent is not None and workflow_event.type is EventType.TEXT:
                    workflow_event.metadata.update(agent=subagent, subagent=True)
                workflow_event = self._apply_hook(workflow_event)
                if workflow_event:
                    yield workflow_event

    def _subagent(self, adk_event: Any) -> str | None:
        """The sub-agent that wrote ``adk_event``, or None (the root, the user)."""
        author = getattr(adk_event, "author", None)
        if self.root is None or not author or author in (self.root, "user"):
            return None
        return author

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

            # Determine success and extract result
            success = True
            result_data = response

            if isinstance(response, dict):
                # Check for error value, not just key presence
                if response.get("error"):
                    success = False
                # Also respect explicit "success" field if present
                if "success" in response:
                    success = response["success"]
                result_data = response

            logger.debug("tool_result", tool_name=tool_name, success=success)

            return WorkflowEvent.tool_result(
                tool_name=tool_name,
                result=result_data,
                success=success,
            )

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
