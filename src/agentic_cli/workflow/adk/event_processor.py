"""ADK event processing: converts ADK events into WorkflowEvents.

Extracted from GoogleADKWorkflowManager to isolate event conversion
logic that runs on every message in the hot path.
"""

from __future__ import annotations

from typing import AsyncGenerator, Any, Callable

from agentic_cli.workflow.events import WorkflowEvent, EventType
from agentic_cli.workflow.thinking import ThinkingDetector
from agentic_cli.logging import Loggers

# Re-export for backward compatibility
from agentic_cli.workflow.retry import is_rate_limit_error as _is_rate_limit_error  # noqa: F401
from agentic_cli.workflow.retry import parse_retry_delay as _parse_retry_delay  # noqa: F401

logger = Loggers.workflow()


class ADKEventProcessor:
    """Converts ADK events into framework-agnostic WorkflowEvents.

    Owns a ThinkingDetector and an optional on_event callback for
    event transformation/filtering.

    Args:
        model: Model name used for usage metadata.
        on_event: Optional hook to transform/filter events before yielding.
    """

    def __init__(
        self,
        model: str,
        on_event: Callable[[WorkflowEvent], WorkflowEvent | None] | None = None,
    ) -> None:
        self._model = model
        self._on_event = on_event

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
            usage_event = self._apply_hook(usage_event)
            if usage_event:
                yield usage_event

        if not adk_event.content or not adk_event.content.parts:
            return

        for part in adk_event.content.parts:
            workflow_event = self.process_part(part, session_id)
            if workflow_event:
                workflow_event = self._apply_hook(workflow_event)
                if workflow_event:
                    yield workflow_event

    def process_part(self, part: Any, session_id: str) -> WorkflowEvent | None:
        """Process a single part from the agent response.

        Args:
            part: ADK response part.
            session_id: Current session ID.

        Returns:
            WorkflowEvent if the part produces one, None otherwise.
        """
        if part.text:
            result = ThinkingDetector.detect_from_part(part)
            if result.is_thinking:
                return WorkflowEvent.thinking(result.content, session_id)
            return WorkflowEvent.text(result.content, session_id)

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
