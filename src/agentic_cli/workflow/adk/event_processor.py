"""ADK event processing: converts ADK events into WorkflowEvents.

Extracted from GoogleADKWorkflowManager to isolate event conversion
logic that runs on every message in the hot path.
"""

from __future__ import annotations

import re
from typing import AsyncGenerator, Any, Callable

from agentic_cli.workflow.events import WorkflowEvent, EventType
from agentic_cli.workflow.thinking import ThinkingDetector
from agentic_cli.constants import truncate, TOOL_SUMMARY_MAX_LENGTH
from agentic_cli.logging import Loggers

logger = Loggers.workflow()


# ---------------------------------------------------------------------------
# Rate-limit helpers (used by event processing and message_processor)
# ---------------------------------------------------------------------------


def _is_rate_limit_error(error: Exception) -> bool:
    """Check if an exception is a 429 rate-limit / RESOURCE_EXHAUSTED error."""
    if getattr(error, "code", None) == 429:
        return True
    if "RESOURCE_EXHAUSTED" in str(error):
        return True
    return False


def _parse_retry_delay(error: Exception) -> float | None:
    """Extract retry delay in seconds from a rate-limit error.

    Looks for retryDelay in the structured error details first,
    then falls back to regex on the error message string.

    Returns:
        Delay in seconds, or None if unparseable.
    """
    # Try structured details: error.details["error"]["details"][*]["retryDelay"]
    details = getattr(error, "details", None)
    if isinstance(details, dict):
        inner_error = details.get("error", {})
        for detail_entry in inner_error.get("details", []):
            if isinstance(detail_entry, dict):
                retry_delay = detail_entry.get("retryDelay")
                if retry_delay:
                    match = re.search(r"([\d.]+)\s*s", str(retry_delay))
                    if match:
                        return float(match.group(1))

    # Fallback: regex on error message string
    match = re.search(r"retry\s+in\s+([\d.]+)\s*s", str(error), re.IGNORECASE)
    if match:
        return float(match.group(1))

    return None


# ---------------------------------------------------------------------------
# ADKEventProcessor
# ---------------------------------------------------------------------------


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

            summary = self.generate_tool_summary(tool_name, result_data, success)
            logger.debug("tool_result", tool_name=tool_name, success=success)

            return WorkflowEvent.tool_result(
                tool_name=tool_name,
                result=summary,
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

    def generate_tool_summary(
        self, tool_name: str, result: Any, success: bool
    ) -> str:
        """Generate human-readable summary for tool result.

        Args:
            tool_name: Name of the tool.
            result: Tool result data.
            success: Whether the tool succeeded.

        Returns:
            Summary string for display.
        """
        if not success:
            if isinstance(result, dict) and "error" in result:
                return f"Failed: {result['error']}"
            return f"Failed: {result}"

        # Try tool-specific formatter first
        if isinstance(result, dict):
            from agentic_cli.workflow.tool_summaries import format_tool_summary

            specific = format_tool_summary(tool_name, result)
            if specific:
                return specific

        # Check for explicit summary in result (only strings, not dicts)
        if isinstance(result, dict):
            if "summary" in result and isinstance(result["summary"], str):
                return result["summary"]
            if "message" in result:
                return str(result["message"])
            # Handle results with a "results" list (e.g., web_search)
            if "results" in result and isinstance(result["results"], list):
                return f"Found {len(result['results'])} results"

        # Auto-generate based on result type
        if result is None:
            return "Completed"
        if isinstance(result, list):
            return f"Returned {len(result)} items"
        if isinstance(result, dict):
            return f"Returned {len(result)} fields"

        # Truncate string results
        text = str(result)
        return truncate(text, TOOL_SUMMARY_MAX_LENGTH)

    def _apply_hook(self, event: WorkflowEvent) -> WorkflowEvent | None:
        """Apply optional event transformation hook."""
        if self._on_event:
            return self._on_event(event)
        return event
