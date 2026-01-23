"""Event processing for workflow events.

Transforms ADK events into WorkflowEvents for consumption by UI components.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Callable

from agentic_cli.workflow.events import WorkflowEvent, UserInputRequest
from agentic_cli.workflow.thinking import ThinkingDetector
from agentic_cli.logging import Loggers

logger = Loggers.workflow()


class EventProcessor:
    """Processes ADK events into WorkflowEvents.

    Handles the transformation of raw ADK event parts into typed
    WorkflowEvent objects, with support for event hooks.

    Example:
        processor = EventProcessor(on_event=my_hook)
        async for event in processor.process_events(events, session_id, check_input):
            handle(event)
    """

    def __init__(
        self,
        on_event: Callable[[WorkflowEvent], WorkflowEvent | None] | None = None,
    ):
        """Initialize event processor.

        Args:
            on_event: Optional hook to transform/filter events before yielding
        """
        self._on_event = on_event

    async def process_events(
        self,
        adk_event: Any,
        session_id: str,
        pending_input_checker: Callable[[], UserInputRequest | None],
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Process a single ADK event and yield WorkflowEvents.

        Args:
            adk_event: ADK event containing content parts
            session_id: Current session ID
            pending_input_checker: Callable that returns pending input request if any

        Yields:
            WorkflowEvent objects for each processed part
        """
        # Check for pending user input requests
        pending_request = pending_input_checker()
        if pending_request:
            yield WorkflowEvent.user_input_required(
                request_id=pending_request.request_id,
                tool_name=pending_request.tool_name,
                prompt=pending_request.prompt,
                input_type=pending_request.input_type,
                choices=pending_request.choices,
                default=pending_request.default,
            )

        if not adk_event.content or not adk_event.content.parts:
            return

        for part in adk_event.content.parts:
            workflow_event = self.process_part(part, session_id)
            if workflow_event:
                # Apply optional event hook
                if self._on_event:
                    workflow_event = self._on_event(workflow_event)
                if workflow_event:
                    yield workflow_event

    def process_part(self, part: Any, session_id: str) -> WorkflowEvent | None:
        """Process a single part from the agent response.

        Args:
            part: ADK response part
            session_id: Current session ID

        Returns:
            WorkflowEvent if the part produces one, None otherwise
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
                if "error" in response:
                    success = False
                result_data = response

            summary = self._generate_tool_summary(tool_name, result_data, success)
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

    def _generate_tool_summary(
        self, tool_name: str, result: Any, success: bool
    ) -> str:
        """Generate human-readable summary for tool result.

        Args:
            tool_name: Name of the tool
            result: Tool result data
            success: Whether the tool succeeded

        Returns:
            Summary string for display
        """
        if not success:
            if isinstance(result, dict) and "error" in result:
                return f"Failed: {result['error']}"
            return f"Failed: {result}"

        # Check for explicit summary in result
        if isinstance(result, dict):
            if "summary" in result:
                return str(result["summary"])
            if "message" in result:
                return str(result["message"])

        # Auto-generate based on result type
        if result is None:
            return "Completed"
        if isinstance(result, list):
            return f"Returned {len(result)} items"
        if isinstance(result, dict):
            return f"Returned {len(result)} fields"

        # Truncate string results
        text = str(result)
        return text[:100] + "..." if len(text) > 100 else text
