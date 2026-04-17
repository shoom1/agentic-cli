"""ADK Plugin for task progress tracking.

Reads ``tool_context.state["tasks"]`` after each tool call and buffers
``TASK_PROGRESS`` workflow events.  The ADK manager drains the buffer
in its process loop (same pattern as ``LLMLoggingPlugin``).
"""

from __future__ import annotations

from collections import deque
from typing import Any, TYPE_CHECKING

from google.adk.plugins.base_plugin import BasePlugin

from agentic_cli.tools._core.tasks import task_progress_data
from agentic_cli.workflow.events import WorkflowEvent

if TYPE_CHECKING:
    from google.adk.tools import BaseTool
    from google.adk.tools.tool_context import ToolContext


class TaskProgressPlugin(BasePlugin):
    """Emit TASK_PROGRESS events after each tool execution."""

    def __init__(self, max_events: int = 100) -> None:
        super().__init__(name="task_progress")
        self._events: deque[WorkflowEvent] = deque(maxlen=max_events)

    async def after_tool_callback(
        self,
        *,
        tool: "BaseTool",
        tool_args: dict[str, Any],
        tool_context: "ToolContext",
        result: dict,
    ) -> dict | None:
        """Check task state after each tool call and buffer a progress event."""
        tasks_data = tool_context.state.get("tasks", [])
        if not tasks_data:
            return None

        progress = task_progress_data(tasks_data)
        if progress is None:
            return None

        event = WorkflowEvent.task_progress(
            display=progress["display"],
            progress=progress["progress"],
            current_task_id=progress.get("current_task_id"),
            current_task_description=progress.get("current_task_description"),
        )
        self._events.append(event)

        # Auto-clear when all tasks are done
        if progress["all_done"]:
            tool_context.state["tasks"] = []

        return None  # don't modify the tool result

    def drain_events(self) -> list[WorkflowEvent]:
        """Get and clear all buffered progress events."""
        events = list(self._events)
        self._events.clear()
        return events
