"""Task progress event building from TaskStore.

Pure function extracted from BaseWorkflowManager for building
TASK_PROGRESS WorkflowEvents from TaskStore data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentic_cli.workflow.events import WorkflowEvent

if TYPE_CHECKING:
    from agentic_cli.tools.task_tools import TaskStore


def build_task_progress_event(
    task_store: "TaskStore | None",
) -> WorkflowEvent | None:
    """Build a TASK_PROGRESS event from TaskStore.

    Returns a WorkflowEvent if TaskStore has tasks, else None.
    Auto-clears the store when all tasks are done (emitting a final snapshot).

    Args:
        task_store: TaskStore instance (may be None).

    Returns:
        A WorkflowEvent.task_progress() if tasks are present, else None.
    """
    if task_store is None or task_store.is_empty():
        return None

    # Auto-clear when all tasks are done — emit final snapshot first
    if task_store.all_done():
        progress = task_store.get_progress()
        display = task_store.to_compact_display()
        task_store.clear()
        return WorkflowEvent.task_progress(
            display=display,
            progress=progress,
            current_task_id=None,
            current_task_description=None,
        )

    progress = task_store.get_progress()
    display = task_store.to_compact_display()
    current = task_store.get_current_task()

    return WorkflowEvent.task_progress(
        display=display,
        progress=progress,
        current_task_id=current.id if current else None,
        current_task_description=current.description if current else None,
    )
