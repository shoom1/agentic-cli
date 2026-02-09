"""Task progress event building from TaskStore and PlanStore.

Pure functions extracted from BaseWorkflowManager for building
TASK_PROGRESS WorkflowEvents from TaskStore or PlanStore data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentic_cli.workflow.events import WorkflowEvent

if TYPE_CHECKING:
    from agentic_cli.tools.planning_tools import PlanStore
    from agentic_cli.tools.task_tools import TaskStore


def build_task_progress_event(
    task_store: "TaskStore | None",
    plan_store: "PlanStore | None",
) -> WorkflowEvent | None:
    """Build a TASK_PROGRESS event from TaskStore or PlanStore.

    Priority:
    1. If TaskStore has tasks -> use TaskStore (auto-clear when all done)
    2. Else if PlanStore has checkboxes -> parse plan for progress
    3. Else -> return None

    Args:
        task_store: TaskStore instance (may be None).
        plan_store: PlanStore instance (may be None).

    Returns:
        A WorkflowEvent.task_progress() if progress is available, else None.
    """
    # Path 1: TaskStore has tasks
    if task_store is not None and not task_store.is_empty():
        # Auto-clear when all tasks are done -- emit final snapshot first
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

    # Path 2: Fall back to PlanStore checkboxes
    plan_result = parse_plan_progress(plan_store)
    if plan_result is None:
        return None

    display, progress = plan_result

    # All checkboxes done -> return None (display clears)
    if progress["total"] > 0 and progress["completed"] == progress["total"]:
        return None

    return WorkflowEvent.task_progress(
        display=display,
        progress=progress,
        current_task_id=None,
        current_task_description=None,
    )


def parse_plan_progress(
    plan_store: "PlanStore | None",
) -> tuple[str, dict[str, int]] | None:
    """Parse PlanStore content for checkbox progress.

    Scans plan markdown for ``- [ ]`` / ``- [x]`` checkboxes,
    grouped under ``##`` or ``###`` section headers.

    Args:
        plan_store: PlanStore instance (may be None).

    Returns:
        (display_str, progress_dict) or None if no checkboxes found.
    """
    if plan_store is None or plan_store.is_empty():
        return None

    content = plan_store.get()
    current_section: str | None = None
    sections: list[tuple[str | None, list[tuple[bool, str]]]] = []
    current_items: list[tuple[bool, str]] = []

    for line in content.splitlines():
        stripped = line.strip()

        # Detect section headers (## or ###)
        if stripped.startswith("##"):
            # Save previous section if it has items
            if current_items:
                sections.append((current_section, current_items))
                current_items = []
            # Extract header text (strip leading #s and whitespace)
            current_section = stripped.lstrip("#").strip()
            continue

        # Detect checkboxes
        if stripped.startswith("- [x]") or stripped.startswith("- [X]"):
            text = stripped[5:].strip()
            current_items.append((True, text))
        elif stripped.startswith("- [ ]"):
            text = stripped[5:].strip()
            current_items.append((False, text))

    # Save last section
    if current_items:
        sections.append((current_section, current_items))

    if not sections:
        return None

    # Build compact display and count progress
    total = 0
    completed = 0
    display_lines: list[str] = []

    for section_name, items in sections:
        if section_name:
            display_lines.append(f"{section_name}:")
        for done, text in items:
            total += 1
            if done:
                completed += 1
                display_lines.append(f"  [x] {text}")
            else:
                display_lines.append(f"  [ ] {text}")

    progress = {
        "total": total,
        "completed": completed,
        "pending": total - completed,
        "in_progress": 0,
        "cancelled": 0,
    }

    return "\n".join(display_lines), progress
