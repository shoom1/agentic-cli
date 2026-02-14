"""Task management tools for agentic workflows.

Provides two tools for tracking execution tasks:
- save_tasks: Write the complete task list (bulk replacement)
- get_tasks: Read the current task list with optional filters

Uses bulk replacement (like Gemini CLI's write_todos / Claude Code's TodoWrite):
the LLM sends the full updated list each time, avoiding N sequential tool calls.

Plans are strategic ("what to do"), tasks are tactical ("track execution").
The TaskStore is auto-created by the workflow manager when these tools are used.

Example:
    from agentic_cli.tools import task_tools

    AgentConfig(
        tools=[task_tools.save_tasks, task_tools.get_tasks],
    )
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from agentic_cli.config import BaseSettings
from agentic_cli.tools import requires, require_context
from agentic_cli.tools.registry import (
    register_tool,
    ToolCategory,
    PermissionLevel,
)
from agentic_cli.workflow.context import get_context_task_store


class TaskStatus(str, Enum):
    """Valid task statuses."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Valid task priorities."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def _safe_enum(enum_cls, value, default):
    """Convert value to enum, returning default if invalid."""
    try:
        return enum_cls(value)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# TaskItem / TaskStore – in-memory task tracking
# ---------------------------------------------------------------------------


@dataclass
class TaskItem:
    """A single task entry."""

    id: str
    description: str
    status: str = TaskStatus.PENDING
    priority: str = TaskPriority.MEDIUM
    tags: list[str] = field(default_factory=list)
    created_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "tags": self.tags,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskItem":
        return cls(
            id=data["id"],
            description=data["description"],
            status=_safe_enum(TaskStatus, data.get("status", "pending"), TaskStatus.PENDING),
            priority=_safe_enum(TaskPriority, data.get("priority", "medium"), TaskPriority.MEDIUM),
            tags=data.get("tags", []),
            created_at=data.get("created_at", ""),
            completed_at=data.get("completed_at", ""),
        )


class TaskStore:
    """In-memory task store using bulk replacement.

    Tasks are ephemeral within a session — no file persistence.
    The LLM writes the full task list each time via replace_all().

    Example:
        >>> store = TaskStore(settings)
        >>> store.replace_all([
        ...     {"description": "Implement feature X", "status": "in_progress"},
        ...     {"description": "Write tests", "status": "pending"},
        ... ])
        >>> tasks = store.list_tasks(status="pending")
    """

    def __init__(self, settings: BaseSettings) -> None:
        self._settings = settings
        self._items: dict[str, TaskItem] = {}

    def replace_all(self, tasks: list[dict[str, Any]]) -> list[str]:
        """Replace the entire task list.

        Each dict must have "description" and "status". Optional keys:
        "id" (preserved if provided), "priority", "tags".
        New items without an id get one auto-assigned.

        Args:
            tasks: Complete list of task dicts.

        Returns:
            List of task IDs in the same order as the input.
        """
        now = datetime.now().isoformat()
        self._items.clear()
        ids: list[str] = []
        for task_data in tasks:
            task_id = task_data.get("id") or str(uuid.uuid4())[:8]
            status = _safe_enum(TaskStatus, task_data.get("status", "pending"), TaskStatus.PENDING)
            completed_at = task_data.get("completed_at", "")
            if status == TaskStatus.COMPLETED and not completed_at:
                completed_at = now
            item = TaskItem(
                id=task_id,
                description=task_data["description"],
                status=status,
                priority=_safe_enum(TaskPriority, task_data.get("priority", "medium"), TaskPriority.MEDIUM),
                tags=task_data.get("tags", []),
                created_at=task_data.get("created_at", now),
                completed_at=completed_at,
            )
            self._items[task_id] = item
            ids.append(task_id)
        return ids

    def get(self, task_id: str) -> TaskItem | None:
        """Get a task by ID."""
        return self._items.get(task_id)

    def list_tasks(
        self,
        status: str | None = None,
        priority: str | None = None,
        tag: str | None = None,
    ) -> list[TaskItem]:
        """List tasks with optional filters.

        Args:
            status: Filter by status.
            priority: Filter by priority.
            tag: Filter by tag.

        Returns:
            List of matching TaskItem objects.
        """
        results = list(self._items.values())
        if status:
            results = [t for t in results if t.status == status]
        if priority:
            results = [t for t in results if t.priority == priority]
        if tag:
            results = [t for t in results if tag in t.tags]
        return results

    def is_empty(self) -> bool:
        """Check if the task store has any items."""
        return not self._items

    def clear(self) -> None:
        """Clear all tasks from memory."""
        self._items.clear()

    def all_done(self) -> bool:
        """Check if all tasks are in a terminal state (completed/cancelled).

        Returns False if empty or any task is pending/in_progress.
        """
        if not self._items:
            return False
        return all(
            item.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED)
            for item in self._items.values()
        )

    def get_progress(self) -> dict[str, int]:
        """Return task count by status.

        Returns:
            Dict with keys: total, pending, in_progress, completed, cancelled.
        """
        counts = {"total": 0, "pending": 0, "in_progress": 0, "completed": 0, "cancelled": 0}
        for item in self._items.values():
            counts["total"] += 1
            if item.status in counts:
                counts[item.status] += 1
        return counts

    def to_compact_display(self) -> str:
        """Format tasks as a compact checklist for the thinking box.

        Returns:
            Multi-line string with status icons: [✓] Done, [▸] Active,
            [ ] Pending, [-] Cancelled.  Sorted by status priority
            (in-progress first, completed last).
        """
        if not self._items:
            return ""
        icons = {
            TaskStatus.COMPLETED: "[✓]",
            TaskStatus.IN_PROGRESS: "[▸]",
            TaskStatus.PENDING: "[ ]",
            TaskStatus.CANCELLED: "[-]",
        }
        status_order = {
            TaskStatus.IN_PROGRESS: 0,
            TaskStatus.PENDING: 1,
            TaskStatus.CANCELLED: 2,
            TaskStatus.COMPLETED: 3,
        }
        sorted_items = sorted(
            self._items.values(), key=lambda t: status_order.get(t.status, 1)
        )
        lines = []
        for item in sorted_items:
            icon = icons.get(item.status, "[ ]")
            lines.append(f"{icon} {item.description}")
        return "\n".join(lines)

    def get_current_task(self) -> TaskItem | None:
        """Return the first in-progress task, or None."""
        for item in self._items.values():
            if item.status == TaskStatus.IN_PROGRESS:
                return item
        return None


@register_tool(
    category=ToolCategory.PLANNING,
    permission_level=PermissionLevel.SAFE,
    description="Write the complete task list. This replaces the existing list. Use this to create initial tasks or update statuses. Each task has a description and status (pending/in_progress/completed/cancelled). At most one task should be in_progress at a time.",
)
@requires("task_store")
@require_context("Task store", get_context_task_store)
def save_tasks(
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write the complete task list (bulk replacement).

    Provide the full updated list of tasks every time. This replaces any
    existing tasks. To update a single task's status, include the entire
    list with that task's status changed.

    Args:
        tasks: Complete list of tasks. Each task dict has keys
            "description" (required), "status" (pending/in_progress/
            completed/cancelled, default pending), "id" (auto-assigned
            if omitted), and "priority" (low/medium/high, default medium).

    Returns:
        A dict with the operation result.
    """
    store = get_context_task_store()
    if not tasks:
        store.replace_all([])
        return {"success": True, "task_ids": [], "count": 0, "message": "Tasks cleared"}

    # Validate all tasks have descriptions and valid enum values
    valid_statuses = {s.value for s in TaskStatus}
    valid_priorities = {p.value for p in TaskPriority}
    for i, task in enumerate(tasks):
        if not task.get("description"):
            return {
                "success": False,
                "error": f"Task at index {i} is missing 'description'",
            }
        status = task.get("status", "pending")
        if status not in valid_statuses:
            return {
                "success": False,
                "error": f"Task at index {i} has invalid status '{status}'. Valid: {', '.join(sorted(valid_statuses))}",
            }
        priority = task.get("priority", "medium")
        if priority not in valid_priorities:
            return {
                "success": False,
                "error": f"Task at index {i} has invalid priority '{priority}'. Valid: {', '.join(sorted(valid_priorities))}",
            }

    task_ids = store.replace_all(tasks)
    return {
        "success": True,
        "task_ids": task_ids,
        "count": len(task_ids),
        "message": f"{len(task_ids)} tasks saved",
    }


@register_tool(
    category=ToolCategory.PLANNING,
    permission_level=PermissionLevel.SAFE,
    description="List execution tasks with optional filters by status, priority, or tag. Use this to check progress or find tasks to work on.",
)
@requires("task_store")
@require_context("Task store", get_context_task_store)
def get_tasks(
    status: str = "",
    priority: str = "",
    tag: str = "",
) -> dict[str, Any]:
    """List execution tasks with optional filters.

    Args:
        status: Filter by status (pending, in_progress, completed, cancelled).
        priority: Filter by priority (low, medium, high).
        tag: Filter by tag.

    Returns:
        A dict with matching tasks.
    """
    store = get_context_task_store()
    tasks = store.list_tasks(
        status=status or None,
        priority=priority or None,
        tag=tag or None,
    )

    items = [
        {
            "id": task.id,
            "description": task.description,
            "status": task.status,
            "priority": task.priority,
            "tags": task.tags,
            "created_at": task.created_at,
            "completed_at": task.completed_at,
        }
        for task in tasks
    ]

    return {
        "success": True,
        "tasks": items,
        "count": len(items),
    }
