"""Task graph for managing agent work plans.

This module provides a TaskGraph class for representing and managing
hierarchical task structures with dependencies, status tracking, and
progress monitoring.

Example:
    >>> graph = TaskGraph()
    >>> task1 = graph.add_task("Gather data")
    >>> task2 = graph.add_task("Process data", dependencies=[task1])
    >>> graph.update_status(task1, TaskStatus.COMPLETED)
    >>> ready = graph.get_ready_tasks()  # Returns task2
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import uuid


class TaskStatus(Enum):
    """Status values for tasks in the graph."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    SKIPPED = "skipped"


# Status icons for display
STATUS_ICONS = {
    TaskStatus.PENDING: "☐",
    TaskStatus.IN_PROGRESS: "◐",
    TaskStatus.COMPLETED: "✓",
    TaskStatus.BLOCKED: "⊘",
    TaskStatus.FAILED: "✗",
    TaskStatus.SKIPPED: "⊝",
}


@dataclass
class Task:
    """A task in the task graph.

    Attributes:
        id: Unique identifier for the task.
        description: Human-readable description of what needs to be done.
        status: Current status of the task.
        dependencies: List of task IDs this task depends on.
        subtasks: List of child task IDs.
        parent: ID of the parent task, if any.
        result: Result data from task completion.
        error: Error message if task failed.
        created_at: When the task was created.
        started_at: When the task was started (status changed to IN_PROGRESS).
        completed_at: When the task was completed/failed/skipped.
        metadata: Additional key-value data associated with the task.
    """

    id: str
    description: str
    created_at: datetime
    status: TaskStatus = TaskStatus.PENDING
    dependencies: list[str] = field(default_factory=list)
    subtasks: list[str] = field(default_factory=list)
    parent: str | None = None
    result: Any = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert task to a dictionary for serialization.

        Returns:
            Dictionary representation of the task.
        """
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "dependencies": self.dependencies.copy(),
            "subtasks": self.subtasks.copy(),
            "parent": self.parent,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        """Create a Task from a dictionary.

        Args:
            data: Dictionary representation of a task.

        Returns:
            A new Task instance.
        """
        return cls(
            id=data["id"],
            description=data["description"],
            status=TaskStatus(data["status"]),
            dependencies=data.get("dependencies", []).copy(),
            subtasks=data.get("subtasks", []).copy(),
            parent=data.get("parent"),
            result=data.get("result"),
            error=data.get("error"),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            metadata=data.get("metadata", {}).copy(),
        )


class TaskGraph:
    """A graph of tasks with dependencies and hierarchy.

    TaskGraph manages a collection of tasks that can have:
    - Dependencies on other tasks (must complete before this task starts)
    - Parent-child relationships (subtasks)
    - Status tracking with timestamps
    - Arbitrary metadata

    Example:
        >>> graph = TaskGraph()
        >>> setup = graph.add_task("Setup environment")
        >>> build = graph.add_task("Build project", dependencies=[setup])
        >>> graph.update_status(setup, TaskStatus.COMPLETED)
        >>> ready = graph.get_ready_tasks()
        >>> assert build in [t.id for t in ready]
    """

    def __init__(self) -> None:
        """Initialize an empty task graph."""
        self._tasks: dict[str, Task] = {}

    def add_task(
        self,
        description: str,
        dependencies: list[str] | None = None,
        parent: str | None = None,
        **metadata: Any,
    ) -> str:
        """Add a new task to the graph.

        Args:
            description: Human-readable description of the task.
            dependencies: List of task IDs this task depends on.
            parent: ID of the parent task (for subtask relationships).
            **metadata: Additional key-value pairs to store with the task.

        Returns:
            The ID of the newly created task.
        """
        task_id = str(uuid.uuid4())[:8]
        task = Task(
            id=task_id,
            description=description,
            dependencies=list(dependencies) if dependencies else [],
            parent=parent,
            created_at=datetime.now(),
            metadata=metadata,
        )
        self._tasks[task_id] = task

        # Update parent's subtasks list
        if parent and parent in self._tasks:
            self._tasks[parent].subtasks.append(task_id)

        return task_id

    def get_task(self, task_id: str) -> Task | None:
        """Get a task by its ID.

        Args:
            task_id: The ID of the task to retrieve.

        Returns:
            The Task object, or None if not found.
        """
        return self._tasks.get(task_id)

    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        """Update a task's status with appropriate timestamp updates.

        Args:
            task_id: The ID of the task to update.
            status: The new status for the task.
            result: Optional result data (typically for COMPLETED status).
            error: Optional error message (typically for FAILED status).

        Raises:
            ValueError: If the task_id does not exist.
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise ValueError(f"Task with id '{task_id}' not found")

        now = datetime.now()
        task.status = status

        # Update timestamps based on status
        if status == TaskStatus.IN_PROGRESS and task.started_at is None:
            task.started_at = now
        elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED):
            task.completed_at = now

        # Update result/error
        if result is not None:
            task.result = result
        if error is not None:
            task.error = error

    def get_ready_tasks(self) -> list[Task]:
        """Get all tasks that are ready to start.

        A task is ready if:
        - Its status is PENDING
        - All its dependencies have COMPLETED status

        Returns:
            List of tasks that are ready to start.
        """
        ready = []
        for task in self._tasks.values():
            if task.status != TaskStatus.PENDING:
                continue

            # Check if all dependencies are completed
            all_deps_completed = all(
                self._tasks.get(dep_id) is not None
                and self._tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            )

            if all_deps_completed:
                ready.append(task)

        return ready

    def get_progress(self) -> dict[str, int]:
        """Get progress statistics for the task graph.

        Returns:
            Dictionary with counts for each status type and total.
        """
        counts = {
            "total": 0,
            "pending": 0,
            "in_progress": 0,
            "completed": 0,
            "blocked": 0,
            "failed": 0,
            "skipped": 0,
        }

        for task in self._tasks.values():
            counts["total"] += 1
            counts[task.status.value] += 1

        return counts

    def get_in_progress_task(self) -> tuple[str, Task] | None:
        """Get the first task that is currently in progress.

        Returns:
            A tuple of (task_id, Task) for the first in-progress task,
            or None if no tasks are in progress.
        """
        for task_id, task in self._tasks.items():
            if task.status == TaskStatus.IN_PROGRESS:
                return task_id, task
        return None

    def clear(self) -> None:
        """Clear all tasks from the graph.

        This removes all tasks, resetting the graph to an empty state.
        """
        self._tasks.clear()

    def all_tasks(self) -> list[Task]:
        """Get all tasks in the graph.

        Returns:
            A list of all Task objects in the graph.
        """
        return list(self._tasks.values())

    def to_display(self) -> str:
        """Generate a formatted display of the task graph.

        Returns:
            A formatted string representation with status icons and hierarchy.
        """
        if not self._tasks:
            return "No tasks."

        lines = []

        # Find root tasks (no parent)
        root_tasks = [t for t in self._tasks.values() if t.parent is None]

        def format_task(task: Task, indent: int = 0) -> None:
            icon = STATUS_ICONS.get(task.status, "?")
            prefix = "  " * indent
            dep_info = ""
            if task.dependencies:
                dep_names = []
                for dep_id in task.dependencies:
                    dep_task = self._tasks.get(dep_id)
                    if dep_task:
                        dep_names.append(dep_task.description[:20])
                if dep_names:
                    dep_info = f" (depends on: {', '.join(dep_names)})"

            lines.append(f"{prefix}{icon} {task.description}{dep_info}")

            # Format subtasks
            for subtask_id in task.subtasks:
                subtask = self._tasks.get(subtask_id)
                if subtask:
                    format_task(subtask, indent + 1)

        for task in root_tasks:
            format_task(task)

        return "\n".join(lines)

    def to_compact_display(self, max_tasks: int = 5, max_desc_len: int = 30) -> str:
        """Generate a compact display suitable for the thinking box status line.

        Shows task progress in a condensed format with truncated descriptions.
        Prioritizes showing in-progress tasks, then pending tasks.

        Args:
            max_tasks: Maximum number of tasks to display
            max_desc_len: Maximum length for task descriptions

        Returns:
            A compact string representation with status icons.
        """
        if not self._tasks:
            return ""

        lines = []

        # Find root tasks (no parent), sorted by status priority
        root_tasks = [t for t in self._tasks.values() if t.parent is None]

        # Sort: in_progress first, then pending, then others
        status_order = {
            TaskStatus.IN_PROGRESS: 0,
            TaskStatus.PENDING: 1,
            TaskStatus.BLOCKED: 2,
            TaskStatus.COMPLETED: 3,
            TaskStatus.FAILED: 4,
            TaskStatus.SKIPPED: 5,
        }
        root_tasks.sort(key=lambda t: status_order.get(t.status, 99))

        def format_task_compact(task: Task, indent: int = 0) -> None:
            if len(lines) >= max_tasks:
                return
            icon = STATUS_ICONS.get(task.status, "?")
            prefix = "  " * indent
            desc = task.description
            if len(desc) > max_desc_len:
                desc = desc[: max_desc_len - 1] + "…"
            lines.append(f"{prefix}{icon} {desc}")

            # Format subtasks (limited)
            for subtask_id in task.subtasks:
                if len(lines) >= max_tasks:
                    break
                subtask = self._tasks.get(subtask_id)
                if subtask:
                    format_task_compact(subtask, indent + 1)

        for task in root_tasks:
            if len(lines) >= max_tasks:
                break
            format_task_compact(task)

        # Add overflow indicator if needed
        total = len(self._tasks)
        if total > max_tasks:
            remaining = total - len(lines)
            lines.append(f"  ... +{remaining} more")

        return "\n".join(lines)

    def revise(self, changes: list[dict[str, Any]]) -> None:
        """Apply a list of changes to the task graph.

        Each change is a dictionary with an "action" key:
        - "add": Add a new task. Keys: description, dependencies, parent, **metadata
        - "remove": Remove a task. Keys: task_id
        - "update": Update a task. Keys: task_id, and any fields to update

        Args:
            changes: List of change dictionaries to apply.
        """
        for change in changes:
            action = change.get("action")

            if action == "add":
                description = change.get("description", "")
                dependencies = change.get("dependencies")
                parent = change.get("parent")
                metadata = change.get("metadata", {})
                self.add_task(description, dependencies=dependencies, parent=parent, **metadata)

            elif action == "remove":
                task_id = change.get("task_id")
                if task_id and task_id in self._tasks:
                    # Remove from parent's subtasks
                    task = self._tasks[task_id]
                    if task.parent and task.parent in self._tasks:
                        parent = self._tasks[task.parent]
                        if task_id in parent.subtasks:
                            parent.subtasks.remove(task_id)
                    del self._tasks[task_id]

            elif action == "update":
                task_id = change.get("task_id")
                if task_id and task_id in self._tasks:
                    task = self._tasks[task_id]

                    if "description" in change:
                        task.description = change["description"]
                    if "status" in change:
                        new_status = TaskStatus(change["status"])
                        self.update_status(task_id, new_status)
                    if "dependencies" in change:
                        task.dependencies = change["dependencies"].copy()
                    if "metadata" in change:
                        task.metadata.update(change["metadata"])

    def to_dict(self) -> dict[str, Any]:
        """Convert the task graph to a dictionary for serialization.

        Returns:
            Dictionary representation of the entire graph.
        """
        return {
            "tasks": {
                task_id: task.to_dict()
                for task_id, task in self._tasks.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskGraph":
        """Create a TaskGraph from a dictionary.

        Args:
            data: Dictionary representation of a task graph.

        Returns:
            A new TaskGraph instance with restored state.
        """
        graph = cls()
        tasks_data = data.get("tasks", {})
        for task_id, task_data in tasks_data.items():
            graph._tasks[task_id] = Task.from_dict(task_data)
        return graph
