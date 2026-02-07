"""Simple file-based task store.

Persistent task tracking following the MemoryStore pattern:
a JSON file with atomic writes. Tasks are tactical ("track
execution progress"), separate from plans ("what to do").
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agentic_cli.config import BaseSettings


@dataclass
class TaskItem:
    """A single task entry."""

    id: str
    description: str
    status: str = "pending"  # pending, in_progress, completed, cancelled
    priority: str = "medium"  # low, medium, high
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
            status=data.get("status", "pending"),
            priority=data.get("priority", "medium"),
            tags=data.get("tags", []),
            created_at=data.get("created_at", ""),
            completed_at=data.get("completed_at", ""),
        )


class TaskStore:
    """Simple persistent task store.

    Stores tasks in a JSON file in the workspace directory.
    Supports create, update, list, and delete operations.

    Example:
        >>> store = TaskStore(settings)
        >>> task_id = store.create("Implement feature X", priority="high")
        >>> store.update_status(task_id, "in_progress")
        >>> tasks = store.list_tasks(status="pending")
    """

    def __init__(self, settings: BaseSettings) -> None:
        self._settings = settings
        self._tasks_dir = settings.workspace_dir / "tasks"
        self._storage_path = self._tasks_dir / "tasks.json"
        self._items: dict[str, TaskItem] = {}
        self._load()

    def _load(self) -> None:
        if self._storage_path.exists():
            try:
                with open(self._storage_path, "r") as f:
                    data = json.load(f)
                for item_data in data.get("items", []):
                    item = TaskItem.from_dict(item_data)
                    self._items[item.id] = item
            except (json.JSONDecodeError, KeyError):
                self._items = {}

    def _save(self) -> None:
        self._tasks_dir.mkdir(parents=True, exist_ok=True)
        data = {"items": [item.to_dict() for item in self._items.values()]}
        tmp_path = self._storage_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        tmp_path.rename(self._storage_path)

    def create(
        self,
        description: str,
        priority: str = "medium",
        tags: list[str] | None = None,
    ) -> str:
        """Create a new task.

        Args:
            description: Task description.
            priority: Priority level (low, medium, high).
            tags: Optional tags for categorization.

        Returns:
            The unique ID of the created task.
        """
        task_id = str(uuid.uuid4())[:8]
        item = TaskItem(
            id=task_id,
            description=description,
            status="pending",
            priority=priority,
            tags=tags or [],
            created_at=datetime.now().isoformat(),
        )
        self._items[task_id] = item
        self._save()
        return task_id

    def update_status(self, task_id: str, status: str) -> bool:
        """Update task status.

        Args:
            task_id: Task ID.
            status: New status (pending, in_progress, completed, cancelled).

        Returns:
            True if updated, False if task not found.
        """
        item = self._items.get(task_id)
        if item is None:
            return False
        item.status = status
        if status == "completed":
            item.completed_at = datetime.now().isoformat()
        self._save()
        return True

    def update(self, task_id: str, **kwargs: Any) -> bool:
        """Update task fields.

        Args:
            task_id: Task ID.
            **kwargs: Fields to update (description, status, priority, tags).

        Returns:
            True if updated, False if task not found.
        """
        item = self._items.get(task_id)
        if item is None:
            return False
        for key, value in kwargs.items():
            if hasattr(item, key) and key != "id":
                setattr(item, key, value)
        if kwargs.get("status") == "completed" and not item.completed_at:
            item.completed_at = datetime.now().isoformat()
        self._save()
        return True

    def delete(self, task_id: str) -> bool:
        """Delete a task.

        Args:
            task_id: Task ID.

        Returns:
            True if deleted, False if not found.
        """
        if task_id not in self._items:
            return False
        del self._items[task_id]
        self._save()
        return True

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
