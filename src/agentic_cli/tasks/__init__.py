"""Task management for agentic workflows.

Provides a simple task store for tracking execution tasks,
separate from strategic plans. Plans are "what to do"; tasks
are "track what you're doing."

Example:
    >>> store = TaskStore(settings)
    >>> task_id = store.create("Implement auth module", priority="high")
    >>> store.update_status(task_id, "in_progress")
    >>> store.list_tasks(status="in_progress")
"""

from agentic_cli.tasks.store import TaskStore, TaskItem

__all__ = ["TaskStore", "TaskItem"]
