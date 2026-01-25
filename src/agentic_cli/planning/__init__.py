"""Planning module for managing agent work plans.

This module provides TaskGraph for representing hierarchical task structures
with dependencies, status tracking, and progress monitoring.

Example:
    >>> from agentic_cli.planning import TaskGraph, TaskStatus
    >>> graph = TaskGraph()
    >>> task1 = graph.add_task("Gather requirements")
    >>> task2 = graph.add_task("Implement feature", dependencies=[task1])
    >>> graph.update_status(task1, TaskStatus.COMPLETED)
    >>> ready = graph.get_ready_tasks()  # Returns [task2]
"""

from agentic_cli.planning.task_graph import (
    Task,
    TaskGraph,
    TaskStatus,
    STATUS_ICONS,
)

__all__ = [
    "Task",
    "TaskGraph",
    "TaskStatus",
    "STATUS_ICONS",
]
