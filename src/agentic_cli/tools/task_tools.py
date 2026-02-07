"""Task management tools for agentic workflows.

Provides two tools for tracking execution tasks:
- save_tasks: Create, update, or delete tasks
- get_tasks: List tasks with optional filters

Plans are strategic ("what to do"), tasks are tactical ("track execution").
The TaskStore is auto-created by the workflow manager when these tools are used.

Example:
    from agentic_cli.tools import task_tools

    AgentConfig(
        tools=[task_tools.save_tasks, task_tools.get_tasks],
    )
"""

from typing import Any

from agentic_cli.tools import requires
from agentic_cli.tools.registry import (
    register_tool,
    ToolCategory,
    PermissionLevel,
)
from agentic_cli.workflow.context import get_context_task_store


@register_tool(
    category=ToolCategory.PLANNING,
    permission_level=PermissionLevel.SAFE,
    description="Create, update, or delete execution tasks for tracking progress. Use this to break work into trackable items with status and priority.",
)
@requires("task_store")
def save_tasks(
    operation: str,
    description: str = "",
    task_id: str = "",
    status: str = "",
    priority: str = "medium",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Create, update, or delete execution tasks.

    Manage tactical tasks for tracking execution progress. Each task has
    an ID, description, status (pending/in_progress/completed/cancelled),
    priority (low/medium/high), and optional tags.

    Args:
        operation: One of "create", "update", "delete".
        description: Task description (required for create, optional for update).
        task_id: Task ID (required for update and delete).
        status: Task status (for update: pending, in_progress, completed, cancelled).
        priority: Priority level (low, medium, high). Default: medium.
        tags: Optional tags for categorization.

    Returns:
        A dict with the operation result.
    """
    store = get_context_task_store()
    if store is None:
        return {"success": False, "error": "Task store not available"}

    if operation == "create":
        if not description:
            return {"success": False, "error": "description is required for create"}
        task_id = store.create(description, priority=priority, tags=tags)
        return {
            "success": True,
            "task_id": task_id,
            "message": "Task created",
        }

    elif operation == "update":
        if not task_id:
            return {"success": False, "error": "task_id is required for update"}
        kwargs: dict[str, Any] = {}
        if description:
            kwargs["description"] = description
        if status:
            kwargs["status"] = status
        if priority != "medium":
            kwargs["priority"] = priority
        if tags is not None:
            kwargs["tags"] = tags
        if not kwargs:
            return {"success": False, "error": "No fields to update"}
        updated = store.update(task_id, **kwargs)
        if not updated:
            return {"success": False, "error": f"Task '{task_id}' not found"}
        return {
            "success": True,
            "task_id": task_id,
            "message": "Task updated",
        }

    elif operation == "delete":
        if not task_id:
            return {"success": False, "error": "task_id is required for delete"}
        deleted = store.delete(task_id)
        if not deleted:
            return {"success": False, "error": f"Task '{task_id}' not found"}
        return {
            "success": True,
            "task_id": task_id,
            "message": "Task deleted",
        }

    else:
        return {
            "success": False,
            "error": f"Unknown operation '{operation}'. Use create, update, or delete.",
        }


@register_tool(
    category=ToolCategory.PLANNING,
    permission_level=PermissionLevel.SAFE,
    description="List execution tasks with optional filters by status, priority, or tag. Use this to check progress or find tasks to work on.",
)
@requires("task_store")
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
    if store is None:
        return {"success": False, "error": "Task store not available"}

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
