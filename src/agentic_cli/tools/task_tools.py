"""Task management tools for agentic workflows.

Provides two tools for tracking execution tasks:
- save_tasks: Write the complete task list (bulk replacement)
- get_tasks: Read the current task list with optional filters

Uses bulk replacement (like Gemini CLI's write_todos / Claude Code's TodoWrite):
the LLM sends the full updated list each time, avoiding N sequential tool calls.

Plans are strategic ("what to do"), tasks are tactical ("track execution").

Example:
    from agentic_cli.tools import task_tools

    AgentConfig(
        tools=[task_tools.save_tasks, task_tools.get_tasks],
    )
"""

from typing import Any

from agentic_cli.tools._core.tasks import (
    TaskStatus,
    TaskPriority,
    validate_tasks,
    normalize_tasks,
    filter_tasks,
)
from agentic_cli.tools.registry import (
    register_tool,
    ToolCategory,
    PermissionLevel,
)
from agentic_cli.workflow.service_registry import get_service_registry, TASKS


@register_tool(
    category=ToolCategory.PLANNING,
    permission_level=PermissionLevel.SAFE,
    description="Write the complete task list. This replaces the existing list. Use this to create initial tasks or update statuses. Each task has a description and status (pending/in_progress/completed/cancelled). At most one task should be in_progress at a time.",
)
def save_tasks(
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write the complete task list (bulk replacement).

    Provide the full updated list of tasks every time. This replaces any
    existing tasks. To update a single task's status, include the entire
    list with that task's status changed.

    IMPORTANT: Always call this tool to mark tasks as completed when you
    finish them. The task progress display depends on this — if you don't
    update the status, the user will see stale progress.

    Args:
        tasks: Complete list of tasks. Each task dict has keys
            "description" (required), "status" (pending/in_progress/
            completed/cancelled, default pending), "id" (auto-assigned
            if omitted), and "priority" (low/medium/high, default medium).

    Returns:
        A dict with the operation result.
    """
    registry = get_service_registry()
    if not tasks:
        registry[TASKS] = []
        return {"success": True, "task_ids": [], "count": 0, "message": "Tasks cleared"}

    error = validate_tasks(tasks)
    if error:
        return error

    normalized, task_ids = normalize_tasks(tasks)
    registry[TASKS] = normalized
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
    tasks_data = get_service_registry().get(TASKS, [])
    if not tasks_data:
        return {"success": True, "tasks": [], "count": 0}

    filtered = filter_tasks(
        tasks_data,
        status=status or None,
        priority=priority or None,
        tag=tag or None,
    )
    return {
        "success": True,
        "tasks": filtered,
        "count": len(filtered),
    }
