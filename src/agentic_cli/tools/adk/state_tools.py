"""ADK-native state tools using ToolContext.state.

These tools use ADK's auto-injected ``tool_context`` parameter to read/write
plan and task state via ``tool_context.state``.  ADK strips ``tool_context``
from the LLM schema and injects it at call time.
"""

from __future__ import annotations

from typing import Any

from google.adk.tools.tool_context import ToolContext

from agentic_cli.tools._core.planning import summarize_checkboxes
from agentic_cli.tools._core.tasks import validate_tasks, normalize_tasks, filter_tasks
from agentic_cli.tools.registry import ToolCategory, register_tool
from agentic_cli.workflow.permissions import EXEMPT


@register_tool(capabilities=EXEMPT, category=ToolCategory.PLANNING)
def save_plan(content: str, tool_context: ToolContext) -> dict[str, Any]:
    """Save or update the execution plan as markdown with checkboxes.

    Use this to record your strategy and track task completion
    (- [ ] pending, - [x] done).

    Args:
        content: Markdown plan string with checkboxes for task tracking.

    Returns:
        A dict confirming the plan was saved.
    """
    tool_context.state["plan"] = content
    summary = summarize_checkboxes(content)
    message = f"Plan saved ({summary})" if summary else "Plan saved"
    return {"success": True, "message": message}


@register_tool(capabilities=EXEMPT, category=ToolCategory.PLANNING)
def get_plan(tool_context: ToolContext) -> dict[str, Any]:
    """Retrieve the current execution plan.

    Use this to check progress or review the plan before updating it.

    Returns:
        A dict with the plan content, or a message if no plan exists.
    """
    plan = tool_context.state.get("plan", "")
    if not plan:
        return {"success": True, "content": "", "message": "No plan created yet"}
    return {"success": True, "content": plan}


@register_tool(capabilities=EXEMPT, category=ToolCategory.PLANNING)
def save_tasks(
    tasks: list[dict[str, Any]], tool_context: ToolContext
) -> dict[str, Any]:
    """Write the complete task list. This replaces the existing list.

    Use this to create initial tasks or update statuses. Each task has a
    description and status (pending/in_progress/completed/cancelled).
    At most one task should be in_progress at a time.

    IMPORTANT: Always call this tool to mark tasks as completed when you
    finish them.

    Args:
        tasks: Complete list of tasks. Each task dict has keys
            "description" (required), "status" (pending/in_progress/
            completed/cancelled, default pending), "id" (auto-assigned
            if omitted), and "priority" (low/medium/high, default medium).

    Returns:
        A dict with the operation result.
    """
    if not tasks:
        tool_context.state["tasks"] = []
        return {"success": True, "task_ids": [], "count": 0, "message": "Tasks cleared"}

    error = validate_tasks(tasks)
    if error:
        return error

    normalized, task_ids = normalize_tasks(tasks)
    tool_context.state["tasks"] = normalized
    return {
        "success": True,
        "task_ids": task_ids,
        "count": len(task_ids),
        "message": f"{len(task_ids)} tasks saved",
    }


@register_tool(capabilities=EXEMPT, category=ToolCategory.PLANNING)
def get_tasks(
    status: str = "",
    priority: str = "",
    tag: str = "",
    tool_context: ToolContext = None,
) -> dict[str, Any]:
    """List execution tasks with optional filters by status, priority, or tag.

    Use this to check progress or find tasks to work on.

    Args:
        status: Filter by status (pending, in_progress, completed, cancelled).
        priority: Filter by priority (low, medium, high).
        tag: Filter by tag.

    Returns:
        A dict with matching tasks.
    """
    tasks_data = tool_context.state.get("tasks", []) if tool_context else []
    if not tasks_data:
        return {"success": True, "tasks": [], "count": 0}

    filtered = filter_tasks(
        tasks_data, status=status or None, priority=priority or None, tag=tag or None
    )
    return {"success": True, "tasks": filtered, "count": len(filtered)}
