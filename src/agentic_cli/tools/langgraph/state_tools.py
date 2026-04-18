"""LangGraph-native state tools using InjectedState and Command.

These tools use LangGraph's ``InjectedState`` for read access and return
``Command`` objects for state writes.  ``InjectedState`` and
``InjectedToolCallId`` are hidden from the LLM schema automatically.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from agentic_cli.tools._core.planning import summarize_checkboxes
from agentic_cli.tools._core.tasks import validate_tasks, normalize_tasks, filter_tasks
from agentic_cli.tools.registry import ToolCategory, register_tool
from agentic_cli.workflow.permissions import EXEMPT


@register_tool(capabilities=EXEMPT, category=ToolCategory.PLANNING)
def save_plan(
    content: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Save or update the execution plan as markdown with checkboxes.

    Use this to record your strategy and track task completion
    (- [ ] pending, - [x] done).

    Args:
        content: Markdown plan string with checkboxes for task tracking.
    """
    summary = summarize_checkboxes(content)
    message = f"Plan saved ({summary})" if summary else "Plan saved"
    result = {"success": True, "message": message}
    return Command(update={
        "plan": content,
        "messages": [ToolMessage(content=json.dumps(result), tool_call_id=tool_call_id)],
    })


@register_tool(capabilities=EXEMPT, category=ToolCategory.PLANNING)
def get_plan(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Retrieve the current execution plan.

    Use this to check progress or review the plan before updating it.
    """
    plan = state.get("plan", "")
    if not plan:
        result = {"success": True, "content": "", "message": "No plan created yet"}
    else:
        result = {"success": True, "content": plan}
    return Command(update={
        "messages": [ToolMessage(content=json.dumps(result), tool_call_id=tool_call_id)],
    })


@register_tool(capabilities=EXEMPT, category=ToolCategory.PLANNING)
def save_tasks(
    tasks: list[dict[str, Any]],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
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
    """
    if not tasks:
        result = {"success": True, "task_ids": [], "count": 0, "message": "Tasks cleared"}
        return Command(update={
            "tasks": [],
            "messages": [ToolMessage(content=json.dumps(result), tool_call_id=tool_call_id)],
        })

    error = validate_tasks(tasks)
    if error:
        return Command(update={
            "messages": [ToolMessage(content=json.dumps(error), tool_call_id=tool_call_id)],
        })

    normalized, task_ids = normalize_tasks(tasks)
    result = {
        "success": True,
        "task_ids": task_ids,
        "count": len(task_ids),
        "message": f"{len(task_ids)} tasks saved",
    }
    return Command(update={
        "tasks": normalized,
        "messages": [ToolMessage(content=json.dumps(result), tool_call_id=tool_call_id)],
    })


@register_tool(capabilities=EXEMPT, category=ToolCategory.PLANNING)
def get_tasks(
    status: str = "",
    priority: str = "",
    tag: str = "",
    state: Annotated[dict, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """List execution tasks with optional filters by status, priority, or tag.

    Use this to check progress or find tasks to work on.

    Args:
        status: Filter by status (pending, in_progress, completed, cancelled).
        priority: Filter by priority (low, medium, high).
        tag: Filter by tag.
    """
    tasks_data = state.get("tasks", []) if state else []
    if not tasks_data:
        result = {"success": True, "tasks": [], "count": 0}
    else:
        filtered = filter_tasks(
            tasks_data, status=status or None, priority=priority or None, tag=tag or None
        )
        result = {"success": True, "tasks": filtered, "count": len(filtered)}
    return Command(update={
        "messages": [ToolMessage(content=json.dumps(result), tool_call_id=tool_call_id)],
    })
