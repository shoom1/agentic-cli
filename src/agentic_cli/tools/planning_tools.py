"""Planning tools for agentic workflows.

Two tools for flat markdown plans: save_plan and get_plan.
The agent manages plan structure (checkboxes, ordering) in-context.
Plans are stored as simple strings in the service registry.

Example:
    from agentic_cli.tools import planning_tools

    # In agent config
    AgentConfig(
        tools=[planning_tools.save_plan, planning_tools.get_plan],
    )
"""

from typing import Any

from agentic_cli.tools.registry import (
    register_tool,
    ToolCategory,
    PermissionLevel,
)
from agentic_cli.workflow.service_registry import get_service_registry, PLAN


from agentic_cli.tools._core.planning import summarize_checkboxes as _summarize_checkboxes


@register_tool(
    category=ToolCategory.PLANNING,
    permission_level=PermissionLevel.SAFE,
    description="Save or update the execution plan as markdown with checkboxes. Use this to record your strategy and track task completion (- [ ] pending, - [x] done).",
)
def save_plan(content: str) -> dict[str, Any]:
    """Save or update the task plan.

    Write a markdown plan with checkboxes to track tasks:
    - [ ] for pending tasks
    - [x] for completed tasks

    Args:
        content: Markdown plan string with checkboxes for task tracking.

    Returns:
        A dict confirming the plan was saved.
    """
    registry = get_service_registry()
    registry[PLAN] = content

    summary = _summarize_checkboxes(content)
    message = f"Plan saved ({summary})" if summary else "Plan saved"

    return {
        "success": True,
        "message": message,
    }


@register_tool(
    category=ToolCategory.PLANNING,
    permission_level=PermissionLevel.SAFE,
    description="Retrieve the current execution plan. Use this to check progress or review the plan before updating it.",
)
def get_plan() -> dict[str, Any]:
    """Retrieve the current plan.

    Returns:
        A dict with the plan content, or a message if no plan exists.
    """
    plan = get_service_registry().get(PLAN, "")
    if not plan:
        return {
            "success": True,
            "content": "",
            "message": "No plan created yet",
        }

    return {
        "success": True,
        "content": plan,
    }
