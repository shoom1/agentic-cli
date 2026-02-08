"""Planning tools for agentic workflows.

Two tools for flat markdown plans: save_plan and get_plan.
The agent manages plan structure (checkboxes, ordering) in-context.
The PlanStore is auto-created by the workflow manager when these tools are used.

Example:
    from agentic_cli.tools import planning_tools

    # In agent config
    AgentConfig(
        tools=[planning_tools.save_plan, planning_tools.get_plan],
    )
"""

from typing import Any

from agentic_cli.tools import requires, require_context
from agentic_cli.tools.registry import (
    register_tool,
    ToolCategory,
    PermissionLevel,
)
from agentic_cli.workflow.context import get_context_plan_store


# ---------------------------------------------------------------------------
# PlanStore – simple string store for agent plans
# ---------------------------------------------------------------------------


class PlanStore:
    """Simple string store for agent plans.

    Follows the same pattern as MemoryStore — minimal backing store
    that lets the LLM handle structure in-context.
    """

    def __init__(self) -> None:
        """Initialize an empty plan store."""
        self._content: str = ""

    def save(self, content: str) -> None:
        """Save or overwrite the plan content.

        Args:
            content: Markdown plan string (use checkboxes for tasks).
        """
        self._content = content

    def get(self) -> str:
        """Get the current plan content.

        Returns:
            The plan string, or empty string if no plan exists.
        """
        return self._content

    def is_empty(self) -> bool:
        """Check if the plan store has content.

        Returns:
            True if no plan has been saved.
        """
        return not self._content

    def clear(self) -> None:
        """Clear the plan content."""
        self._content = ""


def _summarize_checkboxes(content: str) -> str:
    """Parse checkboxes and return a summary like '5 tasks: 2 done, 3 pending'.

    Args:
        content: Markdown plan string.

    Returns:
        Summary string, or empty string if no checkboxes found.
    """
    done = 0
    pending = 0
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("- [x]") or stripped.startswith("- [X]"):
            done += 1
        elif stripped.startswith("- [ ]"):
            pending += 1

    total = done + pending
    if total == 0:
        return ""

    parts: list[str] = []
    if done:
        parts.append(f"{done} done")
    if pending:
        parts.append(f"{pending} pending")
    return f"{total} tasks: {', '.join(parts)}"


@register_tool(
    category=ToolCategory.PLANNING,
    permission_level=PermissionLevel.SAFE,
    description="Save or update the execution plan as markdown with checkboxes. Use this to record your strategy and track task completion (- [ ] pending, - [x] done).",
)
@requires("plan_store")
@require_context("Plan store", get_context_plan_store)
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
    store = get_context_plan_store()
    store.save(content)

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
@requires("plan_store")
@require_context("Plan store", get_context_plan_store)
def get_plan() -> dict[str, Any]:
    """Retrieve the current plan.

    Returns:
        A dict with the plan content, or a message if no plan exists.
    """
    store = get_context_plan_store()
    if store.is_empty():
        return {
            "success": True,
            "content": "",
            "message": "No plan created yet",
        }

    return {
        "success": True,
        "content": store.get(),
    }
