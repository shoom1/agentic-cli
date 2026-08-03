"""Backend-neutral declarations for the plan/task state tools.

These four tools have no neutral implementation: reading and writing plan and
task state is inherently backend-native (ADK's ``ToolContext.state``,
LangGraph's graph state and ``Command`` updates), so the signatures and the
model-visible schemas differ per backend.

What *is* neutral is the contract — the name, the description and the
permission declaration — so it is declared here, once, and each backend
registers its implementation as a variant of it
(``register_tool(..., variant_of="save_plan")``). Without this, the two backend
modules contested the same registry names and whichever imported first decided
what a bare ``"save_plan"`` in an ``AgentConfig`` resolved to.

Importing either backend's state tools imports this module first, so the
declarations always exist before a variant registers against them.
"""

from __future__ import annotations

from agentic_cli.tools.registry import ToolCategory, declare_tool
from agentic_cli.workflow.permissions import EXEMPT

# Plan/task state is the agent's own scratch space — no external side effects,
# so nothing to gate. The declaration is what both backends share.
_STATE_TOOLS = (
    (
        "save_plan",
        "Save or update the execution plan as markdown with checkboxes.",
    ),
    ("get_plan", "Retrieve the current execution plan."),
    (
        "save_tasks",
        "Write the complete task list. This replaces the existing list.",
    ),
    ("get_tasks", "Retrieve the current task list, optionally filtered."),
)


def declare_state_tools() -> None:
    """Declare the state tools' shared contract. Idempotent."""
    for name, description in _STATE_TOOLS:
        declare_tool(
            name,
            description=description,
            capabilities=EXEMPT,
            category=ToolCategory.PLANNING,
        )


declare_state_tools()
