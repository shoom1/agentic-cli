"""The declaration of ``task``, the tool a coordinator delegates with.

On both backends a coordinator delegates to its sub-agents through the
framework's ``task`` tool (``workflow/agents/delegate.py``), which each backend
builds in its own form: ADK's ``workflow/adk/task_tool.py::TaskTool``, Deep
Agents' ``workflow/deepagents/task_tool.py::build_task_tool``. It has no
backend-neutral callable, so only its contract is declared here: the name, the
description and the permission declaration. Each backend binds the object it
builds to this declaration (``bind_tool_identity``), which is how the
permission gate knows it.

Delegation itself needs no permission: each tool the sub-agent calls is gated
in that sub-agent's own stack.

``agentic_cli.tools`` imports this module before any tool module, so the name
is declared before any tool can register: an application tool named ``task``
raises (one name, one tool).
"""

from __future__ import annotations

from agentic_cli.tools.registry import ToolCategory, ToolDefinition, declare_tool
from agentic_cli.workflow.permissions import EXEMPT

#: The delegation tool's name.
TASK_TOOL = "task"


def declare_delegation_tool() -> ToolDefinition:
    """Declare the ``task`` tool's contract. Idempotent.

    Returns:
        The declaration (its ``func`` is None).
    """
    return declare_tool(
        TASK_TOOL,
        description="Delegate a task to a sub-agent.",
        capabilities=EXEMPT,
        category=ToolCategory.PLANNING,
    )


declare_delegation_tool()
