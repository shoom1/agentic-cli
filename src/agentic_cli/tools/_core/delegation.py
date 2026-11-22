"""The declaration of ``task``, the tool a coordinator delegates with, and of the parent's other tools.

On both backends a coordinator delegates to its sub-agents through the
framework's ``task`` tool (``workflow/agents/delegate.py``), which each backend
builds in its own form: ADK's ``workflow/adk/task_tool.py::TaskTool``, Deep
Agents' ``workflow/deepagents/task_tool.py::build_task_tool``. It has no
backend-neutral callable, so only its contract is declared here: the name, the
description and the permission declaration. Each backend binds the object it
builds to this declaration (``bind_tool_identity``), which is how the
permission gate knows it.

The parent's tools for the sub-agents it started, ``message_agent``,
``stop_agent`` and ``check_agents``, are declared the same way, each backend
building its own (ADK's in ``workflow/adk/task_tool.py``, Deep Agents' in
``workflow/deepagents/task_tool.py``), and their names are reserved as
``task``'s is.

Delegation itself needs no permission: each tool the sub-agent calls is gated
in that sub-agent's own stack.

``agentic_cli.tools`` imports this module before any tool module, so the name
is declared before any tool can register: an application tool named ``task``
(or one of the parent's tools)
raises (one name, one tool).
"""

from __future__ import annotations

from agentic_cli.tools.registry import ToolCategory, ToolDefinition, declare_tool
from agentic_cli.workflow.permissions import EXEMPT

#: The delegation tool's name.
TASK_TOOL = "task"
MESSAGE_AGENT_TOOL = "message_agent"
STOP_AGENT_TOOL = "stop_agent"
CHECK_AGENTS_TOOL = "check_agents"
FOLLOW_UP_TOOLS = (MESSAGE_AGENT_TOOL, STOP_AGENT_TOOL, CHECK_AGENTS_TOOL)
#: The skill tool whose call starts a run when its skill names an agent type
#: (``tools/skills/agents.py``); registered with the other skill tools
#: (``tools/skills/permissions.py``).
LOAD_SKILL_TOOL = "load_skill"
#: The tools whose result is a run's envelope, which the event processors
#: show as a delegation (``agents/runs.py::show_as_delegation``). A
#: ``load_skill`` result that is no envelope is shown as any result is.
ENVELOPE_TOOLS = (TASK_TOOL, MESSAGE_AGENT_TOOL, LOAD_SKILL_TOOL)
FOLLOW_UP_CLASH_ERROR = (
    "Agent {agent!r} continues its sub-agents through tools named message_agent, "
    "stop_agent and check_agents, so its own tools cannot include {tool!r}; rename that tool."
)
_FOLLOW_UP_DESCRIPTIONS = {
    MESSAGE_AGENT_TOOL: "Send a message to one of your sub-agents.",
    STOP_AGENT_TOOL: "Stop one of your sub-agents.",
    CHECK_AGENTS_TOOL: "Check on your sub-agents.",
}


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


def declare_follow_up_tools() -> tuple[ToolDefinition, ...]:
    """Declare the parent's tools' contracts. Idempotent.

    Returns:
        The declarations (their ``func`` is None), in ``FOLLOW_UP_TOOLS`` order.
    """
    return tuple(
        declare_tool(
            name,
            description=_FOLLOW_UP_DESCRIPTIONS[name],
            capabilities=EXEMPT,
            category=ToolCategory.PLANNING,
        )
        for name in FOLLOW_UP_TOOLS
    )


declare_delegation_tool()
declare_follow_up_tools()
