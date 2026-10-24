"""Delegation on ADK: a sub-agent is a tool its parent calls, and it returns.

An agent built from an ``AgentConfig`` that is not the root runs in ADK's
``single_turn`` mode (:func:`subagent_fields`). In its parent's constructor
ADK offers it to the parent as a tool named after it, taking a ``request``
string; the parent calls it, the sub-agent runs its own tool loop on its own
model, and its last text comes back as the call's result,
``{"result": <text>}``, in the same turn. The root then answers the user, and
the next turn starts at the root again. Transfers to the parent and peers
are off, so with no chat-mode children ADK offers no agent built from a
config ``transfer_to_agent``. This is how Deep Agents' ``task`` already
works, so both backends delegate alike.

A sub-agent's long-running tools are not wrapped as ADK's
``LongRunningFunctionTool``: ADK pauses the whole invocation on a pending
long-running call, inside a delegation too, so the parent would never be
re-entered. A sub-agent's job starter is an ordinary call that returns the
job's id, and the finished job is told to the root agent
(``GoogleADKWorkflowManager.resume_with_job_result``), as on Deep Agents.

The parent chooses a sub-agent by its tool's description, ADK's
``agent.description``: one without a description is offered as
"The <name> agent.", with a warning, as on Deep Agents.

What the sub-agent sees follows ``AgentConfig.context``: ``fresh`` (or None)
only the request, ``inherit`` its parent's conversation too, other agents'
turns quoted as context and without their thoughts. ADK gives a
``single_turn`` agent ``include_contents="none"`` unless the field was set
explicitly, so ``inherit`` passes ``"default"`` explicitly and ``fresh``
passes nothing. ADK sends an inheriting agent the user's turns as plain user
messages, so its instruction ends with :data:`INHERITED_CONTEXT_NOTE`
(:func:`subagent_instruction`), as Deep Agents frames a fork's task: without
it the agent could take the user's request to its parent as its own and
answer as the parent.

The tool ADK builds is its private ``_SingleTurnAgentTool``, which calls no
``func`` the permission plugin could verify. The framework built the parent
and the child, so it binds each such tool to the exempt delegation
declaration (``tools/_core/delegation.py``, Deep Agents' ``task``) right
after the parent is built (:func:`bind_delegation_tools`), as it binds the
skill tools it builds: only the exact ``_SingleTurnAgentTool`` whose
``agent`` is a child the framework passed in. An application's ``AgentTool``,
a subclass, a look-alike of the same name, or one around an agent the
framework did not build stays unbound, and the gate denies it. Delegation
itself needs no permission: the sub-agent's own tool calls pass the same
gate.

A native ADK YAML app (``adk_config_path``) is built by ADK and none of this
applies: its agents keep transferring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from google.adk.tools import BaseTool
from google.adk.tools.agent_tool import _SingleTurnAgentTool

from agentic_cli.logging import Loggers
from agentic_cli.tools._core.delegation import TASK_TOOL
from agentic_cli.tools.registry import bind_tool_identity, get_registry
from agentic_cli.workflow.config import AgentConfig, AgentGraphError

if TYPE_CHECKING:
    from google.adk.agents import LlmAgent

logger = Loggers.workflow()

#: The end of an inheriting sub-agent's instruction: whose conversation it
#: reads, and which message is its task (Deep Agents' ``_FORK_TASK_PREAMBLE``,
#: told the way ADK shows that conversation). No braces: ADK fills
#: ``{name}`` in an instruction from the session state.
INHERITED_CONTEXT_NOTE = (
    "You are a sub-agent: another agent has handed you a task. The conversation "
    "before the task is that agent's, shown so you know what has been established: "
    "the user's messages in it were addressed to that agent, not to you, and what "
    "that agent did is quoted, its call handing you this task included. Do not "
    "carry out those messages yourself or answer as that agent. Your task is the "
    "request that follows the conversation: complete it directly, and use the "
    "specific facts, figures and identifiers already established in the "
    "conversation rather than answering generically."
)


def subagent_fields(config: AgentConfig) -> dict[str, Any]:
    """The ``LlmAgent`` fields that make ``config``'s agent a delegated sub-agent.

    Its description is what its parent reads in the delegation tool, so an
    agent without one is offered under a placeholder, and the omission is
    logged.
    """
    description = config.description.strip()
    if not description:
        logger.warning("adk_subagent_without_description", agent=config.name)
        description = f"The {config.name} agent."
    fields: dict[str, Any] = {
        "description": description,
        "mode": "single_turn",
        "disallow_transfer_to_parent": True,
        "disallow_transfer_to_peers": True,
    }
    if config.context == "inherit":
        # Explicitly: ADK keeps include_contents only when it was set.
        fields["include_contents"] = "default"
    return fields


def subagent_instruction(config: AgentConfig, prompt: str) -> str:
    """The instruction of ``config``'s sub-agent, whose own prompt is ``prompt``.

    An inheriting one is told whose conversation it reads
    (:data:`INHERITED_CONTEXT_NOTE`); a fresh one sees only its task.
    """
    if config.context != "inherit":
        return prompt
    return f"{prompt}\n\n{INHERITED_CONTEXT_NOTE}" if prompt.strip() else INHERITED_CONTEXT_NOTE


def check_delegation_names(config: AgentConfig, tools: list[Any]) -> None:
    """Refuse a sub-agent named like one of its parent's own tools.

    ADK names the delegation tool after the sub-agent and appends it last,
    so a tool of the parent's with that name would be shadowed: offered to
    the model, but never called. ``tools`` are the parent's assembled tools:
    a function tool counts by its ``__name__``, an ADK tool by its ``name``,
    a skill toolset by its tools' names. An MCP toolset's tools are known
    only once it connects, so they are not checked here.

    Raises:
        AgentGraphError: Naming the parent and the clashing sub-agent(s).
    """
    clashes = sorted(set(config.sub_agents) & _tool_names(tools))
    if clashes:
        raise AgentGraphError(
            f"Agent {config.name!r} delegates to sub-agent(s) "
            f"{', '.join(repr(name) for name in clashes)} through a tool ADK names "
            "after each, and has a tool of its own by that name; rename the "
            "sub-agent or the tool."
        )


def bind_delegation_tools(agent: "LlmAgent", children: list[Any]) -> None:
    """Bind each delegation tool ADK built for one of ``children`` to ``task``.

    Call it right after ``agent`` is constructed: ADK appends the tools in
    its constructor. Only the exact ``_SingleTurnAgentTool`` whose ``agent``
    is one of ``children`` (the objects the framework built and passed in)
    is bound; nothing else in ``agent.tools`` is touched.
    """
    declaration = get_registry().get(TASK_TOOL)
    for tool in agent.tools:
        if type(tool) is _SingleTurnAgentTool and any(
            tool.agent is child for child in children
        ):
            bind_tool_identity(tool, declaration)


def _tool_names(tools: list[Any]) -> set[str]:
    from google.adk.tools.skill_toolset import SkillToolset

    names: set[str] = set()
    for tool in tools:
        if isinstance(tool, SkillToolset):
            names.update(skill_tool.name for skill_tool in tool._tools)
        elif isinstance(tool, BaseTool):
            names.add(tool.name)
        elif callable(tool):
            names.add(getattr(tool, "__name__", ""))
    return names
