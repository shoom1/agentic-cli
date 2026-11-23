"""What may start an agent type at run time, and which agents are built with ``task``.

An app opts in twice: per source, in ``SubAgentPolicy.spawn_sources`` (an
agent's ``task`` call, a skill an agent loads, the user), and per type, in
``AgentConfig.spawnable_by``. ``spawnable`` is the types one source may
start; ``may_spawn`` says whether a run at a depth may start one
(``SubAgentPolicy.max_spawn_depth``, the root's depth being 0); and
``can_have_children`` says whether an agent is built with ``task`` (and, with
follow-ups, the tools and the roster that go with it): it has static
sub-agents, or it may start types at run time.

With ``SubAgentPolicy.general_purpose`` every enabled source may also start
the built-in ``GENERAL_PURPOSE`` type: no config of the app's, but one made
for its spawner type (``general_purpose_config``: the spawner's own tools,
MCP servers, skills and model, the framework's prompt), which starts no
agent itself. Each backend builds one for every type that may start it
(``general_purpose_spawners``). Imports no backend.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

from agentic_cli.workflow.agents.policy import SPAWN_SOURCES

if TYPE_CHECKING:
    from agentic_cli.workflow.agents.policy import SubAgentPolicy
    from agentic_cli.workflow.config import AgentConfig

AGENT_SOURCE, SKILL_SOURCE, USER_SOURCE = SPAWN_SOURCES

#: The built-in general-purpose type's name, its line in ``task``'s list of
#: types and its system prompt.
GENERAL_PURPOSE = "general_purpose"
GENERAL_PURPOSE_DESCRIPTION = (
    "A general-purpose agent with your own tools (but not the ones that start agents) and "
    "your model, for a self-contained task no other agent fits."
)
GENERAL_PURPOSE_PROMPT = (
    "You are a general-purpose agent, started by another agent for one task. The task "
    "message says what to do and what to return. Do it with the tools you have, then "
    "answer with the result itself, in the form the task asks for: the agent that "
    "started you reads only your final answer. If the task cannot be done, say why, "
    "and give what you found."
)


def spawnable(
    configs: Sequence[AgentConfig], policy: SubAgentPolicy, source: str
) -> tuple[str, ...]:
    """The types ``source`` may start at run time, in config order.

    ``source`` must be one the policy enables, and the type must name it in
    its ``spawnable_by``. The built-in ``GENERAL_PURPOSE`` type comes last
    when the policy enables it.
    """
    if source not in policy.spawn_sources:
        return ()
    types = tuple(
        config.name
        for config in configs
        if config.spawnable_by is not None and source in config.spawnable_by
    )
    if policy.general_purpose and GENERAL_PURPOSE not in types:
        return (*types, GENERAL_PURPOSE)
    return types


def may_spawn(depth: int, policy: SubAgentPolicy) -> bool:
    """Whether a run at ``depth`` (the root's is 0) may start a type at run time."""
    return depth < policy.max_spawn_depth


def can_have_children(
    config: AgentConfig,
    configs: Sequence[AgentConfig],
    policy: SubAgentPolicy,
    *,
    root: bool,
) -> bool:
    """Whether ``config`` is built with ``task``: static sub-agents, or types it may start.

    An agent may start types at run time when it does not inherit its
    parent's conversation, some type is spawnable by agents, and it is the
    root or runs below the root may start one too (``max_spawn_depth`` of 2
    or more). The built-in general-purpose type starts none.
    """
    if policy.general_purpose and config.name == GENERAL_PURPOSE:
        return False
    if config.sub_agents:
        return True
    return (
        not config.inherit_context
        and bool(spawnable(configs, policy, AGENT_SOURCE))
        and (root or policy.max_spawn_depth >= 2)
    )


def depth_sentence_applies(
    offered: Mapping[str, AgentConfig],
    own: Collection[str],
    depth: int,
    policy: SubAgentPolicy,
) -> bool:
    """Whether ``task``'s description says sub-agents cannot start sub-agents.

    It does when ``offered`` holds a type started at run time (one not in
    ``own``, the caller's static sub-agents), what that run starts would
    reach ``max_spawn_depth``, and no offered type has static sub-agents of
    its own.
    """
    return (
        any(name not in own for name in offered)
        and not may_spawn(depth + 1, policy)
        and not any(config.sub_agents for config in offered.values())
    )


def root_type(configs: Iterable[AgentConfig]) -> str | None:
    """The root agent's name: the first config no other delegates to that is no free type."""
    configs = list(configs)
    delegated = {sub for config in configs for sub in config.sub_agents}
    for config in configs:
        if config.name not in delegated and config.spawnable_by is None:
            return config.name
    return None


def general_purpose_spawners(
    configs: Sequence[AgentConfig], policy: SubAgentPolicy
) -> tuple[str, ...]:
    """The types a general-purpose agent is built for, in config order.

    None unless the policy enables the type. The root always (every source
    starts from it: the user's start is the root's child). Below the root,
    when ``max_spawn_depth`` is 2 or more, each type that does not inherit
    and may start one: by ``task`` when agents may start types, or by a skill
    it loads when it has skills and skills may.
    """
    if not policy.general_purpose:
        return ()
    root = root_type(configs)
    deeper = policy.max_spawn_depth >= 2
    sources = policy.spawn_sources
    return tuple(
        config.name
        for config in configs
        if config.name == root
        or (
            deeper
            and not config.inherit_context
            and (AGENT_SOURCE in sources or (SKILL_SOURCE in sources and config.skills))
        )
    )


def general_purpose_config(spawner: AgentConfig) -> AgentConfig:
    """The built-in general-purpose type made for ``spawner``.

    The spawner's own tools, MCP servers, skills, model and model settings
    (not a model choice), the framework's prompt and description, the plan's
    tools, a fresh context; no sub-agents and no model choices, so it starts
    no agent. Its lists are its own.
    """
    from agentic_cli.workflow.config import AgentConfig

    return AgentConfig(
        name=GENERAL_PURPOSE,
        prompt=GENERAL_PURPOSE_PROMPT,
        description=GENERAL_PURPOSE_DESCRIPTION,
        tools=list(spawner.tools),
        mcp_servers=list(spawner.mcp_servers),
        skills=list(spawner.skills),
        model=spawner.model,
        model_settings=spawner.model_settings,
        include_state_tools=True,
    )
