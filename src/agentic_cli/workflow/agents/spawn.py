"""What may start an agent type at run time, and which agents are built with ``task``.

An app opts in twice: per source, in ``SubAgentPolicy.spawn_sources`` (an
agent's ``task`` call, a skill an agent loads, the user), and per type, in
``AgentConfig.spawnable_by``. ``spawnable`` is the types one source may
start; ``may_spawn`` says whether a run at a depth may start one
(``SubAgentPolicy.max_spawn_depth``, the root's depth being 0); and
``can_have_children`` says whether an agent is built with ``task`` (and, with
follow-ups, the tools and the roster that go with it): it has static
sub-agents, or it may start types at run time. Imports no backend.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING

from agentic_cli.workflow.agents.policy import SPAWN_SOURCES

if TYPE_CHECKING:
    from agentic_cli.workflow.agents.policy import SubAgentPolicy
    from agentic_cli.workflow.config import AgentConfig

AGENT_SOURCE, SKILL_SOURCE, USER_SOURCE = SPAWN_SOURCES


def spawnable(
    configs: Sequence[AgentConfig], policy: SubAgentPolicy, source: str
) -> tuple[str, ...]:
    """The types ``source`` may start at run time, in config order.

    ``source`` must be one the policy enables, and the type must name it in
    its ``spawnable_by``.
    """
    if source not in policy.spawn_sources:
        return ()
    return tuple(
        config.name
        for config in configs
        if config.spawnable_by is not None and source in config.spawnable_by
    )


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
    or more).
    """
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
