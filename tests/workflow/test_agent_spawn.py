"""What may start an agent type at run time (``workflow/agents/spawn.py``).

``spawnable`` is the types one source may start: the policy enables the
source (``SubAgentPolicy.spawn_sources``) and the type names it
(``AgentConfig.spawnable_by``). ``may_spawn`` says whether a run at a depth
may start one (``max_spawn_depth``), and ``can_have_children`` whether an
agent is built with ``task``: static sub-agents, or types it may start.
"""

from __future__ import annotations

import pytest

from agentic_cli.workflow.agents.policy import SubAgentPolicy
from agentic_cli.workflow.agents.spawn import (
    AGENT_SOURCE,
    SKILL_SOURCE,
    USER_SOURCE,
    can_have_children,
    may_spawn,
    spawnable,
)
from agentic_cli.workflow.config import AgentConfig

LEAD = AgentConfig(name="lead", prompt="p", sub_agents=["researcher"])
RESEARCHER = AgentConfig(name="researcher", prompt="p")
HELPER = AgentConfig(name="helper", prompt="p", spawnable_by=frozenset({"agent", "user"}))
SKILLED = AgentConfig(name="skilled", prompt="p", spawnable_by=frozenset({"skill"}))
READER = AgentConfig(name="reader", prompt="p", spawnable_by=frozenset({"agent"}),
                     inherit_context=True)
CONFIGS = [LEAD, RESEARCHER, HELPER, SKILLED, READER]
EVERY = SubAgentPolicy(spawn_sources={"agent", "skill", "user"})


class TestTheSources:
    def test_their_names(self):
        assert (AGENT_SOURCE, SKILL_SOURCE, USER_SOURCE) == ("agent", "skill", "user")


class TestSpawnable:
    def test_none_with_the_default_policy(self):
        for source in ("agent", "skill", "user"):
            assert spawnable(CONFIGS, SubAgentPolicy(), source) == ()

    def test_the_types_that_name_a_source_the_policy_enables_in_config_order(self):
        assert spawnable(CONFIGS, EVERY, "agent") == ("helper", "reader")
        assert spawnable(CONFIGS, EVERY, "skill") == ("skilled",)
        assert spawnable(CONFIGS, EVERY, "user") == ("helper",)

    def test_a_source_the_policy_leaves_out_starts_none(self):
        policy = SubAgentPolicy(spawn_sources={"user"})
        assert spawnable(CONFIGS, policy, "agent") == ()
        assert spawnable(CONFIGS, policy, "user") == ("helper",)

    def test_a_static_sub_agent_that_names_one_is_spawnable_too(self):
        writer = AgentConfig(name="writer", prompt="p", spawnable_by=["agent"])
        assert spawnable([LEAD, writer], EVERY, "agent") == ("writer",)


class TestMaySpawn:
    @pytest.mark.parametrize("depth, limit, may", [
        (0, 1, True), (1, 1, False), (1, 2, True), (2, 2, False), (5, 9, True),
    ])
    def test_below_the_limit_only(self, depth, limit, may):
        assert may_spawn(depth, SubAgentPolicy(max_spawn_depth=limit)) is may


class TestCanHaveChildren:
    def test_an_agent_with_sub_agents_can(self):
        assert can_have_children(LEAD, CONFIGS, SubAgentPolicy(), root=True)
        middle = AgentConfig(name="middle", prompt="p", sub_agents=["leaf"])
        assert can_have_children(middle, CONFIGS, SubAgentPolicy(), root=False)

    def test_by_default_one_without_cannot(self):
        assert not can_have_children(RESEARCHER, CONFIGS, SubAgentPolicy(), root=False)
        solo = AgentConfig(name="solo", prompt="p")
        assert not can_have_children(solo, [solo], SubAgentPolicy(), root=True)

    def test_the_root_can_when_agents_may_start_a_type(self):
        solo = AgentConfig(name="solo", prompt="p")
        assert can_have_children(solo, [solo, HELPER], EVERY, root=True)
        assert not can_have_children(solo, [solo, SKILLED], EVERY, root=True)
        assert not can_have_children(
            solo, [solo, HELPER], SubAgentPolicy(spawn_sources={"user"}), root=True
        )

    def test_a_sub_agent_can_only_when_runs_below_the_root_may_start_one(self):
        assert not can_have_children(RESEARCHER, CONFIGS, EVERY, root=False)
        deeper = SubAgentPolicy(spawn_sources={"agent"}, max_spawn_depth=2)
        assert can_have_children(RESEARCHER, CONFIGS, deeper, root=False)

    def test_an_inheriting_agent_never_can(self):
        deeper = SubAgentPolicy(spawn_sources={"agent"}, max_spawn_depth=3)
        assert not can_have_children(READER, CONFIGS, deeper, root=False)
