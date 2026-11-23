"""What may start an agent type at run time (``workflow/agents/spawn.py``).

``spawnable`` is the types one source may start: the policy enables the
source (``SubAgentPolicy.spawn_sources``) and the type names it
(``AgentConfig.spawnable_by``), then the built-in ``general_purpose`` when
the policy enables it. ``may_spawn`` says whether a run at a depth may start
one (``max_spawn_depth``), and ``can_have_children`` whether an agent is
built with ``task``: static sub-agents, or types it may start.
``general_purpose_config`` is the general-purpose type made for one spawner
type, and ``general_purpose_spawners`` the types one is made for.
"""

from __future__ import annotations

import pytest

from agentic_cli.workflow.agents.policy import SubAgentPolicy
from agentic_cli.workflow.agents.spawn import (
    AGENT_SOURCE,
    GENERAL_PURPOSE,
    GENERAL_PURPOSE_DESCRIPTION,
    GENERAL_PURPOSE_PROMPT,
    SKILL_SOURCE,
    USER_SOURCE,
    can_have_children,
    general_purpose_config,
    general_purpose_spawners,
    may_spawn,
    root_type,
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


#: The general-purpose type on, for every source.
GENERAL = SubAgentPolicy(spawn_sources={"agent", "skill", "user"}, general_purpose=True)


class TestTheGeneralPurposeType:
    def test_its_name_line_and_prompt(self):
        assert GENERAL_PURPOSE == "general_purpose"
        assert GENERAL_PURPOSE_DESCRIPTION == (
            "A general-purpose agent with your own tools (but not the ones that start "
            "agents) and your model, for a self-contained task no other agent fits."
        )
        assert GENERAL_PURPOSE_PROMPT == (
            "You are a general-purpose agent, started by another agent for one task. The task "
            "message says what to do and what to return. Do it with the tools you have, then "
            "answer with the result itself, in the form the task asks for: the agent that "
            "started you reads only your final answer. If the task cannot be done, say why, "
            "and give what you found."
        )

    @pytest.mark.parametrize("source", ["agent", "skill", "user"])
    def test_spawnable_appends_it_for_each_source_the_policy_enables(self, source):
        assert spawnable(CONFIGS, GENERAL, source)[-1] == GENERAL_PURPOSE
        only = SubAgentPolicy(spawn_sources={source}, general_purpose=True)
        assert spawnable([LEAD, RESEARCHER], only, source) == (GENERAL_PURPOSE,)
        others = [s for s in ("agent", "skill", "user") if s != source]
        assert all(spawnable([LEAD, RESEARCHER], only, other) == () for other in others)

    def test_not_unless_the_policy_enables_it(self):
        assert GENERAL_PURPOSE not in spawnable(CONFIGS, EVERY, "agent")

    def test_the_root_can_have_children_for_it_alone(self):
        solo = AgentConfig(name="solo", prompt="p")
        agents = SubAgentPolicy(spawn_sources={"agent"}, general_purpose=True)
        assert can_have_children(solo, [solo], agents, root=True)
        users = SubAgentPolicy(spawn_sources={"user"}, general_purpose=True)
        assert not can_have_children(solo, [solo], users, root=True)

    def test_it_never_has_children_itself(self):
        deeper = SubAgentPolicy(spawn_sources={"agent"}, max_spawn_depth=3, general_purpose=True)
        made = general_purpose_config(RESEARCHER)
        assert not can_have_children(made, CONFIGS, deeper, root=False)
        # An app's own agent of that name, with the built-in type off, is any agent.
        own = AgentConfig(name=GENERAL_PURPOSE, prompt="p", sub_agents=["researcher"])
        assert can_have_children(own, CONFIGS, EVERY, root=False)


class TestGeneralPurposeConfig:
    def test_the_spawners_tools_servers_skills_and_model_with_its_own_prompt(self):
        from agentic_cli.workflow.mcp import MCPServerConfig
        from agentic_cli.workflow.model_settings import ModelSettings

        def lookup() -> dict:
            return {"success": True}

        server = MCPServerConfig(name="atlas", command="atlas")
        settings = ModelSettings(temperature=0.2)
        spawner = AgentConfig(
            name="lead", prompt="Lead.", tools=[lookup, "write_file"], mcp_servers=[server],
            skills=["/skills/research"], model="fast", model_settings=settings,
            sub_agents=["researcher"], include_state_tools=False, description="The lead.",
        )

        made = general_purpose_config(spawner)

        assert (made.name, made.prompt, made.description) == (
            GENERAL_PURPOSE, GENERAL_PURPOSE_PROMPT, GENERAL_PURPOSE_DESCRIPTION,
        )
        assert (made.tools, made.mcp_servers, made.skills) == (
            [lookup, "write_file"], [server], ["/skills/research"],
        )
        assert (made.model, made.model_settings) == ("fast", settings)
        assert (made.sub_agents, made.model_choices, made.spawnable_by) == ([], (), None)
        assert (made.inherit_context, made.continuable, made.include_state_tools) == (
            False, True, True,
        )
        assert (made.limits, made.brief_required) == (None, ("objective",))

    def test_its_lists_are_its_own(self):
        spawner = AgentConfig(name="lead", prompt="Lead.", tools=["write_file"])
        made = general_purpose_config(spawner)
        made.tools.append("read_file")
        assert spawner.tools == ["write_file"]

    def test_not_from_a_spawner_on_a_model_choice(self):
        researcher = AgentConfig(name="researcher", prompt="p", model="strong",
                                 model_choices=("fast",))
        assert general_purpose_config(researcher).model == "strong"


class TestRootType:
    def test_the_agent_nobody_delegates_to_that_is_no_free_type(self):
        assert root_type([HELPER, RESEARCHER, LEAD]) == "lead"

    def test_none_for_none(self):
        assert root_type([HELPER]) is None


class TestGeneralPurposeSpawners:
    """The types a general-purpose agent is built for: the root whenever the
    type is on; another type when runs below the root may start one (its
    ``task``, or a skill it loads) and it does not inherit."""

    def test_none_unless_the_policy_enables_it(self):
        assert general_purpose_spawners(CONFIGS, EVERY) == ()

    @pytest.mark.parametrize("source", ["agent", "skill", "user"])
    def test_the_root_alone_at_the_default_depth(self, source):
        policy = SubAgentPolicy(spawn_sources={source}, general_purpose=True)
        assert general_purpose_spawners(CONFIGS, policy) == ("lead",)

    def test_deeper_every_type_that_does_not_inherit_for_agents(self):
        policy = SubAgentPolicy(spawn_sources={"agent"}, max_spawn_depth=2, general_purpose=True)
        assert general_purpose_spawners(CONFIGS, policy) == (
            "lead", "researcher", "helper", "skilled",
        )

    def test_deeper_for_skills_only_the_types_with_skills(self):
        with_skills = AgentConfig(name="skilled", prompt="p", skills=["/s/research"],
                                  spawnable_by=frozenset({"skill"}))
        configs = [LEAD, RESEARCHER, with_skills]
        policy = SubAgentPolicy(spawn_sources={"skill"}, max_spawn_depth=2, general_purpose=True)
        assert general_purpose_spawners(configs, policy) == ("lead", "skilled")

    def test_deeper_for_the_user_the_root_alone(self):
        policy = SubAgentPolicy(spawn_sources={"user"}, max_spawn_depth=2, general_purpose=True)
        assert general_purpose_spawners(CONFIGS, policy) == ("lead",)
