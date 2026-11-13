"""A sub-agent's context: the ``AgentConfig.inherit_context`` field and its graph checks.

A sub-agent sees the request alone (``inherit_context=False``, the default)
or its parent's conversation too (``True``). ``validate_agent_graph`` checks
the field where it sits in the tree, for every backend: the root has no
parent to inherit from, and in 0.7.0 inheriting is for the main agent's own
sub-agents, and only those without sub-agents of their own.

What each mode sends a model is pinned per backend: ADK in
``tests/workflow/test_adk_delegation.py``, Deep Agents in
``tests/workflow/deepagents/test_fork.py``.
"""

from __future__ import annotations

import dataclasses

import pytest

from agentic_cli.workflow import config as config_module
from agentic_cli.workflow.config import AgentConfig, AgentGraphError, validate_agent_graph


def tree(*inheriting: str, **values: object) -> list[AgentConfig]:
    """A coordinator, a middle coordinator and two leaves.

    ``lead`` delegates to ``middle`` and ``leaf``; ``middle`` to ``deep``.
    The agents named in ``inheriting`` set ``inherit_context=True``;
    ``values`` sets any other value by name (what a stale caller might pass).
    """
    configs = [
        AgentConfig(name="lead", prompt="p", sub_agents=["middle", "leaf"]),
        AgentConfig(name="middle", prompt="p", sub_agents=["deep"]),
        AgentConfig(name="leaf", prompt="p"),
        AgentConfig(name="deep", prompt="p"),
    ]
    chosen: dict[str, object] = {name: True for name in inheriting} | values
    return [
        dataclasses.replace(config, inherit_context=chosen[config.name])
        if config.name in chosen else config
        for config in configs
    ]


class TestTheField:
    def test_it_is_false_by_default(self):
        assert AgentConfig(name="a", prompt="p").inherit_context is False

    def test_it_is_the_last_field(self):
        # Appended, so a positional AgentConfig(...) written for 0.6 still binds.
        assert [f.name for f in dataclasses.fields(AgentConfig)][-1] == "inherit_context"

    def test_the_old_names_are_gone(self):
        # Never released, so no alias: a config written against a develop
        # build fails at once rather than starting fresh by mistake.
        with pytest.raises(TypeError, match="context"):
            AgentConfig(name="a", prompt="p", context="inherit")
        assert not hasattr(config_module, "SubAgentContext")
        assert not hasattr(config_module, "SUBAGENT_CONTEXTS")


class TestTheGraphChecks:
    @pytest.mark.parametrize("inherit", [False, True])
    def test_a_leaf_of_the_main_agent_takes_either_value(self, inherit):
        configs = tree("leaf") if inherit else tree(leaf=False)
        assert validate_agent_graph(configs).root_name == "lead"

    def test_a_leaf_below_another_sub_agent_sees_the_request_alone(self):
        assert validate_agent_graph(tree(deep=False)).root_name == "lead"

    def test_inheriting_is_for_the_main_agents_own_sub_agents(self):
        # On ADK an inheriting agent is shown the main agent's conversation,
        # whatever its depth: below a fresh coordinator it would see what its
        # parent was kept from. Deep Agents' fork sees its parent's. Neither
        # is offered in 0.7.0.
        with pytest.raises(AgentGraphError) as exc:
            validate_agent_graph(tree("deep"))
        message = str(exc.value)
        assert "'deep'" in message and "'lead'" in message
        assert "inherit_context" in message and "0.7.0" in message

    def test_a_middle_coordinator_sees_the_request_alone(self):
        assert validate_agent_graph(tree(middle=False)).root_name == "lead"

    def test_the_root_may_leave_it_false(self):
        # The default: indistinguishable from not setting it.
        assert validate_agent_graph(tree(lead=False)).root_name == "lead"

    def test_the_root_may_not_inherit(self):
        with pytest.raises(AgentGraphError) as exc:
            validate_agent_graph(tree("lead"))
        message = str(exc.value)
        assert "'lead'" in message and "inherit_context" in message
        assert "sub-agents only" in message

    def test_a_single_agent_is_a_root_too(self):
        with pytest.raises(AgentGraphError, match="'solo'.*sub-agents only"):
            validate_agent_graph(
                [AgentConfig(name="solo", prompt="p", inherit_context=True)]
            )

    def test_inheriting_is_for_agents_without_sub_agents(self):
        with pytest.raises(AgentGraphError) as exc:
            validate_agent_graph(tree("middle"))
        message = str(exc.value)
        assert "'middle'" in message
        assert "inherit_context" in message and "without sub-agents" in message
        assert "0.7.0" in message

    @pytest.mark.parametrize("value", ["fresh", "inherit", 1, None])
    def test_a_value_that_is_not_a_bool_is_named(self, value):
        # "fresh" is truthy: taken as True it would show the leaf the whole
        # conversation, the opposite of what a stale caller meant.
        with pytest.raises(AgentGraphError) as exc:
            validate_agent_graph(tree(leaf=value))
        message = str(exc.value)
        assert "'leaf'" in message and repr(value) in message
        assert "True or False" in message

    def test_the_tree_is_checked_first(self):
        # An inherit_context error never hides a broken tree: the first
        # failure is the most fundamental.
        configs = tree("lead") + [AgentConfig(name="stray", prompt="p")]
        with pytest.raises(AgentGraphError, match="2 roots"):
            validate_agent_graph(configs)
