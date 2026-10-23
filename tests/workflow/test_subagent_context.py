"""A sub-agent's context: the ``AgentConfig.context`` field and its graph checks.

A sub-agent starts ``fresh`` (the brief alone, the default) or ``inherit``s
its parent's conversation. ``validate_agent_graph`` checks the field where
it sits in the tree, for every backend: the root has no parent to inherit
from, and in 0.7.0 ``inherit`` is for agents without sub-agents of their own.

What each mode sends a model is pinned per backend: ADK in
``tests/workflow/test_adk_delegation.py``, Deep Agents in
``tests/workflow/deepagents/test_fork.py``.
"""

from __future__ import annotations

import dataclasses

import pytest

from agentic_cli.workflow.config import AgentConfig, AgentGraphError, validate_agent_graph


def tree(**contexts: str | None) -> list[AgentConfig]:
    """A coordinator, a middle coordinator and two leaves, with ``contexts`` by name.

    ``lead`` delegates to ``middle`` and ``leaf``; ``middle`` to ``deep``.
    """
    configs = [
        AgentConfig(name="lead", prompt="p", sub_agents=["middle", "leaf"]),
        AgentConfig(name="middle", prompt="p", sub_agents=["deep"]),
        AgentConfig(name="leaf", prompt="p"),
        AgentConfig(name="deep", prompt="p"),
    ]
    return [
        dataclasses.replace(config, context=contexts[config.name])
        if config.name in contexts else config
        for config in configs
    ]


class TestTheField:
    def test_it_is_unset_by_default(self):
        assert AgentConfig(name="a", prompt="p").context is None

    def test_it_is_the_last_field(self):
        # Appended, so a positional AgentConfig(...) written for 0.6 still binds.
        assert [f.name for f in dataclasses.fields(AgentConfig)][-1] == "context"


class TestTheGraphChecks:
    @pytest.mark.parametrize("context", [None, "fresh", "inherit"])
    def test_a_leaf_takes_each_value(self, context):
        assert validate_agent_graph(tree(leaf=context, deep=context)).root_name == "lead"

    @pytest.mark.parametrize("context", [None, "fresh"])
    def test_a_middle_coordinator_starts_fresh(self, context):
        assert validate_agent_graph(tree(middle=context)).root_name == "lead"

    @pytest.mark.parametrize("context", ["fresh", "inherit"])
    def test_the_root_takes_none(self, context):
        with pytest.raises(AgentGraphError) as exc:
            validate_agent_graph(tree(lead=context))
        message = str(exc.value)
        assert "'lead'" in message
        assert "sub-agents only" in message

    def test_a_single_agent_is_a_root_too(self):
        with pytest.raises(AgentGraphError, match="'solo'.*sub-agents only"):
            validate_agent_graph([AgentConfig(name="solo", prompt="p", context="fresh")])

    def test_inherit_is_for_agents_without_sub_agents(self):
        with pytest.raises(AgentGraphError) as exc:
            validate_agent_graph(tree(middle="inherit"))
        message = str(exc.value)
        assert "'middle'" in message
        assert "inherit" in message and "without sub-agents" in message
        assert "0.7.0" in message

    def test_an_unknown_value_is_named(self):
        with pytest.raises(AgentGraphError) as exc:
            validate_agent_graph(tree(leaf="shared"))
        message = str(exc.value)
        assert "'leaf'" in message and "'shared'" in message
        assert "'fresh'" in message and "'inherit'" in message

    def test_the_tree_is_checked_first(self):
        # A context error never hides a broken tree: the first failure is the
        # most fundamental.
        configs = tree(lead="inherit") + [AgentConfig(name="stray", prompt="p")]
        with pytest.raises(AgentGraphError, match="2 roots"):
            validate_agent_graph(configs)
