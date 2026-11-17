"""A sub-agent type's delegation fields: ``AgentConfig.brief_required`` and ``AgentConfig.limits``.

``brief_required`` names the brief fields a delegation to the type must fill
(the title and the objective are always needed); ``limits`` are the type's
run limits, over the policy's field by field. ``validate_agent_graph``
checks both for every backend: ``brief_required`` names brief fields,
``limits`` is an ``AgentLimits``, and the root sets neither, since it is
never delegated to.
"""

from __future__ import annotations

import dataclasses

import pytest

from agentic_cli.workflow.agents.policy import AgentLimits
from agentic_cli.workflow.config import AgentConfig, AgentGraphError, validate_agent_graph


def team(**fields: dict[str, object]) -> list[AgentConfig]:
    """A lead with two sub-agents; ``fields`` sets an agent's fields by its name."""
    configs = [
        AgentConfig(name="lead", prompt="p", sub_agents=["researcher", "writer"]),
        AgentConfig(name="researcher", prompt="p"),
        AgentConfig(name="writer", prompt="p"),
    ]
    return [dataclasses.replace(c, **fields.get(c.name, {})) for c in configs]


class TestTheFields:
    def test_a_delegation_needs_the_objective_by_default(self):
        assert AgentConfig(name="a", prompt="p").brief_required == ("objective",)

    def test_no_limits_of_its_own_by_default(self):
        assert AgentConfig(name="a", prompt="p").limits is None

    def test_they_follow_inherit_context(self):
        # Appended: a positional AgentConfig(...) written before them still binds.
        names = [f.name for f in dataclasses.fields(AgentConfig)]
        assert names[-3:] == ["inherit_context", "brief_required", "limits"]

    def test_they_are_kept(self):
        limits = AgentLimits(timeout=60)
        config = AgentConfig(
            name="a", prompt="p", brief_required=("objective", "deliverable"), limits=limits,
        )
        assert (config.brief_required, config.limits) == (("objective", "deliverable"), limits)


class TestTheGraphChecks:
    def test_a_sub_agent_may_ask_for_more_of_the_brief(self):
        configs = team(researcher={"brief_required": ("objective", "deliverable", "done_when")})
        assert validate_agent_graph(configs).root_name == "lead"

    def test_the_title_may_be_named_too(self):
        assert validate_agent_graph(team(writer={"brief_required": ("title",)})).root_name == "lead"

    def test_a_sub_agent_may_ask_for_nothing_more(self):
        assert validate_agent_graph(team(writer={"brief_required": ()})).root_name == "lead"

    def test_a_list_is_taken_as_well(self):
        configs = team(writer={"brief_required": ["objective", "known"]})
        assert validate_agent_graph(configs).root_name == "lead"

    def test_a_sub_agent_may_have_limits_of_its_own(self):
        configs = team(researcher={"limits": AgentLimits(max_model_calls=10, timeout=120)})
        assert validate_agent_graph(configs).root_name == "lead"

    @pytest.mark.parametrize("value", [("objective", "goal"), ("Deliverable",), (1,)])
    def test_brief_required_names_brief_fields(self, value):
        with pytest.raises(AgentGraphError) as exc:
            validate_agent_graph(team(writer={"brief_required": value}))
        message = str(exc.value)
        assert "brief_required" in message and "'writer'" in message and repr(value) in message
        assert "title, objective, deliverable, constraints, known, done_when" in message

    @pytest.mark.parametrize("value", ["objective", None, {"objective"}])
    def test_brief_required_is_a_tuple_or_a_list(self, value):
        # A string would be read letter by letter.
        with pytest.raises(AgentGraphError) as exc:
            validate_agent_graph(team(writer={"brief_required": value}))
        message = str(exc.value)
        assert "brief_required" in message and "'writer'" in message and repr(value) in message
        assert "must be a tuple or list of brief field names" in message

    @pytest.mark.parametrize("value", [{"timeout": 60}, 60, "fast"])
    def test_limits_is_an_agent_limits(self, value):
        with pytest.raises(AgentGraphError) as exc:
            validate_agent_graph(team(researcher={"limits": value}))
        message = str(exc.value)
        assert "limits" in message and "'researcher'" in message and repr(value) in message
        assert "AgentLimits" in message

    def test_the_root_keeps_the_default_brief(self):
        # The default is indistinguishable from not setting it.
        configs = team(lead={"brief_required": ["objective"]})
        assert validate_agent_graph(configs).root_name == "lead"

    @pytest.mark.parametrize("fields, named", [
        ({"brief_required": ("objective", "deliverable")}, "brief_required"),
        ({"limits": AgentLimits(timeout=60)}, "limits"),
        ({"brief_required": (), "limits": AgentLimits()}, "brief_required and limits"),
    ])
    def test_the_root_sets_neither(self, fields, named):
        with pytest.raises(AgentGraphError) as exc:
            validate_agent_graph(team(lead=fields))
        message = str(exc.value)
        assert "'lead'" in message and f"sets {named}" in message
        assert "never delegated to" in message

    def test_a_single_agent_is_a_root_too(self):
        with pytest.raises(AgentGraphError, match="'solo'.*sets limits"):
            validate_agent_graph(
                [AgentConfig(name="solo", prompt="p", limits=AgentLimits(timeout=1))]
            )

    def test_the_inherit_context_checks_come_first(self):
        configs = team(lead={"inherit_context": True, "limits": AgentLimits(timeout=1)})
        with pytest.raises(AgentGraphError, match="inherit_context"):
            validate_agent_graph(configs)
