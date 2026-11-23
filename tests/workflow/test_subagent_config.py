"""A sub-agent type's delegation fields: ``brief_required``, ``limits``,
``continuable``, ``spawnable_by`` and ``model_choices``.

``brief_required`` names the brief fields a delegation to the type must fill
(the title and the objective are always needed); ``limits`` are the type's
run limits, over the policy's field by field; ``continuable`` says whether
an instance of the type takes follow-ups (False: it is closed when its run
ends). ``validate_agent_graph`` checks all three for every backend:
``brief_required`` names brief fields, ``limits`` is an ``AgentLimits``,
``continuable`` is True or False, and the root sets none, since it is
never delegated to.

``spawnable_by`` names who may start the type at run time: a type that sets
it and is in no ``sub_agents`` list is a free type, never the root, and
``validate_agent_graph`` checks its sources, refuses it on the root and warns
of a free type the policy lets no source start.

``model_choices`` are the models (or roles) a delegation may run the type on
instead of its own: a tuple or list of texts, none named twice, and none on
the root.
"""

from __future__ import annotations

import dataclasses

import pytest

from agentic_cli.workflow import config as config_module
from agentic_cli.workflow.agents.policy import AgentLimits, SubAgentPolicy
from agentic_cli.workflow.config import (
    AgentConfig,
    AgentGraph,
    AgentGraphError,
    validate_agent_graph,
)


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

    def test_an_instance_takes_follow_ups_by_default(self):
        assert AgentConfig(name="a", prompt="p").continuable is True

    def test_they_follow_inherit_context(self):
        # Appended: a positional AgentConfig(...) written before them still binds.
        names = [f.name for f in dataclasses.fields(AgentConfig)]
        assert names[-6:] == [
            "inherit_context", "brief_required", "limits", "continuable",
            "spawnable_by", "model_choices",
        ]

    def test_no_model_choices_by_default(self):
        assert AgentConfig(name="a", prompt="p").model_choices == ()

    def test_only_its_parent_starts_it_by_default(self):
        assert AgentConfig(name="a", prompt="p").spawnable_by is None

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

    def test_a_sub_agent_may_be_one_shot(self):
        assert validate_agent_graph(team(writer={"continuable": False})).root_name == "lead"

    @pytest.mark.parametrize("value", ["no", 0, None])
    def test_continuable_is_true_or_false(self, value):
        with pytest.raises(AgentGraphError) as exc:
            validate_agent_graph(team(writer={"continuable": value}))
        message = str(exc.value)
        assert "continuable must be True or False" in message
        assert f"'writer' ({value!r})" in message

    def test_the_root_is_not_one_shot(self):
        with pytest.raises(AgentGraphError) as exc:
            validate_agent_graph(team(lead={"continuable": False}))
        message = str(exc.value)
        assert "'lead'" in message and "sets continuable" in message
        assert "never delegated to" in message

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


@pytest.fixture
def logged(monkeypatch) -> list[dict]:
    """Every log call of ``workflow/config.py``, with its level."""
    events: list[dict] = []

    class Recorder:
        def __getattr__(self, level: str):
            def log(event: str, **fields) -> None:
                events.append({"event": event, "log_level": level, **fields})

            return log

    monkeypatch.setattr(config_module, "logger", Recorder())
    return events


AGENTS = SubAgentPolicy(spawn_sources={"agent"})


def with_free(*free: AgentConfig, first: bool = False) -> list[AgentConfig]:
    """``team()``'s lead and its sub-agents, with ``free`` before them (``first``) or after."""
    return [*free, *team()] if first else [*team(), *free]


class TestFreeTypes:
    """A type that sets ``spawnable_by`` and is in no ``sub_agents`` list is a
    free type: started at run time only, so it is no root."""

    def test_a_free_type_is_no_second_root(self):
        graph = validate_agent_graph(
            with_free(AgentConfig(name="helper", prompt="p", spawnable_by=frozenset({"agent"}))),
            policy=AGENTS,
        )
        assert (graph.root_name, graph.free_types) == ("lead", ("helper",))

    def test_listed_before_the_root_it_is_still_no_root(self):
        graph = validate_agent_graph(
            with_free(AgentConfig(name="helper", prompt="p", spawnable_by={"agent"}), first=True),
            policy=AGENTS,
        )
        assert (graph.root_name, graph.free_types) == ("lead", ("helper",))

    def test_free_types_are_in_config_order_and_built_with_the_rest(self):
        graph = validate_agent_graph(
            [AgentConfig(name="b", prompt="p", spawnable_by=["agent"]), *team(),
             AgentConfig(name="a", prompt="p", spawnable_by=("agent", "user"))],
            policy=SubAgentPolicy(spawn_sources={"agent", "user"}),
        )
        assert graph.free_types == ("b", "a")
        assert {"a", "b"} <= set(graph.build_order) and set(graph.config_map) >= {"a", "b"}

    def test_a_free_type_may_have_sub_agents_of_its_own(self):
        configs = with_free(
            AgentConfig(name="planner", prompt="p", sub_agents=["checker"],
                        spawnable_by={"agent"}),
            AgentConfig(name="checker", prompt="p"),
        )
        graph = validate_agent_graph(configs, policy=AGENTS)
        assert (graph.root_name, graph.free_types) == ("lead", ("planner",))
        assert graph.build_order.index("checker") < graph.build_order.index("planner")

    def test_a_sub_agent_may_be_started_at_run_time_too(self):
        # In a sub_agents list: its parent's, and no free type.
        graph = validate_agent_graph(team(writer={"spawnable_by": {"agent"}}), policy=AGENTS)
        assert (graph.root_name, graph.free_types) == ("lead", ())

    def test_without_spawnable_by_a_second_unreferenced_agent_is_still_a_second_root(self):
        with pytest.raises(AgentGraphError, match="2 roots"):
            validate_agent_graph(with_free(AgentConfig(name="helper", prompt="p")))

    def test_two_roots_among_the_rest_are_still_refused(self):
        with pytest.raises(AgentGraphError, match="2 roots: lead, other"):
            validate_agent_graph(
                with_free(AgentConfig(name="other", prompt="p"),
                          AgentConfig(name="helper", prompt="p", spawnable_by={"agent"})),
                policy=AGENTS,
            )

    def test_a_free_inheriting_type_with_sub_agents_is_refused(self):
        configs = with_free(
            AgentConfig(name="reader", prompt="p", sub_agents=["checker"],
                        inherit_context=True, spawnable_by={"agent"}),
            AgentConfig(name="checker", prompt="p"),
        )
        with pytest.raises(AgentGraphError, match="'reader' set inherit_context=True and have"):
            validate_agent_graph(configs, policy=AGENTS)

    def test_a_free_inheriting_type_without_sub_agents_is_valid(self):
        configs = with_free(AgentConfig(
            name="reader", prompt="p", inherit_context=True, spawnable_by={"agent"},
        ))
        graph = validate_agent_graph(configs, policy=AGENTS)
        assert (graph.root_name, graph.free_types) == ("lead", ("reader",))

    def test_a_free_types_own_sub_agent_does_not_inherit(self):
        configs = with_free(
            AgentConfig(name="planner", prompt="p", sub_agents=["reader"], spawnable_by={"agent"}),
            AgentConfig(name="reader", prompt="p", inherit_context=True),
        )
        with pytest.raises(AgentGraphError, match="'reader' set inherit_context=True below"):
            validate_agent_graph(configs, policy=AGENTS)

    def test_a_free_type_may_set_what_a_delegation_sets(self):
        configs = with_free(AgentConfig(
            name="helper", prompt="p", spawnable_by={"agent"}, continuable=False,
            limits=AgentLimits(timeout=60), brief_required=("objective", "deliverable"),
        ))
        assert validate_agent_graph(configs, policy=AGENTS).free_types == ("helper",)

    def test_a_graph_has_no_free_types_by_default(self):
        assert validate_agent_graph(team()).free_types == ()
        assert AgentGraph(config_map={}, build_order=(), root_name="r").free_types == ()


class TestSpawnableBy:
    """What ``spawnable_by`` may hold, and that the root sets none."""

    @pytest.mark.parametrize("value", [{"agent"}, ["skill", "user"], ("agent", "skill", "user"),
                                       frozenset({"user"})])
    def test_a_set_list_or_tuple_of_sources(self, value):
        assert validate_agent_graph(team(writer={"spawnable_by": value})).root_name == "lead"

    @pytest.mark.parametrize("value", [{"agents"}, ["agent", "tool"], "agent", True, {1}])
    def test_a_source_that_is_not_one_of_the_three_is_refused(self, value):
        with pytest.raises(AgentGraphError) as exc:
            validate_agent_graph(team(writer={"spawnable_by": value}))
        assert str(exc.value) == (
            f"Agent 'writer' has spawnable_by {value!r}: each source must be one of "
            "'agent', 'skill', 'user'."
        )

    @pytest.mark.parametrize("value", [frozenset(), set(), [], ()])
    def test_an_empty_one_is_refused(self, value):
        with pytest.raises(AgentGraphError) as exc:
            validate_agent_graph(with_free(AgentConfig(name="helper", prompt="p",
                                                       spawnable_by=value)))
        assert str(exc.value) == (
            "Agent 'helper' has an empty spawnable_by: name at least one source, or leave it "
            "out for a type only its parent starts."
        )

    def test_the_root_cannot_set_it(self):
        # The agent nobody delegates to, the only one: the root, whatever it sets.
        with pytest.raises(AgentGraphError) as exc:
            validate_agent_graph(team(lead={"spawnable_by": {"agent"}}), policy=AGENTS)
        assert str(exc.value) == (
            "Agent 'lead' is the root agent and cannot set spawnable_by: only a sub-agent "
            "type is started at run time."
        )

    def test_a_single_agent_cannot_set_it(self):
        with pytest.raises(AgentGraphError, match="'solo' is the root agent and cannot set"):
            validate_agent_graph(
                [AgentConfig(name="solo", prompt="p", spawnable_by={"user"})], policy=AGENTS
            )

    def test_with_every_unreferenced_agent_free_there_is_no_root(self):
        configs = [
            AgentConfig(name="helper", prompt="p", spawnable_by={"agent"}),
            *team(lead={"spawnable_by": {"agent"}}),
        ]
        with pytest.raises(AgentGraphError) as exc:
            validate_agent_graph(configs, policy=AGENTS)
        assert str(exc.value) == (
            "Agent graph has no root: every agent nobody delegates to (helper, lead) sets "
            "spawnable_by, which makes it a type started at run time. Leave spawnable_by off "
            "the agent the runner starts from."
        )

    def test_the_values_are_checked_before_the_root(self):
        # A bad source on a sub-agent is named, not the shape of the tree.
        with pytest.raises(AgentGraphError, match="'writer' has spawnable_by"):
            validate_agent_graph(team(writer={"spawnable_by": {"bogus"}},
                                      lead={"spawnable_by": {"agent"}}))


class TestAFreeTypeNoSourceCanStart:
    """A free type the policy lets no source start is warned of, once for each
    validation: an app may enable its sources per install."""

    def test_is_a_warning_naming_it(self, logged):
        configs = with_free(AgentConfig(name="helper", prompt="p", spawnable_by={"skill", "user"}))

        graph = validate_agent_graph(configs, policy=AGENTS)

        assert graph.free_types == ("helper",)
        assert logged == [{
            "event": "subagent_type_unreachable", "log_level": "warning", "agent": "helper",
            "spawnable_by": ["skill", "user"], "spawn_sources": ["agent"],
        }]

    def test_with_the_default_policy_every_free_type_is(self, logged):
        validate_agent_graph(with_free(
            AgentConfig(name="a", prompt="p", spawnable_by={"agent"}),
            AgentConfig(name="b", prompt="p", spawnable_by={"user"}),
        ))
        assert [(e["event"], e["agent"]) for e in logged] == [
            ("subagent_type_unreachable", "a"), ("subagent_type_unreachable", "b"),
        ]

    def test_one_a_source_can_start_is_not(self, logged):
        validate_agent_graph(
            with_free(AgentConfig(name="helper", prompt="p", spawnable_by={"agent", "user"})),
            policy=AGENTS,
        )
        assert logged == []

    def test_each_validation_says_so_again(self, logged):
        # A model switch validates the same configs once more: a warning each time.
        configs = with_free(AgentConfig(name="helper", prompt="p", spawnable_by={"user"}))

        validate_agent_graph(configs, policy=AGENTS)
        validate_agent_graph(configs, policy=AGENTS)

        assert [e["event"] for e in logged] == ["subagent_type_unreachable"] * 2

    def test_a_sub_agent_its_parent_starts_is_not(self, logged):
        validate_agent_graph(team(writer={"spawnable_by": {"user"}}), policy=AGENTS)
        assert logged == []

    def test_the_policy_must_be_a_policy(self):
        with pytest.raises(TypeError, match="subagents must be a SubAgentPolicy or None"):
            validate_agent_graph(team(), policy={"spawn_sources": {"agent"}})


class TestModelChoices:
    """``model_choices``: the models a delegation may pick instead of the type's own."""

    @pytest.mark.parametrize("value", [
        ("fast",), ["fast", "anthropic:claude-haiku-4-5"], ("claude-opus-4-7",),
    ])
    def test_a_tuple_or_list_of_models_or_roles(self, value):
        assert validate_agent_graph(
            team(writer={"model_choices": value})
        ).root_name == "lead"

    @pytest.mark.parametrize("value", ["fast", ("fast", ""), ("fast", "  "), (1,), None,
                                       {"fast"}])
    def test_each_is_text(self, value):
        # A string would be read letter by letter.
        with pytest.raises(AgentGraphError) as exc:
            validate_agent_graph(team(writer={"model_choices": value}))
        assert str(exc.value) == (
            "model_choices must be a tuple or list of models or roles, each as text: "
            f"'writer' ({value!r})."
        )

    def test_none_is_named_twice(self):
        with pytest.raises(AgentGraphError) as exc:
            validate_agent_graph(team(
                writer={"model_choices": ("fast", "strong", "fast")},
                researcher={"model_choices": ["strong", "strong"]},
            ))
        assert str(exc.value) == (
            "model_choices names a model twice: 'researcher' ('strong'), 'writer' ('fast')."
        )

    def test_the_root_sets_none(self):
        with pytest.raises(AgentGraphError) as exc:
            validate_agent_graph(team(lead={"model_choices": ("fast",)}))
        assert str(exc.value) == (
            "Agent 'lead' is the root agent, but sets model_choices: they apply to a "
            "delegation, and the root is never delegated to. Remove them from the root."
        )

    def test_a_free_type_may_set_them(self):
        configs = with_free(AgentConfig(
            name="helper", prompt="p", spawnable_by={"agent"}, model_choices=("fast",),
        ))
        assert validate_agent_graph(configs, policy=AGENTS).free_types == ("helper",)


class TestTheGeneralPurposeName:
    """With the built-in ``general_purpose`` type on, no agent of the app's may
    take its name; with it off, the name is any agent's."""

    GENERAL = SubAgentPolicy(spawn_sources={"agent"}, general_purpose=True)

    #: The root named so, and a sub-agent named so.
    NAMED = {
        "root": {"lead": {"name": "general_purpose"}},
        "sub-agent": {"lead": {"sub_agents": ["researcher", "general_purpose"]},
                      "writer": {"name": "general_purpose"}},
    }

    @pytest.mark.parametrize("named", NAMED.values(), ids=NAMED.keys())
    def test_an_agent_of_that_name_is_refused_while_it_is_on(self, named):
        with pytest.raises(AgentGraphError) as exc:
            validate_agent_graph(team(**named), policy=self.GENERAL)
        assert str(exc.value) == (
            "An agent named 'general_purpose' clashes with the built-in general_purpose "
            "type, which this app's SubAgentPolicy enables."
        )

    def test_with_it_off_the_name_is_any_agents(self):
        graph = validate_agent_graph(team(**self.NAMED["sub-agent"]), policy=AGENTS)
        assert "general_purpose" in graph.config_map

    def test_the_names_are_checked_for_duplicates_first(self):
        configs = [*team(), AgentConfig(name="writer", prompt="p"),
                   AgentConfig(name="general_purpose", prompt="p")]
        with pytest.raises(AgentGraphError, match="Duplicate agent name"):
            validate_agent_graph(configs, policy=self.GENERAL)
