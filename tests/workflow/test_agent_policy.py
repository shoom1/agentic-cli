"""How an app's sub-agents run: ``SubAgentPolicy`` and ``AgentLimits`` (``workflow/agents/policy.py``).

Both are frozen and check their values as they are made: a limit is a
positive int (a timeout: a positive number of seconds), True and False are
no numbers there, and a policy's ``limits`` is an ``AgentLimits``. A type's
limits lie over the policy's field by field (``over``): a field left None
takes the policy's. ``as_policy`` is the one reading of a ``subagents=``
argument.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

import agentic_cli.workflow as workflow
from agentic_cli.workflow import agents
from agentic_cli.workflow.agents.policy import (
    DEFAULT_LIMITS,
    SPAWN_SOURCES,
    AgentLimits,
    SubAgentPolicy,
    as_policy,
)


class TestAgentLimits:
    def test_every_field_is_none_by_default(self):
        assert AgentLimits() == AgentLimits(max_model_calls=None, timeout=None, max_tokens=None)

    def test_it_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            AgentLimits().timeout = 5  # type: ignore[misc]

    def test_positive_values_are_kept(self):
        limits = AgentLimits(max_model_calls=3, timeout=1.5, max_tokens=10_000)
        assert (limits.max_model_calls, limits.timeout, limits.max_tokens) == (3, 1.5, 10_000)

    def test_a_whole_number_of_seconds_is_a_timeout(self):
        assert AgentLimits(timeout=600).timeout == 600

    @pytest.mark.parametrize("field", ["max_model_calls", "max_tokens"])
    @pytest.mark.parametrize("value", [0, -1, 2.5, "5", True, False])
    def test_a_count_must_be_a_positive_int(self, field, value):
        with pytest.raises(ValueError) as exc:
            AgentLimits(**{field: value})
        message = str(exc.value)
        assert f"AgentLimits.{field}" in message and "positive int" in message
        assert repr(value) in message

    @pytest.mark.parametrize("value", [0, -0.5, math.inf, math.nan, "30", True])
    def test_a_timeout_must_be_a_positive_number_of_seconds(self, value):
        with pytest.raises(ValueError) as exc:
            AgentLimits(timeout=value)
        message = str(exc.value)
        assert "AgentLimits.timeout" in message and "positive number" in message

    def test_an_int_too_large_for_a_float_is_no_timeout(self):
        # Refused as inf is, naming the field: no float holds it, so no
        # clock can wait for it.
        with pytest.raises(ValueError) as exc:
            AgentLimits(timeout=10**400)
        message = str(exc.value)
        assert "AgentLimits.timeout" in message and "positive number of seconds" in message

    def test_over_takes_its_own_fields_where_set_and_the_base_elsewhere(self):
        base = AgentLimits(max_model_calls=100, timeout=60)
        mine = AgentLimits(timeout=600, max_tokens=50_000)
        assert mine.over(base) == AgentLimits(max_model_calls=100, timeout=600, max_tokens=50_000)

    def test_over_cannot_lift_a_limit_the_base_sets(self):
        # None is "not set": the base's limit stands.
        assert AgentLimits().over(DEFAULT_LIMITS) == DEFAULT_LIMITS

    def test_over_a_base_without_limits_is_itself(self):
        mine = AgentLimits(max_model_calls=5)
        assert mine.over(AgentLimits()) == mine


class TestSubAgentPolicy:
    def test_the_defaults(self):
        policy = SubAgentPolicy()
        assert policy.max_active == 4
        assert policy.max_runs_per_turn == 20
        assert policy.limits is DEFAULT_LIMITS
        assert policy.max_open == 20
        assert policy.follow_ups is False

    def test_max_open_and_follow_ups_come_after_the_first_ones(self):
        # Appended: a positional SubAgentPolicy(...) written before them still binds.
        assert [f.name for f in dataclasses.fields(SubAgentPolicy)][3:5] == [
            "max_open", "follow_ups",
        ]
        assert SubAgentPolicy(max_open=3).max_open == 3
        assert SubAgentPolicy(follow_ups=True).follow_ups is True

    @pytest.mark.parametrize("value", [1, 0, None, "yes"])
    def test_follow_ups_is_true_or_false(self, value):
        with pytest.raises(ValueError) as exc:
            SubAgentPolicy(follow_ups=value)
        message = str(exc.value)
        assert "SubAgentPolicy.follow_ups must be True or False" in message
        assert repr(value) in message

    def test_the_default_limits_are_a_hundred_model_calls(self):
        assert DEFAULT_LIMITS == AgentLimits(max_model_calls=100)

    def test_it_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            SubAgentPolicy().max_active = 8  # type: ignore[misc]

    def test_its_values_are_kept(self):
        limits = AgentLimits(timeout=30)
        policy = SubAgentPolicy(max_active=1, max_runs_per_turn=5, limits=limits)
        assert (policy.max_active, policy.max_runs_per_turn, policy.limits) == (1, 5, limits)

    @pytest.mark.parametrize("field", ["max_active", "max_runs_per_turn", "max_open"])
    @pytest.mark.parametrize("value", [0, -3, 1.0, None, "4", True])
    def test_a_count_must_be_a_positive_int(self, field, value):
        with pytest.raises(ValueError) as exc:
            SubAgentPolicy(**{field: value})
        message = str(exc.value)
        assert f"SubAgentPolicy.{field}" in message and "positive int" in message
        assert repr(value) in message

    @pytest.mark.parametrize("value", [None, {"max_model_calls": 5}, 100])
    def test_limits_must_be_an_agent_limits(self, value):
        with pytest.raises(ValueError) as exc:
            SubAgentPolicy(limits=value)
        message = str(exc.value)
        assert "SubAgentPolicy.limits" in message and "AgentLimits" in message

    def test_policies_with_the_same_values_are_equal(self):
        assert SubAgentPolicy() == SubAgentPolicy(limits=AgentLimits(max_model_calls=100))

    def test_its_own_limits_replace_the_default_limits_whole(self):
        # As the docstring says: laid over DEFAULT_LIMITS, they keep its 100 calls.
        assert SubAgentPolicy(limits=AgentLimits(timeout=60)).limits.max_model_calls is None
        kept = SubAgentPolicy(limits=AgentLimits(timeout=60).over(DEFAULT_LIMITS)).limits
        assert kept == AgentLimits(max_model_calls=100, timeout=60)


class TestRunTimeStarts:
    """``spawn_sources``: who may start an agent type at run time (none by
    default); ``max_spawn_depth``: from how deep a run may start one."""

    def test_the_defaults(self):
        policy = SubAgentPolicy()
        assert policy.spawn_sources == frozenset()
        assert policy.max_spawn_depth == 1

    def test_the_sources(self):
        assert SPAWN_SOURCES == ("agent", "skill", "user")

    def test_they_come_after_follow_ups(self):
        # Appended: a positional SubAgentPolicy(...) written before them still binds.
        assert [f.name for f in dataclasses.fields(SubAgentPolicy)][-2:] == [
            "spawn_sources", "max_spawn_depth",
        ]

    @pytest.mark.parametrize("given", [{"agent", "user"}, ["user", "agent"], ("agent", "user"),
                                       frozenset({"agent", "user"})])
    def test_a_set_list_or_tuple_of_sources_is_kept_as_a_frozenset(self, given):
        policy = SubAgentPolicy(spawn_sources=given)
        assert policy.spawn_sources == frozenset({"agent", "user"})
        assert type(policy.spawn_sources) is frozenset
        hash(policy)  # still hashable, as a frozen dataclass

    def test_every_source_may_be_named(self):
        assert SubAgentPolicy(spawn_sources=SPAWN_SOURCES).spawn_sources == frozenset(SPAWN_SOURCES)

    @pytest.mark.parametrize("value", [{"agents"}, ["agent", "tool"], {1}, "agent", None, True])
    def test_a_source_that_is_not_one_of_the_three_is_refused(self, value):
        with pytest.raises(ValueError) as exc:
            SubAgentPolicy(spawn_sources=value)
        assert str(exc.value) == (
            f"SubAgentPolicy.spawn_sources has {value!r}: each source must be one of "
            "'agent', 'skill', 'user'."
        )

    def test_a_depth_is_kept(self):
        assert SubAgentPolicy(max_spawn_depth=3).max_spawn_depth == 3

    @pytest.mark.parametrize("value", [0, -1, 1.0, None, "2", True])
    def test_max_spawn_depth_is_a_count(self, value):
        with pytest.raises(ValueError) as exc:
            SubAgentPolicy(max_spawn_depth=value)
        assert str(exc.value) == f"SubAgentPolicy.max_spawn_depth must be a positive int, not {value!r}."

    def test_policies_with_the_same_sources_are_equal(self):
        assert SubAgentPolicy(spawn_sources=["agent"]) == SubAgentPolicy(spawn_sources={"agent"})


class TestAsPolicy:
    def test_none_is_the_default_policy(self):
        assert as_policy(None) == SubAgentPolicy()

    def test_a_policy_is_itself(self):
        policy = SubAgentPolicy(max_active=2)
        assert as_policy(policy) is policy

    @pytest.mark.parametrize("value", [{"max_active": 2}, SubAgentPolicy, AgentLimits(), 4])
    def test_anything_else_is_a_type_error_naming_its_type(self, value):
        with pytest.raises(TypeError) as exc:
            as_policy(value)
        message = str(exc.value)
        assert "subagents must be a SubAgentPolicy or None" in message
        assert type(value).__name__ in message


class TestTheExports:
    @pytest.mark.parametrize("package", [workflow, agents])
    def test_both_packages_export_the_policy_and_the_limits(self, package):
        assert {"SubAgentPolicy", "AgentLimits"} <= set(package.__all__)
        assert package.SubAgentPolicy is SubAgentPolicy
        assert package.AgentLimits is AgentLimits

    def test_the_agents_package_exports_the_default_limits(self):
        assert "DEFAULT_LIMITS" in agents.__all__
        assert agents.DEFAULT_LIMITS is DEFAULT_LIMITS
