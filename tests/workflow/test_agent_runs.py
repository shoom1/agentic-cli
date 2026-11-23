"""A manager's sub-agent runs, kept by conversation (``workflow/agents/runs.py``).

``run_for`` names a run once per (conversation, parent run, call id), within
its conversation: instance ``<agent>-<n>``, numbered per agent, and run
``<instance>#1``; a key whose run has ended names a new one (a provider that
numbers its call ids repeats them each turn). ``observe`` records a run from
its ``AGENT_RUN`` events, under the turn they are reported in;
``interrupt_open`` ends the runs a stopped or failed turn left open. A
manager's ``agent_runs`` is its registry, and its turn entry (``_turn``)
ends the runs a turn leaves open. ``export`` and ``restore`` are a
conversation's records as its store keeps them, and ``on_change`` is told
each change.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable

import pytest

from agentic_cli.workflow.agents import (
    INSTANCE_CLOSED,
    INSTANCE_IDLE,
    INSTANCE_RUNNING,
    INSTANCE_STATUSES,
    INSTANCE_STOPPED,
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_INPUT_REQUIRED,
    RUN_INTERRUPTED,
    RUN_QUEUED,
    RUN_RUNNING,
    RUN_STARTED,
    RUN_WORKING_STATUSES,
    AgentRuns,
    InstanceRecord,
    RunRecord,
    result_head,
    run_title,
)
from agentic_cli.workflow.agents.envelope import INTERRUPTED_ERROR
from agentic_cli.workflow.agents.runs import RECORDS_VERSION, run_instance, stored_instance_ids
from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.sessions import (
    RunRef,
    SessionRef,
    reset_active_turn,
    root_run,
    set_active_turn,
)
from tests.conftest import MockContext

ONE = SessionRef("app", "u", "s1")
TWO = SessionRef("app", "u", "s2")


def in_turn(conversation: SessionRef, work: Callable[[], Any]) -> Any:
    """``work()`` as the turn of ``conversation`` runs it."""
    token = set_active_turn(conversation)
    try:
        return work()
    finally:
        reset_active_turn(token)


def started(runs: AgentRuns, conversation: SessionRef, call_id: str, agent: str) -> RunRef:
    ref = runs.run_for(conversation, None, call_id, agent)
    in_turn(conversation, lambda: runs.observe(
        WorkflowEvent.agent_run(ref, RUN_STARTED, title=f"{agent} work", call_id=call_id)
    ))
    return ref


class TestRunFor:
    def test_a_call_starts_its_agents_first_instance(self):
        assert AgentRuns().run_for(ONE, None, "c1", "researcher") == RunRef(
            agent="researcher", instance="researcher-1", run="researcher-1#1",
            parent_run=None, depth=1,
        )

    def test_the_root_as_parent_is_no_parent_run(self):
        runs = AgentRuns()

        assert runs.run_for(ONE, root_run("coordinator"), "c1", "researcher") is runs.run_for(
            ONE, None, "c1", "researcher"
        )

    def test_the_same_call_is_the_same_run(self):
        runs = AgentRuns()

        assert runs.run_for(ONE, None, "c1", "researcher") is runs.run_for(
            ONE, None, "c1", "researcher"
        )

    def test_each_agent_is_numbered_on_its_own(self):
        runs = AgentRuns()

        named = [
            runs.run_for(ONE, None, call_id, agent).run
            for call_id, agent in (("c1", "researcher"), ("c2", "writer"), ("c3", "researcher"))
        ]

        assert named == ["researcher-1#1", "writer-1#1", "researcher-2#1"]

    def test_each_conversation_is_numbered_on_its_own(self):
        runs = AgentRuns()
        runs.run_for(ONE, None, "c1", "researcher")

        assert runs.run_for(TWO, None, "c1", "researcher").run == "researcher-1#1"

    def test_a_child_runs_under_its_parents_run(self):
        runs = AgentRuns()
        middle = runs.run_for(ONE, None, "c1", "middle")

        assert runs.run_for(ONE, middle, "m1", "leaf") == RunRef(
            agent="leaf", instance="leaf-1", run="leaf-1#1", parent_run="middle-1#1", depth=2
        )

    def test_one_call_id_under_two_parents_is_two_runs(self):
        runs = AgentRuns()
        middle = runs.run_for(ONE, None, "call_0", "middle")

        assert runs.run_for(ONE, middle, "call_0", "leaf").run == "leaf-1#1"

    def test_a_call_id_used_again_after_its_run_ended_is_a_new_run(self):
        runs = AgentRuns()
        first = started(runs, ONE, "call_0", "researcher")
        in_turn(ONE, lambda: runs.observe(
            WorkflowEvent.agent_run(first, RUN_COMPLETED, call_id="call_0", result="ok")
        ))

        assert runs.run_for(ONE, None, "call_0", "researcher").run == "researcher-2#1"

    def test_no_conversation_is_one_too(self):
        runs = AgentRuns()

        assert runs.run_for(None, None, "c1", "researcher").run == "researcher-1#1"

    def test_a_run_id_names_its_instance(self):
        """What a job's ``started_by`` is matched by, for a run named by its id."""
        runs = AgentRuns()
        middle = runs.run_for(ONE, None, "c1", "middle")
        leaf = runs.run_for(ONE, middle, "m1", "leaf")

        assert [run_instance(ref.run) for ref in (middle, leaf)] == ["middle-1", "leaf-1"]

    def test_without_a_call_each_is_a_new_instance_under_the_root(self):
        """The user's start (``/agent new``) has no call: each is its own run,
        an open one too, numbered with the rest."""
        runs = AgentRuns()
        first = started(runs, ONE, "c1", "helper")

        named = [runs.run_for(ONE, None, None, "helper") for _ in range(2)]

        assert named == [
            RunRef(agent="helper", instance="helper-2", run="helper-2#1", parent_run=None,
                   depth=1),
            RunRef(agent="helper", instance="helper-3", run="helper-3#1", parent_run=None,
                   depth=1),
        ]
        assert runs.run_for(ONE, None, "c1", "helper") is first  # the call's own, still open


class TestNumberFrom:
    """A conversation's store may keep instances an earlier manager numbered
    (a restart, a switch of backend or store): ``number_from`` numbers the
    next ones past them."""

    def test_the_next_instance_is_numbered_past_the_count(self):
        runs = AgentRuns()

        runs.number_from(ONE, {"researcher": 3})

        assert runs.run_for(ONE, None, "c1", "researcher").run == "researcher-4#1"

    def test_another_agent_or_conversation_is_not_moved(self):
        runs = AgentRuns()

        runs.number_from(ONE, {"researcher": 3})

        assert runs.run_for(ONE, None, "c1", "writer").run == "writer-1#1"
        assert runs.run_for(TWO, None, "c1", "researcher").run == "researcher-1#1"

    def test_a_count_is_never_lowered(self):
        runs = AgentRuns()
        for call_id in ("c1", "c2", "c3"):
            runs.run_for(ONE, None, call_id, "researcher")

        runs.number_from(ONE, {"researcher": 1})

        assert runs.run_for(ONE, None, "c4", "researcher").run == "researcher-4#1"

    def test_an_open_run_keeps_its_name(self):
        runs = AgentRuns()
        open_run = runs.run_for(ONE, None, "c1", "researcher")

        runs.number_from(ONE, {"researcher": 5})

        assert runs.run_for(ONE, None, "c1", "researcher") is open_run
        assert runs.run_for(ONE, None, "c2", "researcher").run == "researcher-6#1"

    def test_no_conversation_is_one_too(self):
        runs = AgentRuns()

        runs.number_from(None, {"researcher": 2})

        assert runs.run_for(None, None, "c1", "researcher").run == "researcher-3#1"


class TestObserve:
    def test_a_start_is_recorded(self):
        runs = AgentRuns()
        ref = runs.run_for(ONE, None, "c1", "researcher")
        event = WorkflowEvent.agent_run(ref, RUN_STARTED, title="Find X.", call_id="c1")

        in_turn(ONE, lambda: runs.observe(event))

        assert runs.records(ONE) == [RunRecord(
            ref=ref, call_id="c1", title="Find X.", status=RUN_RUNNING, started=event.timestamp,
        )]

    def test_a_working_runs_status_is_the_latest_it_reported(self):
        runs = AgentRuns()
        ref = started(runs, ONE, "c1", "researcher")
        seen: list[str] = []

        for status in (RUN_QUEUED, RUN_RUNNING, RUN_INPUT_REQUIRED, RUN_RUNNING):
            in_turn(ONE, lambda: runs.observe(WorkflowEvent.agent_run(ref, status, call_id="c1")))
            seen.append(runs.records(ONE)[0].status)

        assert seen == [RUN_QUEUED, RUN_RUNNING, RUN_INPUT_REQUIRED, RUN_RUNNING]
        assert RUN_WORKING_STATUSES == {RUN_QUEUED, RUN_RUNNING, RUN_INPUT_REQUIRED}

    def test_a_working_status_after_the_end_or_without_a_start_changes_nothing(self):
        runs = AgentRuns()
        ended = started(runs, ONE, "c1", "researcher")
        in_turn(ONE, lambda: runs.observe(
            WorkflowEvent.agent_run(ended, RUN_COMPLETED, call_id="c1", result="ok")
        ))
        unknown = runs.run_for(ONE, None, "c2", "researcher")

        in_turn(ONE, lambda: runs.observe(WorkflowEvent.agent_run(ended, RUN_RUNNING)))
        in_turn(ONE, lambda: runs.observe(WorkflowEvent.agent_run(unknown, RUN_QUEUED)))

        assert [(r.ref, r.status) for r in runs.records(ONE)] == [(ended, RUN_COMPLETED)]

    def test_an_end_ends_the_record(self):
        runs = AgentRuns()
        ref = started(runs, ONE, "c1", "researcher")
        end = WorkflowEvent.agent_run(ref, RUN_FAILED, call_id="c1", result="", error="overloaded")

        in_turn(ONE, lambda: runs.observe(end))

        (record,) = runs.records(ONE)
        assert (record.status, record.ended, record.result, record.error, record.partial) == (
            RUN_FAILED, end.timestamp, "", "overloaded", False,
        )

    def test_an_end_keeps_what_the_run_used(self):
        runs = AgentRuns()
        ref = started(runs, ONE, "c1", "researcher")
        used = {"m": {"input": 140, "output": 14}}

        in_turn(ONE, lambda: runs.observe(
            WorkflowEvent.agent_run(ref, RUN_COMPLETED, call_id="c1", result="ok", usage=used)
        ))

        assert runs.records(ONE)[0].usage == used

    def test_an_end_keeps_the_runs_own_plan_and_tasks(self):
        runs = AgentRuns()
        ref = started(runs, ONE, "c1", "researcher")
        tasks = [{"id": "1", "description": "find X", "status": "completed"}]

        in_turn(ONE, lambda: runs.observe(WorkflowEvent.agent_run(
            ref, RUN_COMPLETED, call_id="c1", result="ok", plan="- [x] find X", tasks=tasks
        )))

        record = runs.records(ONE)[0]
        assert (record.plan, record.tasks) == ("- [x] find X", tasks)

    def test_a_plan_that_is_no_string_and_tasks_that_are_no_dicts_are_left_out(self):
        from agentic_cli.workflow.agents.runs import own_plan_and_tasks

        assert own_plan_and_tasks({"plan": ["x"], "tasks": "none"}) == ("", [])
        assert own_plan_and_tasks({"plan": 5, "tasks": [{"description": "a"}, "b"]}) == ("", [])
        assert own_plan_and_tasks({"plan": "p", "tasks": [{"description": "a"}]}) == (
            "p", [{"description": "a"}],
        )
        assert own_plan_and_tasks({}) == ("", [])

        runs = AgentRuns()
        ref = started(runs, ONE, "c1", "researcher")
        in_turn(ONE, lambda: runs.observe(WorkflowEvent.agent_run(
            ref, RUN_COMPLETED, call_id="c1", plan=5, tasks="x"  # type: ignore[arg-type]
        )))
        record = runs.records(ONE)[0]
        assert (record.plan, record.tasks) == ("", [])

    def test_an_end_without_them_leaves_the_record_empty(self):
        runs = AgentRuns()
        ref = started(runs, ONE, "c1", "researcher")

        in_turn(ONE, lambda: runs.observe(
            WorkflowEvent.agent_run(ref, RUN_COMPLETED, call_id="c1", result="ok")
        ))

        record = runs.records(ONE)[0]
        assert (record.plan, record.tasks) == ("", [])

    def test_a_partial_end_says_so(self):
        runs = AgentRuns()
        ref = started(runs, ONE, "c1", "researcher")

        in_turn(ONE, lambda: runs.observe(
            WorkflowEvent.agent_run(ref, RUN_COMPLETED, call_id="c1", result="X is", partial=True)
        ))

        assert runs.records(ONE)[0].partial is True

    def test_an_end_without_its_start_is_recorded_too(self):
        runs = AgentRuns()
        ref = runs.run_for(ONE, None, "c1", "researcher")

        in_turn(ONE, lambda: runs.observe(
            WorkflowEvent.agent_run(ref, RUN_COMPLETED, call_id="c1", result="ok")
        ))

        (record,) = runs.records(ONE)
        assert (record.status, record.title, record.result) == (RUN_COMPLETED, "", "ok")

    def test_records_are_kept_by_the_turns_conversation(self):
        runs = AgentRuns()
        started(runs, ONE, "c1", "researcher")

        assert runs.records(TWO) == []
        assert [r.ref.run for r in runs.records(ONE)] == ["researcher-1#1"]

    def test_records_are_in_start_order(self):
        runs = AgentRuns()
        for call_id, agent in (("c1", "writer"), ("c2", "researcher"), ("c3", "writer")):
            started(runs, ONE, call_id, agent)

        assert [r.ref.run for r in runs.records(ONE)] == [
            "writer-1#1", "researcher-1#1", "writer-2#1",
        ]

    def test_other_events_change_nothing(self):
        runs = AgentRuns()

        in_turn(ONE, lambda: runs.observe(WorkflowEvent.text("hi")))

        assert runs.records(ONE) == []

    def test_a_record_is_found_by_its_run_or_its_instance(self):
        runs = AgentRuns()
        started(runs, ONE, "c1", "researcher")

        record = runs.record(ONE, "researcher-1#1")
        assert record is not None
        assert runs.record(ONE, "researcher-1") is record
        assert runs.record(ONE, "researcher-2") is None


class TestInterruptOpen:
    def test_the_open_runs_end_interrupted(self):
        runs = AgentRuns()
        open_one = started(runs, ONE, "c1", "researcher")
        done = started(runs, ONE, "c2", "researcher")
        in_turn(ONE, lambda: runs.observe(
            WorkflowEvent.agent_run(done, RUN_COMPLETED, call_id="c2", result="ok")
        ))

        assert runs.interrupt_open(ONE) == [open_one]
        assert [(r.ref, r.status) for r in runs.records(ONE)] == [
            (open_one, RUN_INTERRUPTED), (done, RUN_COMPLETED),
        ]
        assert runs.records(ONE)[0].ended is not None

    def test_none_open_is_none(self):
        assert AgentRuns().interrupt_open(ONE) == []

    def test_an_interrupted_run_keeps_what_it_used(self):
        runs = AgentRuns()
        open_one = started(runs, ONE, "c1", "researcher")
        done = started(runs, ONE, "c2", "researcher")
        in_turn(ONE, lambda: runs.observe(
            WorkflowEvent.agent_run(done, RUN_COMPLETED, call_id="c2", result="ok")
        ))
        asked: list[str] = []

        def usage(run: str) -> dict[str, dict[str, int]]:
            asked.append(run)
            return {"m": {"input": 50, "output": 5}}

        runs.interrupt_open(ONE, usage=usage)

        assert asked == [open_one.run]
        assert [r.usage for r in runs.records(ONE)] == [{"m": {"input": 50, "output": 5}}, {}]


class TestForget:
    def test_a_forgotten_conversation_starts_again(self):
        runs = AgentRuns()
        started(runs, ONE, "c1", "researcher")
        started(runs, TWO, "c1", "researcher")

        runs.forget(ONE)

        assert runs.records(ONE) == []
        assert runs.run_for(ONE, None, "c9", "researcher").run == "researcher-1#1"
        assert [r.ref.run for r in runs.records(TWO)] == ["researcher-1#1"]


def ended(
    runs: AgentRuns, conversation: SessionRef, ref: RunRef, **fields: Any
) -> None:
    """``ref``'s run ended completed, as ``fields`` say."""
    in_turn(conversation, lambda: runs.observe(WorkflowEvent.agent_run(
        ref, RUN_COMPLETED, call_id="c1", result="ok", **fields
    )))


class TestInstances:
    """Each instance's status, kept from its runs' events: a start makes it
    (or reopens it) ``running``, an end sets the status it carries, and
    ``AGENT_INSTANCE`` a change without a run."""

    def test_the_statuses(self):
        assert INSTANCE_STATUSES == {"running", "idle", "stopped", "closed"}
        assert (INSTANCE_RUNNING, INSTANCE_IDLE, INSTANCE_STOPPED, INSTANCE_CLOSED) == (
            "running", "idle", "stopped", "closed",
        )

    def test_a_first_runs_start_makes_its_instance_running(self):
        runs = AgentRuns()
        ref = runs.run_for(ONE, None, "c1", "researcher")

        in_turn(ONE, lambda: runs.observe(WorkflowEvent.agent_run(
            ref, RUN_STARTED, title="Find X.", call_id="c1",
            inherit_context=True, continuable=False,
        )))

        assert runs.instances(ONE) == [InstanceRecord(
            instance="researcher-1", agent="researcher", parent="root", depth=1,
            inherit_context=True, title="Find X.", status=INSTANCE_RUNNING,
            runs=["researcher-1#1"], continuable=False,
        )]
        assert runs.instance(ONE, "researcher-1") is runs.instances(ONE)[0]

    def test_a_start_without_them_keeps_the_defaults(self):
        runs = AgentRuns()
        started(runs, ONE, "c1", "researcher")

        record = runs.instance(ONE, "researcher-1")
        assert (record.inherit_context, record.continuable) == (False, True)
        assert (record.tools, record.model) == (None, None)

    def test_a_first_runs_start_keeps_the_tools_and_model_it_was_given(self):
        runs = AgentRuns()
        ref = runs.run_for(ONE, None, "c1", "researcher")

        in_turn(ONE, lambda: runs.observe(WorkflowEvent.agent_run(
            ref, RUN_STARTED, title="Find X.", call_id="c1", tools=["lookup", "fetch"],
            model="fast",
        )))

        record = runs.instance(ONE, "researcher-1")
        assert (record.tools, record.model) == (("lookup", "fetch"), "fast")

    def test_a_first_runs_start_keeps_its_spawner_type(self):
        runs = AgentRuns()
        ref = runs.run_for(ONE, None, "c1", "general_purpose")

        in_turn(ONE, lambda: runs.observe(WorkflowEvent.agent_run(
            ref, RUN_STARTED, title="Sum it up.", call_id="c1", based_on="lead",
        )))

        record = runs.instance(ONE, "general_purpose-1")
        assert (record.agent, record.based_on) == ("general_purpose", "lead")

    def test_a_follow_ups_start_changes_neither(self):
        runs = AgentRuns()
        ref = runs.run_for(ONE, None, "c1", "researcher")
        in_turn(ONE, lambda: runs.observe(WorkflowEvent.agent_run(
            ref, RUN_STARTED, title="Find X.", call_id="c1", tools=[], model="fast",
        )))
        ended(runs, ONE, ref)
        second = runs.next_run(ONE, "researcher-1", parent_run=None)

        in_turn(ONE, lambda: runs.observe(WorkflowEvent.agent_run(
            second, RUN_STARTED, title="Go on.", call_id="c2", input="Go on.",
        )))

        record = runs.instance(ONE, "researcher-1")
        assert (record.tools, record.model) == ((), "fast")

    def test_a_childs_parent_is_its_parent_runs_instance(self):
        runs = AgentRuns()
        middle = started(runs, ONE, "c1", "middle")
        leaf = runs.run_for(ONE, middle, "m1", "leaf")

        in_turn(ONE, lambda: runs.observe(
            WorkflowEvent.agent_run(leaf, RUN_STARTED, title="Look.", call_id="m1")
        ))

        record = runs.instance(ONE, "leaf-1")
        assert (record.parent, record.depth) == ("middle-1", 2)

    @pytest.mark.parametrize("status", [INSTANCE_IDLE, INSTANCE_CLOSED, INSTANCE_STOPPED])
    def test_an_end_leaves_it_as_its_instance_status_says(self, status):
        runs = AgentRuns()
        ref = started(runs, ONE, "c1", "researcher")

        ended(runs, ONE, ref, instance_status=status)

        assert runs.instance(ONE, "researcher-1").status == status

    def test_an_end_that_says_none_leaves_it_idle(self):
        runs = AgentRuns()
        ref = started(runs, ONE, "c1", "researcher")

        ended(runs, ONE, ref)

        assert runs.instance(ONE, "researcher-1").status == INSTANCE_IDLE

    def test_its_next_run_is_numbered_past_its_runs(self):
        runs = AgentRuns()
        middle = started(runs, ONE, "c1", "middle")
        leaf = runs.run_for(ONE, middle, "m1", "leaf")
        in_turn(ONE, lambda: runs.observe(
            WorkflowEvent.agent_run(leaf, RUN_STARTED, title="Look.", call_id="m1")
        ))
        ended(runs, ONE, leaf, instance_status=INSTANCE_IDLE)

        assert runs.next_run(ONE, "leaf-1", parent_run="middle-1#1") == RunRef(
            agent="leaf", instance="leaf-1", run="leaf-1#2", parent_run="middle-1#1", depth=2,
        )
        # The root's own message, or the user's, has no parent run.
        assert runs.next_run(ONE, "leaf-1", parent_run=None).parent_run is None

    def test_an_unknown_instance_has_no_next_run(self):
        with pytest.raises(KeyError):
            AgentRuns().next_run(ONE, "researcher-1", parent_run=None)

    def test_a_follow_ups_start_reopens_it_and_keeps_its_first_title(self):
        runs = AgentRuns()
        first = started(runs, ONE, "c1", "researcher")
        ended(runs, ONE, first, instance_status=INSTANCE_IDLE)
        second = runs.next_run(ONE, "researcher-1", parent_run=None)

        in_turn(ONE, lambda: runs.observe(WorkflowEvent.agent_run(
            second, RUN_STARTED, title="Go on.", call_id="c2", input="Go on.\nAnd on.",
        )))

        record = runs.instance(ONE, "researcher-1")
        assert (record.status, record.runs, record.title) == (
            INSTANCE_RUNNING, ["researcher-1#1", "researcher-1#2"], "researcher work",
        )
        run = runs.record(ONE, "researcher-1#2")
        assert (run.title, run.input) == ("Go on.", "Go on.\nAnd on.")
        assert runs.record(ONE, "researcher-1#1").input == ""
        # The instance's latest run is its record.
        assert runs.record(ONE, "researcher-1") is run

    def test_agent_instance_sets_its_status_without_a_run(self):
        runs = AgentRuns()
        ref = started(runs, ONE, "c1", "researcher")
        ended(runs, ONE, ref, instance_status=INSTANCE_IDLE)

        in_turn(ONE, lambda: runs.observe(
            WorkflowEvent.agent_instance(ref, INSTANCE_CLOSED, reason="closed by its parent")
        ))

        assert runs.instance(ONE, "researcher-1").status == INSTANCE_CLOSED
        assert [r.status for r in runs.records(ONE)] == [RUN_COMPLETED]

    def test_agent_instance_for_an_unknown_instance_or_status_changes_nothing(self):
        runs = AgentRuns()
        ref = started(runs, ONE, "c1", "researcher")
        other = runs.run_for(ONE, None, "c2", "writer")

        in_turn(ONE, lambda: runs.observe(WorkflowEvent.agent_instance(other, INSTANCE_CLOSED)))
        in_turn(ONE, lambda: runs.observe(WorkflowEvent.agent_instance(ref, "asleep")))

        assert [i.instance for i in runs.instances(ONE)] == ["researcher-1"]
        assert runs.instance(ONE, "researcher-1").status == INSTANCE_RUNNING

    def test_a_turn_that_stops_its_run_leaves_it_idle(self):
        runs = AgentRuns()
        started(runs, ONE, "c1", "researcher")

        runs.interrupt_open(ONE)

        assert runs.instance(ONE, "researcher-1").status == INSTANCE_IDLE

    def test_a_turn_that_stops_a_one_shot_run_closes_its_instance(self):
        runs = AgentRuns()
        ref = runs.run_for(ONE, None, "c1", "checker")
        in_turn(ONE, lambda: runs.observe(WorkflowEvent.agent_run(
            ref, RUN_STARTED, title="Check", call_id="c1", continuable=False
        )))

        runs.interrupt_open(ONE)

        assert runs.instance(ONE, "checker-1").status == INSTANCE_CLOSED

    def test_the_order_runs_ended_in_is_kept(self):
        runs = AgentRuns()
        first = started(runs, ONE, "c1", "researcher")
        second = started(runs, ONE, "c2", "writer")
        third = started(runs, ONE, "c3", "reader")
        ended(runs, ONE, second)
        ended(runs, ONE, first)

        order = [runs.end_order(ONE, ref.run) for ref in (second, first, third)]

        assert 0 < order[0] < order[1] and order[2] == 0
        assert runs.end_order(TWO, second.run) == 0

    def test_a_run_ended_twice_does_not_give_the_next_end_its_number(self):
        """A turn that ends a run open (``interrupt_open``) and then the run's own
        late end: each end still gets a number of its own."""
        runs = AgentRuns()
        first = started(runs, ONE, "c1", "researcher")
        second = started(runs, ONE, "c2", "writer")
        runs.interrupt_open(ONE)
        ended(runs, ONE, first)
        third = started(runs, ONE, "c3", "reader")
        ended(runs, ONE, third)

        order = [runs.end_order(ONE, ref.run) for ref in (first, second, third)]

        assert len(set(order)) == 3
        assert order[2] > order[0] > order[1]

    def test_they_are_kept_by_conversation_in_start_order(self):
        runs = AgentRuns()
        for call_id, agent in (("c1", "writer"), ("c2", "researcher")):
            started(runs, ONE, call_id, agent)

        assert [i.instance for i in runs.instances(ONE)] == ["writer-1", "researcher-1"]
        assert runs.instances(TWO) == [] and runs.instance(TWO, "writer-1") is None

    def test_forget_drops_them(self):
        runs = AgentRuns()
        started(runs, ONE, "c1", "researcher")

        runs.forget(ONE)

        assert runs.instances(ONE) == []


class TestTitlesAndResults:
    def test_a_title_is_the_requests_first_line_cut(self):
        assert run_title("\n  Find X.  \nThen report.") == "Find X."
        assert run_title("y" * 120) == "y" * 80
        assert run_title("") == ""

    def test_a_result_keeps_its_first_200_characters(self):
        assert result_head("z" * 500) == "z" * 200
        assert result_head("short") == "short"


class _Delegating(BaseWorkflowManager):
    """A backend whose turn starts one sub-agent run, then ends it, unless stopped first.

    ``one_shot``: the run's type is not continuable.
    """

    one_shot = False

    @property
    def backend_type(self) -> str:
        return "delegating"

    def _get_state_tools(self) -> list:
        return []

    async def _do_initialize(self) -> None:
        return None

    async def process(self, message, user_id, session_id=None):
        async with self._turn(session_id, user_id) as ref:
            child = self.agent_runs.run_for(ref, None, "c1", "researcher")
            start = WorkflowEvent.agent_run(
                child, RUN_STARTED, title=message, call_id="c1", continuable=not self.one_shot
            )
            self.agent_runs.observe(start)
            yield start
            end = WorkflowEvent.agent_run(child, RUN_COMPLETED, call_id="c1", result="done")
            self.agent_runs.observe(end)
            yield end


class TestTheTurnEndsItsOpenRuns:
    @pytest.fixture
    def manager(self):
        with MockContext(google_api_key="test-key") as context:
            yield _Delegating([AgentConfig(name="a", prompt="p")], settings=context.settings)

    async def test_a_turn_stopped_with_a_run_open_leaves_it_interrupted(self, manager):
        events = manager.process("Find X.", "u", "s")
        await anext(events)  # the run has started; the turn holds
        await events.aclose()  # the consumer stops: the turn ends now

        (record,) = manager.agent_runs.records(manager.session_ref("s", "u"))
        assert record.status == RUN_INTERRUPTED

    async def test_a_one_shot_run_it_stops_closes_its_instance_and_frees_its_stop(self, manager):
        manager.one_shot = True
        ref = manager.session_ref("s", "u")
        events = manager.process("Find X.", "u", "s")
        await anext(events)
        manager._open_calls.cancelled(ref, instance="researcher-1")
        await events.aclose()

        assert manager.agent_runs.instance(ref, "researcher-1").status == "closed"
        assert manager._open_calls.take_cancel(ref, instance="researcher-1") is False

    async def test_each_turn_is_counted_in_its_conversations_ledger(self, manager):
        for _ in range(2):
            [event async for event in manager.process("Find X.", "u", "s")]

        assert manager.usage_of(manager.session_ref("s", "u")).turns == 2
        assert manager.usage_of(manager.session_ref("t", "u")).turns == 0

    async def test_a_turn_that_ends_its_runs_leaves_them_as_they_ended(self, manager):
        [event async for event in manager.process("Find X.", "u", "s")]

        (record,) = manager.agent_runs.records(manager.session_ref("s", "u"))
        assert (record.status, record.title, record.result) == (RUN_COMPLETED, "Find X.", "done")


# -- the records a conversation keeps beside it: export, restore, and each change told ------


def a_conversation(runs: AgentRuns) -> None:
    """ONE's records: researcher-1 ran twice (a brief, then a follow-up that
    was cut and left a plan and tasks), writer-1 still works (queued), and
    checker-1 was closed when its one run ended; TWO has a run of its own."""
    first = runs.run_for(ONE, None, "c1", "researcher")
    in_turn(ONE, lambda: runs.observe(WorkflowEvent.agent_run(
        first, RUN_STARTED, title="Find X.", call_id="c1", brief={"title": "Find X.",
        "objective": "Find X."}, started_by="agent:root", mode="blocking",
        inherit_context=False, continuable=True, tools=["lookup"], model="fast",
    )))
    ended(runs, ONE, first, usage={"m": {"input": 120, "output": 30}}, instance_status="idle")
    second = runs.next_run(ONE, "researcher-1", parent_run=None)
    in_turn(ONE, lambda: runs.observe(WorkflowEvent.agent_run(
        second, RUN_STARTED, title="Go on.", call_id="c2", started_by="user",
        mode="blocking", input="Go on.\nAnd on.",
    )))
    in_turn(ONE, lambda: runs.observe(WorkflowEvent.agent_run(
        second, "cancelled", call_id="c2", result="Half.", error="stopped by the user",
        partial=True, cancelled_by="user", plan="1. Look", usage={"m": {"input": 7, "output": 2}},
        tasks=[{"description": "Look", "status": "completed"}], instance_status="stopped",
    )))
    writer = started(runs, ONE, "c3", "writer")
    in_turn(ONE, lambda: runs.observe(WorkflowEvent.agent_run(writer, RUN_QUEUED, call_id="c3")))
    checker = started(runs, ONE, "c4", "checker")
    ended(runs, ONE, checker, instance_status="closed")
    started(runs, TWO, "c1", "researcher")


class TestExportAndRestore:
    """``export`` gives a conversation's records as JSON-safe data (every
    instance, every run's record, the id counters, ``RECORDS_VERSION``), and
    ``restore`` reads it back into another registry: what a conversation's
    store keeps beside it, and what a new manager loads."""

    def test_the_version(self):
        assert RECORDS_VERSION == 1

    def test_a_restore_reads_back_what_an_export_wrote(self):
        runs = AgentRuns()
        a_conversation(runs)
        in_turn(ONE, lambda: runs.observe(WorkflowEvent.agent_run(
            runs.record(ONE, "writer-1#1").ref, RUN_COMPLETED, call_id="c3", result="ok",
        )))
        again = AgentRuns()

        interrupted = again.restore(ONE, json.loads(json.dumps(runs.export(ONE))))

        assert interrupted == []
        assert again.records(ONE) == runs.records(ONE)
        assert again.instances(ONE) == runs.instances(ONE)
        assert again.records(TWO) == []
        # The counters come back: the next instance of each type is numbered past them.
        assert again.run_for(ONE, None, "c9", "researcher").instance == "researcher-2"
        assert again.run_for(ONE, None, "c10", "checker").instance == "checker-2"

    def test_its_shape(self):
        runs = AgentRuns()
        ref = started(runs, ONE, "c1", "researcher")
        ended(runs, ONE, ref, usage={"m": {"input": 120, "output": 30}})
        record = runs.record(ONE, "researcher-1#1")

        data = runs.export(ONE)

        assert data == {
            "v": RECORDS_VERSION,
            "saved": data["saved"],
            "numbered": {"researcher": 1},
            "instances": [{
                "instance": "researcher-1", "agent": "researcher", "parent": "root",
                "depth": 1, "inherit_context": False, "title": "researcher work",
                "status": INSTANCE_IDLE, "runs": ["researcher-1#1"], "continuable": True,
                "tools": None, "model": None, "based_on": None,
            }],
            "runs": [{
                "run": "researcher-1#1", "agent": "researcher", "instance": "researcher-1",
                "parent_run": None, "depth": 1, "call_id": "c1", "title": "researcher work",
                "status": RUN_COMPLETED, "started": record.started.isoformat(),
                "ended": record.ended.isoformat(), "result": "ok", "error": None,
                "partial": False, "usage": {"m": {"input": 120, "output": 30}}, "brief": {},
                "started_by": None, "mode": "blocking", "cancelled_by": None, "plan": "",
                "tasks": [], "input": "", "end": 1,
            }],
        }
        assert datetime.fromisoformat(data["saved"]) >= record.ended

    def test_a_run_found_working_ends_interrupted_and_its_instance_is_idle(self):
        runs = AgentRuns()
        a_conversation(runs)
        data = runs.export(ONE)
        again = AgentRuns()

        interrupted = again.restore(ONE, data)

        assert interrupted == ["writer-1#1"]
        writer = again.record(ONE, "writer-1#1")
        assert (writer.status, writer.error) == (RUN_INTERRUPTED, INTERRUPTED_ERROR)
        # It ended when its records were last saved: it was working then.
        assert writer.ended == datetime.fromisoformat(data["saved"])
        assert [(i.instance, i.status) for i in again.instances(ONE)] == [
            ("researcher-1", INSTANCE_STOPPED), ("writer-1", INSTANCE_IDLE),
            ("checker-1", INSTANCE_CLOSED),
        ]
        # The other runs are as they were.
        assert again.records(ONE)[:2] == runs.records(ONE)[:2]

    @pytest.mark.parametrize("status", [RUN_STARTED, RUN_RUNNING, RUN_INPUT_REQUIRED])
    def test_each_working_status_is_found_interrupted(self, status):
        runs = AgentRuns()
        ref = started(runs, ONE, "c1", "researcher")
        if status != RUN_RUNNING:
            runs.records(ONE)[0].status = status
        again = AgentRuns()

        assert again.restore(ONE, runs.export(ONE)) == [ref.run]
        assert again.record(ONE, ref.run).status == RUN_INTERRUPTED

    def test_it_replaces_its_conversations_records_alone(self):
        runs = AgentRuns()
        a_conversation(runs)
        other = AgentRuns()
        started(other, ONE, "c1", "writer")
        started(other, ONE, "c2", "writer")
        started(other, TWO, "c1", "writer")

        other.restore(ONE, runs.export(ONE))

        assert [r.ref.run for r in other.records(ONE)] == [
            "researcher-1#1", "researcher-1#2", "writer-1#1", "checker-1#1",
        ]
        assert [r.ref.run for r in other.records(TWO)] == ["writer-1#1"]
        # Its counters are the records', not the ones replaced.
        assert other.run_for(ONE, None, "c9", "writer").instance == "writer-2"

    def test_the_counters_never_fall_below_the_instances_kept(self):
        runs = AgentRuns()
        for call_id in ("c1", "c2", "c3"):
            started(runs, ONE, call_id, "researcher")
        data = {**runs.export(ONE), "numbered": {}}
        again = AgentRuns()

        again.restore(ONE, data)

        assert again.run_for(ONE, None, "c9", "researcher").instance == "researcher-4"

    @pytest.mark.parametrize("data", [
        None,
        "records",
        {},
        {"v": 2, "saved": "2026-11-19T10:00:00", "numbered": {}, "instances": [], "runs": []},
        {"v": 1, "saved": "2026-11-19T10:00:00", "numbered": {"a": "1"}, "instances": [],
         "runs": []},
        {"v": 1, "saved": "2026-11-19T10:00:00", "numbered": {}, "instances": [{"instance": 1}],
         "runs": []},
        {"v": 1, "saved": "2026-11-19T10:00:00", "numbered": {}, "instances": [],
         "runs": [{"run": "researcher-1#1"}]},
    ], ids=["none", "text", "empty", "another-version", "a-counter", "an-instance", "a-run"])
    def test_records_it_cannot_read_change_nothing(self, data):
        runs = AgentRuns()
        started(runs, ONE, "c1", "writer")
        before = (runs.records(ONE), runs.instances(ONE))

        with pytest.raises(ValueError):
            runs.restore(ONE, data)

        assert (runs.records(ONE), runs.instances(ONE)) == before

    def test_an_instances_tools_are_a_list_and_read_back_as_given(self):
        runs = AgentRuns()
        a_conversation(runs)

        data = json.loads(json.dumps(runs.export(ONE)))

        assert (data["instances"][0]["tools"], data["instances"][0]["model"]) == (
            ["lookup"], "fast"
        )
        again = AgentRuns()
        again.restore(ONE, data)
        assert (again.instance(ONE, "researcher-1").tools,
                again.instance(ONE, "researcher-1").model) == (("lookup",), "fast")

    def test_records_written_before_tools_and_model_read_them_as_none(self):
        runs = AgentRuns()
        a_conversation(runs)
        data = json.loads(json.dumps(runs.export(ONE)))
        for instance in data["instances"]:
            del instance["tools"], instance["model"]
        again = AgentRuns()

        assert again.restore(ONE, data) == ["writer-1#1"]

        assert {(i.tools, i.model) for i in again.instances(ONE)} == {(None, None)}

    def test_a_spawner_type_is_kept_and_read_back(self):
        runs = AgentRuns()
        ref = started(runs, ONE, "c1", "general_purpose")
        in_turn(ONE, lambda: runs.observe(WorkflowEvent.agent_run(
            runs.run_for(ONE, None, "c2", "general_purpose"), RUN_STARTED, title="More.",
            call_id="c2", based_on="lead",
        )))
        ended(runs, ONE, ref)

        data = json.loads(json.dumps(runs.export(ONE)))

        assert [i["based_on"] for i in data["instances"]] == [None, "lead"]
        again = AgentRuns()
        again.restore(ONE, data)
        assert [i.based_on for i in again.instances(ONE)] == [None, "lead"]

    def test_records_written_before_a_spawner_type_read_it_as_none(self):
        runs = AgentRuns()
        a_conversation(runs)
        data = json.loads(json.dumps(runs.export(ONE)))
        for instance in data["instances"]:
            del instance["based_on"]
        again = AgentRuns()

        assert again.restore(ONE, data) == ["writer-1#1"]

        assert {i.based_on for i in again.instances(ONE)} == {None}

    @pytest.mark.parametrize("field, value", [
        ("tools", "lookup"), ("tools", [1]), ("tools", {"lookup": True}), ("model", 3),
        ("model", ["fast"]), ("based_on", 1), ("based_on", ["lead"]),
    ])
    def test_tools_or_a_model_that_do_not_read_are_records_it_cannot_read(
        self, field, value
    ):
        runs = AgentRuns()
        a_conversation(runs)
        data = json.loads(json.dumps(runs.export(ONE)))
        data["instances"][0][field] = value

        with pytest.raises(ValueError, match="do not read"):
            AgentRuns().restore(ONE, data)

    def test_the_instances_records_name(self):
        runs = AgentRuns()
        a_conversation(runs)

        assert stored_instance_ids(runs.export(ONE)) == ["researcher-1", "writer-1", "checker-1"]
        assert stored_instance_ids({"v": 2, "instances": []}) is None
        assert stored_instance_ids("records") is None


def a_nested_conversation(runs: AgentRuns, *, working: bool) -> None:
    """ONE's records with every non-default value: an inheriting analyst, its
    helper (depth 2, a ``parent_run``), and a one-shot scribe (not
    continuable). ``working``: the analyst and the scribe are still working."""
    analyst = runs.run_for(ONE, None, "c1", "analyst")
    in_turn(ONE, lambda: runs.observe(WorkflowEvent.agent_run(
        analyst, RUN_STARTED, title="Analyse.", call_id="c1", inherit_context=True,
        continuable=True, started_by="agent:root",
    )))
    helper = runs.run_for(ONE, analyst, "n1", "helper")
    in_turn(ONE, lambda: runs.observe(WorkflowEvent.agent_run(
        helper, RUN_STARTED, title="Help.", call_id="n1", started_by="agent:analyst-1",
    )))
    ended(runs, ONE, helper, instance_status="idle")
    scribe = runs.run_for(ONE, None, "c2", "scribe")
    in_turn(ONE, lambda: runs.observe(WorkflowEvent.agent_run(
        scribe, RUN_STARTED, title="Write.", call_id="c2", continuable=False,
    )))
    if not working:
        ended(runs, ONE, scribe, instance_status="closed")
        ended(runs, ONE, analyst, instance_status="idle")


class TestEveryFieldRoundTrips:
    """Each field of every instance and run comes back as it was, the
    non-default values too (an inheriting instance, a one-shot type, a nested
    run's parent and depth)."""

    def test_an_export_restores_to_equal_records(self):
        runs = AgentRuns()
        a_nested_conversation(runs, working=False)
        again = AgentRuns()

        again.restore(ONE, json.loads(json.dumps(runs.export(ONE))))

        assert again.instances(ONE) == runs.instances(ONE)
        assert again.records(ONE) == runs.records(ONE)
        analyst, helper, scribe = again.instances(ONE)
        assert (analyst.inherit_context, analyst.continuable) == (True, True)
        assert (scribe.inherit_context, scribe.continuable) == (False, False)
        assert (helper.parent, helper.depth) == ("analyst-1", 2)
        nested = again.record(ONE, "helper-1#1")
        assert nested.ref.parent_run == "analyst-1#1" and nested.ref.depth == 2

    def test_a_one_shot_instance_found_working_is_closed_not_idle(self):
        runs = AgentRuns()
        a_nested_conversation(runs, working=True)
        again = AgentRuns()

        interrupted = again.restore(ONE, runs.export(ONE))

        assert interrupted == ["analyst-1#1", "scribe-1#1"]
        assert [(i.instance, i.status) for i in again.instances(ONE)] == [
            ("analyst-1", INSTANCE_IDLE), ("helper-1", INSTANCE_IDLE),
            ("scribe-1", INSTANCE_CLOSED),
        ]

    def test_the_order_runs_ended_in_survives_equal_times(self):
        runs = AgentRuns()
        first = started(runs, ONE, "c1", "researcher")
        second = started(runs, ONE, "c2", "writer")
        moment = datetime(2026, 11, 19, 10, 0, 0)
        for ref, call in ((second, "c2"), (first, "c1")):  # the later-started ends first
            event = WorkflowEvent.agent_run(ref, RUN_COMPLETED, call_id=call, result="ok")
            event.timestamp = moment
            in_turn(ONE, lambda: runs.observe(event))
        again = AgentRuns()

        again.restore(ONE, runs.export(ONE))

        assert again.end_order(ONE, "writer-1#1") == 1
        assert again.end_order(ONE, "researcher-1#1") == 2

    def test_a_time_with_a_zone_is_records_it_cannot_read(self):
        runs = AgentRuns()
        started(runs, ONE, "c1", "researcher")
        data = runs.export(ONE)
        data["saved"] = "2026-11-19T10:00:00+02:00"
        before = runs.records(ONE)

        with pytest.raises(ValueError):
            runs.restore(ONE, data)

        assert runs.records(ONE) == before


class TestEachChangeIsTold:
    """``on_change`` is told the conversation each time its records change
    (what a store's writes follow), and not for an event that changes
    nothing."""

    def test_a_change_and_no_change(self):
        told: list[SessionRef | None] = []
        runs = AgentRuns(on_change=told.append)

        ref = started(runs, ONE, "c1", "researcher")
        assert told == [ONE]
        for status, changes in [(RUN_RUNNING, False), (RUN_QUEUED, True), (RUN_QUEUED, False)]:
            in_turn(ONE, lambda: runs.observe(WorkflowEvent.agent_run(ref, status, call_id="c1")))
            assert len(told) == 1 + changes
            told[:] = [ONE]
        ended(runs, ONE, ref)
        assert told == [ONE, ONE]
        for _ in range(2):
            in_turn(ONE, lambda: runs.observe(WorkflowEvent.agent_instance(ref, INSTANCE_CLOSED)))
        assert told == [ONE, ONE, ONE]

    def test_interrupt_open_tells_when_it_ends_one(self):
        told: list[SessionRef | None] = []
        runs = AgentRuns(on_change=told.append)
        assert runs.interrupt_open(ONE) == [] and told == []
        started(runs, ONE, "c1", "researcher")
        told.clear()

        runs.interrupt_open(ONE)

        assert told == [ONE]

    def test_a_restore_tells_nothing(self):
        told: list[SessionRef | None] = []
        source = AgentRuns()
        started(source, ONE, "c1", "researcher")

        AgentRuns(on_change=told.append).restore(ONE, source.export(ONE))

        assert told == []
