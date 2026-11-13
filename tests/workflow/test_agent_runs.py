"""A manager's sub-agent runs, kept by conversation (``workflow/agents/runs.py``).

``run_for`` names a run once per (conversation, parent run, call id), within
its conversation: instance ``<agent>-<n>``, numbered per agent, and run
``<instance>#1``; a key whose run has ended names a new one (a provider that
numbers its call ids repeats them each turn). ``observe`` records a run from
its ``AGENT_RUN`` events, under the turn they are reported in;
``interrupt_open`` ends the runs a stopped or failed turn left open. A
manager's ``agent_runs`` is its registry, and its turn entry (``_turn``)
ends the runs a turn leaves open.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from agentic_cli.workflow.agents import (
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_INTERRUPTED,
    RUN_STARTED,
    AgentRuns,
    RunRecord,
    result_head,
    run_title,
)
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


class TestObserve:
    def test_a_start_is_recorded(self):
        runs = AgentRuns()
        ref = runs.run_for(ONE, None, "c1", "researcher")
        event = WorkflowEvent.agent_run(ref, RUN_STARTED, title="Find X.", call_id="c1")

        in_turn(ONE, lambda: runs.observe(event))

        assert runs.records(ONE) == [RunRecord(
            ref=ref, call_id="c1", title="Find X.", status=RUN_STARTED, started=event.timestamp,
        )]

    def test_an_end_ends_the_record(self):
        runs = AgentRuns()
        ref = started(runs, ONE, "c1", "researcher")
        end = WorkflowEvent.agent_run(ref, RUN_FAILED, call_id="c1", result="", error="overloaded")

        in_turn(ONE, lambda: runs.observe(end))

        (record,) = runs.records(ONE)
        assert (record.status, record.ended, record.result, record.error, record.partial) == (
            RUN_FAILED, end.timestamp, "", "overloaded", False,
        )

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


class TestForget:
    def test_a_forgotten_conversation_starts_again(self):
        runs = AgentRuns()
        started(runs, ONE, "c1", "researcher")
        started(runs, TWO, "c1", "researcher")

        runs.forget(ONE)

        assert runs.records(ONE) == []
        assert runs.run_for(ONE, None, "c9", "researcher").run == "researcher-1#1"
        assert [r.ref.run for r in runs.records(TWO)] == ["researcher-1#1"]


class TestTitlesAndResults:
    def test_a_title_is_the_requests_first_line_cut(self):
        assert run_title("\n  Find X.  \nThen report.") == "Find X."
        assert run_title("y" * 120) == "y" * 80
        assert run_title("") == ""

    def test_a_result_keeps_its_first_200_characters(self):
        assert result_head("z" * 500) == "z" * 200
        assert result_head("short") == "short"


class _Delegating(BaseWorkflowManager):
    """A backend whose turn starts one sub-agent run, then ends it, unless stopped first."""

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
            start = WorkflowEvent.agent_run(child, RUN_STARTED, title=message, call_id="c1")
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

    async def test_a_turn_that_ends_its_runs_leaves_them_as_they_ended(self, manager):
        [event async for event in manager.process("Find X.", "u", "s")]

        (record,) = manager.agent_runs.records(manager.session_ref("s", "u"))
        assert (record.status, record.title, record.result) == (RUN_COMPLETED, "Find X.", "done")
