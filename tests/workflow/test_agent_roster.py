"""The main agent's roster of its open sub-agents (``workflow/agents/roster.py``).

The roster opens the main agent's requests, after the memory index: one line
per open instance (idle, stopped, running) of the main agent's own, newest
first, at most ``ROSTER_MAX`` characters in all, whole lines only. It is a
snapshot per conversation (``Rosters``), taken when the conversation's first
request asks for it and kept until the manager forgets it, where it forgets
the memory index's, so the head of the request stays the same for caching.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from agentic_cli.workflow.agents import (
    INSTANCE_CLOSED,
    INSTANCE_IDLE,
    INSTANCE_RUNNING,
    INSTANCE_STOPPED,
    RUN_COMPLETED,
    RUN_RUNNING,
    AgentRuns,
    InstanceRecord,
    RunRecord,
    SubAgentPolicy,
)
from agentic_cli.workflow.agents.roster import (
    ROSTER_CLOSE,
    ROSTER_HEADER,
    ROSTER_MAX,
    ROSTER_OPEN,
    Rosters,
    roster_text,
)
from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.sessions import (
    RunRef,
    SessionRef,
    reset_active_turn,
    set_active_turn,
)
from tests.conftest import MockContext
from tests.workflow.test_agent_records import TEAM, _Kept, turn

FOLLOWING = SubAgentPolicy(follow_ups=True)
REF = SessionRef(app_name="app", user_id="u", session_id="s")
OTHER = SessionRef(app_name="app", user_id="u", session_id="t")
AT = datetime(2026, 11, 19, tzinfo=timezone.utc)


def instance(n: int, status: str = INSTANCE_IDLE, *, runs: int = 1, agent: str = "researcher",
             parent: str = "root", title: str | None = None) -> InstanceRecord:
    name = f"{agent}-{n}"
    return InstanceRecord(
        instance=name, agent=agent, parent=parent, depth=1, inherit_context=False,
        title=title if title is not None else f"Find thing {n}.", status=status,
        runs=[f"{name}#{k}" for k in range(1, runs + 1)],
    )


def record(run: str, result: str = "", status: str = RUN_COMPLETED) -> RunRecord:
    name = run.rpartition("#")[0]
    ref = RunRef(agent=name.rpartition("-")[0], instance=name, run=run, parent_run=None, depth=1)
    return RunRecord(ref=ref, call_id=None, title="", status=status, started=AT, result=result)


class TestTheText:
    def test_the_specs_lines_newest_first(self):
        instances = [instance(1), instance(2, INSTANCE_RUNNING), instance(3, INSTANCE_STOPPED)]
        records = {
            "researcher-1#1": record("researcher-1#1", "X is 42.\n\nMore on X."),
            "researcher-2#1": record("researcher-2#1", "", RUN_RUNNING),
            "researcher-3#1": record("researcher-3#1", "Y is 7."),
        }

        assert roster_text(instances, records) == "\n".join([
            "<agents>",
            "Your sub-agents in this conversation (message_agent continues one):",
            "- researcher-3 (researcher, stopped): Find thing 3. — Y is 7.",
            "- researcher-2 (researcher, running): Find thing 2.",
            "- researcher-1 (researcher, idle): Find thing 1. — X is 42.",
            "</agents>",
        ])
        assert (ROSTER_OPEN, ROSTER_CLOSE, ROSTER_MAX) == ("<agents>", "</agents>", 2_000)
        assert ROSTER_HEADER == "Your sub-agents in this conversation (message_agent continues one):"

    def test_a_closed_instance_is_left_out_and_none_open_is_no_roster(self):
        instances = [instance(1, INSTANCE_CLOSED), instance(2)]

        assert "researcher-1" not in roster_text(instances, {})
        assert "- researcher-2 (researcher, idle): Find thing 2." in roster_text(instances, {})
        assert roster_text([instance(1, INSTANCE_CLOSED)], {}) == ""
        assert roster_text([], {}) == ""

    def test_the_last_result_is_the_newest_runs_that_has_one(self):
        # A follow-up at work has no result yet: its previous run's is the last.
        working = instance(1, INSTANCE_RUNNING, runs=2)
        records = {
            "researcher-1#1": record("researcher-1#1", "\n  X is 42.  \nMore."),
            "researcher-1#2": record("researcher-1#2", "", RUN_RUNNING),
        }

        assert "- researcher-1 (researcher, running): Find thing 1. — X is 42." in (
            roster_text([working], records)
        )

    def test_at_most_its_cap_in_whole_lines_the_oldest_left_out(self):
        long_title = "T" * 80
        result = "R" * 200
        instances = [instance(n, title=long_title) for n in range(1, 13)]
        records = {f"researcher-{n}#1": record(f"researcher-{n}#1", result) for n in range(1, 13)}

        text = roster_text(instances, records)

        lines = text.splitlines()
        assert lines[:2] == [ROSTER_OPEN, ROSTER_HEADER] and lines[-1] == ROSTER_CLOSE
        listed = lines[2:-1]
        line = "- researcher-{n} (researcher, idle): " + long_title + " — " + result
        assert listed == [line.format(n=n) for n in range(12, 12 - len(listed), -1)]
        assert len(text) <= ROSTER_MAX
        # One line more would pass the cap.
        assert len(text) + len(line.format(n=12 - len(listed))) + 1 > ROSTER_MAX
        assert 0 < len(listed) < 12


def started(ref: RunRef) -> WorkflowEvent:
    return WorkflowEvent.agent_run(
        ref, "started", call_id="c1", title=f"Find {ref.instance}.", inherit_context=False,
        continuable=True,
    )


def ended(ref: RunRef, result: str) -> WorkflowEvent:
    return WorkflowEvent.agent_run(
        ref, RUN_COMPLETED, call_id="c1", result=result, instance_status=INSTANCE_IDLE,
    )


def run_of(instance_id: str, *, parent_run: str | None = None, depth: int = 1) -> RunRef:
    return RunRef(agent=instance_id.rpartition("-")[0], instance=instance_id,
                  run=f"{instance_id}#1", parent_run=parent_run, depth=depth)


def observed(runs: AgentRuns, conversation: SessionRef, *events: WorkflowEvent) -> None:
    token = set_active_turn(conversation)
    try:
        for event in events:
            runs.observe(event)
    finally:
        reset_active_turn(token)


class TestTheSnapshots:
    def test_taken_once_per_conversation_and_kept_until_forgotten(self):
        runs = AgentRuns()
        rosters = Rosters(lambda: runs)
        first = run_of("researcher-1")
        observed(runs, REF, started(first), ended(first, "X is 42."))
        token = set_active_turn(REF)
        try:
            taken = rosters.now()
            assert "- researcher-1 (researcher, idle): Find researcher-1. — X is 42." in taken
            # A new instance changes nothing until the snapshot is forgotten.
            second = run_of("researcher-2")
            observed(runs, REF, started(second))
            assert rosters.now() == taken and rosters.snapshots == {REF: taken}
            rosters.forget()
            assert "- researcher-2 (researcher, running): Find researcher-2." in rosters.now()
        finally:
            reset_active_turn(token)

    def test_none_is_a_snapshot_too_and_outside_a_turn_there_is_none(self):
        runs = AgentRuns()
        rosters = Rosters(lambda: runs)
        assert rosters.now() == "" and rosters.snapshots == {}
        token = set_active_turn(REF)
        try:
            assert rosters.now() == ""
            observed(runs, REF, started(run_of("researcher-1")))
            assert rosters.now() == ""  # taken: empty until forgotten
        finally:
            reset_active_turn(token)
        assert rosters.snapshots == {REF: ""}

    def test_only_the_main_agents_own_and_only_the_conversations(self):
        runs = AgentRuns()
        rosters = Rosters(lambda: runs)
        mine = run_of("researcher-1")
        nested = run_of("checker-1", parent_run="researcher-1#1", depth=2)
        theirs = run_of("researcher-1")
        observed(runs, REF, started(mine), started(nested))
        observed(runs, OTHER, started(theirs), ended(theirs, "Elsewhere."))
        token = set_active_turn(REF)
        try:
            text = rosters.now()
        finally:
            reset_active_turn(token)

        assert "researcher-1 (researcher, running)" in text
        assert "checker-1" not in text and "Elsewhere." not in text

    def test_forget_by_conversation_and_clear(self):
        rosters = Rosters(AgentRuns)
        rosters.snapshots.update({REF: "a", OTHER: "b"})
        rosters.forget(OTHER)
        assert rosters.snapshots == {REF: "a"}
        rosters.clear()
        assert rosters.snapshots == {}


# -- the manager: taken in a turn, forgotten where the memory index's is ------------------------


@pytest.fixture
def context():
    with MockContext(google_api_key="test-key") as made:
        yield made


@pytest.fixture
async def following(context):
    """``following(subagents=FOLLOWING)``: a manager whose turn delegates once
    (``test_agent_records._Kept``), with ``subagents``; cleaned up after the test."""
    made: list[_Kept] = []

    def make(subagents: SubAgentPolicy | None = FOLLOWING) -> _Kept:
        manager = _Kept(TEAM, settings=context.settings, stored={}, subagents=subagents)
        made.append(manager)
        return manager

    yield make
    for manager in made:
        await asyncio.wait_for(manager.cleanup(), 10)


def roster_in(manager: _Kept, ref: SessionRef) -> str:
    """What the manager's ``_roster_now`` gives in ``ref``'s turn."""
    token = set_active_turn(ref)
    try:
        return manager._roster_now()
    finally:
        reset_active_turn(token)


class TestTheManagersRoster:
    def test_only_a_main_agent_that_continues_its_sub_agents_gets_one(self, following):
        assert following()._roster_wanted("lead") is True
        assert following()._roster_wanted("researcher") is False  # no sub-agents
        assert following(None)._roster_wanted("lead") is False  # the policy's defaults

    async def test_taken_in_a_turn_and_kept_until_the_memory_index_is_forgotten(self, following):
        manager = following()
        await turn(manager, "Find X.")
        ref = manager.session_ref("s", "u")

        taken = roster_in(manager, ref)
        assert "- researcher-1 (researcher, idle): Find X. — Done." in taken
        assert manager._rosters.snapshots == {ref: taken}
        await turn(manager, "Find Y.")
        assert roster_in(manager, ref) == taken
        manager._forget_memory_index(ref)
        assert "- researcher-2 (researcher, idle): Find Y. — Done." in roster_in(manager, ref)

    async def test_opening_the_conversation_takes_a_new_one(self, following):
        manager = following()
        await turn(manager, "Find X.")
        ref = manager.session_ref("s", "u")
        assert roster_in(manager, ref)

        await manager.load_session("s", user_id="u")

        assert manager._rosters.snapshots == {}

    async def test_the_release_clears_them_and_a_delete_forgets_its_own(self, following):
        manager = following()
        await turn(manager, "Find X.")
        ref = manager.session_ref("s", "u")
        other = manager.session_ref("t", "u")
        roster_in(manager, ref)
        manager._rosters.snapshots[other] = "kept"

        await manager.delete_session("s", user_id="u")
        assert manager._rosters.snapshots == {other: "kept"}
        await manager._release_resources()
        assert manager._rosters.snapshots == {}

    def test_a_manager_built_without_its_constructor_has_them(self):
        manager = _Kept.__new__(_Kept)
        assert isinstance(manager._rosters, Rosters) and manager._rosters is manager._rosters
