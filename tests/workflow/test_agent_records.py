"""A conversation's sub-agent records, kept beside it in its store.

``RecordsWriter`` (``workflow/agents/records.py``) writes each conversation's
records as they change: one write at a time per conversation, each taking
the records as they are when it starts (changes made while one is written
are written together, by the next), a failure warned of once per
conversation and never raised. A manager (``BaseWorkflowManager``) writes
through its backend's store hooks at each run event that changes the
records, loads them with the conversation (``load_agents``: once, at a
turn's start; again whenever ``load_session`` opens it), has written them
all before a turn returns, and forgets a conversation it deleted.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
from collections.abc import Callable
from typing import Any

import pytest

from agentic_cli.workflow.agents import (
    INSTANCE_IDLE,
    RUN_COMPLETED,
    RUN_INTERRUPTED,
    RUN_RUNNING,
)
from agentic_cli.workflow.agents.delegate import AgentRun, DelegationRequest, RunAnswer
from agentic_cli.workflow.agents.envelope import INTERRUPTED_ERROR
from agentic_cli.workflow import base_manager as base_manager_module
from agentic_cli.workflow.agents import records as records_module
from agentic_cli.workflow.agents.records import RecordsWriter
from agentic_cli.workflow.agents.runs import RECORDS_VERSION
from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.sessions import SessionRef
from tests.conftest import MockContext
from tests.loop_guard import no_blocking_calls

ONE = SessionRef("app", "u", "s1")
TWO = SessionRef("app", "u", "s2")


class _LogRecorder:
    """Stands in for a module's logger and records every call, with its level.

    ``structlog.testing.capture_logs()`` sees nothing once an earlier test has
    run ``configure_logging()`` (a logger is cached on first use).
    """

    def __init__(self) -> None:
        self.events: list[dict] = []

    def __getattr__(self, level: str):
        def log(event: str, **fields) -> None:
            self.events.append({"event": event, "log_level": level, **fields})

        return log


@pytest.fixture
def logged(monkeypatch) -> list[dict]:
    """Every log call of the writer (``agents/records.py``) and the manager (``base_manager.py``)."""
    recorder = _LogRecorder()
    monkeypatch.setattr(records_module, "logger", recorder)
    monkeypatch.setattr(base_manager_module, "logger", recorder)
    return recorder.events


def warned(logged: list[dict]) -> list[tuple[str, str]]:
    """The warnings logged: (event, the conversation's session id)."""
    return [(e["event"], e["session_id"]) for e in logged if e["log_level"] == "warning"]


class Store:
    """A store the writer saves to: what each save wrote, and a gate a test may close."""

    def __init__(self) -> None:
        self.saved: list[tuple[SessionRef, Any]] = []
        self.open = asyncio.Event()
        self.open.set()
        self.entered = asyncio.Event()
        self.fails: Exception | None = None

    async def save(self, conversation: SessionRef, data: Any) -> None:
        self.entered.set()
        await self.open.wait()
        if self.fails is not None:
            raise self.fails
        self.saved.append((conversation, data))


def writer_over(store: Store, records: dict[SessionRef, int]) -> RecordsWriter:
    """A writer whose records of a conversation are the number ``records`` holds for it."""
    return RecordsWriter(export=lambda conversation: records[conversation], save=store.save)


class TestRecordsWriter:
    async def test_a_change_is_written(self):
        store, records = Store(), {ONE: 1}
        writer = writer_over(store, records)

        writer.changed(ONE)
        await writer.flush(ONE)

        assert store.saved == [(ONE, 1)]

    async def test_changes_made_while_one_is_written_are_written_together_the_latest(self):
        store, records = Store(), {ONE: 1}
        writer = writer_over(store, records)
        store.open.clear()
        writer.changed(ONE)
        await store.entered.wait()

        for value in (2, 3, 4):
            records[ONE] = value
            writer.changed(ONE)
        store.open.set()
        await writer.flush(ONE)

        assert store.saved == [(ONE, 1), (ONE, 4)]

    async def test_one_conversations_write_never_waits_for_anothers(self):
        store, records = Store(), {ONE: 1, TWO: 2}
        writer = writer_over(store, records)
        store.open.clear()
        writer.changed(ONE)
        await store.entered.wait()
        other = Store()
        writer._save = lambda conversation, data: (  # TWO's goes to a store that is open
            other.save(conversation, data) if conversation == TWO else store.save(conversation, data)
        )

        writer.changed(TWO)
        await asyncio.wait_for(writer.flush(TWO), 5)

        assert (store.saved, other.saved) == ([], [(TWO, 2)])
        store.open.set()
        await writer.flush(ONE)

    async def test_a_failed_write_is_warned_of_once_per_conversation_and_never_raised(
        self, logged
    ):
        store, records = Store(), {ONE: 1, TWO: 2}
        store.fails = OSError("disk full")
        writer = writer_over(store, records)

        for conversation in (ONE, ONE, TWO):
            writer.changed(conversation)
            await writer.flush(conversation)

        assert warned(logged) == [
            ("subagent_records_not_saved", "s1"), ("subagent_records_not_saved", "s2"),
        ]
        assert [e["error"] for e in logged if e["log_level"] == "warning"][0] == "disk full"

    async def test_a_flush_with_nothing_to_write_returns_at_once(self):
        writer = writer_over(Store(), {})

        await asyncio.wait_for(writer.flush(ONE), 1)
        await asyncio.wait_for(writer.flush_all(), 1)

    async def test_flush_all_waits_for_every_conversation(self):
        store, records = Store(), {ONE: 1, TWO: 2}
        writer = writer_over(store, records)
        store.open.clear()
        writer.changed(ONE)
        writer.changed(TWO)

        waiting = asyncio.ensure_future(writer.flush_all())
        await asyncio.sleep(0)
        assert not waiting.done()
        store.open.set()
        await waiting

        assert sorted(store.saved, key=lambda saved: saved[1]) == [(ONE, 1), (TWO, 2)]

    async def test_a_discard_drops_what_waits_after_the_write_under_way(self):
        store, records = Store(), {ONE: 1}
        writer = writer_over(store, records)
        store.open.clear()
        writer.changed(ONE)
        await store.entered.wait()
        records[ONE] = 2
        writer.changed(ONE)

        discarding = asyncio.ensure_future(writer.discard(ONE))
        await asyncio.sleep(0)
        assert not discarding.done()  # the write under way ends first
        store.open.set()
        await discarding

        assert store.saved == [(ONE, 1)]

    async def test_a_cancelled_flush_leaves_the_write_to_finish(self):
        store, records = Store(), {ONE: 1}
        writer = writer_over(store, records)
        store.open.clear()
        writer.changed(ONE)
        flushing = asyncio.ensure_future(writer.flush(ONE))
        await store.entered.wait()

        flushing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await flushing
        store.open.set()
        await writer.flush(ONE)

        assert store.saved == [(ONE, 1)]

    def test_outside_an_event_loop_nothing_is_written(self):
        store = Store.__new__(Store)  # no event loop to make its events on
        store.saved = []
        writer = RecordsWriter(export=lambda conversation: 1, save=store.save)

        writer.changed(ONE)

        assert store.saved == [] and writer._writing == {}


# -- a manager: written at each run event, loaded with the conversation, forgotten on delete --


class _Kept(BaseWorkflowManager):
    """A backend that keeps each conversation's records in ``stored`` (as JSON
    reads them back), and whose turn delegates once, its brief the message."""

    def __init__(self, *args: Any, stored: dict[SessionRef, Any], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.stored = stored
        self.loads: list[SessionRef] = []
        self.load_fails: Exception | None = None
        self.save_fails: Exception | None = None
        self.save_gate: asyncio.Event | None = None
        self.hold: asyncio.Event | None = None

    @property
    def backend_type(self) -> str:
        return "kept"

    def _get_state_tools(self) -> list:
        return []

    async def _do_initialize(self) -> None:
        return None

    async def session_exists(self, session_id: str, *, user_id: str | None = None) -> bool:
        return True

    async def delete_session(self, session_id: str, *, user_id: str | None = None) -> bool:
        ref = self.session_ref(session_id, user_id)
        async with self._deleting(ref):
            found = self.stored.pop(ref, None) is not None
        return found

    async def _load_agent_records(self, conversation: SessionRef) -> Any:
        self.loads.append(conversation)
        if self.load_fails is not None:
            raise self.load_fails
        return self.stored.get(conversation)

    async def _save_agent_records(self, conversation: SessionRef, data: Any) -> None:
        if self.save_gate is not None:
            await self.save_gate.wait()
        if self.save_fails is not None:
            raise self.save_fails
        self.stored[conversation] = json.loads(json.dumps(data))

    async def process(self, message, user_id, session_id=None):
        async with self._turn(session_id, user_id) as ref:
            delegated = await self._delegations.delegate(DelegationRequest(
                caller=None, caller_agent="lead", call_id="c1",
                arguments={"agent_type": "researcher", "title": message, "objective": message},
            ))
            yield WorkflowEvent.text(json.dumps(delegated.result), ref.session_id)

    async def _run_agent(self, run: AgentRun) -> RunAnswer:
        if self.hold is not None:
            await self.hold.wait()
        return RunAnswer(text="Done.")


TEAM = [
    AgentConfig(name="lead", prompt="p", sub_agents=["researcher"]),
    AgentConfig(name="researcher", prompt="p", description="Finds."),
]


@pytest.fixture
def context():
    with MockContext(google_api_key="test-key") as made:
        yield made


@pytest.fixture
async def kept(context):
    """``kept(stored)``: a manager whose store is ``stored``; each cleaned up after the test."""
    made: list[_Kept] = []

    def make(stored: dict[SessionRef, Any]) -> _Kept:
        manager = _Kept(TEAM, settings=context.settings, stored=stored)
        made.append(manager)
        return manager

    yield make
    for manager in made:
        await asyncio.wait_for(manager.cleanup(), 10)


async def turn(manager: _Kept, text: str = "Find X.") -> list[WorkflowEvent]:
    return [event async for event in manager.process(text, "u", "s")]


def runs_in(stored: dict[SessionRef, Any], ref: SessionRef) -> list[tuple[str, str]]:
    """The runs the store keeps of ``ref``: (run, status), in start order."""
    return [(run["run"], run["status"]) for run in stored[ref]["runs"]]


def runs_of(manager: _Kept, ref: SessionRef) -> list[tuple[str, str]]:
    return [(r.ref.run, r.status) for r in manager.agent_runs.records(ref)]


async def until(condition: Callable[[], bool]) -> None:
    for _ in range(1000):
        if condition():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("never")


class TestAManagersRecords:
    async def test_each_change_is_written_as_it_happens(self, kept):
        stored: dict[SessionRef, Any] = {}
        manager = kept(stored)
        manager.hold = asyncio.Event()
        ref = manager.session_ref("s", "u")

        running = asyncio.ensure_future(turn(manager))
        try:
            await until(lambda: ref in stored)
            while_it_works = runs_in(stored, ref)
        finally:
            manager.hold.set()
            await asyncio.wait_for(running, 5)

        assert while_it_works == [("researcher-1#1", RUN_RUNNING)]
        # The turn returned once its end was written.
        assert runs_in(stored, ref) == [("researcher-1#1", RUN_COMPLETED)]
        assert stored[ref]["instances"][0]["status"] == INSTANCE_IDLE

    async def test_a_new_manager_loads_them_with_the_conversation_once(self, kept):
        stored: dict[SessionRef, Any] = {}
        first = kept(stored)
        await turn(first)
        await first.cleanup()
        second = kept(stored)
        ref = second.session_ref("s", "u")

        await second.initialize_services()
        assert await second.load_session("s", user_id="u") is True
        loaded = runs_of(second, ref)
        await turn(second, "Find Y.")

        assert loaded == [("researcher-1#1", RUN_COMPLETED)]
        assert second.loads == [ref]  # not again at the turn's start
        assert runs_of(second, ref) == [
            ("researcher-1#1", RUN_COMPLETED), ("researcher-2#1", RUN_COMPLETED),
        ]
        assert runs_in(stored, ref) == runs_of(second, ref)

    async def test_a_turns_start_loads_them_when_nothing_did(self, kept):
        stored: dict[SessionRef, Any] = {}
        await turn(kept(stored))
        second = kept(stored)
        ref = second.session_ref("s", "u")

        with no_blocking_calls():
            await turn(second, "Find Y.")

        assert second.loads == [ref]
        assert [run for run, _ in runs_of(second, ref)] == ["researcher-1#1", "researcher-2#1"]

    async def test_a_load_before_the_store_is_open_waits_for_the_turn(self, kept):
        stored: dict[SessionRef, Any] = {}
        await turn(kept(stored))
        second = kept(stored)

        await second.load_agents(second.session_ref("s", "u"))

        assert second.loads == []
        await turn(second)
        assert second.loads == [second.session_ref("s", "u")]

    async def test_a_run_found_working_reads_interrupted_and_is_written_so(self, kept):
        stored: dict[SessionRef, Any] = {}
        first = kept(stored)
        first.hold = asyncio.Event()
        ref = first.session_ref("s", "u")
        running = asyncio.ensure_future(turn(first))
        try:
            await until(lambda: ref in stored)
            # The process is killed: the store keeps what was written before.
            kept_at_the_kill = copy.deepcopy(stored)
        finally:
            running.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await running

        second = kept(kept_at_the_kill)
        await second.initialize_services()
        await second.load_session("s", user_id="u")
        record = second.agent_runs.record(ref, "researcher-1#1")
        await second._agent_records_written(ref)

        assert (record.status, record.error) == (RUN_INTERRUPTED, INTERRUPTED_ERROR)
        assert second.agent_runs.instance(ref, "researcher-1").status == INSTANCE_IDLE
        assert runs_in(kept_at_the_kill, ref) == [("researcher-1#1", RUN_INTERRUPTED)]

    async def test_records_of_another_version_are_ignored_with_one_warning(self, kept, logged):
        manager = kept({})
        ref = manager.session_ref("s", "u")
        manager.stored[ref] = {"v": RECORDS_VERSION + 1, "runs": "?"}

        await turn(manager)
        await turn(manager)

        assert warned(logged) == [("subagent_records_ignored", "s")]
        assert runs_of(manager, ref) == [
            ("researcher-1#1", RUN_COMPLETED), ("researcher-2#1", RUN_COMPLETED),
        ]
        assert manager.stored[ref]["v"] == RECORDS_VERSION

    async def test_a_store_that_cannot_be_read_stops_the_turn_and_is_read_again(
        self, kept, logged
    ):
        stored: dict[SessionRef, Any] = {}
        await turn(kept(stored))
        manager = kept(stored)
        ref = manager.session_ref("s", "u")
        await manager.initialize_services()
        manager.load_fails = OSError("disk unreadable")

        opened = await manager.load_session("s", user_id="u")
        with pytest.raises(OSError):
            await turn(manager)
        manager.load_fails = None
        await turn(manager, "Find Y.")

        assert opened is True
        assert warned(logged) == [("subagent_records_unread", "s")]
        assert manager.loads == [ref, ref, ref]
        # Nothing was written over them meanwhile.
        assert [run for run, _ in runs_in(stored, ref)] == ["researcher-1#1", "researcher-2#1"]

    async def test_a_failing_write_is_warned_of_once_and_every_turn_completes(
        self, kept, logged
    ):
        manager = kept({})
        manager.save_fails = OSError("disk full")

        answers = [json.loads((await turn(manager))[-1].content) for _ in range(2)]

        assert [answer["status"] for answer in answers] == [RUN_COMPLETED, RUN_COMPLETED]
        assert warned(logged) == [("subagent_records_not_saved", "s")]

    async def test_a_release_waits_for_the_records_being_written(self, kept):
        stored: dict[SessionRef, Any] = {}
        manager = kept(stored)
        await turn(manager)
        ref = manager.session_ref("s", "u")
        manager.save_gate = asyncio.Event()
        manager.agent_runs.records(ref)[0].title = "Renamed."
        manager._agent_records_changed(ref)

        cleaning = asyncio.ensure_future(manager.cleanup())
        await asyncio.sleep(0.01)
        waited = not cleaning.done()
        manager.save_gate.set()
        await cleaning

        assert waited
        assert stored[ref]["runs"][0]["title"] == "Renamed."

    async def test_a_backend_without_a_store_keeps_them_in_memory(self, context):
        class Unkept(_Kept):
            _load_agent_records = BaseWorkflowManager._load_agent_records
            _save_agent_records = BaseWorkflowManager._save_agent_records

        manager = Unkept(TEAM, settings=context.settings, stored={})
        try:
            await turn(manager)
            assert await BaseWorkflowManager._load_agent_records(manager, ONE) is None
            assert runs_of(manager, manager.session_ref("s", "u")) == [
                ("researcher-1#1", RUN_COMPLETED),
            ]
        finally:
            await manager.cleanup()

    async def test_opening_it_again_reads_what_another_manager_wrote_since(self, kept):
        """A conversation another process had in between (it holds the
        conversation only while it has it open): opening it again reads
        the store, after this manager's own writes."""
        stored: dict[SessionRef, Any] = {}
        one, two = kept(stored), kept(stored)
        ref = one.session_ref("s", "u")
        await turn(one)
        await two.initialize_services()
        await two.load_session("s", user_id="u")
        await turn(two, "Find Y.")

        await one.load_session("s", user_id="u")
        await turn(one, "Find Z.")

        assert [run for run, _ in runs_of(one, ref)] == [
            "researcher-1#1", "researcher-2#1", "researcher-3#1",
        ]
        assert runs_in(stored, ref) == runs_of(one, ref)


class TestADeletedConversation:
    """Deleting a conversation writes nothing of it after the delete, and the
    manager forgets it: its records, the calls it recorded, its usage, its
    numbering; a turn of the same id starts as a new conversation's."""

    async def test_it_is_forgotten_and_its_id_starts_again(self, kept):
        stored: dict[SessionRef, Any] = {}
        manager = kept(stored)
        ref = manager.session_ref("s", "u")
        await turn(manager)
        await turn(manager)

        assert await manager.delete_session("s", user_id="u") is True

        assert manager.agent_runs.records(ref) == [] and manager.agent_runs.instances(ref) == []
        assert manager.usage_of(ref).turns == 0
        await turn(manager)
        assert runs_of(manager, ref) == [("researcher-1#1", RUN_COMPLETED)]
        assert manager.loads == [ref, ref]

    async def test_a_write_under_way_ends_before_the_delete_and_none_follows(self, kept):
        stored: dict[SessionRef, Any] = {}
        manager = kept(stored)
        ref = manager.session_ref("s", "u")
        await turn(manager)
        manager.save_gate = asyncio.Event()
        manager._agent_records_changed(ref)
        await asyncio.sleep(0)
        manager._agent_records_changed(ref)  # waits behind the one under way

        deleting = asyncio.ensure_future(manager.delete_session("s", user_id="u"))
        await asyncio.sleep(0.01)
        waited = not deleting.done()
        manager.save_gate.set()
        await deleting
        await asyncio.sleep(0.01)

        assert waited
        assert ref not in stored
