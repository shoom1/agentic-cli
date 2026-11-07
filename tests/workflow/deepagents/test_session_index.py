"""The session index on Deep Agents (``workflow/deepagents/sessions.py``).

A conversation's row (its title, its turns, when it last changed) is kept in
a LangGraph store beside the checkpointer, written as each turn starts (the
row the turn will leave, so a turn the process does not survive still lists
its conversation) and as it ends, however it ends, and after ``/compact``. A listing reads those rows, one
small value per conversation, instead of every checkpoint of the user, each
of which holds the whole conversation; ``session_info`` reads one. A
database written before the index is scanned once per app and user, then
never again. Sub-agents store no checkpoints.

What a listing reads is counted by wrapping the saver (``CountingSaver``):
after the backfill it reads none. Every row is compared with the row of the
conversation's latest root checkpoint (``_row``), which is what a full scan
gives.
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

pytest.importorskip("deepagents")

from langchain_core.messages import HumanMessage  # noqa: E402

from agentic_cli.tools.jobs import JobState  # noqa: E402
from agentic_cli.workflow.checkpointers import open_checkpointer  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.deepagents import manager as manager_module  # noqa: E402
from agentic_cli.workflow.deepagents import sessions as sessions_module  # noqa: E402
from agentic_cli.workflow.deepagents.agent_builder import build_agent_tree  # noqa: E402
from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager  # noqa: E402
from agentic_cli.workflow.deepagents.sessions import (  # noqa: E402
    CheckpointSessions,
    _row,
    run_config,
    thread_id,
)
from agentic_cli.workflow.events import EventType  # noqa: E402
from agentic_cli.workflow.sessions import SessionInfo, SessionRef  # noqa: E402
from tests.conftest import MockContext  # noqa: E402
from tests.workflow.compacting import compact_now  # noqa: E402
from tests.workflow.deepagents.fakes import ScriptedChat, ScriptedModels, ai, task  # noqa: E402
from tests.workflow.deepagents.test_compact import THREE, answers, solo  # noqa: E402
from tests.workflow.deepagents.test_context import (  # noqa: E402,F401
    GatedChat,
    hermetic,
    small,
)
from tests.workflow.deepagents.test_context import turn as long_turn  # noqa: E402

SOLO = [AgentConfig(name="solo", prompt="p")]


class CountingSaver:
    """A saver that records each read of its checkpoints, and does it."""

    def __init__(self, saver) -> None:
        self.saver = saver
        self.reads: list[str] = []

    def alist(self, config, **kwargs):
        self.reads.append("alist")
        return self.saver.alist(config, **kwargs)

    async def aget_tuple(self, config):
        self.reads.append("aget_tuple")
        return await self.saver.aget_tuple(config)

    def __getattr__(self, name: str):
        return getattr(self.saver, name)


async def unread(manager, call):
    """What ``call()`` returns, asserting it read no checkpoint of ``manager``'s."""
    sessions = manager._sessions
    counting = CountingSaver(sessions._saver)
    sessions._saver = counting
    try:
        value = await call()
    finally:
        sessions._saver = counting.saver
    assert counting.reads == [], f"it read checkpoints: {counting.reads}"
    return value


async def listed(manager, user: str = "u", **kwargs) -> list[SessionInfo]:
    """``manager``'s list, asserting it read no checkpoint."""
    return await unread(manager, lambda: manager.list_sessions(user_id=user, **kwargs))


async def truth(manager, session: str = "s", user: str = "u") -> SessionInfo:
    """The conversation's row as a full scan gives it: its latest root checkpoint's."""
    ref = manager.session_ref(session, user)
    item = await manager._checkpointer.aget_tuple(
        {"configurable": {"thread_id": thread_id(ref), "checkpoint_ns": ""}}
    )
    return _row(session, item)


async def say(saver, session: SessionRef, text: str) -> None:
    """One turn in ``session`` written straight to ``saver``, as a version
    without the index wrote it: no row."""
    root = build_agent_tree(
        SOLO, chat_models={"solo": ScriptedChat(script=[ai("Hi.")])}, resolved={},
        tools={}, prompts={"solo": "p"}, settings=None, checkpointer=saver,
    )
    await root.ainvoke({"messages": [HumanMessage(text)]}, run_config(session))


def his(count: int) -> list:
    """``count`` answers (one message each: LangChain stamps an id on each)."""
    return [ai(f"Hi {n}.") for n in range(count)]


def scripted(monkeypatch, **scripts) -> ScriptedModels:
    models = ScriptedModels(**scripts)
    monkeypatch.setattr(manager_module, "build_chat_model", models)
    return models


def _manager(settings, configs=SOLO, **kwargs) -> DeepAgentsWorkflowManager:
    return DeepAgentsWorkflowManager(
        agent_configs=configs, settings=settings, model="gemini-2.5-flash", **kwargs
    )


@pytest.fixture
async def managers():
    """Clean up every manager a test hands it."""
    made: list[DeepAgentsWorkflowManager] = []

    def keep(manager: DeepAgentsWorkflowManager) -> DeepAgentsWorkflowManager:
        made.append(manager)
        return manager

    yield keep
    for manager in made:
        await manager.cleanup()


@pytest.fixture(params=["memory", "sqlite"])
def store_kind(request, hermetic):
    """Settings on the in-memory and the sqlite session store."""
    with MockContext(google_api_key="test-key", session_store=request.param) as context:
        yield context.settings


async def turn(manager, text: str = "Hello.", *, user: str = "u", session: str = "s") -> list:
    return [event async for event in manager.process(text, user, session_id=session)]


def _record(**fields) -> SimpleNamespace:
    """A finished job of the conversation ``("u", "s")``."""
    return SimpleNamespace(**{
        "job_id": "j1", "tool": "build", "name": "build", "call_id": "c1",
        "call_name": "build", "session_id": "s", "user_id": "u",
        "state": JobState.SUCCEEDED, "exit_code": 0, "error": None, **fields,
    })


# --- listing ----------------------------------------------------------------------------


class TestListing:
    @pytest.mark.parametrize("turns", [1, 10])
    async def test_after_the_first_listing_none_reads_a_checkpoint(
        self, store_kind, monkeypatch, managers, turns
    ):
        """However long the conversations, a listing reads one row each."""
        scripted(monkeypatch, solo=his(3 * turns))
        manager = managers(_manager(store_kind))
        for session in ["s1", "s2", "s3"]:
            for n in range(turns):
                await turn(manager, f"{session}, turn {n}.", session=session)

        first = await manager.list_sessions(user_id="u")
        every = await listed(manager)
        newest = await listed(manager, limit=1)

        expected = [await truth(manager, session) for session in ["s3", "s2", "s1"]]
        assert first == every == expected
        assert newest == expected[:1]
        assert [(row.title, row.turns) for row in every] == [
            (f"{session}, turn 0.", turns) for session in ["s3", "s2", "s1"]
        ]

    async def test_on_the_memory_kind_no_listing_reads_a_checkpoint(
        self, hermetic, monkeypatch, managers
    ):
        """An in-memory saver is born with its index: there is nothing to scan."""
        scripted(monkeypatch, solo=[ai("Hi.")])
        with MockContext(google_api_key="test-key", session_store="memory") as context:
            manager = managers(_manager(context.settings))
            await manager.initialize_services()
            assert await listed(manager) == []
            await turn(manager)

            assert await listed(manager) == [await truth(manager)]

    async def test_more_rows_than_one_search_returns(self, store_kind, monkeypatch, managers):
        scripted(monkeypatch, solo=his(5))
        monkeypatch.setattr(sessions_module, "_SEARCH_LIMIT", 2)
        manager = managers(_manager(store_kind))
        for n in range(5):
            await turn(manager, f"Number {n}.", session=f"s{n}")
        await manager.list_sessions(user_id="u")

        assert [r.title for r in await listed(manager)] == [
            f"Number {n}." for n in reversed(range(5))
        ]

    async def test_other_users_and_apps_are_left_out(self, store_kind, monkeypatch, managers):
        """The stores match a namespace by its text's prefix (case-blind in
        sqlite): ``u`` must not list ``U``'s or ``u.x``'s rows."""
        scripted(monkeypatch, solo=his(5))
        manager = managers(_manager(store_kind))
        other_app = managers(_manager(store_kind, app_name="other"))
        for user in ["u", "U", "u.x", "u%"]:
            await turn(manager, f"From {user}.", user=user)
        await turn(other_app, "From the other app.")
        for user in ["u", "U", "u.x", "u%"]:
            await manager.list_sessions(user_id=user)
        await other_app.list_sessions(user_id="u")

        assert [(r.session_id, r.title) for r in await listed(manager)] == [("s", "From u.")]
        assert [r.title for r in await listed(manager, user="U")] == ["From U."]
        assert [r.title for r in await listed(manager, user="u.x")] == ["From u.x."]
        assert [r.title for r in await listed(manager, user="u%")] == ["From u%."]
        assert [r.title for r in await listed(other_app)] == ["From the other app."]


# --- the row after every kind of turn ---------------------------------------------------


@pytest.fixture
def ctx(hermetic):
    with MockContext(google_api_key="test-key", session_store="memory") as context:
        yield context


class TestTheRowAfterEveryTurn:
    """On the memory kind, which never scans: only what the turns wrote is listed."""

    async def test_a_turn(self, ctx, monkeypatch, managers):
        scripted(monkeypatch, solo=[ai("Hi."), ai("Again.")])
        manager = managers(_manager(ctx.settings))
        await turn(manager, "  Plan the\ntrip. ")
        await turn(manager, "And the budget?")

        (row,) = await listed(manager)

        assert row == await truth(manager)
        assert (row.title, row.turns) == ("Plan the trip.", 2)

    async def test_a_turn_that_failed(self, ctx, monkeypatch, managers):
        scripted(monkeypatch, solo=[ai("Hi."), RuntimeError("provider said 500")])
        manager = managers(_manager(ctx.settings))
        await turn(manager, "Hello.")
        (before,) = await listed(manager)

        failed = await turn(manager, "Still there?")

        assert [e.type for e in failed if e.type is EventType.ERROR] == [EventType.ERROR]
        (row,) = await listed(manager)
        assert row == await truth(manager)
        assert row.turns == 2 and row.updated > before.updated

    async def test_a_turn_cancelled_mid_model_call(self, ctx, monkeypatch, managers):
        chat = GatedChat(label="solo", script=[ai("Hi."), ai("Late.")], gate_at=2)
        models = ScriptedModels()
        models.chats["solo"] = chat
        monkeypatch.setattr(manager_module, "build_chat_model", models)
        manager = managers(_manager(ctx.settings))
        await turn(manager, "Hello.")
        (before,) = await listed(manager)

        consumer = asyncio.create_task(turn(manager, "Second question."))
        await asyncio.wait_for(chat._started.wait(), 5)
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(consumer, 5)
        chat._release.set()

        (row,) = await listed(manager)
        assert row == await truth(manager)
        assert row.turns == 2 and row.updated > before.updated

    async def test_a_cancel_while_the_row_is_written_lets_it_finish(
        self, ctx, monkeypatch, managers
    ):
        """Ctrl+C pressed as the turn ends stops the turn, not its row."""
        scripted(monkeypatch, solo=his(1))
        manager = managers(_manager(ctx.settings))
        await manager.initialize_services()

        # Write 1 as the turn starts, write 2 (held) as it ends.
        slow = await stopped_while_writing(manager, turn(manager, "Hello."), hold=2)
        await slow.settled(2)

        (row,) = await listed(manager)
        assert row == await truth(manager)
        assert (row.title, row.turns) == ("Hello.", 1)

    async def test_a_turn_stopped_as_it_starts_leaves_the_checkpoints_row(
        self, ctx, monkeypatch, managers
    ):
        """The row written as the turn starts counts an input the stopped
        turn never stored; the end's row, from the checkpoint, is written
        after it, however late it lands."""
        scripted(monkeypatch, solo=his(1))
        manager = managers(_manager(ctx.settings))
        await turn(manager, "Hello.")

        # Write 1 (held) as the second turn starts, write 2 as it ends.
        slow = await stopped_while_writing(manager, turn(manager, "Again?"), hold=1)
        await slow.settled(2)

        assert slow.done == ["put", "put"]
        (row,) = await listed(manager)
        assert row == await truth(manager)
        assert (row.title, row.turns) == ("Hello.", 1)

    async def test_a_first_turn_stopped_as_it_starts_lists_nothing(
        self, ctx, monkeypatch, managers
    ):
        scripted(monkeypatch, solo=[])
        manager = managers(_manager(ctx.settings))
        await manager.initialize_services()

        slow = await stopped_while_writing(manager, turn(manager, "Hello."), hold=1)
        await slow.settled(2)

        assert slow.done == ["put", "delete"]
        assert await listed(manager) == []
        assert await manager.session_exists("s", user_id="u") is False

    async def test_a_job_told_to_the_conversation(self, ctx, monkeypatch, managers):
        scripted(monkeypatch, solo=[ai("Building."), ai("Built.")])
        manager = managers(_manager(ctx.settings))
        await turn(manager, "Build it.")
        (before,) = await listed(manager)

        async for _ in manager.resume_with_job_result(_record(), result="ok"):
            pass

        (row,) = await listed(manager)
        assert row == await truth(manager)
        assert (row.title, row.turns) == ("Build it.", 1)
        assert row.updated > before.updated

    async def test_a_compact(self, small, monkeypatch, managers):
        manager, _ = solo(monkeypatch, small, [*answers(3), ai("A summary.")])
        managers(manager)
        for text in THREE:
            await long_turn(manager, text)
        (before,) = await listed(manager)

        result, _ = await compact_now(manager)

        assert result is not None
        (row,) = await listed(manager)
        assert row == await truth(manager)
        assert row.turns == 3 and row.updated > before.updated


# --- the row while a turn runs ---------------------------------------------------------


async def listed_in_flight(manager, chat: GatedChat, run) -> list[SessionInfo]:
    """The list while ``run`` (a turn) waits in its held model call; then
    the turn is stopped."""
    consumer = asyncio.create_task(run)
    try:
        await asyncio.wait_for(chat._started.wait(), 5)
        return await listed(manager)
    finally:
        consumer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await consumer
        chat._release.set()


def gated(monkeypatch, script: list, gate_at: int) -> GatedChat:
    chat = GatedChat(label="solo", script=script, gate_at=gate_at)
    models = ScriptedModels()
    models.chats["solo"] = chat
    monkeypatch.setattr(manager_module, "build_chat_model", models)
    return chat


async def drain(events) -> None:
    async for _ in events:
        pass


class TestTheRowWhileATurnRuns:
    """Written as the turn starts too: a process killed mid-turn (SIGKILL, the
    terminal closed, out of memory) runs no end of turn, and its
    conversation must still be listed, and counted."""

    async def test_a_new_conversation_is_listed_as_its_first_turn_starts(
        self, store_kind, monkeypatch, managers
    ):
        chat = gated(monkeypatch, [ai("Late.")], gate_at=1)
        manager = managers(_manager(store_kind))
        await manager.initialize_services()
        await manager.list_sessions(user_id="u")
        started = datetime.now(timezone.utc)

        rows = await listed_in_flight(
            manager, chat, drain(manager.process("  Brand\nnew. ", "u", session_id="new"))
        )

        assert [(r.session_id, r.title, r.turns) for r in rows] == [("new", "Brand new.", 1)]
        assert rows[0].updated >= started

    async def test_a_later_turn_is_counted_as_it_starts(
        self, store_kind, monkeypatch, managers
    ):
        chat = gated(monkeypatch, [ai("Hi."), ai("Other."), ai("Late.")], gate_at=3)
        manager = managers(_manager(store_kind))
        await turn(manager, "Hello.")
        await turn(manager, "Elsewhere.", session="other")
        await manager.list_sessions(user_id="u")
        (_, before) = await listed(manager)

        rows = await listed_in_flight(manager, chat, turn(manager, "Second question."))

        assert [(r.session_id, r.title, r.turns) for r in rows] == [
            ("s", "Hello.", 2), ("other", "Elsewhere.", 1)
        ]
        assert rows[0].updated > before.updated
        # The turn's end (here its cancel) writes the row from the checkpoint.
        assert (await listed(manager))[0] == await truth(manager)

    async def test_a_job_told_to_the_conversation_is_no_turn_as_it_starts(
        self, ctx, monkeypatch, managers
    ):
        chat = gated(monkeypatch, [ai("Building."), ai("Late.")], gate_at=2)
        manager = managers(_manager(ctx.settings))
        await turn(manager, "Build it.")
        (before,) = await listed(manager)

        (row,) = await listed_in_flight(
            manager, chat, drain(manager.resume_with_job_result(_record(), result="ok"))
        )

        assert (row.title, row.turns) == ("Build it.", 1)
        assert row.updated > before.updated


# --- info and delete --------------------------------------------------------------------


class TestInfo:
    async def test_it_is_the_row_and_falls_back_to_the_checkpoint(
        self, ctx, monkeypatch, managers
    ):
        scripted(monkeypatch, solo=[ai("Hi.")])
        manager = managers(_manager(ctx.settings))
        await turn(manager, "Hello there.")

        info = await unread(manager, lambda: manager.session_info("s", user_id="u"))
        assert info == await truth(manager)

        # A conversation the index does not hold is read from its checkpoint.
        await say(manager._checkpointer, SessionRef(manager.app_name, "u", "older"), "Old.")
        older = await manager.session_info("older", user_id="u")
        assert (older.session_id, older.title, older.turns) == ("older", "Old.", 1)
        assert await manager.session_info("nope", user_id="u") is None


class TestDelete:
    async def test_it_removes_the_row(self, store_kind, monkeypatch, managers):
        scripted(monkeypatch, solo=his(2))
        manager = managers(_manager(store_kind))
        await turn(manager, session="s1")
        await turn(manager, session="s2")
        await manager.list_sessions(user_id="u")

        assert await manager.delete_session("s1", user_id="u") is True

        assert [r.session_id for r in await listed(manager)] == ["s2"]
        assert await manager.session_info("s1", user_id="u") is None

    async def test_a_row_whose_thread_is_gone_is_deleted_too(
        self, ctx, monkeypatch, managers
    ):
        scripted(monkeypatch, solo=[ai("Hi.")])
        manager = managers(_manager(ctx.settings))
        await turn(manager)
        await manager._checkpointer.adelete_thread(thread_id(manager.session_ref("s", "u")))

        assert await manager.delete_session("s", user_id="u") is True

        assert await listed(manager) == []
        assert await manager.delete_session("s", user_id="u") is False


# --- databases written before the index, and sharing one --------------------------------


class TestBackfill:
    async def test_a_database_written_before_the_index_is_scanned_once(
        self, hermetic, monkeypatch, managers
    ):
        """Conversations an older version wrote (no rows) are listed, from one
        scan; the mark it leaves in the file spares every later listing,
        another process's included."""
        db = hermetic / "old.db"
        saver, connection = await open_checkpointer("sqlite", sqlite_path=db)
        try:
            await say(saver, SessionRef("app", "u", "s1"), "First.")
            await say(saver, SessionRef("app", "u", "s2"), "Second.")
            await say(saver, SessionRef("app", "u", "s1"), "First, again.")
        finally:
            await connection.__aexit__(None, None, None)
        scripted(monkeypatch, solo=[])
        with MockContext(google_api_key="test-key", sqlite_uri=f"sqlite:///{db}") as context:
            one = managers(_manager(context.settings, app_name="app"))
            two = managers(_manager(context.settings, app_name="app"))
            await one.initialize_services()
            await two.initialize_services()

            first = await one.list_sessions(user_id="u")
            again = await listed(one)
            elsewhere = await listed(two)

            expected = [await truth(one, "s1"), await truth(one, "s2")]
            assert first == again == elsewhere == expected
            assert [(r.title, r.turns) for r in first] == [("First.", 2), ("Second.", 1)]

    @pytest.mark.parametrize("kind", ["memory", "sqlite"])
    async def test_it_writes_older_rows_over_and_never_a_newer_one(self, tmp_path, kind):
        """A row an earlier turn wrote is replaced by the scan's newer one; a
        row another process wrote while the scan ran is kept."""
        from agentic_cli.workflow.checkpointers import open_store

        db = tmp_path / "c.db"
        saver, saver_connection = await open_checkpointer(kind, sqlite_path=db)
        store, store_connection = await open_store(kind, sqlite_path=db)
        try:
            stale, raced = SessionRef("app", "u", "stale"), SessionRef("app", "u", "raced")
            ours = CheckpointSessions(saver, store)
            theirs = CheckpointSessions(saver, store)
            # Indexed after one turn, then a turn an older version ran.
            await say(saver, stale, "Stale one.")
            await ours.record(stale)
            await say(saver, stale, "Unindexed turn.")
            await say(saver, raced, "Raced one.")

            async def their_turn_ends() -> None:
                await say(saver, raced, "Their newer turn.")
                await theirs.record(raced)

            ours._saver = TurnDuringScan(saver, their_turn_ends)
            rows = {row.session_id: row for row in await ours.list("app", "u")}

            truth_of = {
                ref.session_id: _row(ref.session_id, await saver.aget_tuple(
                    {"configurable": {"thread_id": thread_id(ref), "checkpoint_ns": ""}}
                ))
                for ref in (stale, raced)
            }
            assert rows == truth_of
            assert (rows["stale"].turns, rows["raced"].turns) == (2, 2)
        finally:
            for connection in (store_connection, saver_connection):
                if connection is not None:
                    await connection.__aexit__(None, None, None)


class SlowStore:
    """A store whose ``hold``-th write waits for ``release``, as a slow disk
    would; ``done`` names each write and delete as it completes."""

    def __init__(self, store, hold: int) -> None:
        self.store = store
        self.hold = hold
        self.puts = 0
        self.writing = asyncio.Event()
        self.release = asyncio.Event()
        self.done: list[str] = []

    async def aput(self, *args, **kwargs):
        self.puts += 1
        if self.puts == self.hold:
            self.writing.set()
            await self.release.wait()
        await self.store.aput(*args, **kwargs)
        self.done.append("put")

    async def adelete(self, *args, **kwargs):
        await self.store.adelete(*args, **kwargs)
        self.done.append("delete")

    async def settled(self, count: int) -> None:
        """Wait until ``count`` writes and deletes completed."""
        async def wait() -> None:
            while len(self.done) < count:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait(), 5)

    def __getattr__(self, name: str):
        return getattr(self.store, name)


async def stopped_while_writing(manager, run, hold: int) -> SlowStore:
    """``run`` (a turn) cancelled while the ``hold``-th write of the index
    waits; the write then lands (a cancelled turn waits for its writes)."""
    slow = SlowStore(manager._sessions._store, hold)
    manager._sessions._store = slow
    consumer = asyncio.create_task(run)
    await asyncio.wait_for(slow.writing.wait(), 5)
    consumer.cancel()
    await asyncio.sleep(0.05)
    slow.release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(consumer, 5)
    return slow


class TurnDuringScan:
    """A saver whose full scan lets another process's turn end as it finishes."""

    def __init__(self, saver, during) -> None:
        self.saver = saver
        self.during = during

    async def alist(self, config, **kwargs):
        async for item in self.saver.alist(config, **kwargs):
            yield item
        if config is None and self.during is not None:
            during, self.during = self.during, None
            await during()

    def __getattr__(self, name: str):
        return getattr(self.saver, name)


class TestOneFileTwoManagers:
    async def test_each_lists_the_others_rows(self, hermetic, monkeypatch, managers):
        scripted(monkeypatch, solo=his(3))
        db = hermetic / "shared.db"
        with MockContext(google_api_key="test-key", sqlite_uri=f"sqlite:///{db}") as context:
            one = managers(_manager(context.settings, app_name="app"))
            two = managers(_manager(context.settings, app_name="app"))
            await turn(one, "From one.", session="s1")
            await turn(two, "From two.", session="s2")
            await one.list_sessions(user_id="u")

            await turn(one, "One again.", session="s1")

            for manager in (one, two):
                assert [(r.session_id, r.turns) for r in await listed(manager)] == [
                    ("s1", 2), ("s2", 1)
                ]
            assert await listed(two) == [await truth(one, "s1"), await truth(one, "s2")]

        with sqlite3.connect(db) as connection:
            tables = {row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )}
        assert {"checkpoints", "writes", "store", "store_migrations"} <= tables


# --- sub-agents -------------------------------------------------------------------------


class TestSubAgentsStoreNoCheckpoints:
    @pytest.mark.parametrize("context", [None, "inherit"])
    async def test_after_a_delegated_turn_only_the_root_namespace_is_stored(
        self, ctx, monkeypatch, managers, context
    ):
        models = scripted(
            monkeypatch,
            lead=[ai(calls=[task("helper", "Do it.")]), ai("Done, says the helper.")],
            helper=[ai("Worked.")],
        )
        manager = managers(_manager(ctx.settings, [
            AgentConfig(name="lead", prompt="Lead.", sub_agents=["helper"]),
            AgentConfig(name="helper", prompt="Help.", description="Helps.", context=context),
        ]))

        events = await turn(manager, "Delegate this.")

        assert [e.content for e in events if e.type is EventType.TEXT][-1] == (
            "Done, says the helper."
        )
        namespaces = {
            item.config["configurable"]["checkpoint_ns"]
            async for item in manager._checkpointer.alist(None)
        }
        assert namespaces == {""}
        # A fork still reads its parent's conversation (from the parent's state).
        sent = [m.text for m in models.chats["helper"].conversation(0)]
        assert ("Delegate this." in sent) is (context == "inherit")
        (row,) = await listed(manager)
        assert row == await truth(manager)
        assert (row.title, row.turns) == ("Delegate this.", 1)
