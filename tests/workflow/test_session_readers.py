"""A backend's stored conversations, read without building its manager.

A conversation belongs to one backend for its whole life, and each backend
keeps its own store: ADK its session service on ``session_db_url``, Deep
Agents a checkpointer and the session index beside it. To list, find or
delete a conversation of a backend that is not running, the CLI opens that
backend's ``SessionReader`` (``workflow/session_readers.py``) from the
settings and the run's store: it answers what the manager's own session
hooks answer, each row naming its backend, and builds no agents, resolves
no model, builds no service and opens nothing but the store.

A store with nothing in it yet is not made by reading it; the memory store
has no reader (its conversations live in the manager that made them), nor
has LangGraph (deprecated: its manager reads its own); a backend whose
extra is not installed has none either, and asking for it says how to
install it.

Each backend's conversations are stored by its real manager (ADK on Claude
through ``FakeAPI``, Deep Agents on scripted models), on sqlite.
"""

from __future__ import annotations

import asyncio
import dataclasses
import sqlite3
import sys
import threading

import pytest

pytest.importorskip("google.adk")
pytest.importorskip("deepagents")

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.workflow import session_readers  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.base_manager import BaseWorkflowManager  # noqa: E402
from agentic_cli.workflow.checkpointers import deepagents_sqlite_path  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.deepagents import manager as deepagents_manager  # noqa: E402
from agentic_cli.workflow.deepagents import sessions as deepagents_sessions  # noqa: E402
from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager  # noqa: E402
from agentic_cli.workflow.session_readers import open_session_reader  # noqa: E402
from agentic_cli.workflow.sessions import SessionInfo  # noqa: E402
from tests.workflow.deepagents.fakes import ScriptedModels, ai  # noqa: E402
from tests.workflow.test_adk_compaction import (  # noqa: E402,F401
    PROMPTS,
    Claude,
    project,
    says,
    serve,
)

REPLIES = ["One.", "Two.", "Three."]


def sqlite_settings(**values) -> BaseSettings:
    return BaseSettings(**{
        "session_store": "sqlite", "default_model": "claude-sonnet-4-6", **values,
    })


def _manager(backend: str, app_settings: BaseSettings) -> BaseWorkflowManager:
    cls = {"adk": GoogleADKWorkflowManager, "deepagents": DeepAgentsWorkflowManager}[backend]
    return cls(
        agent_configs=[AgentConfig(name="solo", prompt=PROMPTS["solo"], include_state_tools=False)],
        settings=app_settings,
    )


@pytest.fixture(params=["adk", "deepagents"])
def backend(request, project, monkeypatch) -> str:
    """Each backend, its model answering ``REPLIES`` in turn."""
    if request.param == "adk":
        serve(monkeypatch, Claude(solo=[says(reply) for reply in REPLIES]))
    else:
        monkeypatch.setattr(deepagents_manager, "build_chat_model",
                            ScriptedModels(solo=[ai(reply) for reply in REPLIES]))
    return request.param


async def stored_by_its_manager(backend: str, app_settings: BaseSettings) -> list[SessionInfo]:
    """Three turns in two conversations, stored by the backend's own manager,
    which is then released: what its ``list_sessions`` listed."""
    manager = _manager(backend, app_settings)
    user = app_settings.default_user
    try:
        for text, session_id in (("Plan the trip.", "s1"), ("Fix the build.", "s2"),
                                 ("And the budget?", "s1")):
            async for _ in manager.process(text, user, session_id=session_id):
                pass
        return await manager.list_sessions()
    finally:
        await manager.cleanup()


class TestItReadsWhatTheManagerStored:
    async def test_the_rows_name_their_backend(self, backend):
        app_settings = sqlite_settings()
        listed = await stored_by_its_manager(backend, app_settings)

        reader = await open_session_reader(backend, app_settings)
        try:
            rows = await reader.list()
            newest = await reader.list(limit=1)
            one = await reader.info("s2")
        finally:
            await reader.aclose()

        assert reader.backend == backend
        assert [(r.session_id, r.title, r.turns) for r in rows] == [
            ("s1", "Plan the trip.", 2), ("s2", "Fix the build.", 1),
        ]
        assert rows == [dataclasses.replace(row, backend=backend) for row in listed]
        assert newest == rows[:1]
        assert one == rows[1]

    async def test_exists_info_and_delete(self, backend):
        app_settings = sqlite_settings()
        await stored_by_its_manager(backend, app_settings)

        async with await open_session_reader(backend, app_settings) as reader:
            assert await reader.exists("s1") is True
            assert await reader.exists("missing") is False
            assert await reader.info("missing") is None
            assert await reader.delete("missing") is False

            assert await reader.delete("s1") is True

            assert await reader.exists("s1") is False
            assert [r.session_id for r in await reader.list()] == ["s2"]

        # What the reader deleted is gone for the manager too.
        manager = _manager(backend, app_settings)
        await manager.initialize_services()
        try:
            assert [r.session_id for r in await manager.list_sessions()] == ["s2"]
        finally:
            await manager.cleanup()

    async def test_another_users_conversations_are_not_read(self, backend):
        await stored_by_its_manager(backend, sqlite_settings())

        reader = await open_session_reader(backend, sqlite_settings(default_user="someone-else"))
        try:
            assert await reader.list() == []
            assert await reader.info("s1") is None
        finally:
            await reader.aclose()


class TestOpeningOneBuildsNothing:
    async def test_no_manager_is_built(self, backend, monkeypatch):
        app_settings = sqlite_settings()
        await stored_by_its_manager(backend, app_settings)

        def built(*args, **kwargs):
            raise AssertionError("a manager was built to read the conversations")

        monkeypatch.setattr(BaseWorkflowManager, "__init__", built)
        reader = await open_session_reader(backend, app_settings)
        try:
            assert [r.session_id for r in await reader.list()] == ["s1", "s2"]
        finally:
            await reader.aclose()

    async def test_a_store_with_nothing_in_it_yet_is_not_made(self, backend, project):
        app_settings = sqlite_settings()
        before = sorted(p.name for p in app_settings.sessions_dir.parent.rglob("*"))

        reader = await open_session_reader(backend, app_settings)
        try:
            assert await reader.list() == []
            assert await reader.info("s1") is None
            assert await reader.exists("s1") is False
            assert await reader.delete("s1") is False
        finally:
            await reader.aclose()

        after = sorted(p.name for p in app_settings.sessions_dir.parent.rglob("*"))
        assert after == before, "reading made the store"


def _tables(path) -> list[str]:
    with sqlite3.connect(path) as db:
        rows = db.execute("select name from sqlite_master where type='table'")
        return sorted(name for (name,) in rows)


def _journal_mode(path) -> str:
    with sqlite3.connect(path) as db:
        return db.execute("pragma journal_mode").fetchone()[0]


class TestAStoreNotYetItsBackends:
    """A reader writes into no store that is not yet its backend's: the
    file ``sqlite_uri`` names may be another backend's (both read it), or a
    team's. It checks, read-only, that its backend's tables are there, and
    reads nothing otherwise: no new tables, no change of journal mode."""

    @pytest.mark.parametrize("backend, other", [("adk", "deepagents"), ("deepagents", "adk")])
    async def test_the_other_backends_file_keeps_its_tables_and_journal_mode(
        self, backend, other, project, tmp_path, monkeypatch
    ):
        shared = tmp_path / "shared.db"
        app_settings = sqlite_settings(sqlite_uri=str(shared))
        if other == "adk":
            serve(monkeypatch, Claude(solo=[says(reply) for reply in REPLIES]))
        else:
            monkeypatch.setattr(deepagents_manager, "build_chat_model",
                                ScriptedModels(solo=[ai(reply) for reply in REPLIES]))
        await stored_by_its_manager(other, app_settings)
        tables, mode = _tables(shared), _journal_mode(shared)

        async with await open_session_reader(backend, app_settings) as reader:
            assert await reader.list() == []
            assert await reader.info("s1") is None
            assert await reader.exists("s1") is False

        assert (_tables(shared), _journal_mode(shared)) == (tables, mode)

    @pytest.mark.parametrize("backend", ["adk", "deepagents"])
    async def test_an_empty_file_a_failed_start_left_is_left_as_it_is(self, backend, project):
        app_settings = sqlite_settings()
        file = (
            deepagents_sqlite_path(app_settings) if backend == "deepagents"
            else app_settings.sessions_dir / "sessions.db"
        )
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()

        async with await open_session_reader(backend, app_settings) as reader:
            assert await reader.list() == []

        assert file.stat().st_size == 0


def _threads() -> set[int]:
    return {thread.ident for thread in threading.enumerate()}


async def _left_after(before: set[int]) -> set[str]:
    """The threads started since ``before`` still running, a moment after
    (a connection's thread ends as it closes)."""
    for _ in range(100):
        left = _threads() - before
        if not left:
            return set()
        await asyncio.sleep(0.02)
    return {thread.name for thread in threading.enumerate() if thread.ident in left}


class TestItClosesWhatItOpened:
    """A reader holds its store's connections (ADK's engine, Deep Agents'
    checkpointer and index, each on a thread of its own): ``aclose`` closes
    them, and so does an open that fails or is cancelled half way."""

    async def test_aclose_closes_every_connection(self, backend):
        app_settings = sqlite_settings()
        await stored_by_its_manager(backend, app_settings)
        await asyncio.sleep(0.1)
        before = _threads()

        reader = await open_session_reader(backend, app_settings)
        assert [row.session_id for row in await reader.list()] == ["s1", "s2"]
        assert await reader.exists("s1")
        opened = _threads() - before
        await reader.aclose()

        assert opened, "the reader opened no connection to close"
        assert await _left_after(before) == set()

    @pytest.mark.parametrize("stops", ["fails", "is cancelled"])
    async def test_a_deep_agents_open_stopped_after_the_checkpointer_closes_it(
        self, stops, project, monkeypatch
    ):
        app_settings = sqlite_settings()
        monkeypatch.setattr(deepagents_manager, "build_chat_model",
                            ScriptedModels(solo=[ai(reply) for reply in REPLIES]))
        await stored_by_its_manager("deepagents", app_settings)
        await asyncio.sleep(0.1)
        before = _threads()
        opening = asyncio.Event()

        async def store_stops(*args, **kwargs):
            opening.set()
            if stops == "fails":
                raise RuntimeError("the index cannot open")
            await asyncio.Event().wait()

        monkeypatch.setattr(deepagents_sessions, "open_store", store_stops)
        task = asyncio.create_task(open_session_reader("deepagents", app_settings))
        await asyncio.wait_for(opening.wait(), 10)
        assert _threads() - before, "the checkpointer was not open yet"
        if stops == "is cancelled":
            task.cancel()
        with pytest.raises((RuntimeError, asyncio.CancelledError)):
            await task

        assert await _left_after(before) == set()


class TestBackendsWithoutOne:
    @pytest.mark.parametrize("backend", ["adk", "deepagents"])
    async def test_the_memory_store_has_none(self, backend, project):
        """Its conversations live in the manager that made them."""
        assert await open_session_reader(backend, sqlite_settings(session_store="memory")) is None

    async def test_the_store_given_is_the_one_read(self, project):
        """The run keeps the store it started with, whatever the settings say now."""
        assert await open_session_reader("adk", sqlite_settings(), store="memory") is None

    async def test_langgraph_has_none(self, project):
        """Deprecated: its running manager reads its own."""
        assert await open_session_reader("langgraph", sqlite_settings()) is None

    async def test_a_backend_whose_extra_is_missing_says_how_to_install_it(
        self, project, monkeypatch
    ):
        """Its store holds something (left by an earlier install, say)."""
        app_settings = sqlite_settings()
        deepagents_sqlite_path(app_settings)  # made
        monkeypatch.setitem(sys.modules, "agentic_cli.workflow.deepagents", None)
        monkeypatch.delitem(sys.modules, "agentic_cli.workflow.deepagents.sessions", raising=False)

        with pytest.raises(ImportError, match=r"needs the 'deepagents' extra"):
            await open_session_reader("deepagents", app_settings)

    async def test_deep_agents_is_not_imported_to_read_a_store_never_made(
        self, project, monkeypatch
    ):
        """Its import takes a second: a run that never used Deep Agents does
        not pay it to find nothing there."""
        monkeypatch.setitem(sys.modules, "agentic_cli.workflow.deepagents", None)
        monkeypatch.delitem(sys.modules, "agentic_cli.workflow.deepagents.sessions", raising=False)

        async with await open_session_reader("deepagents", sqlite_settings()) as reader:
            assert await reader.list() == [] and await reader.info("s1") is None

    async def test_an_unknown_backend_is_refused(self, project):
        with pytest.raises(ValueError):
            await open_session_reader("nope", sqlite_settings())

    def test_the_backends_with_one(self):
        assert session_readers.reader_backends() == ("adk", "deepagents")


def test_a_row_names_no_backend_unless_told():
    """``backend`` is additive: a row made as before names none."""
    row = SessionInfo("abc", "Plan the trip", 2)

    assert row.backend is None
    assert dataclasses.replace(row, backend="adk").backend == "adk"
