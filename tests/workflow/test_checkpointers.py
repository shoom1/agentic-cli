"""The checkpointer factory the LangChain-based backends share.

``workflow/checkpointers.py`` opens a LangGraph checkpointer for a
``session_store`` kind and returns it with the context manager that owns its
connection. Each backend says where its sqlite file is: LangGraph keeps its
own path (``{workspace}/checkpoints.db``, ``sqlite_uri`` taken as a file name)
through ``create_checkpointer``, and ``deepagents_sqlite_path`` puts Deep
Agents' file next to ADK's ``sessions.db``, reading ``sqlite_uri`` the way
ADK does.
"""

from __future__ import annotations

import sqlite3
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("langgraph.checkpoint.memory")

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.workflow.checkpointers import (  # noqa: E402
    deepagents_sqlite_path,
    open_checkpointer,
    sqlite_file,
)

LANGGRAPH_MODULE = "agentic_cli.workflow.langgraph.persistence.checkpointers"


@pytest.fixture
def isolated(tmp_path, monkeypatch) -> Path:
    """No user or project settings file reaches the settings under test."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _settings(root: Path, **kwargs) -> BaseSettings:
    return BaseSettings(workspace_dir=root / "ws", **kwargs)


def _tables(path: Path) -> set[str]:
    with sqlite3.connect(path) as conn:
        rows = conn.execute("select name from sqlite_master where type='table'")
        return {name for (name,) in rows}


class _FakeSaver:
    def __init__(self, fail_setup: bool = False) -> None:
        self.fail_setup = fail_setup
        self.set_up = False

    async def setup(self) -> None:
        if self.fail_setup:
            raise RuntimeError("relation already exists")
        self.set_up = True


class _FakeConnection:
    """What ``AsyncPostgresSaver.from_conn_string`` returns: a context manager."""

    def __init__(self, saver: _FakeSaver) -> None:
        self.saver = saver
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> _FakeSaver:
        self.entered = True
        return self.saver

    async def __aexit__(self, *exc) -> bool:
        self.exited = True
        return False


@pytest.fixture
def fake_postgres(monkeypatch):
    """Stand in for ``langgraph.checkpoint.postgres.aio`` (no server, no libpq)."""
    opened: list[tuple[str, _FakeConnection]] = []
    state = {"fail_setup": False}

    class AsyncPostgresSaver:
        @staticmethod
        def from_conn_string(uri: str) -> _FakeConnection:
            connection = _FakeConnection(_FakeSaver(fail_setup=state["fail_setup"]))
            opened.append((uri, connection))
            return connection

    module = types.ModuleType("langgraph.checkpoint.postgres.aio")
    module.AsyncPostgresSaver = AsyncPostgresSaver
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres.aio", module)
    return types.SimpleNamespace(opened=opened, state=state)


class TestOpenCheckpointer:
    async def test_memory_is_an_in_memory_saver_with_nothing_to_close(self):
        saver, cm = await open_checkpointer("memory")

        assert isinstance(saver, InMemorySaver)
        assert cm is None

    async def test_none_opens_nothing(self):
        assert await open_checkpointer(None) == (None, None)

    async def test_sqlite_creates_its_tables_and_closing_releases_the_connection(
        self, tmp_path
    ):
        pytest.importorskip("langgraph.checkpoint.sqlite.aio")
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        path = tmp_path / "checkpoints.db"
        saver, cm = await open_checkpointer("sqlite", sqlite_path=path)
        try:
            assert isinstance(saver, AsyncSqliteSaver)
            assert {"checkpoints", "writes"} <= _tables(path)
            await saver.conn.execute("select 1")
        finally:
            await cm.__aexit__(None, None, None)

        with pytest.raises(ValueError, match="no active connection"):
            await saver.conn.execute("select 1")

    async def test_sqlite_without_a_path_is_an_error(self):
        with pytest.raises(ValueError, match="sqlite_path"):
            await open_checkpointer("sqlite")

    async def test_postgres_without_a_uri_names_the_setting(self, fake_postgres):
        with pytest.raises(ValueError, match="postgres_uri"):
            await open_checkpointer("postgres")
        assert fake_postgres.opened == []

    async def test_postgres_opens_enters_and_sets_up(self, fake_postgres):
        saver, cm = await open_checkpointer("postgres", postgres_uri="postgresql://db/x")

        [(uri, connection)] = fake_postgres.opened
        assert uri == "postgresql://db/x"
        assert cm is connection and connection.entered and not connection.exited
        assert saver is connection.saver and saver.set_up

    async def test_a_failed_setup_closes_the_connection_it_opened(self, fake_postgres):
        fake_postgres.state["fail_setup"] = True

        with pytest.raises(RuntimeError, match="relation already exists"):
            await open_checkpointer("postgres", postgres_uri="postgresql://db/x")

        [(_, connection)] = fake_postgres.opened
        assert connection.entered and connection.exited

    async def test_an_unknown_kind_is_an_error(self):
        with pytest.raises(ValueError, match="Unknown checkpointer type: redis"):
            await open_checkpointer("redis")  # type: ignore[arg-type]


class TestSqliteFile:
    @pytest.mark.parametrize(
        "value, path",
        [
            ("/data/checkpoints.db", "/data/checkpoints.db"),
            ("relative/checkpoints.db", "relative/checkpoints.db"),
            ("sqlite:////data/checkpoints.db", "/data/checkpoints.db"),
            ("sqlite+aiosqlite:////data/checkpoints.db", "/data/checkpoints.db"),
            ("sqlite:///relative.db", "relative.db"),
            # A URI's path is taken as written, as SQLAlchemy (ADK) takes it.
            ("sqlite:///~/s.db", "~/s.db"),
        ],
    )
    def test_a_uri_becomes_its_path_and_a_path_stays(self, value, path):
        assert sqlite_file(value) == path

    def test_a_plain_path_expands_the_home_directory(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))

        assert sqlite_file("~/sessions/c.db") == f"{tmp_path}/sessions/c.db"


class TestDeepAgentsSqlitePath:
    def test_the_default_is_next_to_adks_sessions_db(self, isolated):
        settings = _settings(isolated)

        path = deepagents_sqlite_path(settings)

        assert path == settings.sessions_dir / "checkpoints.db"
        assert path.parent.is_dir()
        adk_file = Path(settings.session_db_url().split(":///", 1)[1])
        assert adk_file.parent == path.parent

    @pytest.mark.parametrize("form", ["sqlite:///", "sqlite+aiosqlite:///", ""])
    async def test_every_sqlite_uri_form_names_the_file_adk_opens(self, isolated, form):
        from google.adk.sessions import DatabaseSessionService

        target = isolated / "shared" / "app.db"
        settings = _settings(isolated, sqlite_uri=f"{form}{target}")

        path = deepagents_sqlite_path(settings)

        assert path == target
        assert path.parent.is_dir()
        assert settings.session_db_url() == f"sqlite+aiosqlite:///{target}"
        service = DatabaseSessionService(db_url=settings.session_db_url())
        try:
            await service.create_session(app_name="app", user_id="u", session_id="s")
        finally:
            await service.close()
        assert target.is_file()

    def test_a_plain_path_is_used_as_it_is(self, isolated):
        target = isolated / "elsewhere" / "c.db"

        path = deepagents_sqlite_path(_settings(isolated, sqlite_uri=str(target)))

        assert path == target
        assert path.parent.is_dir()


class TestLangGraphCreateCheckpointer:
    """LangGraph's wrapper keeps its own sqlite location, exactly as before."""

    @pytest.fixture
    def opened(self, monkeypatch) -> list[tuple[tuple, dict]]:
        calls: list[tuple[tuple, dict]] = []

        async def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return "saver", "cm"

        monkeypatch.setattr(f"{LANGGRAPH_MODULE}.open_checkpointer", spy)
        return calls

    async def test_sqlite_defaults_to_the_workspace(self, isolated, opened):
        from agentic_cli.workflow.langgraph.persistence import create_checkpointer

        settings = _settings(isolated)

        assert await create_checkpointer("sqlite", settings) == ("saver", "cm")
        [(args, kwargs)] = opened
        assert args == ("sqlite",)
        assert kwargs["sqlite_path"] == str(settings.workspace_dir / "checkpoints.db")

    async def test_a_sqlite_uri_is_passed_through_raw(self, isolated, opened):
        from agentic_cli.workflow.langgraph.persistence import create_checkpointer

        uri = f"sqlite:///{isolated}/shared.db"

        await create_checkpointer("sqlite", _settings(isolated, sqlite_uri=uri))

        [(_, kwargs)] = opened
        assert kwargs["sqlite_path"] == uri

    async def test_postgres_passes_the_uri(self, isolated, opened):
        from agentic_cli.workflow.langgraph.persistence import create_checkpointer

        uri = "postgresql://db/x"

        await create_checkpointer("postgres", _settings(isolated, postgres_uri=uri))

        [(args, kwargs)] = opened
        assert args == ("postgres",)
        assert kwargs["postgres_uri"] == uri

    async def test_sqlite_writes_the_workspace_file(self, isolated):
        pytest.importorskip("langgraph.checkpoint.sqlite.aio")
        from agentic_cli.workflow.langgraph.persistence import create_checkpointer

        settings = _settings(isolated)
        settings.workspace_dir.mkdir(parents=True, exist_ok=True)

        saver, cm = await create_checkpointer("sqlite", settings)
        try:
            assert {"checkpoints", "writes"} <= _tables(settings.workspace_dir / "checkpoints.db")
            assert not (settings.sessions_dir / "checkpoints.db").exists()
        finally:
            await cm.__aexit__(None, None, None)

    async def test_postgres_without_a_uri_is_still_refused(self, isolated):
        from agentic_cli.workflow.langgraph.persistence import create_checkpointer

        with pytest.raises(ValueError, match="postgres_uri"):
            await create_checkpointer("postgres", _settings(isolated))
