"""Resumable sessions: session_db_url resolution + ADK session-service selection.

Milestone 1 of the durable-sessions feature: the unified `session_store` setting
resolves to one async SQLAlchemy URL (shared by both backends), and the ADK
manager builds a DatabaseSessionService (durable) vs InMemory (ephemeral).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_cli.config import BaseSettings


def _settings(tmp_path: Path, **over) -> BaseSettings:
    return BaseSettings(workspace_dir=tmp_path, **over)


class TestSessionDbUrl:
    def test_sqlite_default(self, tmp_path: Path):
        url = _settings(tmp_path, session_store="sqlite").session_db_url()
        assert url is not None
        assert url.startswith("sqlite+aiosqlite:///")
        assert url.endswith("sessions/sessions.db")

    def test_memory_is_none(self, tmp_path: Path):
        assert _settings(tmp_path, session_store="memory").session_db_url() is None

    def test_postgres_normalized_to_async(self, tmp_path: Path):
        url = _settings(
            tmp_path, session_store="postgres", postgres_uri="postgresql://u@h/db"
        ).session_db_url()
        assert url == "postgresql+asyncpg://u@h/db"

    def test_postgres_requires_uri(self, tmp_path: Path):
        with pytest.raises(ValueError):
            _settings(tmp_path, session_store="postgres").session_db_url()

    def test_explicit_sqlite_uri_normalized(self, tmp_path: Path):
        url = _settings(
            tmp_path, session_store="sqlite", sqlite_uri="sqlite:////tmp/custom.db"
        ).session_db_url()
        assert url == "sqlite+aiosqlite:////tmp/custom.db"


class TestAdkSessionServiceSelection:
    @pytest.fixture(autouse=True)
    def _require_adk(self):
        pytest.importorskip("google.adk")

    def _manager(self, settings):
        from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager

        mgr = GoogleADKWorkflowManager.__new__(GoogleADKWorkflowManager)
        mgr._settings = settings
        return mgr

    def test_memory_uses_in_memory_service(self, tmp_path: Path):
        from google.adk.sessions import InMemorySessionService

        mgr = self._manager(_settings(tmp_path, session_store="memory"))
        assert isinstance(mgr._make_session_service(), InMemorySessionService)

    def test_sqlite_uses_database_service_and_creates_dir(self, tmp_path: Path):
        from google.adk.sessions import DatabaseSessionService

        mgr = self._manager(_settings(tmp_path, session_store="sqlite"))
        svc = mgr._make_session_service()
        assert isinstance(svc, DatabaseSessionService)
        assert (tmp_path / "sessions").is_dir()


class TestAdkNativeSessions:
    """Native session query/manage against a real sqlite DatabaseSessionService."""

    @pytest.fixture(autouse=True)
    def _require_adk(self):
        pytest.importorskip("google.adk")

    def _manager(self, tmp_path: Path):
        from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager

        settings = _settings(tmp_path, session_store="sqlite")
        mgr = GoogleADKWorkflowManager.__new__(GoogleADKWorkflowManager)
        mgr._settings = settings
        mgr._app_name = "test_app"
        mgr.session_id = "default_session"
        mgr._session_service = mgr._make_session_service()
        return mgr, settings

    async def _seed(self, mgr, settings, sid: str, text: str):
        from google.adk.events import Event
        from google.genai import types

        s = await mgr._session_service.create_session(
            app_name=mgr.app_name, user_id=settings.default_user, session_id=sid
        )
        await mgr._session_service.append_event(
            session=s,
            event=Event(
                author="user",
                content=types.Content(role="user", parts=[types.Part.from_text(text=text)]),
            ),
        )

    async def test_exists_list_recent_delete(self, tmp_path: Path):
        mgr, settings = self._manager(tmp_path)
        await self._seed(mgr, settings, "sess-x", "remember the alpha value")

        assert await mgr.session_exists("sess-x") is True
        assert await mgr.session_exists("missing") is False

        listed = await mgr.list_sessions()
        assert any(s["session_id"] == "sess-x" for s in listed)

        recent = await mgr.recent_messages("sess-x")
        assert recent and recent[-1]["content"] == "remember the alpha value"

        assert await mgr.delete_session("sess-x") is True
        assert await mgr.session_exists("sess-x") is False

    async def test_load_session_reports_resume(self, tmp_path: Path):
        mgr, settings = self._manager(tmp_path)
        await self._seed(mgr, settings, "sess-y", "hi")
        # Existing session → resumed=True and id adopted.
        assert await mgr.load_session("sess-y") is True
        assert mgr.session_id == "sess-y"
        # Unknown session → new (False) but still adopted.
        assert await mgr.load_session("brand-new") is False
        assert mgr.session_id == "brand-new"

    async def test_persists_across_fresh_manager(self, tmp_path: Path):
        mgr, settings = self._manager(tmp_path)
        await self._seed(mgr, settings, "sess-z", "durable")
        # A second manager over the same sqlite file sees the session.
        mgr2, _ = self._manager(tmp_path)
        assert await mgr2.session_exists("sess-z") is True
