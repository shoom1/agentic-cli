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
