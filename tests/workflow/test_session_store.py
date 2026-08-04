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


@pytest.fixture
async def _closing_session_services(monkeypatch):
    """Close every session service the test creates.

    ``DatabaseSessionService`` owns a SQLAlchemy async engine whose connection
    worker thread outlives a service that is merely dropped. When the engine is
    eventually finalized that thread raises, and pytest reports it as a
    ``PytestUnhandledThreadExceptionWarning`` against whichever *unrelated*
    test happens to be running at the time. Closing them here keeps the failure
    attributable — and matches the ownership contract the manager itself obeys
    (see ``BaseWorkflowManager._aclose_owned``).
    """
    import inspect

    from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager

    created: list[object] = []
    original = GoogleADKWorkflowManager._make_session_service

    def _tracked(self):
        service = original(self)
        created.append(service)
        return service

    monkeypatch.setattr(
        GoogleADKWorkflowManager, "_make_session_service", _tracked
    )
    yield created

    for service in created:
        close = getattr(service, "aclose", None) or getattr(service, "close", None)
        if close is None:
            continue
        result = close()
        if inspect.isawaitable(result):
            await result


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
    async def _require_adk(self, _closing_session_services):
        pytest.importorskip("google.adk")

    def _manager(self, settings):
        from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager

        mgr = GoogleADKWorkflowManager.__new__(GoogleADKWorkflowManager)
        mgr._settings = settings
        return mgr

    async def test_memory_uses_in_memory_service(self, tmp_path: Path):
        from google.adk.sessions import InMemorySessionService

        mgr = self._manager(_settings(tmp_path, session_store="memory"))
        assert isinstance(mgr._make_session_service(), InMemorySessionService)

    async def test_sqlite_uses_database_service_and_creates_dir(self, tmp_path: Path):
        from google.adk.sessions import DatabaseSessionService

        mgr = self._manager(_settings(tmp_path, session_store="sqlite"))
        svc = mgr._make_session_service()
        assert isinstance(svc, DatabaseSessionService)
        assert (tmp_path / "sessions").is_dir()


class TestAdkNativeSessions:
    """Native session query/manage against a real sqlite DatabaseSessionService."""

    @pytest.fixture(autouse=True)
    async def _require_adk(self, _closing_session_services):
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


class TestAdkSessionUserScope:
    """Session APIs accept an explicit user_id, defaulting to settings.default_user.

    process() always accepted arbitrary user_id (and job-resume threads
    record.user_id), but the query/manage APIs hard-coded default_user —
    sessions created for another user were invisible to them.
    """

    @pytest.fixture(autouse=True)
    async def _require_adk(self, _closing_session_services):
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

    async def _seed_as(self, mgr, user_id: str, sid: str, text: str):
        from google.adk.events import Event
        from google.genai import types

        s = await mgr._session_service.create_session(
            app_name=mgr.app_name, user_id=user_id, session_id=sid
        )
        await mgr._session_service.append_event(
            session=s,
            event=Event(
                author="user",
                content=types.Content(
                    role="user", parts=[types.Part.from_text(text=text)]
                ),
            ),
        )

    async def test_session_apis_scope_to_given_user(self, tmp_path: Path):
        mgr, settings = self._manager(tmp_path)
        await self._seed_as(mgr, "alice", "sess-a", "alice message")
        await self._seed_as(mgr, settings.default_user, "sess-d", "default message")

        # Default scope: unchanged behavior, sees only default_user's sessions
        assert await mgr.session_exists("sess-d") is True
        assert await mgr.session_exists("sess-a") is False

        # Explicit user scope reaches alice's session through every API
        assert await mgr.session_exists("sess-a", user_id="alice") is True

        listed = await mgr.list_sessions(user_id="alice")
        assert [s["session_id"] for s in listed] == ["sess-a"]

        recent = await mgr.recent_messages("sess-a", user_id="alice")
        assert recent and recent[-1]["content"] == "alice message"

        assert await mgr.delete_session("sess-a", user_id="alice") is True
        assert await mgr.session_exists("sess-a", user_id="alice") is False

        # Default user's session untouched by alice-scoped operations
        assert await mgr.session_exists("sess-d") is True

    async def test_session_ref_resolves_partial_identity(self, tmp_path: Path):
        mgr, settings = self._manager(tmp_path)

        ref = mgr.session_ref()
        assert (ref.app_name, ref.user_id, ref.session_id) == (
            "test_app", settings.default_user, "default_session",
        )

        explicit = mgr.session_ref("sess-x", "alice")
        assert (explicit.user_id, explicit.session_id) == ("alice", "sess-x")

    async def test_load_session_honours_explicit_user(self, tmp_path: Path):
        """A session adopted for another user must be seen as a real resume."""
        mgr, _ = self._manager(tmp_path)
        await self._seed_as(mgr, "alice", "sess-a", "alice message")

        assert await mgr.load_session("sess-a") is False  # default user: not theirs
        assert await mgr.load_session("sess-a", user_id="alice") is True
        assert mgr.session_id == "sess-a"

    async def test_adk_reports_session_support(self, tmp_path: Path):
        mgr, _ = self._manager(tmp_path)
        assert mgr.supports_sessions is True


class TestSessionsUnsupportedFailExplicitly:
    """A backend without durable sessions must not answer with a bare False/[]."""

    def _manager(self):
        from agentic_cli.workflow.base_manager import BaseWorkflowManager

        class _NoSessions(BaseWorkflowManager):
            def _get_state_tools(self):
                return []

            @property
            def backend_type(self) -> str:
                return "test"

            async def _do_initialize(self) -> None:
                return None

            async def process(self, message, user_id, session_id=None):
                raise NotImplementedError

            async def reinitialize(self, model=None, preserve_sessions=True):
                return None

            async def cleanup(self):
                return None

        from unittest.mock import MagicMock

        settings = MagicMock()
        settings.app_name = "test-app"
        settings.default_user = "default_user"
        return _NoSessions(agent_configs=[], settings=settings)

    def test_supports_sessions_is_false(self):
        assert self._manager().supports_sessions is False

    async def test_hooks_raise_not_implemented(self):
        mgr = self._manager()
        for call in (
            mgr.session_exists("s"),
            mgr.list_sessions(),
            mgr.delete_session("s"),
            mgr.recent_messages("s"),
        ):
            with pytest.raises(NotImplementedError, match="durable sessions"):
                await call

    async def test_load_session_adopts_without_raising(self):
        """Adoption still works — there is simply nothing to resume."""
        mgr = self._manager()
        assert await mgr.load_session("sess-1") is False


class TestSessionEndScope:
    """Fact extraction must read the session it actually belongs to."""

    @pytest.fixture(autouse=True)
    def _require_adk(self):
        pytest.importorskip("google.adk")

    def _manager(self, tmp_path: Path):
        from unittest.mock import MagicMock

        from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager

        settings = _settings(tmp_path, session_store="sqlite")
        object.__setattr__(settings, "auto_extract_session_facts", True)
        mgr = GoogleADKWorkflowManager.__new__(GoogleADKWorkflowManager)
        mgr._settings = settings
        mgr._app_name = "test_app"
        mgr.session_id = "sess-current"
        mgr._services = {"memory_store": MagicMock()}
        mgr._session_service = None
        return mgr, settings

    async def test_reads_the_default_user_session_by_default(self, tmp_path: Path):
        """The default user is passed positionally, so pre-``user_id``
        backend overrides keep working."""
        mgr, settings = self._manager(tmp_path)
        seen: list[tuple[str, str | None]] = []

        async def _recent(session_id, limit=20, *, user_id=None):
            seen.append((session_id, user_id))
            return []

        mgr.recent_messages = _recent
        await mgr.on_session_end()

        assert seen == [("sess-current", None)]

    async def test_legacy_override_without_user_id_still_works(self, tmp_path: Path):
        """A backend that predates the parameter must not break (LangGraph)."""
        mgr, _ = self._manager(tmp_path)
        seen: list[str] = []

        async def _legacy_recent(session_id, limit=20):
            seen.append(session_id)
            return []

        mgr.recent_messages = _legacy_recent
        await mgr.on_session_end()

        assert seen == ["sess-current"]

    async def test_explicit_session_ref_is_honoured(self, tmp_path: Path):
        from agentic_cli.workflow.sessions import SessionRef

        mgr, _ = self._manager(tmp_path)
        seen: list[tuple[str, str | None]] = []

        async def _recent(session_id, limit=20, *, user_id=None):
            seen.append((session_id, user_id))
            return []

        mgr.recent_messages = _recent
        await mgr.on_session_end(
            session=SessionRef(app_name="test_app", user_id="alice", session_id="sess-a")
        )

        assert seen == [("sess-a", "alice")], "another user's session was not read"

    async def test_active_turn_identity_is_used_when_available(self, tmp_path: Path):
        mgr, _ = self._manager(tmp_path)
        seen: list[tuple[str, str | None]] = []

        async def _recent(session_id, limit=20, *, user_id=None):
            seen.append((session_id, user_id))
            return []

        mgr.recent_messages = _recent
        with mgr._workflow_context(session_id="sess-live", user_id="bob"):
            await mgr.on_session_end()

        assert seen == [("sess-live", "bob")]

    async def test_save_session_reports_full_identity(self, tmp_path: Path):
        mgr, settings = self._manager(tmp_path)

        assert await mgr.save_session() == {
            "success": True,
            "session_id": "sess-current",
            "user_id": settings.default_user,
        }
        assert await mgr.save_session("s2", user_id="alice") == {
            "success": True,
            "session_id": "s2",
            "user_id": "alice",
        }
