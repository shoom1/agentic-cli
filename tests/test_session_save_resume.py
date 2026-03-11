"""Tests for session save/resume feature.

Covers:
- SessionPersistence.save_snapshot()
- BaseWorkflowManager.save_session() / load_session() / list_sessions()
- Round-trip: save then load produces same messages
- Cross-backend load logs warning but succeeds
- SessionsCommand displays table
"""

import json
from datetime import datetime
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_cli.config import BaseSettings
from agentic_cli.persistence.session import SessionPersistence, SessionSnapshot
from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.events import WorkflowEvent

from tests.conftest import MockContext


# ---------------------------------------------------------------------------
# Concrete test subclass of BaseWorkflowManager
# ---------------------------------------------------------------------------

class _TestWorkflowManager(BaseWorkflowManager):
    """Minimal concrete subclass for testing base class session methods."""

    def __init__(self, settings: BaseSettings, **kwargs):
        super().__init__(agent_configs=[], settings=settings, **kwargs)
        # In-memory message store for testing
        self._stored_messages: list[dict] = []
        self._stored_agent: str | None = None
        self.session_id = "default_session"

    @property
    def backend_type(self) -> str:
        return "test"

    async def _do_initialize(self) -> None:
        pass

    async def process(
        self, message: str, user_id: str, session_id: str | None = None
    ) -> AsyncGenerator[WorkflowEvent, None]:
        if False:
            yield  # type: ignore[misc]

    async def reinitialize(
        self, model: str | None = None, preserve_sessions: bool = True
    ) -> None:
        pass

    async def cleanup(self) -> None:
        pass

    async def _extract_session_messages(self, session_id: str) -> list[dict]:
        return self._stored_messages

    async def _extract_current_agent(self, session_id: str) -> str | None:
        return self._stored_agent

    async def _inject_session_messages(
        self,
        session_id: str,
        messages: list[dict],
        current_agent: str | None = None,
    ) -> None:
        self._stored_messages = messages
        self._stored_agent = current_agent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_MESSAGES = [
    {"role": "user", "content": "Research quantum computing"},
    {
        "role": "assistant",
        "content": "I'll search for that.",
        "tool_calls": [
            {"id": "tc_1", "name": "web_search", "args": {"query": "quantum computing"}},
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_1",
        "name": "web_search",
        "content": '{"success": true, "results": []}',
    },
    {"role": "assistant", "content": "Based on my research..."},
]


# ---------------------------------------------------------------------------
# Tests: SessionPersistence.save_snapshot()
# ---------------------------------------------------------------------------


class TestSaveSnapshot:
    """Tests for the new save_snapshot() method."""

    def test_save_snapshot_creates_file(self, mock_context: MockContext):
        persistence = SessionPersistence(mock_context.settings)
        snapshot = SessionSnapshot(
            session_id="snap-test",
            created_at=datetime(2024, 1, 1),
            saved_at=datetime(2024, 1, 1, 1, 0),
            messages=SAMPLE_MESSAGES,
            metadata={"model": "test-model", "backend_type": "test"},
        )

        path = persistence.save_snapshot(snapshot)

        assert path.exists()
        data = json.loads(path.read_text())
        assert data["session_id"] == "snap-test"
        assert len(data["messages"]) == 4
        assert data["metadata"]["backend_type"] == "test"

    def test_save_snapshot_updates_index(self, mock_context: MockContext):
        persistence = SessionPersistence(mock_context.settings)
        snapshot = SessionSnapshot(
            session_id="indexed-test",
            created_at=datetime(2024, 1, 1),
            saved_at=datetime(2024, 1, 1, 1, 0),
            messages=[{"role": "user", "content": "hello"}],
        )

        persistence.save_snapshot(snapshot)

        sessions = persistence.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "indexed-test"
        assert sessions[0]["message_count"] == 1

    def test_save_snapshot_overwrites_existing(self, mock_context: MockContext):
        persistence = SessionPersistence(mock_context.settings)

        # Save first version
        snap1 = SessionSnapshot(
            session_id="overwrite-test",
            created_at=datetime(2024, 1, 1),
            saved_at=datetime(2024, 1, 1, 1, 0),
            messages=[{"role": "user", "content": "v1"}],
        )
        persistence.save_snapshot(snap1)

        # Save second version
        snap2 = SessionSnapshot(
            session_id="overwrite-test",
            created_at=datetime(2024, 1, 1),
            saved_at=datetime(2024, 1, 1, 2, 0),
            messages=[
                {"role": "user", "content": "v1"},
                {"role": "assistant", "content": "v2"},
            ],
        )
        persistence.save_snapshot(snap2)

        # Should have latest data
        loaded = persistence.load_session("overwrite-test")
        assert loaded is not None
        assert len(loaded.messages) == 2

    def test_save_snapshot_load_roundtrip(self, mock_context: MockContext):
        persistence = SessionPersistence(mock_context.settings)
        now = datetime.now()
        snapshot = SessionSnapshot(
            session_id="roundtrip",
            created_at=now,
            saved_at=now,
            messages=SAMPLE_MESSAGES,
            metadata={"model": "gemini-2.5-flash", "backend_type": "adk"},
        )

        persistence.save_snapshot(snapshot)
        loaded = persistence.load_session("roundtrip")

        assert loaded is not None
        assert loaded.session_id == "roundtrip"
        assert loaded.messages == SAMPLE_MESSAGES
        assert loaded.metadata["model"] == "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Tests: BaseWorkflowManager session methods
# ---------------------------------------------------------------------------


class TestWorkflowManagerSession:
    """Tests for save_session / load_session / list_sessions on BaseWorkflowManager."""

    async def test_save_session_persists_to_disk(self, mock_context: MockContext):
        mgr = _TestWorkflowManager(settings=mock_context.settings)
        mgr._stored_messages = SAMPLE_MESSAGES
        mgr._stored_agent = "researcher"
        mgr._model = "test-model"
        mgr._model_resolved = True

        result = await mgr.save_session("my-session")

        assert result["success"] is True
        assert result["session_id"] == "my-session"
        assert result["message_count"] == 4

        # Verify file on disk
        persistence = SessionPersistence(mock_context.settings)
        loaded = persistence.load_session("my-session")
        assert loaded is not None
        assert loaded.metadata["current_agent"] == "researcher"
        assert loaded.metadata["backend_type"] == "test"

    async def test_save_session_uses_default_session_id(self, mock_context: MockContext):
        mgr = _TestWorkflowManager(settings=mock_context.settings)
        mgr.session_id = "custom-default"
        mgr._stored_messages = [{"role": "user", "content": "hi"}]
        mgr._model = "test-model"
        mgr._model_resolved = True

        result = await mgr.save_session()

        assert result["success"] is True
        assert result["session_id"] == "custom-default"

    async def test_load_session_injects_messages(self, mock_context: MockContext):
        mgr = _TestWorkflowManager(settings=mock_context.settings)
        mgr._model = "test-model"
        mgr._model_resolved = True

        # First save
        mgr._stored_messages = SAMPLE_MESSAGES
        mgr._stored_agent = "researcher"
        await mgr.save_session("inject-test")

        # Clear and load into fresh manager
        mgr2 = _TestWorkflowManager(settings=mock_context.settings)
        mgr2._model = "test-model"
        mgr2._model_resolved = True
        loaded = await mgr2.load_session("inject-test")

        assert loaded is True
        assert mgr2._stored_messages == SAMPLE_MESSAGES
        assert mgr2._stored_agent == "researcher"
        assert mgr2.session_id == "inject-test"

    async def test_load_nonexistent_session_returns_false(self, mock_context: MockContext):
        mgr = _TestWorkflowManager(settings=mock_context.settings)

        loaded = await mgr.load_session("does-not-exist")

        assert loaded is False

    async def test_cross_backend_load_succeeds_with_warning(self, mock_context: MockContext):
        """Loading a session saved by a different backend should work but log a warning."""
        # Save with "adk" backend_type in metadata
        persistence = SessionPersistence(mock_context.settings)
        snapshot = SessionSnapshot(
            session_id="cross-backend",
            created_at=datetime.now(),
            saved_at=datetime.now(),
            messages=SAMPLE_MESSAGES,
            metadata={"backend_type": "adk", "model": "gemini-2.5-flash"},
        )
        persistence.save_snapshot(snapshot)

        # Load into "test" backend manager
        mgr = _TestWorkflowManager(settings=mock_context.settings)
        mgr._model = "test-model"
        mgr._model_resolved = True

        with patch("agentic_cli.workflow.base_manager.logger") as mock_logger:
            loaded = await mgr.load_session("cross-backend")

        assert loaded is True
        assert mgr._stored_messages == SAMPLE_MESSAGES
        # Should have logged a warning about backend mismatch
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert call_args[0][0] == "session_backend_mismatch"

    async def test_list_sessions(self, mock_context: MockContext):
        mgr = _TestWorkflowManager(settings=mock_context.settings)
        mgr._model = "test-model"
        mgr._model_resolved = True

        # Save two sessions
        mgr._stored_messages = [{"role": "user", "content": "a"}]
        await mgr.save_session("session-a")

        mgr._stored_messages = [
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "reply"},
        ]
        await mgr.save_session("session-b")

        sessions = mgr.list_sessions()

        assert len(sessions) == 2
        ids = {s["session_id"] for s in sessions}
        assert ids == {"session-a", "session-b"}

    async def test_save_session_round_trip_preserves_tool_calls(self, mock_context: MockContext):
        """Full round-trip: messages with tool_calls survive save/load."""
        mgr = _TestWorkflowManager(settings=mock_context.settings)
        mgr._model = "test-model"
        mgr._model_resolved = True
        mgr._stored_messages = SAMPLE_MESSAGES

        await mgr.save_session("tool-roundtrip")

        mgr2 = _TestWorkflowManager(settings=mock_context.settings)
        mgr2._model = "test-model"
        mgr2._model_resolved = True
        await mgr2.load_session("tool-roundtrip")

        assert mgr2._stored_messages == SAMPLE_MESSAGES
        # Verify tool_calls are preserved
        assistant_msg = mgr2._stored_messages[1]
        assert "tool_calls" in assistant_msg
        assert assistant_msg["tool_calls"][0]["name"] == "web_search"

        # Verify tool result message
        tool_msg = mgr2._stored_messages[2]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "tc_1"


# ---------------------------------------------------------------------------
# Tests: SessionsCommand
# ---------------------------------------------------------------------------


class TestSessionsCommand:
    """Tests for the /sessions command."""

    async def test_sessions_command_empty(self, mock_context: MockContext):
        from agentic_cli.cli.builtin_commands import SessionsCommand

        cmd = SessionsCommand()
        app = MagicMock()
        app.settings = mock_context.settings
        app.session = MagicMock()
        app._session_id = None

        await cmd.execute("", app)

        app.session.add_message.assert_called_once_with("system", "No saved sessions.")

    async def test_sessions_command_lists_sessions(self, mock_context: MockContext):
        from agentic_cli.cli.builtin_commands import SessionsCommand

        # Save a session first
        persistence = SessionPersistence(mock_context.settings)
        snapshot = SessionSnapshot(
            session_id="listed-session",
            created_at=datetime(2024, 6, 1),
            saved_at=datetime(2024, 6, 1, 12, 0),
            messages=[{"role": "user", "content": "hi"}],
        )
        persistence.save_snapshot(snapshot)

        cmd = SessionsCommand()
        app = MagicMock()
        app.settings = mock_context.settings
        app.session = MagicMock()
        app._session_id = "listed-session"

        await cmd.execute("", app)

        # Should have called add_rich with a table
        app.session.add_rich.assert_called_once()

    async def test_sessions_command_delete(self, mock_context: MockContext):
        from agentic_cli.cli.builtin_commands import SessionsCommand

        # Save a session to delete
        persistence = SessionPersistence(mock_context.settings)
        snapshot = SessionSnapshot(
            session_id="to-delete",
            created_at=datetime(2024, 6, 1),
            saved_at=datetime(2024, 6, 1, 12, 0),
            messages=[{"role": "user", "content": "bye"}],
        )
        persistence.save_snapshot(snapshot)

        cmd = SessionsCommand()
        app = MagicMock()
        app.settings = mock_context.settings
        app.session = MagicMock()

        await cmd.execute("--delete=to-delete", app)

        app.session.add_success.assert_called_once()
        # Verify it's actually gone
        assert persistence.load_session("to-delete") is None

    async def test_sessions_command_delete_nonexistent(self, mock_context: MockContext):
        from agentic_cli.cli.builtin_commands import SessionsCommand

        cmd = SessionsCommand()
        app = MagicMock()
        app.settings = mock_context.settings
        app.session = MagicMock()

        await cmd.execute("--delete=nope", app)

        app.session.add_error.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: StatusCommand session info
# ---------------------------------------------------------------------------


class TestStatusCommandSessionInfo:
    """Tests that /status shows session info."""

    async def test_status_shows_persistent_session(self, mock_context: MockContext):
        from agentic_cli.cli.builtin_commands import StatusCommand

        cmd = StatusCommand()
        app = MagicMock()
        app._session_id = "my-research"
        app.message_history = []
        app.usage_tracker = MagicMock(invocation_count=0)
        # Make workflow access raise to skip workflow section
        type(app).workflow = property(lambda self: (_ for _ in ()).throw(RuntimeError))

        await cmd.execute("", app)

        # The table should have been added via add_rich
        app.session.add_rich.assert_called_once()

    async def test_status_shows_ephemeral_session(self, mock_context: MockContext):
        from agentic_cli.cli.builtin_commands import StatusCommand

        cmd = StatusCommand()
        app = MagicMock()
        app._session_id = None
        app.message_history = []
        app.usage_tracker = MagicMock(invocation_count=0)
        type(app).workflow = property(lambda self: (_ for _ in ()).throw(RuntimeError))

        await cmd.execute("", app)

        app.session.add_rich.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: BaseCLIApp session_id parameter
# ---------------------------------------------------------------------------


class TestBaseCLIAppSessionId:
    """Tests for session_id parameter on BaseCLIApp."""

    def test_session_id_defaults_to_none(self):
        """session_id should default to None when not provided."""
        from agentic_cli.cli.app import BaseCLIApp

        # Just verify the parameter exists in the signature
        import inspect
        sig = inspect.signature(BaseCLIApp.__init__)
        param = sig.parameters.get("session_id")
        assert param is not None
        assert param.default is None

    def test_session_id_property(self, mock_context: MockContext):
        """session_id property should be gettable and settable."""
        from agentic_cli.cli.app import BaseCLIApp
        from thinking_prompt import AppInfo

        with patch("agentic_cli.cli.app.ThinkingPromptSession"):
            app = BaseCLIApp(
                app_info=AppInfo(name="test", version="0.1.0"),
                agent_configs=[],
                settings=mock_context.settings,
                session_id="my-sess",
            )

        assert app.session_id == "my-sess"
        app.session_id = "other"
        assert app.session_id == "other"
