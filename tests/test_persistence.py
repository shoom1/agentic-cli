"""Tests for session persistence."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agentic_cli.persistence.session import (
    SessionPersistence,
    SessionSnapshot,
    StateSnapshot,
    ToolCallRecord,
)
from tests.conftest import MockContext


class TestSessionSnapshot:
    """Tests for SessionSnapshot dataclass."""

    def test_snapshot_to_dict(self):
        """Test snapshot serialization."""
        snapshot = SessionSnapshot(
            session_id="test-session",
            created_at=datetime(2024, 1, 1, 10, 0, 0),
            saved_at=datetime(2024, 1, 1, 12, 0, 0),
            messages=[
                {"content": "Hello", "message_type": "user"},
                {"content": "Hi there", "message_type": "assistant"},
            ],
            metadata={"model": "gemini-2.5-pro"},
        )

        data = snapshot.to_dict()

        assert data["session_id"] == "test-session"
        assert data["created_at"] == "2024-01-01T10:00:00"
        assert data["saved_at"] == "2024-01-01T12:00:00"
        assert len(data["messages"]) == 2
        assert data["metadata"]["model"] == "gemini-2.5-pro"

    def test_snapshot_from_dict(self):
        """Test snapshot deserialization."""
        data = {
            "session_id": "test-session",
            "created_at": "2024-01-01T10:00:00",
            "saved_at": "2024-01-01T12:00:00",
            "messages": [{"content": "Test", "message_type": "user"}],
            "metadata": {"key": "value"},
        }

        snapshot = SessionSnapshot.from_dict(data)

        assert snapshot.session_id == "test-session"
        assert snapshot.created_at == datetime(2024, 1, 1, 10, 0, 0)
        assert snapshot.saved_at == datetime(2024, 1, 1, 12, 0, 0)
        assert len(snapshot.messages) == 1
        assert snapshot.metadata == {"key": "value"}

    def test_snapshot_from_dict_no_metadata(self):
        """Test snapshot deserialization without metadata."""
        data = {
            "session_id": "test",
            "created_at": "2024-01-01T00:00:00",
            "saved_at": "2024-01-01T00:00:00",
            "messages": [],
        }

        snapshot = SessionSnapshot.from_dict(data)

        assert snapshot.metadata == {}

    def test_snapshot_roundtrip(self):
        """Test serialization roundtrip."""
        original = SessionSnapshot(
            session_id="roundtrip-test",
            created_at=datetime(2024, 6, 15, 14, 30, 0),
            saved_at=datetime(2024, 6, 15, 15, 0, 0),
            messages=[{"content": "Test message", "message_type": "user"}],
            metadata={"test": True},
        )

        data = original.to_dict()
        restored = SessionSnapshot.from_dict(data)

        assert restored.session_id == original.session_id
        assert restored.created_at == original.created_at
        assert restored.saved_at == original.saved_at
        assert restored.messages == original.messages
        assert restored.metadata == original.metadata


class TestSessionPersistence:
    """Tests for SessionPersistence class."""

    def test_init_creates_directory(self, mock_context: MockContext):
        """Test that initialization creates sessions directory."""
        persistence = SessionPersistence(mock_context.settings)

        assert persistence.sessions_path.exists()
        assert persistence.sessions_path == mock_context.workspace_dir / "sessions"

    def test_session_id_sanitization(self, mock_context: MockContext):
        """Test that session IDs are sanitized for filesystem."""
        persistence = SessionPersistence(mock_context.settings)

        # Test various problematic session IDs
        test_cases = [
            ("normal-session", "normal-session.json"),
            ("session with spaces", "session_with_spaces.json"),
            ("session/with/slashes", "session_with_slashes.json"),
            ("session:with:colons", "session_with_colons.json"),
        ]

        for session_id, expected_filename in test_cases:
            path = persistence._get_session_path(session_id)
            assert path.name == expected_filename

    def test_list_sessions_empty(self, mock_context: MockContext):
        """Test listing sessions when none exist."""
        persistence = SessionPersistence(mock_context.settings)

        sessions = persistence.list_sessions()

        assert sessions == []

    def test_delete_nonexistent_session(self, mock_context: MockContext):
        """Test deleting a session that doesn't exist."""
        persistence = SessionPersistence(mock_context.settings)

        result = persistence.delete_session("nonexistent")

        assert result is False

    def test_load_nonexistent_session(self, mock_context: MockContext):
        """Test loading a session that doesn't exist."""
        persistence = SessionPersistence(mock_context.settings)

        snapshot = persistence.load_session("nonexistent")

        assert snapshot is None

    def test_save_and_load_manually(self, mock_context: MockContext):
        """Test saving and loading session data manually."""
        persistence = SessionPersistence(mock_context.settings)

        # Create a session file manually
        session_id = "manual-test"
        session_path = persistence._get_session_path(session_id)

        snapshot_data = {
            "session_id": session_id,
            "created_at": "2024-01-01T00:00:00",
            "saved_at": "2024-01-01T01:00:00",
            "messages": [{"content": "Hello", "message_type": "user"}],
            "metadata": {"test": True},
        }

        with open(session_path, "w") as f:
            json.dump(snapshot_data, f)

        # Load it back
        loaded = persistence.load_session(session_id)

        assert loaded is not None
        assert loaded.session_id == session_id
        assert len(loaded.messages) == 1

    def test_list_sessions(self, mock_context: MockContext):
        """Test listing multiple sessions."""
        persistence = SessionPersistence(mock_context.settings)

        # Create multiple session files
        for i in range(3):
            session_id = f"session-{i}"
            session_path = persistence._get_session_path(session_id)

            snapshot_data = {
                "session_id": session_id,
                "created_at": f"2024-01-0{i+1}T00:00:00",
                "saved_at": f"2024-01-0{i+1}T01:00:00",
                "messages": [{"content": f"Message {i}"}],
                "metadata": {},
            }

            with open(session_path, "w") as f:
                json.dump(snapshot_data, f)

        sessions = persistence.list_sessions()

        assert len(sessions) == 3
        # Should be sorted by saved_at descending
        assert sessions[0]["session_id"] == "session-2"
        assert sessions[1]["session_id"] == "session-1"
        assert sessions[2]["session_id"] == "session-0"

    def test_delete_session(self, mock_context: MockContext):
        """Test deleting a session."""
        persistence = SessionPersistence(mock_context.settings)

        # Create a session file
        session_id = "to-delete"
        session_path = persistence._get_session_path(session_id)

        with open(session_path, "w") as f:
            json.dump(
                {
                    "session_id": session_id,
                    "created_at": "2024-01-01T00:00:00",
                    "saved_at": "2024-01-01T00:00:00",
                    "messages": [],
                },
                f,
            )

        assert session_path.exists()

        # Delete it
        result = persistence.delete_session(session_id)

        assert result is True
        assert not session_path.exists()

    def test_list_sessions_skips_invalid(self, mock_context: MockContext):
        """Test that list_sessions skips invalid JSON files."""
        persistence = SessionPersistence(mock_context.settings)

        # Create a valid session
        valid_path = persistence._get_session_path("valid")
        with open(valid_path, "w") as f:
            json.dump(
                {
                    "session_id": "valid",
                    "created_at": "2024-01-01T00:00:00",
                    "saved_at": "2024-01-01T00:00:00",
                    "messages": [],
                },
                f,
            )

        # Create an invalid session file
        invalid_path = persistence.sessions_path / "invalid.json"
        with open(invalid_path, "w") as f:
            f.write("not valid json{")

        sessions = persistence.list_sessions()

        # Should only return the valid session
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "valid"


# ============================================================================
# State Snapshot Tests
# ============================================================================


class TestToolCallRecord:
    """Tests for ToolCallRecord dataclass."""

    def test_create_record(self):
        """Test creating a tool call record."""
        record = ToolCallRecord(
            tool_name="search_kb",
            arguments={"query": "machine learning"},
            result={"matches": 5},
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            success=True,
        )

        assert record.tool_name == "search_kb"
        assert record.arguments == {"query": "machine learning"}
        assert record.result == {"matches": 5}
        assert record.success is True
        assert record.error is None

    def test_create_failed_record(self):
        """Test creating a failed tool call record."""
        record = ToolCallRecord(
            tool_name="web_search",
            arguments={"query": "test"},
            result=None,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            success=False,
            error="API rate limited",
        )

        assert record.success is False
        assert record.error == "API rate limited"

    def test_record_to_dict(self):
        """Test tool call record serialization."""
        record = ToolCallRecord(
            tool_name="execute",
            arguments={"code": "1+1"},
            result=2,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            success=True,
        )

        data = record.to_dict()

        assert data["tool_name"] == "execute"
        assert data["arguments"] == {"code": "1+1"}
        assert data["result"] == 2
        assert data["timestamp"] == "2024-01-01T12:00:00"
        assert data["success"] is True

    def test_record_from_dict(self):
        """Test tool call record deserialization."""
        data = {
            "tool_name": "search",
            "arguments": {"q": "test"},
            "result": [1, 2, 3],
            "timestamp": "2024-01-01T12:00:00",
            "success": True,
            "error": None,
        }

        record = ToolCallRecord.from_dict(data)

        assert record.tool_name == "search"
        assert record.arguments == {"q": "test"}
        assert record.result == [1, 2, 3]
        assert record.timestamp == datetime(2024, 1, 1, 12, 0, 0)


class TestStateSnapshot:
    """Tests for StateSnapshot dataclass."""

    def test_create_empty_snapshot(self):
        """Test creating empty state snapshot."""
        state = StateSnapshot()

        assert state.model is None
        assert state.thinking_effort is None
        assert state.tool_calls == []
        assert state.ingested_documents == []
        assert state.custom_state == {}
        assert state.checkpoint_name is None

    def test_create_full_snapshot(self):
        """Test creating full state snapshot."""
        state = StateSnapshot(
            model="gemini-3-flash-preview",
            thinking_effort="high",
            custom_state={"research_topic": "ML"},
            checkpoint_name="after_search",
        )

        assert state.model == "gemini-3-flash-preview"
        assert state.thinking_effort == "high"
        assert state.custom_state == {"research_topic": "ML"}
        assert state.checkpoint_name == "after_search"

    def test_add_tool_call(self):
        """Test adding tool calls."""
        state = StateSnapshot()

        state.add_tool_call(
            tool_name="search_kb",
            arguments={"query": "test"},
            result={"matches": 3},
        )

        assert len(state.tool_calls) == 1
        assert state.tool_calls[0].tool_name == "search_kb"
        assert state.tool_calls[0].success is True

    def test_add_failed_tool_call(self):
        """Test adding failed tool call."""
        state = StateSnapshot()

        state.add_tool_call(
            tool_name="web_search",
            arguments={"query": "test"},
            result=None,
            success=False,
            error="Connection timeout",
        )

        assert len(state.tool_calls) == 1
        assert state.tool_calls[0].success is False
        assert state.tool_calls[0].error == "Connection timeout"

    def test_add_ingested_document(self):
        """Test adding ingested documents."""
        state = StateSnapshot()

        state.add_ingested_document(
            document_id="doc-123",
            title="ML Survey",
            source_type="arxiv",
            source_url="https://arxiv.org/abs/1234",
        )

        assert len(state.ingested_documents) == 1
        assert state.ingested_documents[0]["document_id"] == "doc-123"
        assert state.ingested_documents[0]["title"] == "ML Survey"
        assert state.ingested_documents[0]["source_type"] == "arxiv"

    def test_state_to_dict(self):
        """Test state snapshot serialization."""
        state = StateSnapshot(
            model="claude-sonnet-4",
            thinking_effort="medium",
            checkpoint_name="checkpoint1",
        )
        state.add_tool_call("test_tool", {"arg": 1}, {"result": "ok"})

        data = state.to_dict()

        assert data["model"] == "claude-sonnet-4"
        assert data["thinking_effort"] == "medium"
        assert data["checkpoint_name"] == "checkpoint1"
        assert len(data["tool_calls"]) == 1

    def test_state_from_dict(self):
        """Test state snapshot deserialization."""
        data = {
            "model": "gemini-3-pro-preview",
            "thinking_effort": "low",
            "tool_calls": [
                {
                    "tool_name": "search",
                    "arguments": {},
                    "result": [],
                    "timestamp": "2024-01-01T12:00:00",
                    "success": True,
                    "error": None,
                }
            ],
            "ingested_documents": [
                {
                    "document_id": "doc-1",
                    "title": "Paper",
                    "source_type": "arxiv",
                    "source_url": None,
                    "ingested_at": "2024-01-01T12:00:00",
                }
            ],
            "custom_state": {"key": "value"},
            "checkpoint_name": "test",
        }

        state = StateSnapshot.from_dict(data)

        assert state.model == "gemini-3-pro-preview"
        assert state.thinking_effort == "low"
        assert len(state.tool_calls) == 1
        assert len(state.ingested_documents) == 1
        assert state.custom_state == {"key": "value"}
        assert state.checkpoint_name == "test"

    def test_state_roundtrip(self):
        """Test state snapshot serialization roundtrip."""
        original = StateSnapshot(
            model="test-model",
            thinking_effort="high",
            custom_state={"test": True},
        )
        original.add_tool_call("tool1", {"a": 1}, "result1")
        original.add_ingested_document("doc-1", "Title", "web")

        data = original.to_dict()
        restored = StateSnapshot.from_dict(data)

        assert restored.model == original.model
        assert restored.thinking_effort == original.thinking_effort
        assert len(restored.tool_calls) == len(original.tool_calls)
        assert len(restored.ingested_documents) == len(original.ingested_documents)


class TestSessionSnapshotWithState:
    """Tests for SessionSnapshot with StateSnapshot."""

    def test_snapshot_without_state(self):
        """Test session snapshot without state."""
        snapshot = SessionSnapshot(
            session_id="test",
            created_at=datetime(2024, 1, 1, 0, 0, 0),
            saved_at=datetime(2024, 1, 1, 0, 0, 0),
            messages=[],
        )

        assert snapshot.state is None

        data = snapshot.to_dict()
        assert "state" not in data

    def test_snapshot_with_state(self):
        """Test session snapshot with state."""
        state = StateSnapshot(model="test-model")
        snapshot = SessionSnapshot(
            session_id="test",
            created_at=datetime(2024, 1, 1, 0, 0, 0),
            saved_at=datetime(2024, 1, 1, 0, 0, 0),
            messages=[],
            state=state,
        )

        assert snapshot.state is not None
        assert snapshot.state.model == "test-model"

        data = snapshot.to_dict()
        assert "state" in data
        assert data["state"]["model"] == "test-model"

    def test_snapshot_with_state_roundtrip(self):
        """Test session snapshot with state roundtrip."""
        state = StateSnapshot(
            model="gemini-3-flash-preview",
            thinking_effort="medium",
        )
        state.add_tool_call("search", {}, [])

        original = SessionSnapshot(
            session_id="test",
            created_at=datetime(2024, 1, 1, 0, 0, 0),
            saved_at=datetime(2024, 1, 1, 0, 0, 0),
            messages=[{"content": "test"}],
            state=state,
        )

        data = original.to_dict()
        restored = SessionSnapshot.from_dict(data)

        assert restored.state is not None
        assert restored.state.model == "gemini-3-flash-preview"
        assert len(restored.state.tool_calls) == 1

    def test_load_legacy_snapshot_without_state(self):
        """Test loading legacy snapshot without state field."""
        data = {
            "session_id": "legacy",
            "created_at": "2024-01-01T00:00:00",
            "saved_at": "2024-01-01T00:00:00",
            "messages": [],
            "metadata": {},
            # No "state" field - legacy format
        }

        snapshot = SessionSnapshot.from_dict(data)

        assert snapshot.state is None
