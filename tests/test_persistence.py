"""Tests for session persistence."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agentic_cli.persistence.session import (
    SessionPersistence,
    SessionSnapshot,
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
