"""Session persistence for agentic CLI applications.

Manages saving and loading of session state including message history.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentic_cli.logging import Loggers
from agentic_cli.persistence._utils import atomic_write_json, file_lock, sanitize_filename

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings

logger = Loggers.persistence()


@dataclass
class SessionSnapshot:
    """A snapshot of session state that can be persisted.

    Contains message history and optional workflow state.
    """

    session_id: str
    created_at: datetime
    saved_at: datetime
    messages: list[dict]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "saved_at": self.saved_at.isoformat(),
            "messages": self.messages,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionSnapshot":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            saved_at=datetime.fromisoformat(data["saved_at"]),
            messages=data["messages"],
            metadata=data.get("metadata", {}),
        )


class SessionPersistence:
    """Manages session state persistence.

    Sessions are saved to:
        {workspace_dir}/sessions/
        ├── {session_id}.json
        └── ...
    """

    def __init__(self, settings: "BaseSettings"):
        """Initialize session persistence.

        Args:
            settings: Application settings with workspace_dir
        """
        self.sessions_path = settings.sessions_dir
        self._ensure_directory_exists()

    def _ensure_directory_exists(self) -> None:
        """Create sessions directory if it doesn't exist."""
        self.sessions_path.mkdir(parents=True, exist_ok=True)

    def _get_session_path(self, session_id: str) -> Path:
        """Get path for a session file."""
        safe_id = sanitize_filename(session_id)
        return self.sessions_path / f"{safe_id}.json"

    def _get_sessions_index_path(self) -> Path:
        """Get path to sessions index file."""
        return self.sessions_path / "_sessions_index.json"

    def _load_sessions_index(self) -> dict[str, dict]:
        """Load sessions index from disk.

        Returns:
            Dict mapping session_id to metadata dict.
        """
        index_path = self._get_sessions_index_path()
        if index_path.exists():
            try:
                data = json.loads(index_path.read_text())
                return data.get("sessions", {})
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("sessions_index_load_failed", error=str(exc))
        return {}

    def _save_sessions_index(self, index: dict[str, dict]) -> None:
        """Save sessions index to disk."""
        atomic_write_json(self._get_sessions_index_path(), {"sessions": index})

    def _update_sessions_index(
        self,
        session_id: str,
        created_at: str,
        saved_at: str,
        message_count: int,
    ) -> None:
        """Update sessions index with metadata for a single session."""
        with file_lock(self._get_sessions_index_path()):
            sessions_index = self._load_sessions_index()
            sessions_index[session_id] = {
                "session_id": session_id,
                "created_at": created_at,
                "saved_at": saved_at,
                "message_count": message_count,
            }
            self._save_sessions_index(sessions_index)

    def _rebuild_sessions_index(self) -> dict[str, dict]:
        """Rebuild sessions index from individual session files (migration)."""
        index: dict[str, dict] = {}
        for session_file in self.sessions_path.glob("*.json"):
            if session_file.name.startswith("_"):
                continue
            try:
                with open(session_file, "r") as f:
                    data = json.load(f)
                sid = data["session_id"]
                index[sid] = {
                    "session_id": sid,
                    "created_at": data["created_at"],
                    "saved_at": data["saved_at"],
                    "message_count": len(data["messages"]),
                }
            except (json.JSONDecodeError, KeyError):
                continue
        self._save_sessions_index(index)
        return index

    def save_snapshot(self, snapshot: SessionSnapshot) -> Path:
        """Save a SessionSnapshot to disk.

        Args:
            snapshot: The session snapshot to persist.

        Returns:
            Path to saved session file.
        """
        session_path = self._get_session_path(snapshot.session_id)
        atomic_write_json(session_path, snapshot.to_dict())

        self._update_sessions_index(
            session_id=snapshot.session_id,
            created_at=snapshot.created_at.isoformat(),
            saved_at=snapshot.saved_at.isoformat(),
            message_count=len(snapshot.messages),
        )

        logger.info(
            "snapshot_saved",
            session_id=snapshot.session_id,
            message_count=len(snapshot.messages),
            path=str(session_path),
        )
        return session_path

    def load_session(self, session_id: str) -> SessionSnapshot | None:
        """Load a saved session.

        Args:
            session_id: Session identifier to load

        Returns:
            SessionSnapshot if found, None otherwise
        """
        session_path = self._get_session_path(session_id)
        if not session_path.exists():
            logger.debug("session_not_found", session_id=session_id)
            return None

        with open(session_path, "r") as f:
            data = json.load(f)

        snapshot = SessionSnapshot.from_dict(data)
        logger.info(
            "session_loaded",
            session_id=session_id,
            message_count=len(snapshot.messages),
        )
        return snapshot

    def list_sessions(self) -> list[dict]:
        """List all saved sessions.

        Returns:
            List of session summaries (id, created_at, message_count)
        """
        # Try reading from sessions index first
        sessions_index = self._load_sessions_index()
        if not sessions_index:
            # Fallback: rebuild from files (migration path)
            sessions_index = self._rebuild_sessions_index()

        sessions = list(sessions_index.values())

        # Sort by saved_at descending
        sessions.sort(key=lambda x: x["saved_at"], reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a saved session.

        Args:
            session_id: Session identifier to delete

        Returns:
            True if deleted, False if not found
        """
        session_path = self._get_session_path(session_id)
        try:
            session_path.unlink()
        except FileNotFoundError:
            return False

        # Update sessions index (lock to prevent TOCTOU race)
        with file_lock(self._get_sessions_index_path()):
            sessions_index = self._load_sessions_index()
            sessions_index.pop(session_id, None)
            self._save_sessions_index(sessions_index)

        return True
