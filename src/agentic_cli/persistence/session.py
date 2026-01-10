"""Session persistence for agentic CLI applications.

Manages saving and loading of session state including message history,
workflow state, and tool execution history.

State Snapshots:
    Sessions can include optional state snapshots that capture:
    - Workflow configuration (model, settings)
    - Tool execution history (calls and results)
    - Knowledge base ingestion records
    - Custom state data

Example:
    # Save with state snapshot
    state = StateSnapshot(
        model="gemini-3-flash-preview",
        thinking_effort="medium",
        tool_calls=[
            {"tool": "search_kb", "args": {"query": "ML"}, "result": {...}},
        ],
    )
    persistence.save_session(session_id, history, state=state)

    # Load and restore
    snapshot = persistence.load_session(session_id)
    if snapshot.state:
        print(f"Model used: {snapshot.state.model}")
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from agentic_cli.cli.app import MessageHistory
    from agentic_cli.config import BaseSettings

logger = Loggers.persistence()


@dataclass
class ToolCallRecord:
    """Record of a tool call during a session."""

    tool_name: str
    arguments: dict[str, Any]
    result: Any
    timestamp: datetime
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ToolCallRecord":
        """Create from dictionary."""
        return cls(
            tool_name=data["tool_name"],
            arguments=data["arguments"],
            result=data["result"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            success=data.get("success", True),
            error=data.get("error"),
        )


@dataclass
class StateSnapshot:
    """Snapshot of workflow state at a point in time.

    Captures configuration and execution context that can be used
    to understand or resume a session.
    """

    model: str | None = None
    """Model used during the session."""

    thinking_effort: str | None = None
    """Thinking effort level used."""

    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    """History of tool calls during the session."""

    ingested_documents: list[dict] = field(default_factory=list)
    """Documents ingested to knowledge base during session."""

    custom_state: dict[str, Any] = field(default_factory=dict)
    """Custom state data from domain applications."""

    checkpoint_name: str | None = None
    """Optional name for this checkpoint."""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model": self.model,
            "thinking_effort": self.thinking_effort,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "ingested_documents": self.ingested_documents,
            "custom_state": self.custom_state,
            "checkpoint_name": self.checkpoint_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StateSnapshot":
        """Create from dictionary."""
        return cls(
            model=data.get("model"),
            thinking_effort=data.get("thinking_effort"),
            tool_calls=[
                ToolCallRecord.from_dict(tc) for tc in data.get("tool_calls", [])
            ],
            ingested_documents=data.get("ingested_documents", []),
            custom_state=data.get("custom_state", {}),
            checkpoint_name=data.get("checkpoint_name"),
        )

    def add_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        result: Any,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Record a tool call."""
        self.tool_calls.append(
            ToolCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                timestamp=datetime.now(),
                success=success,
                error=error,
            )
        )

    def add_ingested_document(
        self,
        document_id: str,
        title: str,
        source_type: str,
        source_url: str | None = None,
    ) -> None:
        """Record a document ingested to knowledge base."""
        self.ingested_documents.append(
            {
                "document_id": document_id,
                "title": title,
                "source_type": source_type,
                "source_url": source_url,
                "ingested_at": datetime.now().isoformat(),
            }
        )


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
    state: StateSnapshot | None = None
    """Optional workflow state snapshot."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "saved_at": self.saved_at.isoformat(),
            "messages": self.messages,
            "metadata": self.metadata,
        }
        if self.state:
            result["state"] = self.state.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "SessionSnapshot":
        """Create from dictionary."""
        state = None
        if "state" in data and data["state"]:
            state = StateSnapshot.from_dict(data["state"])

        return cls(
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            saved_at=datetime.fromisoformat(data["saved_at"]),
            messages=data["messages"],
            metadata=data.get("metadata", {}),
            state=state,
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
        # Sanitize session_id for filesystem
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
        return self.sessions_path / f"{safe_id}.json"

    def save_session(
        self,
        session_id: str,
        message_history: "MessageHistory",
        metadata: dict | None = None,
        state: StateSnapshot | None = None,
    ) -> Path:
        """Save current session state.

        Args:
            session_id: Unique session identifier
            message_history: MessageHistory instance with messages
            metadata: Optional additional metadata to save
            state: Optional workflow state snapshot

        Returns:
            Path to saved session file
        """
        # Convert message history to serializable format
        messages = []
        all_messages = message_history.get_all()
        for msg in all_messages:
            messages.append(
                {
                    "content": msg.content,
                    "message_type": msg.message_type.value,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata,
                }
            )

        # Create snapshot
        snapshot = SessionSnapshot(
            session_id=session_id,
            created_at=all_messages[0].timestamp if all_messages else datetime.now(),
            saved_at=datetime.now(),
            messages=messages,
            metadata=metadata or {},
            state=state,
        )

        # Save to file
        session_path = self._get_session_path(session_id)
        with open(session_path, "w") as f:
            json.dump(snapshot.to_dict(), f, indent=2)

        logger.info(
            "session_saved",
            session_id=session_id,
            message_count=len(messages),
            has_state=state is not None,
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

    def restore_to_history(
        self, snapshot: SessionSnapshot, message_history: "MessageHistory"
    ) -> int:
        """Restore session messages to a MessageHistory instance.

        Args:
            snapshot: Session snapshot to restore
            message_history: MessageHistory instance to restore to

        Returns:
            Number of messages restored
        """
        from agentic_cli.cli.app import MessageType

        message_history.clear()

        for msg_data in snapshot.messages:
            message_history.add(
                content=msg_data["content"],
                message_type=MessageType(msg_data["message_type"]),
                timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                **msg_data.get("metadata", {}),
            )

        return len(snapshot.messages)

    def list_sessions(self) -> list[dict]:
        """List all saved sessions.

        Returns:
            List of session summaries (id, created_at, message_count)
        """
        sessions = []
        for session_file in self.sessions_path.glob("*.json"):
            try:
                with open(session_file, "r") as f:
                    data = json.load(f)
                sessions.append(
                    {
                        "session_id": data["session_id"],
                        "created_at": data["created_at"],
                        "saved_at": data["saved_at"],
                        "message_count": len(data["messages"]),
                    }
                )
            except (json.JSONDecodeError, KeyError):
                continue  # Skip invalid files

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
        if session_path.exists():
            session_path.unlink()
            return True
        return False
