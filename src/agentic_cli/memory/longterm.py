"""Long-term memory for persistent storage across sessions.

Provides persistent storage of facts, preferences, learnings, and references
that survive across sessions. Data is stored in JSON format in the workspace.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from agentic_cli.config import BaseSettings


class MemoryType(Enum):
    """Types of long-term memory entries."""

    FACT = "fact"
    PREFERENCE = "preference"
    LEARNING = "learning"
    REFERENCE = "reference"


@dataclass
class MemoryEntry:
    """A long-term memory entry with metadata and tracking.

    Attributes:
        id: Unique identifier for the entry.
        type: Type of memory (FACT, PREFERENCE, LEARNING, REFERENCE).
        content: The actual content stored.
        source: Session or document that created this entry.
        kb_references: List of knowledge base document IDs related to this entry.
        tags: List of tags for categorization.
        created_at: When the entry was created.
        accessed_at: When the entry was last accessed.
        access_count: Number of times the entry has been accessed.
        confidence: Confidence level of the information (0.0 to 1.0).
    """

    id: str
    type: MemoryType
    content: str
    source: str
    kb_references: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to dictionary for serialization.

        Returns:
            Dictionary representation of the entry.
        """
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "source": self.source,
            "kb_references": self.kb_references,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        """Create entry from dictionary.

        Args:
            data: Dictionary previously created by to_dict().

        Returns:
            A new MemoryEntry instance.
        """
        return cls(
            id=data["id"],
            type=MemoryType(data["type"]),
            content=data["content"],
            source=data["source"],
            kb_references=data.get("kb_references", []),
            tags=data.get("tags", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            accessed_at=datetime.fromisoformat(data["accessed_at"]),
            access_count=data.get("access_count", 0),
            confidence=data.get("confidence", 1.0),
        )


class LongTermMemory:
    """Persistent memory storage that survives across sessions.

    Stores facts, preferences, learnings, and references in a JSON file
    within the workspace directory. Supports search by query, type, and tags.

    Example:
        >>> from agentic_cli.memory import LongTermMemory, MemoryType
        >>> memory = LongTermMemory(settings)
        >>> entry_id = memory.store(
        ...     type=MemoryType.FACT,
        ...     content="Basel III requires 99% confidence",
        ...     source="session_1"
        ... )
        >>> results = memory.recall("Basel")
        >>> print(results[0].content)
        Basel III requires 99% confidence
    """

    def __init__(self, settings: BaseSettings) -> None:
        """Initialize long-term memory with settings.

        Args:
            settings: BaseSettings instance providing workspace_dir.
        """
        self._settings = settings
        self._memory_dir = settings.workspace_dir / "memory"
        self._storage_path = self._memory_dir / "longterm.json"
        self._entries: dict[str, MemoryEntry] = {}
        self._load()

    def _load(self) -> None:
        """Load entries from storage file."""
        if self._storage_path.exists():
            try:
                with open(self._storage_path, "r") as f:
                    data = json.load(f)
                    for entry_data in data.get("entries", []):
                        entry = MemoryEntry.from_dict(entry_data)
                        self._entries[entry.id] = entry
            except (json.JSONDecodeError, KeyError):
                # Corrupted file, start fresh
                self._entries = {}

    def _save(self) -> None:
        """Save entries to storage file."""
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        data = {"entries": [entry.to_dict() for entry in self._entries.values()]}
        with open(self._storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def store(
        self,
        type: MemoryType,
        content: str,
        source: str,
        kb_references: list[str] | None = None,
        tags: list[str] | None = None,
        confidence: float = 1.0,
    ) -> str:
        """Store a new memory entry.

        Args:
            type: Type of memory entry.
            content: The content to store.
            source: Session or document that created this entry.
            kb_references: Optional list of knowledge base document IDs.
            tags: Optional list of tags for categorization.
            confidence: Confidence level (0.0 to 1.0), defaults to 1.0.

        Returns:
            The unique ID of the created entry.
        """
        entry_id = str(uuid.uuid4())
        now = datetime.now()
        entry = MemoryEntry(
            id=entry_id,
            type=type,
            content=content,
            source=source,
            kb_references=kb_references or [],
            tags=tags or [],
            created_at=now,
            accessed_at=now,
            access_count=0,
            confidence=confidence,
        )
        self._entries[entry_id] = entry
        self._save()
        return entry_id

    def recall(
        self,
        query: str,
        type: MemoryType | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Search for memory entries matching criteria.

        Performs simple substring search on content. Can filter by type and tags.

        Args:
            query: Substring to search for in content (empty string matches all).
            type: Optional filter by memory type.
            tags: Optional filter by tags (entries must have ALL specified tags).
            limit: Maximum number of results to return.

        Returns:
            List of matching entries, up to limit.
        """
        results: list[MemoryEntry] = []
        query_lower = query.lower()

        for entry in self._entries.values():
            # Filter by type if specified
            if type is not None and entry.type != type:
                continue

            # Filter by tags if specified (must have ALL tags)
            if tags is not None:
                entry_tags_set = set(entry.tags)
                if not set(tags).issubset(entry_tags_set):
                    continue

            # Filter by query (substring search, case-insensitive)
            if query and query_lower not in entry.content.lower():
                continue

            results.append(entry)

            if len(results) >= limit:
                break

        return results

    def get(self, entry_id: str) -> MemoryEntry | None:
        """Retrieve a specific entry by ID.

        Updates the access tracking (accessed_at and access_count).

        Args:
            entry_id: The unique ID of the entry.

        Returns:
            The entry if found, None otherwise.
        """
        entry = self._entries.get(entry_id)
        if entry is not None:
            entry.accessed_at = datetime.now()
            entry.access_count += 1
            self._save()
        return entry

    def update(self, entry_id: str, **kwargs: Any) -> None:
        """Update an existing entry.

        Args:
            entry_id: The unique ID of the entry to update.
            **kwargs: Fields to update (content, tags, confidence, kb_references).

        Raises:
            KeyError: If entry_id is not found.
        """
        if entry_id not in self._entries:
            raise KeyError(f"Entry not found: {entry_id}")

        entry = self._entries[entry_id]

        # Update allowed fields
        if "content" in kwargs:
            entry.content = kwargs["content"]
        if "tags" in kwargs:
            entry.tags = kwargs["tags"]
        if "confidence" in kwargs:
            entry.confidence = kwargs["confidence"]
        if "kb_references" in kwargs:
            entry.kb_references = kwargs["kb_references"]

        self._save()

    def forget(self, entry_id: str) -> None:
        """Remove an entry from memory.

        Args:
            entry_id: The unique ID of the entry to remove.
                     No error if entry doesn't exist.
        """
        if entry_id in self._entries:
            del self._entries[entry_id]
            self._save()

    def get_preferences(self) -> list[MemoryEntry]:
        """Get all preference entries.

        Convenience method for retrieving user preferences.

        Returns:
            List of all PREFERENCE type entries.
        """
        return [
            entry
            for entry in self._entries.values()
            if entry.type == MemoryType.PREFERENCE
        ]
