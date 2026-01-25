"""Working memory for session-scoped key-value storage."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryEntry:
    """A memory entry with value and optional tags."""

    value: Any
    tags: set[str] = field(default_factory=set)


class WorkingMemory:
    """Session-scoped key-value store with tag-based filtering.

    Provides a simple in-memory storage for session data with support
    for tagging entries and filtering by tags.

    Example:
        >>> memory = WorkingMemory()
        >>> memory.set("paper1", {"title": "ML Paper"}, tags=["research", "ml"])
        >>> memory.get("paper1")
        {'title': 'ML Paper'}
        >>> memory.list(tags=["research"])
        ['paper1']
    """

    def __init__(self) -> None:
        """Initialize an empty working memory."""
        self._entries: dict[str, MemoryEntry] = {}

    def set(self, key: str, value: Any, tags: list[str] | None = None) -> None:
        """Store a value with optional tags.

        Args:
            key: The key to store the value under.
            value: The value to store (can be any type).
            tags: Optional list of tags to associate with this entry.
        """
        tag_set = set(tags) if tags else set()
        self._entries[key] = MemoryEntry(value=value, tags=tag_set)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by key.

        Args:
            key: The key to retrieve.
            default: Value to return if key is not found.

        Returns:
            The stored value or the default if key doesn't exist.
        """
        entry = self._entries.get(key)
        if entry is None:
            return default
        return entry.value

    def list(self, tags: list[str] | None = None) -> list[str]:
        """List keys, optionally filtered by tags.

        Args:
            tags: If provided, only return keys that have ALL specified tags.

        Returns:
            List of matching keys.
        """
        if tags is None:
            return list(self._entries.keys())

        required_tags = set(tags)
        return [
            key
            for key, entry in self._entries.items()
            if required_tags.issubset(entry.tags)
        ]

    def delete(self, key: str) -> None:
        """Remove a key from memory.

        Args:
            key: The key to remove. No error if key doesn't exist.
        """
        self._entries.pop(key, None)

    def clear(self) -> None:
        """Clear all entries from memory."""
        self._entries.clear()

    def to_snapshot(self) -> dict[str, Any]:
        """Export memory state for session persistence.

        Returns:
            A dictionary representation of the memory state that can be
            serialized and later restored with from_snapshot().
        """
        return {
            "entries": {
                key: {"value": entry.value, "tags": list(entry.tags)}
                for key, entry in self._entries.items()
            }
        }

    @classmethod
    def from_snapshot(cls, data: dict[str, Any]) -> "WorkingMemory":
        """Restore memory from a persistence snapshot.

        Args:
            data: A dictionary previously created by to_snapshot().

        Returns:
            A new WorkingMemory instance with restored state.
        """
        memory = cls()
        entries = data.get("entries", {})
        for key, entry_data in entries.items():
            memory.set(
                key, entry_data["value"], tags=entry_data.get("tags", [])
            )
        return memory
