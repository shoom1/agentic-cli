"""Simple file-based memory store.

Persistent storage following the Claude Code / Gemini CLI pattern:
a single append-to-file store with substring search. The LLM's
context window serves as working memory; this module handles
long-term persistence only.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agentic_cli.config import BaseSettings


@dataclass
class MemoryItem:
    """A single persistent memory entry."""

    id: str
    content: str
    tags: list[str] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "tags": self.tags,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryItem":
        return cls(
            id=data["id"],
            content=data["content"],
            tags=data.get("tags", []),
            created_at=data.get("created_at", ""),
        )


class MemoryStore:
    """Simple persistent memory store.

    Appends memories to a JSON file in the workspace directory.
    Supports substring search and optional tags.

    Example:
        >>> store = MemoryStore(settings)
        >>> item_id = store.store("User prefers markdown output", tags=["preference"])
        >>> results = store.search("markdown")
        >>> print(results[0].content)
        User prefers markdown output
    """

    def __init__(self, settings: BaseSettings) -> None:
        self._settings = settings
        self._memory_dir = settings.workspace_dir / "memory"
        self._storage_path = self._memory_dir / "memories.json"
        self._items: dict[str, MemoryItem] = {}
        self._load()

    def _load(self) -> None:
        if self._storage_path.exists():
            try:
                with open(self._storage_path, "r") as f:
                    data = json.load(f)
                for item_data in data.get("items", []):
                    item = MemoryItem.from_dict(item_data)
                    self._items[item.id] = item
            except (json.JSONDecodeError, KeyError):
                self._items = {}

    def _save(self) -> None:
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        data = {"items": [item.to_dict() for item in self._items.values()]}
        tmp_path = self._storage_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        tmp_path.rename(self._storage_path)

    def store(self, content: str, tags: list[str] | None = None) -> str:
        """Append a memory to the persistent store.

        Args:
            content: The text content to remember.
            tags: Optional tags for categorization.

        Returns:
            The unique ID of the stored memory.
        """
        item_id = str(uuid.uuid4())
        item = MemoryItem(
            id=item_id,
            content=content,
            tags=tags or [],
            created_at=datetime.now().isoformat(),
        )
        self._items[item_id] = item
        self._save()
        return item_id

    def search(self, query: str, limit: int = 10) -> list[MemoryItem]:
        """Search memories by substring match (case-insensitive).

        Args:
            query: Substring to search for. Empty string matches all.
            limit: Maximum results to return.

        Returns:
            List of matching MemoryItem objects.
        """
        query_lower = query.lower()
        results: list[MemoryItem] = []
        for item in self._items.values():
            if not query or query_lower in item.content.lower():
                results.append(item)
                if len(results) >= limit:
                    break
        return results

    def load_all(self) -> str:
        """Load all memories as a formatted string for system prompt injection.

        Returns:
            Formatted string of all memories, or empty string if none.
        """
        if not self._items:
            return ""
        lines = []
        for item in self._items.values():
            tag_str = f" [{', '.join(item.tags)}]" if item.tags else ""
            lines.append(f"- {item.content}{tag_str}")
        return "\n".join(lines)
