"""Memory tools for agentic workflows.

Provides two tools for persistent memory:
- save_memory: Store information that persists across sessions
- search_memory: Search stored memories by substring

The MemoryStore is auto-created by the workflow manager when
these tools are used (detected via _TOOL_SERVICE_MAP).

Example:
    from agentic_cli.tools import memory_tools

    AgentConfig(
        tools=[memory_tools.save_memory, memory_tools.search_memory],
    )
"""

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from agentic_cli.config import BaseSettings
from agentic_cli.logging import get_logger
from agentic_cli.file_utils import atomic_write_json

logger = get_logger("agentic_cli.tools.memory")
from agentic_cli.tools.registry import (
    register_tool,
    ToolCategory,
    PermissionLevel,
)
from agentic_cli.workflow.service_registry import require_service, MEMORY_STORE


# ---------------------------------------------------------------------------
# MemoryItem / MemoryStore – simple file-based memory persistence
# ---------------------------------------------------------------------------

_SENTINEL = object()


@dataclass
class MemoryItem:
    """A single memory entry."""

    id: str
    content: str
    tags: list[str] | None = None
    created_at: str = ""
    updated_at: str = ""
    last_accessed_at: str = ""
    access_count: int = 0
    importance: int = 5
    embedding: list[float] | None = None
    archived: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict. Excludes embedding (stored separately)."""
        return {
            "id": self.id,
            "content": self.content,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "importance": self.importance,
            "archived": self.archived,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryItem":
        """Deserialize from dict. Backward-compatible with old format."""
        created_at = data.get("created_at", "")
        return cls(
            id=data["id"],
            content=data["content"],
            tags=data.get("tags"),
            created_at=created_at,
            updated_at=data.get("updated_at", created_at),
            last_accessed_at=data.get("last_accessed_at", created_at),
            access_count=data.get("access_count", 0),
            importance=data.get("importance", 5),
            embedding=data.get("embedding"),
            archived=data.get("archived", False),
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
                logger.warning("memory_load_failed", path=str(self._storage_path))
                self._items = {}

    def _save(self) -> None:
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        data = {"items": [item.to_dict() for item in self._items.values()]}
        atomic_write_json(self._storage_path, data)

    def store(self, content: str, tags: list[str] | None = None, importance: int = 5) -> str:
        """Append a memory to the persistent store.

        Args:
            content: The text content to remember.
            tags: Optional tags for categorization.
            importance: Importance level from 1-10 (default 5).

        Returns:
            The unique ID of the stored memory.
        """
        now = datetime.now().isoformat()
        item = MemoryItem(
            id=str(uuid.uuid4()),
            content=content,
            tags=tags,
            created_at=now,
            updated_at=now,
            last_accessed_at=now,
            access_count=0,
            importance=max(1, min(10, importance)),
        )
        self._items[item.id] = item
        self._save()
        return item.id

    def update(self, item_id: str, content: str | None = None, tags: list[str] | None = _SENTINEL) -> bool:
        """Update an existing memory item.

        Args:
            item_id: The ID of the memory to update.
            content: New content, or None to leave unchanged.
            tags: New tags, or _SENTINEL to leave unchanged. Pass None to clear tags.

        Returns:
            True if updated, False if item not found.
        """
        item = self._items.get(item_id)
        if item is None:
            return False
        if content is not None:
            item.content = content
        if tags is not _SENTINEL:
            item.tags = tags
        item.updated_at = datetime.now().isoformat()
        item.embedding = None  # invalidate cached embedding
        self._save()
        return True

    def delete(self, item_id: str, purge: bool = False) -> bool:
        """Delete or archive a memory item.

        Args:
            item_id: The ID of the memory to delete.
            purge: If True, permanently remove. If False (default), soft-delete (archive).

        Returns:
            True if deleted/archived, False if item not found.
        """
        item = self._items.get(item_id)
        if item is None:
            return False
        if purge:
            del self._items[item_id]
        else:
            item.archived = True
        self._save()
        return True

    def search(self, query: str, limit: int = 10, include_archived: bool = False) -> list[MemoryItem]:
        """Search memories by substring match (case-insensitive).

        Args:
            query: Substring to search for. Empty string matches all.
            limit: Maximum results to return.
            include_archived: If True, include archived (soft-deleted) items.

        Returns:
            List of matching MemoryItem objects.
        """
        results = []
        q = query.lower()
        for item in self._items.values():
            if not include_archived and item.archived:
                continue
            if not q or q in item.content.lower():
                results.append(item)
        return results[:limit]

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


@register_tool(
    category=ToolCategory.MEMORY,
    permission_level=PermissionLevel.SAFE,
    description="Save information to persistent memory that survives across sessions. Use this to remember user preferences, important facts, or learnings for future conversations.",
)
def save_memory(
    content: str,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Save information to persistent memory.

    Use this to remember important facts, preferences, or learnings
    that should persist across sessions.

    Args:
        content: The content to store.
        tags: Optional tags for categorization.

    Returns:
        A dict with the stored item ID.
    """
    store = require_service(MEMORY_STORE)
    if isinstance(store, dict):
        return store
    item_id = store.store(content, tags=tags)
    return {
        "success": True,
        "item_id": item_id,
        "message": "Saved to persistent memory",
    }


@register_tool(
    category=ToolCategory.MEMORY,
    permission_level=PermissionLevel.SAFE,
    description="Search persistent memory by keyword/substring. Use this to recall previously saved facts, preferences, or learnings.",
)
def search_memory(
    query: str,
    limit: int = 10,
) -> dict[str, Any]:
    """Search persistent memory for stored information.

    Args:
        query: The search query (substring match, case-insensitive).
        limit: Maximum number of results to return.

    Returns:
        A dict with matching memory items.
    """
    store = require_service(MEMORY_STORE)
    if isinstance(store, dict):
        return store
    results = store.search(query, limit=limit)
    items = [
        {
            "id": item.id,
            "content": item.content,
            "tags": item.tags,
        }
        for item in results
    ]

    return {
        "success": True,
        "query": query,
        "items": items,
        "count": len(items),
    }
