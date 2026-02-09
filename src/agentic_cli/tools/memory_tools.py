"""Memory tools for agentic workflows.

Provides two tools for persistent memory:
- save_memory: Store information that persists across sessions
- search_memory: Search stored memories by substring

The MemoryStore is auto-created by the workflow manager when
these tools are used (via @requires("memory_manager")).

Example:
    from agentic_cli.tools import memory_tools

    AgentConfig(
        tools=[memory_tools.save_memory, memory_tools.search_memory],
    )
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agentic_cli.config import BaseSettings
from agentic_cli.logging import get_logger
from agentic_cli.persistence._utils import atomic_write_json
from agentic_cli.tools import requires, require_context

logger = get_logger("agentic_cli.tools.memory")
from agentic_cli.tools.registry import (
    register_tool,
    ToolCategory,
    PermissionLevel,
)
from agentic_cli.workflow.context import get_context_memory_store


# ---------------------------------------------------------------------------
# MemoryItem / MemoryStore – simple file-based memory persistence
# ---------------------------------------------------------------------------


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
                logger.warning("memory_load_failed", path=str(self._storage_path))
                self._items = {}

    def _save(self) -> None:
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        data = {"items": [item.to_dict() for item in self._items.values()]}
        atomic_write_json(self._storage_path, data)

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


@register_tool(
    category=ToolCategory.MEMORY,
    permission_level=PermissionLevel.SAFE,
    description="Save information to persistent memory that survives across sessions. Use this to remember user preferences, important facts, or learnings for future conversations.",
)
@requires("memory_manager")
@require_context("Memory store", get_context_memory_store)
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
    store = get_context_memory_store()
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
@requires("memory_manager")
@require_context("Memory store", get_context_memory_store)
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
    store = get_context_memory_store()
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
