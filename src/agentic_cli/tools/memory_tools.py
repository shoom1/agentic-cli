"""Memory tools for agentic workflows.

Provides two tools for persistent memory:
- save_memory: Store information that persists across sessions
- search_memory: Search stored memories by substring or semantic similarity

The MemoryStore is auto-created by the workflow manager when
these tools are used (detected via _TOOL_SERVICE_MAP).

Example:
    from agentic_cli.tools import memory_tools

    AgentConfig(
        tools=[memory_tools.save_memory, memory_tools.search_memory],
    )
"""

import json
import math
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
    Supports substring search and optional semantic search when an
    embedding_service is provided.

    Example:
        >>> store = MemoryStore(settings)
        >>> item_id = store.store("User prefers markdown output", tags=["preference"])
        >>> results = store.search("markdown")
        >>> print(results[0].content)
        User prefers markdown output
    """

    def __init__(self, settings: BaseSettings, embedding_service=None) -> None:
        self._settings = settings
        self._embedding_service = embedding_service
        mem_dir = settings.workspace_dir / "memory"
        mem_dir.mkdir(parents=True, exist_ok=True)
        self._path = mem_dir / "memories.json"
        self._embeddings_path = mem_dir / "memories_embeddings.json"
        self._items: dict[str, MemoryItem] = {}
        self._load()
        if self._embedding_service:
            self._ensure_embeddings()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                # Support both flat list (new format) and {"items": [...]} (old format)
                if isinstance(data, dict):
                    items_list = data.get("items", [])
                else:
                    items_list = data
                for item_data in items_list:
                    item = MemoryItem.from_dict(item_data)
                    self._items[item.id] = item
            except (json.JSONDecodeError, KeyError):
                logger.warning("corrupted_memory_file", path=str(self._path))
                self._items = {}
        if self._embeddings_path.exists():
            try:
                emb_data = json.loads(self._embeddings_path.read_text())
                for item_id, embedding in emb_data.items():
                    if item_id in self._items:
                        self._items[item_id].embedding = embedding
            except (json.JSONDecodeError, KeyError):
                logger.warning("corrupted_embeddings_file", path=str(self._embeddings_path))

    def _save(self) -> None:
        data = [item.to_dict() for item in self._items.values()]
        atomic_write_json(self._path, data)
        self._save_embeddings()

    def _save_embeddings(self) -> None:
        emb_data = {}
        for item_id, item in self._items.items():
            if item.embedding is not None:
                emb_data[item_id] = item.embedding
        if emb_data:
            atomic_write_json(self._embeddings_path, emb_data)

    def _ensure_embeddings(self) -> None:
        """Batch-embed any items missing embeddings."""
        to_embed = [item for item in self._items.values() if item.embedding is None]
        if not to_embed:
            return
        texts = [item.content for item in to_embed]
        embeddings = self._embedding_service.embed_batch(texts)
        for item, emb in zip(to_embed, embeddings):
            item.embedding = emb
        self._save_embeddings()

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
        if self._embedding_service:
            item.embedding = self._embedding_service.embed_text(content)
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
        """Search memories by substring or semantic similarity.

        When an embedding_service is available and items have embeddings,
        uses cosine similarity with recency-importance-relevance scoring.
        Otherwise falls back to case-insensitive substring match.

        Args:
            query: The search query. Empty string matches all (substring mode).
            limit: Maximum results to return.
            include_archived: If True, include archived (soft-deleted) items.

        Returns:
            List of matching MemoryItem objects.
        """
        candidates = [
            item for item in self._items.values()
            if include_archived or not item.archived
        ]
        if not candidates:
            return []
        if self._embedding_service and any(item.embedding for item in candidates):
            results = self._semantic_search(query, candidates, limit)
        else:
            results = self._substring_search(query, candidates, limit)
        # Update access tracking
        now = datetime.now().isoformat()
        for item in results:
            item.access_count += 1
            item.last_accessed_at = now
        self._save()
        return results

    def _substring_search(self, query: str, candidates: list[MemoryItem], limit: int) -> list[MemoryItem]:
        """Case-insensitive substring search."""
        q = query.lower()
        if not q:
            return candidates[:limit]
        return [item for item in candidates if q in item.content.lower()][:limit]

    def _semantic_search(self, query: str, candidates: list[MemoryItem], limit: int) -> list[MemoryItem]:
        """Cosine similarity search scored with recency and importance."""
        query_embedding = self._embedding_service.embed_text(query)
        now = datetime.now()
        scored = []
        for item in candidates:
            if item.embedding is None:
                continue
            relevance = self._cosine_similarity(query_embedding, item.embedding)
            try:
                last_access = datetime.fromisoformat(item.last_accessed_at)
                hours_since = max(0, (now - last_access).total_seconds() / 3600)
            except (ValueError, TypeError):
                hours_since = 0
            recency = math.exp(-0.01 * hours_since)  # decay=0.01, ~3 day half-life
            importance_norm = item.importance / 10.0
            score = 0.7 * relevance + 0.15 * recency + 0.15 * importance_norm
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

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
