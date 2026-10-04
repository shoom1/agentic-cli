"""Memory tools for agentic workflows.

``save_memory``, ``search_memory``, ``update_memory`` and ``delete_memory``
over the workflow's ``MemoryStore`` (``agentic_cli.memory``), which the
workflow creates when an agent has these tools.

Example:
    from agentic_cli.tools import memory_tools

    AgentConfig(
        tools=[memory_tools.save_memory, memory_tools.search_memory],
    )
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from agentic_cli.memory import ForgettingPolicy, MemoryItem
from agentic_cli.memory import MemoryStore as _MemoryStore
from agentic_cli.tools.registry import ToolCategory, register_tool
from agentic_cli.workflow.permissions import Capability
from agentic_cli.workflow.service_registry import MEMORY_STORE, require_service

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings

__all__ = [
    "ForgettingPolicy",
    "MemoryItem",
    "MemoryStore",
    "delete_memory",
    "save_memory",
    "search_memory",
    "update_memory",
]


class MemoryStore(_MemoryStore):
    """Deprecated: use ``agentic_cli.memory.MemoryStore(base_dir)``. Removed in 0.7.0."""

    def __init__(self, settings: "BaseSettings", embedding_service=None) -> None:
        warnings.warn(
            "agentic_cli.tools.memory_tools.MemoryStore(settings) is deprecated and "
            "will be removed in 0.7.0; use "
            "agentic_cli.memory.MemoryStore(settings.workspace_dir / 'memory')",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(settings.workspace_dir / "memory", embedding_service=embedding_service)


# ---------------------------------------------------------------------------
# Shared helpers — the single source of truth for tool behavior.
#
# Both the @register_tool wrappers below (service-registry bound) and the
# make_memory_tools() closures in tools/factories.py delegate to these
# helpers, so the two entry points cannot drift.
# ---------------------------------------------------------------------------


def _save_memory_with_store(
    store: "MemoryStore",
    content: str,
    tags: list[str] | None,
    importance: int,
) -> dict[str, Any]:
    result = store.store_with_similarity_check(
        content, tags=tags, importance=importance
    )
    return {
        "success": True,
        "item_id": result["item_id"],
        "message": "Saved to persistent memory",
        "similar_existing": result["similar_existing"],
    }


def _search_memory_with_store(
    store: "MemoryStore",
    query: str,
    limit: int,
    include_archived: bool,
) -> dict[str, Any]:
    results = store.search(query, limit=limit, include_archived=include_archived)
    items = [
        {
            "id": item.id,
            "content": item.content,
            "tags": item.tags,
            "importance": item.importance,
        }
        for item in results
    ]
    return {
        "success": True,
        "query": query,
        "items": items,
        "count": len(items),
    }


def _update_memory_with_store(
    store: "MemoryStore",
    item_id: str,
    content: str | None,
    tags: list[str] | None,
) -> dict[str, Any]:
    # Tool contract: None (or omitted) leaves tags alone, [] clears them.
    # Models often send null for an optional argument they mean to omit, so
    # null must not wipe tags; and the declaration default must be plain JSON.
    if tags is None:
        updated = store.update(item_id, content=content)
    else:
        updated = store.update(item_id, content=content, tags=tags or None)
    return {"success": True, "updated": updated}


def _delete_memory_with_store(
    store: "MemoryStore",
    item_id: str,
    purge: bool,
) -> dict[str, Any]:
    deleted = store.delete(item_id, purge=purge)
    return {"success": True, "deleted": deleted}


# ---------------------------------------------------------------------------
# Registry-bound tool functions (@register_tool) — thin wrappers that resolve
# the store through the service registry and delegate to the helpers above.
# ---------------------------------------------------------------------------


@register_tool(
    category=ToolCategory.MEMORY,
    capabilities=[Capability("memory.write")],
    description="Save information to persistent memory that survives across sessions. Use this to remember user preferences, important facts, or learnings for future conversations.",
    requires="memory_store",
)
def save_memory(
    content: str,
    tags: list[str] | None = None,
    importance: int = 5,
) -> dict[str, Any]:
    """Save information to persistent memory.

    Args:
        content: The content to store.
        tags: Optional tags for categorization.
        importance: Importance rating 1-10 (default 5).
    """
    store = require_service(MEMORY_STORE)
    if isinstance(store, dict):
        return store
    return _save_memory_with_store(store, content, tags, importance)


@register_tool(
    category=ToolCategory.MEMORY,
    capabilities=[Capability("memory.read")],
    description="Search persistent memory by keyword/substring. Use this to recall previously saved facts, preferences, or learnings.",
    requires="memory_store",
)
def search_memory(
    query: str,
    limit: int = 10,
    include_archived: bool = False,
) -> dict[str, Any]:
    """Search persistent memory for stored information.

    Args:
        query: The search query (substring match, case-insensitive).
        limit: Maximum number of results to return.
        include_archived: If True, include archived (soft-deleted) memories.
    """
    store = require_service(MEMORY_STORE)
    if isinstance(store, dict):
        return store
    return _search_memory_with_store(store, query, limit, include_archived)


@register_tool(
    category=ToolCategory.MEMORY,
    capabilities=[Capability("memory.write")],
    description="Update an existing memory item",
    requires="memory_store",
)
def update_memory(
    item_id: str,
    content: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Update an existing memory item.

    Args:
        item_id: ID of the memory to update.
        content: New content (optional).
        tags: New tags. Omit to leave unchanged; pass an empty list to clear.
    """
    store = require_service(MEMORY_STORE)
    if isinstance(store, dict):
        return store
    return _update_memory_with_store(store, item_id, content, tags)


@register_tool(
    category=ToolCategory.MEMORY,
    capabilities=[Capability("memory.write")],
    description="Delete a memory item",
    requires="memory_store",
)
def delete_memory(
    item_id: str,
    purge: bool = False,
) -> dict[str, Any]:
    """Delete a memory item (soft-delete by default).

    Args:
        item_id: ID of the memory to delete.
        purge: If True, permanently remove. If False, archive.
    """
    store = require_service(MEMORY_STORE)
    if isinstance(store, dict):
        return store
    return _delete_memory_with_store(store, item_id, purge)
