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

from typing import Any

from agentic_cli.tools import requires
from agentic_cli.tools.registry import (
    register_tool,
    ToolCategory,
    PermissionLevel,
)
from agentic_cli.workflow.context import get_context_memory_manager


@register_tool(
    category=ToolCategory.MEMORY,
    permission_level=PermissionLevel.SAFE,
    description="Save information to persistent memory that survives across sessions. Use this to remember user preferences, important facts, or learnings for future conversations.",
)
@requires("memory_manager")
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
    store = get_context_memory_manager()
    if store is None:
        return {"success": False, "error": "Memory store not available"}

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
    store = get_context_memory_manager()
    if store is None:
        return {"success": False, "error": "Memory store not available"}

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
