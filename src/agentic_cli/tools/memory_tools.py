"""Memory tools for agentic workflows.

These tools provide working memory (session-scoped) and long-term memory
(persistent) capabilities. The MemoryManager is auto-created by the
workflow manager when these tools are used.

Example:
    from agentic_cli.tools import memory_tools

    # In agent config
    AgentConfig(
        tools=[memory_tools.remember_context, memory_tools.recall_context, ...],
    )
"""

from typing import Any

from agentic_cli.tools import requires
from agentic_cli.workflow.context import get_context_memory_manager


@requires("memory_manager")
def remember_context(
    key: str,
    value: str,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Store context in working memory for the current session.

    Use this to remember important context like the current research topic,
    user preferences, or intermediate results.

    Args:
        key: A unique identifier for this context (e.g., "current_topic").
        value: The value to store.
        tags: Optional tags for categorization.

    Returns:
        A dict with success status and stored key.
    """
    manager = get_context_memory_manager()
    if manager is None:
        return {"success": False, "error": "Memory manager not available"}

    manager.working.set(key, value, tags=tags)
    return {
        "success": True,
        "key": key,
        "message": f"Stored '{key}' in working memory",
    }


@requires("memory_manager")
def recall_context(key: str) -> dict[str, Any]:
    """Recall a specific context from working memory.

    Args:
        key: The key to look up.

    Returns:
        A dict with the value if found, or an error message.
    """
    manager = get_context_memory_manager()
    if manager is None:
        return {"success": False, "error": "Memory manager not available"}

    value = manager.working.get(key)
    if value is None:
        return {"success": False, "error": f"Key '{key}' not found in working memory"}

    return {
        "success": True,
        "key": key,
        "value": value,
    }


@requires("memory_manager")
def search_memory(
    query: str,
    memory_type: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Search across working and/or long-term memory.

    Searches both working memory (session) and long-term memory (persistent).

    Args:
        query: The search query.
        memory_type: Optional filter for long-term memory type
            ("fact", "preference", "learning", "reference").
        limit: Maximum number of results to return.

    Returns:
        A dict with matching results from both memory tiers.
    """
    manager = get_context_memory_manager()
    if manager is None:
        return {"success": False, "error": "Memory manager not available"}

    # Search across tiers
    results = manager.search(query, include_working=True, include_longterm=True)

    # Format working memory results
    working_results = [
        {"key": key, "value": value}
        for key, value in results.working_results[:limit]
    ]

    # Format long-term memory results with optional type filter
    longterm_results = []
    for entry in results.longterm_results:
        if memory_type is None or entry.type.value == memory_type:
            longterm_results.append({
                "id": entry.id,
                "type": entry.type.value,
                "content": entry.content,
                "tags": entry.tags,
            })
            if len(longterm_results) >= limit:
                break

    return {
        "success": True,
        "query": query,
        "working_memory": working_results,
        "longterm_memory": longterm_results,
        "count": len(working_results) + len(longterm_results),
    }


@requires("memory_manager")
def save_to_longterm(
    content: str,
    memory_type: str = "learning",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Save information to long-term memory (persists across sessions).

    Use this when you discover important information that should be
    remembered across sessions.

    Args:
        content: The content to store.
        memory_type: Type of memory entry ("fact", "preference", "learning", "reference").
        tags: Optional tags for categorization.

    Returns:
        A dict with the stored entry ID.
    """
    manager = get_context_memory_manager()
    if manager is None:
        return {"success": False, "error": "Memory manager not available"}

    from agentic_cli.memory.longterm import MemoryType

    # Validate and convert type
    try:
        mem_type = MemoryType(memory_type)
    except ValueError:
        valid_types = [t.value for t in MemoryType]
        return {
            "success": False,
            "error": f"Invalid memory_type: {memory_type}. Valid types: {', '.join(valid_types)}",
        }

    entry_id = manager.longterm.store(
        type=mem_type,
        content=content,
        source="framework_tool",
        tags=tags,
    )

    return {
        "success": True,
        "entry_id": entry_id,
        "type": memory_type,
        "message": f"Saved to long-term memory ({memory_type})",
    }


@requires("memory_manager")
def clear_working_memory() -> dict[str, Any]:
    """Clear all entries from working memory.

    Use this to reset session state when starting a new task.

    Returns:
        A dict with success status.
    """
    manager = get_context_memory_manager()
    if manager is None:
        return {"success": False, "error": "Memory manager not available"}

    manager.working.clear()
    return {
        "success": True,
        "message": "Working memory cleared",
    }
