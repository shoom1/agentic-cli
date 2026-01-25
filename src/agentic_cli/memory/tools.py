"""Memory tools for agent use.

Provides tools that agents can use to interact with working memory
and long-term memory during their operation.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from agentic_cli.memory.working import WorkingMemory
from agentic_cli.memory.longterm import LongTermMemory, MemoryType

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings


# Module-level working memory instance (singleton per process)
_working_memory: WorkingMemory | None = None


def _get_working_memory() -> WorkingMemory:
    """Get or create the module-level WorkingMemory instance."""
    global _working_memory
    if _working_memory is None:
        _working_memory = WorkingMemory()
    return _working_memory


def reset_working_memory() -> None:
    """Reset the module-level working memory instance.

    Primarily used for testing to ensure a clean state between tests.
    """
    global _working_memory
    _working_memory = None


def working_memory_tool(
    operation: str,  # "set", "get", "list", "delete", "clear"
    key: str | None = None,
    value: Any = None,
    tags: list[str] | None = None,
    *,
    settings: "BaseSettings | None" = None,
) -> dict[str, Any]:
    """Tool for agents to interact with working memory.

    Provides a simple key-value store with tag-based filtering for
    session-scoped data that agents need to track during their operation.

    Args:
        operation: One of "set", "get", "list", "delete", "clear"
        key: Key for set/get/delete operations
        value: Value for set operation
        tags: Optional tags for set/list operations
        settings: Optional settings (not used for working memory but
            included for API consistency)

    Returns:
        Dict with "success" key and operation-specific results:
        - set: {"success": True, "key": key}
        - get: {"success": True, "key": key, "value": value}
        - list: {"success": True, "keys": [...]}
        - delete: {"success": True, "key": key}
        - clear: {"success": True}
        - error: {"success": False, "error": "message"}

    Example:
        >>> working_memory_tool(operation="set", key="task", value="analyzing")
        {'success': True, 'key': 'task'}
        >>> working_memory_tool(operation="get", key="task")
        {'success': True, 'key': 'task', 'value': 'analyzing'}
    """
    try:
        memory = _get_working_memory()

        if operation == "set":
            if key is None:
                return {"success": False, "error": "key is required for set operation"}
            memory.set(key, value, tags=tags)
            return {"success": True, "key": key}

        elif operation == "get":
            if key is None:
                return {"success": False, "error": "key is required for get operation"}
            value = memory.get(key)
            return {"success": True, "key": key, "value": value}

        elif operation == "list":
            keys = memory.list(tags=tags)
            return {"success": True, "keys": keys}

        elif operation == "delete":
            if key is None:
                return {"success": False, "error": "key is required for delete operation"}
            memory.delete(key)
            return {"success": True, "key": key}

        elif operation == "clear":
            memory.clear()
            return {"success": True}

        else:
            return {
                "success": False,
                "error": f"Unknown operation: {operation}. Valid operations: set, get, list, delete, clear",
            }

    except Exception as e:
        return {"success": False, "error": str(e)}


def long_term_memory_tool(
    operation: str,  # "store", "recall", "update", "forget"
    content: str | None = None,
    query: str | None = None,
    type: str | None = None,  # "fact", "preference", "learning", "reference"
    entry_id: str | None = None,
    kb_references: list[str] | None = None,
    tags: list[str] | None = None,
    *,
    settings: "BaseSettings | None" = None,
) -> dict[str, Any]:
    """Tool for agents to interact with long-term memory.

    Provides persistent storage for facts, preferences, learnings, and
    references that survive across sessions.

    Args:
        operation: One of "store", "recall", "update", "forget"
        content: Content for store/update operations
        query: Search query for recall operation
        type: Memory type for store/recall ("fact", "preference", "learning", "reference")
        entry_id: Entry ID for update/forget operations
        kb_references: Optional knowledge base document references
        tags: Optional tags for categorization
        settings: Optional settings instance. If not provided, uses get_settings()

    Returns:
        Dict with "success" key and operation-specific results:
        - store: {"success": True, "entry_id": id}
        - recall: {"success": True, "entries": [{id, type, content, tags, kb_references}, ...]}
        - update: {"success": True, "entry_id": id}
        - forget: {"success": True, "entry_id": id}
        - error: {"success": False, "error": "message"}

    Example:
        >>> long_term_memory_tool(operation="store", content="User prefers markdown", type="preference")
        {'success': True, 'entry_id': 'abc-123'}
        >>> long_term_memory_tool(operation="recall", query="markdown")
        {'success': True, 'entries': [{'id': 'abc-123', 'type': 'preference', ...}]}
    """
    try:
        # Get settings if not provided
        if settings is None:
            from agentic_cli.config import get_settings
            settings = get_settings()

        memory = LongTermMemory(settings)

        if operation == "store":
            if content is None:
                return {"success": False, "error": "content is required for store operation"}
            if type is None:
                return {"success": False, "error": "type is required for store operation"}

            # Validate and convert type
            try:
                memory_type = MemoryType(type)
            except ValueError:
                valid_types = [t.value for t in MemoryType]
                return {
                    "success": False,
                    "error": f"Invalid type: {type}. Valid types: {', '.join(valid_types)}",
                }

            entry_id = memory.store(
                type=memory_type,
                content=content,
                source="agent_tool",  # Default source for tool-created entries
                kb_references=kb_references,
                tags=tags,
            )
            return {"success": True, "entry_id": entry_id}

        elif operation == "recall":
            # Convert type string to MemoryType if provided
            memory_type = None
            if type is not None:
                try:
                    memory_type = MemoryType(type)
                except ValueError:
                    valid_types = [t.value for t in MemoryType]
                    return {
                        "success": False,
                        "error": f"Invalid type: {type}. Valid types: {', '.join(valid_types)}",
                    }

            entries = memory.recall(
                query=query or "",
                type=memory_type,
                tags=tags,
            )

            # Convert entries to serializable format
            serialized_entries = [
                {
                    "id": entry.id,
                    "type": entry.type.value,
                    "content": entry.content,
                    "tags": entry.tags,
                    "kb_references": entry.kb_references,
                }
                for entry in entries
            ]
            return {"success": True, "entries": serialized_entries}

        elif operation == "update":
            if entry_id is None:
                return {"success": False, "error": "entry_id is required for update operation"}

            # Build update kwargs
            update_kwargs = {}
            if content is not None:
                update_kwargs["content"] = content
            if tags is not None:
                update_kwargs["tags"] = tags
            if kb_references is not None:
                update_kwargs["kb_references"] = kb_references

            memory.update(entry_id, **update_kwargs)
            return {"success": True, "entry_id": entry_id}

        elif operation == "forget":
            if entry_id is None:
                return {"success": False, "error": "entry_id is required for forget operation"}

            memory.forget(entry_id)
            return {"success": True, "entry_id": entry_id}

        else:
            return {
                "success": False,
                "error": f"Unknown operation: {operation}. Valid operations: store, recall, update, forget",
            }

    except KeyError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": str(e)}
