"""Memory tools for agentic workflows.

``save_memory``, ``read_memory``, ``search_memory``, ``update_memory`` and
``delete_memory`` over the workflow's ``MemoryStore`` (``agentic_cli.memory``):
markdown files in a user scope and a project scope, which the workflow creates
when an agent has these tools. The main agent also gets an index of the
memories at the start of each conversation.

Example:
    from agentic_cli.tools import memory_tools

    AgentConfig(
        tools=[memory_tools.save_memory, memory_tools.read_memory, memory_tools.search_memory],
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentic_cli.memory import DuplicateMemory, MemoryStoreError
from agentic_cli.tools.registry import ToolCategory, register_tool
from agentic_cli.workflow.permissions import Capability
from agentic_cli.workflow.service_registry import MEMORY_STORE, require_service
from agentic_cli.workflow.sessions import get_active_turn

if TYPE_CHECKING:
    from agentic_cli.memory import Memory, MemoryStore

__all__ = ["delete_memory", "read_memory", "save_memory", "search_memory", "update_memory"]


# ---------------------------------------------------------------------------
# Shared helpers — the single source of truth for tool behavior.
#
# Both the @register_tool wrappers below (service-registry bound) and the
# make_memory_tools() closures in tools/factories.py delegate to these
# helpers, so the two entry points cannot drift.
# ---------------------------------------------------------------------------


def _memory_dict(memory: "Memory") -> dict[str, Any]:
    """What a tool result shows of a memory (never its path)."""
    return {
        "id": memory.id,
        "description": memory.description,
        "content": memory.content,
        "type": memory.type,
        "scope": memory.scope,
        "source": memory.source,
        "created": memory.created.isoformat(),
        "updated": memory.updated.isoformat(),
    }


def _failed(error: object) -> dict[str, Any]:
    return {"success": False, "error": str(error)}


def _save_memory_with_store(
    store: "MemoryStore",
    description: str,
    content: str,
    memory_type: str,
    scope: str,
    user_requested: bool,
) -> dict[str, Any]:
    turn = get_active_turn()
    try:
        memory = store.save(
            description, content, memory_type, scope=scope,
            source="user" if user_requested else "agent",
            conversation=turn.session_id if turn is not None else None,
        )
    except DuplicateMemory as e:
        return _failed(f"{e}: use update_memory to change it")
    except (MemoryStoreError, OSError) as e:
        return _failed(e)
    return {
        "success": True,
        "item_id": memory.id,
        "scope": memory.scope,
        "message": f"Saved to {memory.scope} memory: {memory.description}",
    }


def _read_memory_with_store(store: "MemoryStore", item_id: str) -> dict[str, Any]:
    try:
        memory = store.read(item_id)
    except (MemoryStoreError, OSError) as e:
        return _failed(e)
    if memory is None:
        return _failed(f"No memory with id {item_id}")
    return {"success": True, "memory": _memory_dict(memory)}


def _search_memory_with_store(store: "MemoryStore", query: str, limit: int) -> dict[str, Any]:
    try:
        found = store.search(query, limit=limit)
    except (MemoryStoreError, OSError) as e:
        return _failed(e)
    items = [_memory_dict(memory) for memory in found]
    return {"success": True, "query": query, "items": items, "count": len(items)}


def _update_memory_with_store(
    store: "MemoryStore",
    item_id: str,
    description: str | None,
    content: str | None,
    memory_type: str | None,
) -> dict[str, Any]:
    if description is None and content is None and memory_type is None:
        return _failed("Give a new description, content or type")
    try:
        memory = store.update(item_id, description=description, content=content, type=memory_type)
    except DuplicateMemory as e:
        return _failed(f"{e}: update or delete that one instead")
    except (MemoryStoreError, OSError) as e:
        return _failed(e)
    if memory is None:
        return _failed(f"No memory with id {item_id}")
    return {"success": True, "updated": True, "message": f"Updated memory: {memory.description}"}


def _delete_memory_with_store(store: "MemoryStore", item_id: str) -> dict[str, Any]:
    try:
        memory = store.read(item_id)
        deleted = memory is not None and store.delete(item_id)
    except (MemoryStoreError, OSError) as e:
        return _failed(e)
    if not deleted:
        return _failed(f"No memory with id {item_id}")
    return {"success": True, "deleted": True, "message": f"Deleted memory: {memory.description}"}


# ---------------------------------------------------------------------------
# Registry-bound tool functions (@register_tool) — thin wrappers that resolve
# the store through the service registry and delegate to the helpers above.
# ---------------------------------------------------------------------------


@register_tool(
    category=ToolCategory.MEMORY,
    capabilities=[Capability("memory.write")],
    description=(
        "Save a memory later conversations will see: a preference or correction the user "
        "stated, a lasting fact not in the project's files, or a pointer to an external resource."
    ),
    requires="memory_store",
)
def save_memory(
    description: str,
    content: str,
    type: str,
    scope: str = "project",
    user_requested: bool = False,
) -> dict[str, Any]:
    """Save a memory that later conversations will see.

    Memories are short notes kept across conversations; the main agent gets an
    index of them (by id) at the start of each conversation. Save only what
    will matter later: a preference or correction the user stated; a lasting
    fact about the user or the project that is not written in the project's
    files; a pointer to an external resource (a URL, a dashboard, a document).
    Never save secrets (keys, tokens, passwords), anything that can be read
    from the files, or what matters only in this conversation. To change a
    memory that exists, use update_memory.

    Args:
        description: One line, at most 150 characters: what the memory is
            about (the index shows it).
        content: The memory itself, at most 4,000 characters.
        type: "user" (who the user is: role, expertise), "feedback" (how the
            user wants the work done: corrections and confirmed approaches),
            "project" (ongoing work, goals, decisions not recorded in files)
            or "reference" (where something lives).
        scope: "project" (the default: this project only) or "user" (true
            across all of the user's projects).
        user_requested: True only when the user asked you to remember this.
    """
    store = require_service(MEMORY_STORE)
    if isinstance(store, dict):
        return store
    return _save_memory_with_store(store, description, content, type, scope, user_requested)


@register_tool(
    category=ToolCategory.MEMORY,
    capabilities=[Capability("memory.read")],
    description="Read a memory in full by its id.",
    requires="memory_store",
)
def read_memory(item_id: str) -> dict[str, Any]:
    """Read a memory in full by its id (the memory index lists ids).

    A memory is a dated claim: check it against the current state before
    acting on it, and update or delete it when it is wrong.

    Args:
        item_id: The memory's id.
    """
    store = require_service(MEMORY_STORE)
    if isinstance(store, dict):
        return store
    return _read_memory_with_store(store, item_id)


@register_tool(
    category=ToolCategory.MEMORY,
    capabilities=[Capability("memory.read")],
    description="Search saved memories by keywords.",
    requires="memory_store",
)
def search_memory(query: str, limit: int = 10) -> dict[str, Any]:
    """Search memories by keywords, for what the memory index does not show.

    A memory is a dated claim: check it against the current state before
    acting on it.

    Args:
        query: Words to look for in descriptions and contents; empty lists the newest.
        limit: At most this many memories (1 to 50).
    """
    store = require_service(MEMORY_STORE)
    if isinstance(store, dict):
        return store
    return _search_memory_with_store(store, query, limit)


@register_tool(
    category=ToolCategory.MEMORY,
    capabilities=[Capability("memory.write")],
    description="Change a memory that is wrong or out of date.",
    requires="memory_store",
)
def update_memory(
    item_id: str,
    description: str | None = None,
    content: str | None = None,
    type: str | None = None,
) -> dict[str, Any]:
    """Change a memory that is wrong or out of date. Its previous version is kept for 30 days.

    Args:
        item_id: The memory's id.
        description: A new one-line description (omit to keep it).
        content: New content (omit to keep it).
        type: A new type: "user", "feedback", "project" or "reference" (omit to keep it).
    """
    store = require_service(MEMORY_STORE)
    if isinstance(store, dict):
        return store
    return _update_memory_with_store(store, item_id, description, content, type)


@register_tool(
    category=ToolCategory.MEMORY,
    capabilities=[Capability("memory.write")],
    description="Delete a memory that is wrong or no longer useful.",
    requires="memory_store",
)
def delete_memory(item_id: str) -> dict[str, Any]:
    """Delete a memory that is wrong or no longer useful. It can be restored for 30 days.

    Args:
        item_id: The memory's id.
    """
    store = require_service(MEMORY_STORE)
    if isinstance(store, dict):
        return store
    return _delete_memory_with_store(store, item_id)
