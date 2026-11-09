"""Unified service registry for workflow execution.

Single ContextVar holding a dict[str, Any] that replaces the previous
8 individual ContextVar pairs.  Tools access services and simple state
via ``get_service(key)`` instead of per-service getter functions.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any

# ---- Well-known registry keys (defined with the table of services) ----

from agentic_cli.workflow.services import (
    ARXIV_SOURCE,
    EMBEDDING_SERVICE,
    JOB_MANAGER,
    KNOWLEDGE_BASE,
    LLM_SUMMARIZER,
    MEMORY_STORE,
    PERMISSION_ENGINE,
    SANDBOX_MANAGER,
    SERVICES,
    WORKFLOW,
)

# Service keys a tool may declare via ``register_tool(requires=...)``: the
# table's declarable services (``workflow/services.py``), each one a manager
# builds on demand. Keys that are always present (PERMISSION_ENGINE,
# WORKFLOW) say nothing when declared; the embedding service is built with
# the knowledge base, never on its own, so no tool can declare it.
KNOWN_SERVICE_KEYS = frozenset(spec.key for spec in SERVICES if spec.declarable)

# Extra guidance for keys that look declarable but are not.
SERVICE_KEY_HINTS = {
    "kb_manager": (
        f"The knowledge base is one service: declare requires={KNOWLEDGE_BASE!r} "
        "(its project scope is knowledge_base.manager('project'))."
    ),
    "user_kb_manager": (
        f"The knowledge base is one service: declare requires={KNOWLEDGE_BASE!r} "
        "(its user scope is knowledge_base.manager('user'))."
    ),
    PERMISSION_ENGINE: f"{PERMISSION_ENGINE!r} is always available.",
    WORKFLOW: f"{WORKFLOW!r} is always available.",
    EMBEDDING_SERVICE: (
        f"{EMBEDDING_SERVICE!r} is private: the workflow builds it with the "
        "knowledge bases and the memory store."
    ),
}


# ---- ContextVar and accessors ----

_registry_var: ContextVar[dict[str, Any] | None] = ContextVar(
    "service_registry", default=None
)


def set_service_registry(registry: dict[str, Any]) -> Token:
    """Set the service registry for the current context.

    Called by ``BaseWorkflowManager._workflow_context()`` at the start
    of each ``process()`` call.

    Returns:
        Token for resetting the ContextVar on exit.
    """
    return _registry_var.set(registry)


def clear_service_registry() -> Token:
    """Reset the registry ContextVar to None.

    Useful in tests to simulate "no workflow context".
    """
    return _registry_var.set(None)


def get_service_registry() -> dict[str, Any]:
    """Get the current service registry dict.

    Returns the live dict — callers may read and mutate it.
    Auto-creates an empty dict on first access to avoid
    shared-mutable-default issues.
    """
    registry = _registry_var.get()
    if registry is None:
        registry = {}
        _registry_var.set(registry)
    return registry


def get_service(key: str) -> Any | None:
    """Look up a single service or state value by key.

    Convenience wrapper around ``get_service_registry().get(key)``.
    """
    registry = _registry_var.get()
    if registry is None:
        return None
    return registry.get(key)


def require_service(key: str) -> Any:
    """Get a service, returning an error dict if unavailable.

    Returns the service object, or ``{"success": False, "error": "..."}``
    if the service is not registered. Callers use ``isinstance(result, dict)``
    to detect the error case.
    """
    svc = get_service(key)
    if svc is None:
        label = key.replace("_", " ")
        return {"success": False, "error": f"{label} not available"}
    return svc


