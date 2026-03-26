"""Backward-compatible context accessors.

These functions delegate to the unified service registry.
Existing code that calls ``get_context_kb_manager()`` etc. continues
to work without changes.

New code should use ``get_service(key)`` from
:mod:`agentic_cli.workflow.service_registry` directly.
"""

from __future__ import annotations

from typing import Any

from agentic_cli.workflow.service_registry import (
    get_service,
    get_service_registry,
)


class _CompatToken:
    """Minimal token that supports ``token.var.reset(token)``.

    Used by the compat setters so that test code doing
    ``token = set_context_kb_manager(mock); ... ; token.var.reset(token)``
    continues to work.  Only modifies the service registry dict contents;
    does not reset the ContextVar itself.
    """

    def __init__(self, registry: dict[str, Any], key: str, old_value: Any) -> None:
        self._registry = registry
        self._key = key
        self._old_value = old_value
        self.var = self  # token.var.reset(token) calls self.reset(self)

    def reset(self, _token: Any) -> None:
        if self._old_value is _SENTINEL:
            self._registry.pop(self._key, None)
        else:
            self._registry[self._key] = self._old_value


_SENTINEL = object()


def _make_compat_setter(key: str):
    """Create a setter that writes into the service registry."""
    def setter(value: Any) -> _CompatToken:
        registry = get_service_registry()
        old = registry.get(key, _SENTINEL)
        registry[key] = value
        return _CompatToken(registry, key, old)
    setter.__name__ = setter.__qualname__ = f"set_context_{key}"
    return setter


def _make_compat_getter(key: str):
    """Create a getter that reads from the service registry."""
    def getter() -> Any:
        return get_service(key)
    getter.__name__ = getter.__qualname__ = f"get_context_{key}"
    return getter


# ---- Public accessors (backward-compat for tests) ----

set_context_memory_store = _make_compat_setter("memory_store")
get_context_memory_store = _make_compat_getter("memory_store")

set_context_llm_summarizer = _make_compat_setter("llm_summarizer")
get_context_llm_summarizer = _make_compat_getter("llm_summarizer")

set_context_kb_manager = _make_compat_setter("kb_manager")
get_context_kb_manager = _make_compat_getter("kb_manager")

set_context_user_kb_manager = _make_compat_setter("user_kb_manager")
get_context_user_kb_manager = _make_compat_getter("user_kb_manager")

set_context_workflow = _make_compat_setter("workflow")
get_context_workflow = _make_compat_getter("workflow")

set_context_sandbox_manager = _make_compat_setter("sandbox_manager")
get_context_sandbox_manager = _make_compat_getter("sandbox_manager")
