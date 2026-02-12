"""Context variables for workflow execution.

Provides ContextVars that allow tools to access managers during workflow execution.
The workflow manager sets these context variables before processing, and tools
can access them via the getter functions.
"""

from contextvars import ContextVar, Token
from typing import Any, Callable


def _make_context_accessors(name: str) -> tuple[Callable[..., Token], Callable[..., Any]]:
    """Create a (setter, getter) pair backed by a ContextVar."""
    var: ContextVar[Any] = ContextVar(f"{name}_context", default=None)

    def setter(value: Any) -> Token:
        return var.set(value)

    def getter() -> Any:
        return var.get()

    setter.__name__ = setter.__qualname__ = f"set_context_{name}"
    getter.__name__ = getter.__qualname__ = f"get_context_{name}"
    return setter, getter


set_context_memory_store, get_context_memory_store = _make_context_accessors("memory_store")
set_context_plan_store, get_context_plan_store = _make_context_accessors("plan_store")
set_context_approval_manager, get_context_approval_manager = _make_context_accessors("approval_manager")
set_context_task_store, get_context_task_store = _make_context_accessors("task_store")
set_context_llm_summarizer, get_context_llm_summarizer = _make_context_accessors("llm_summarizer")
set_context_kb_manager, get_context_kb_manager = _make_context_accessors("kb_manager")
set_context_user_kb_manager, get_context_user_kb_manager = _make_context_accessors("user_kb_manager")
