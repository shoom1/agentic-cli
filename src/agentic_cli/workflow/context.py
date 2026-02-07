"""Context variables for workflow execution.

Provides ContextVars that allow tools to access managers during workflow execution.
The workflow manager sets these context variables before processing, and tools
can access them via the getter functions.
"""

from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentic_cli.memory import MemoryStore
    from agentic_cli.planning import PlanStore
    from agentic_cli.hitl import ApprovalManager, CheckpointManager

# Context variables for manager instances
_memory_manager_context: ContextVar[Any] = ContextVar(
    "memory_manager_context", default=None
)
_task_graph_context: ContextVar[Any] = ContextVar(
    "task_graph_context", default=None
)
_approval_manager_context: ContextVar[Any] = ContextVar(
    "approval_manager_context", default=None
)
_checkpoint_manager_context: ContextVar[Any] = ContextVar(
    "checkpoint_manager_context", default=None
)
_llm_summarizer_context: ContextVar[Any] = ContextVar(
    "llm_summarizer_context", default=None
)


# Setters (used by workflow manager)


def set_context_memory_manager(manager: "MemoryStore | None") -> Token:
    """Set the memory manager in the current context."""
    return _memory_manager_context.set(manager)


def set_context_task_graph(store: "PlanStore | None") -> Token:
    """Set the plan store in the current context."""
    return _task_graph_context.set(store)


def set_context_approval_manager(manager: "ApprovalManager | None") -> Token:
    """Set the approval manager in the current context."""
    return _approval_manager_context.set(manager)


def set_context_checkpoint_manager(manager: "CheckpointManager | None") -> Token:
    """Set the checkpoint manager in the current context."""
    return _checkpoint_manager_context.set(manager)


def set_context_llm_summarizer(summarizer: Any | None) -> Token:
    """Set the LLM summarizer in the current context."""
    return _llm_summarizer_context.set(summarizer)


# Getters (used by tools)


def get_context_memory_manager() -> "MemoryStore | None":
    """Get the memory manager from the current context.

    Returns:
        The MemoryStore instance set by the workflow manager, or None if not set.
    """
    return _memory_manager_context.get()


def get_context_task_graph() -> "PlanStore | None":
    """Get the plan store from the current context.

    Returns:
        The PlanStore instance set by the workflow manager, or None if not set.
    """
    return _task_graph_context.get()


def get_context_approval_manager() -> "ApprovalManager | None":
    """Get the approval manager from the current context.

    Returns:
        The ApprovalManager instance set by the workflow manager, or None if not set.
    """
    return _approval_manager_context.get()


def get_context_checkpoint_manager() -> "CheckpointManager | None":
    """Get the checkpoint manager from the current context.

    Returns:
        The CheckpointManager instance set by the workflow manager, or None if not set.
    """
    return _checkpoint_manager_context.get()


def get_context_llm_summarizer() -> Any | None:
    """Get the LLM summarizer from the current context.

    Returns:
        The LLMSummarizer instance set by the workflow manager, or None if not set.
    """
    return _llm_summarizer_context.get()
