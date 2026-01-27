"""Workflow management for agentic CLI applications.

This module uses lazy loading for GoogleADKWorkflowManager to avoid slow
Google ADK imports at startup. Light types (EventType, AgentConfig, etc.)
are available immediately.
"""

# Light imports - always available (fast)
from agentic_cli.workflow.events import WorkflowEvent, EventType, UserInputRequest
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.thinking import ThinkingDetector, ThinkingResult
from agentic_cli.workflow.settings import WorkflowSettingsMixin
from agentic_cli.workflow.context import (
    # Getters for tools
    get_context_memory_manager,
    get_context_task_graph,
    get_context_approval_manager,
    get_context_checkpoint_manager,
    # Setters for workflow managers
    set_context_memory_manager,
    set_context_task_graph,
    set_context_approval_manager,
    set_context_checkpoint_manager,
)

# Heavy imports - lazy loaded on first access
_lazy_imports = {
    "GoogleADKWorkflowManager": "agentic_cli.workflow.adk_manager",
}


def __getattr__(name: str):
    """Lazy import for heavy modules."""
    if name in _lazy_imports:
        import importlib

        module = importlib.import_module(_lazy_imports[name])
        value = getattr(module, name)
        globals()[name] = value  # Cache for future access
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "WorkflowEvent",
    "EventType",
    "UserInputRequest",
    "AgentConfig",
    "GoogleADKWorkflowManager",  # lazy
    "ThinkingDetector",
    "ThinkingResult",
    "WorkflowSettingsMixin",
    # Context getters (for tools)
    "get_context_memory_manager",
    "get_context_task_graph",
    "get_context_approval_manager",
    "get_context_checkpoint_manager",
    # Context setters (for workflow managers)
    "set_context_memory_manager",
    "set_context_task_graph",
    "set_context_approval_manager",
    "set_context_checkpoint_manager",
]
