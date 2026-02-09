"""Workflow management for agentic CLI applications.

This module uses lazy loading for heavy workflow managers to avoid slow
imports at startup. Light types (EventType, AgentConfig, etc.)
are available immediately.

Submodules:
- langgraph/: LangGraph-specific implementation (manager, state, middleware, persistence, tools)
- adk/: Google ADK-specific implementation (placeholder for future refactoring)
"""

# Light imports - always available (fast)
from agentic_cli.workflow.events import WorkflowEvent, EventType, UserInputRequest
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.thinking import ThinkingDetector, ThinkingResult
from agentic_cli.workflow.settings import WorkflowSettingsMixin
from agentic_cli.workflow.context import (
    # Getters for tools
    get_context_memory_store,
    get_context_plan_store,
    get_context_approval_manager,
    get_context_checkpoint_manager,
    get_context_task_store,
    get_context_llm_summarizer,
    # Setters for workflow managers
    set_context_memory_store,
    set_context_plan_store,
    set_context_approval_manager,
    set_context_checkpoint_manager,
    set_context_task_store,
    set_context_llm_summarizer,
)

# Heavy imports - lazy loaded on first access
_lazy_imports = {
    # ADK manager
    "GoogleADKWorkflowManager": "agentic_cli.workflow.adk.manager",
    # LangGraph manager (both old and new paths)
    "LangGraphWorkflowManager": "agentic_cli.workflow.langgraph.manager",
    "LangGraphManager": "agentic_cli.workflow.langgraph.manager",
    # LangGraph state types (for backward compatibility)
    "AgentState": "agentic_cli.workflow.langgraph.state",
    "ResearchState": "agentic_cli.workflow.langgraph.state",
    "ApprovalState": "agentic_cli.workflow.langgraph.state",
    "FinanceResearchState": "agentic_cli.workflow.langgraph.state",
    "CheckpointData": "agentic_cli.workflow.langgraph.state",
    "add_messages": "agentic_cli.workflow.langgraph.state",
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
    # Events
    "WorkflowEvent",
    "EventType",
    "UserInputRequest",
    # Config
    "AgentConfig",
    # Thinking
    "ThinkingDetector",
    "ThinkingResult",
    # Settings mixin
    "WorkflowSettingsMixin",
    # Managers (lazy)
    "GoogleADKWorkflowManager",
    "LangGraphWorkflowManager",
    "LangGraphManager",
    # LangGraph state (lazy, for backward compatibility)
    "AgentState",
    "ResearchState",
    "ApprovalState",
    "FinanceResearchState",
    "CheckpointData",
    "add_messages",
    # Context getters (for tools)
    "get_context_memory_store",
    "get_context_plan_store",
    "get_context_approval_manager",
    "get_context_checkpoint_manager",
    "get_context_task_store",
    "get_context_llm_summarizer",
    # Context setters (for workflow managers)
    "set_context_memory_store",
    "set_context_plan_store",
    "set_context_approval_manager",
    "set_context_checkpoint_manager",
    "set_context_task_store",
    "set_context_llm_summarizer",
]
