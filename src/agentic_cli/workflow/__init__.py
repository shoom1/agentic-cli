"""Workflow management for agentic CLI applications.

This module uses lazy loading for WorkflowManager to avoid slow
Google ADK imports at startup. Light types (EventType, AgentConfig, etc.)
are available immediately.
"""

# Light imports - always available (fast)
from agentic_cli.workflow.events import WorkflowEvent, EventType, UserInputRequest
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.thinking import ThinkingDetector, ThinkingResult
from agentic_cli.workflow.memory import ConversationMemory
from agentic_cli.workflow.retry import RetryConfig, RetryHandler
from agentic_cli.workflow.session_handler import SessionHandler
from agentic_cli.workflow.event_processor import EventProcessor

# Heavy imports - lazy loaded on first access
_lazy_imports = {
    "WorkflowManager": "agentic_cli.workflow.manager",
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
    "WorkflowManager",  # lazy
    "ThinkingDetector",
    "ThinkingResult",
    "ConversationMemory",
    "RetryConfig",
    "RetryHandler",
    "SessionHandler",
    "EventProcessor",
]
