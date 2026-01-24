"""Agentic CLI - A framework for building domain-specific agentic CLI applications.

This package provides the core infrastructure for building CLI applications
powered by LLM agents, including:

- CLI framework with thinking boxes and rich output
- Workflow management for agent orchestration
- Generic tools (search, code execution, document generation)
- Knowledge base with vector search
- Session persistence

Domain-specific applications extend the base classes to provide their own
agents, prompts, and configuration.

Note: WorkflowManager is lazy-loaded to avoid slow Google ADK imports at startup.
"""

from agentic_cli.cli.app import BaseCLIApp
from agentic_cli.cli.commands import Command, CommandRegistry
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.events import WorkflowEvent, EventType
from agentic_cli.config import (
    BaseSettings,
    SettingsContext,
    SettingsValidationError,
    get_settings,
    set_settings,
    set_context_settings,
    get_context_settings,
    validate_settings,
    reload_settings,
)
from agentic_cli.resolvers import ModelResolver, PathResolver
from agentic_cli.config_mixins import (
    KnowledgeBaseMixin,
    PythonExecutorMixin,
    PersistenceMixin,
    ArtifactsMixin,
    FullFeaturesMixin,
)

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
    # CLI
    "BaseCLIApp",
    "Command",
    "CommandRegistry",
    # Workflow
    "WorkflowManager",  # lazy
    "AgentConfig",
    "WorkflowEvent",
    "EventType",
    # Settings
    "BaseSettings",
    "SettingsContext",
    "SettingsValidationError",
    "get_settings",
    "set_settings",
    "set_context_settings",
    "get_context_settings",
    "validate_settings",
    "reload_settings",
    # Resolvers (SRP-compliant model/path resolution)
    "ModelResolver",
    "PathResolver",
    # Configuration Mixins (ISP-compliant optional features)
    "KnowledgeBaseMixin",
    "PythonExecutorMixin",
    "PersistenceMixin",
    "ArtifactsMixin",
    "FullFeaturesMixin",
]

__version__ = "0.1.2"
