"""Agentic CLI - A framework for building domain-specific agentic CLI applications.

This package provides the core infrastructure for building CLI applications
powered by LLM agents, including:

- CLI framework with thinking boxes and rich output
- Workflow management for agent orchestration (Google ADK or LangGraph)
- Generic tools (search, code execution, document generation)
- Knowledge base with vector search
- Session persistence

Domain-specific applications extend the base classes to provide their own
agents, prompts, and configuration.

Orchestrator Selection:
- Default: Google ADK (GoogleADKWorkflowManager)
- Optional: LangGraph (LangGraphWorkflowManager) - requires `pip install agentic-cli[langgraph]`

Use settings.orchestrator = "langgraph" or the factory function
`create_workflow_manager_from_settings()` to switch orchestrators.

Note: GoogleADKWorkflowManager and LangGraphWorkflowManager are lazy-loaded to avoid slow imports.
"""

from agentic_cli.cli.app import BaseCLIApp
from agentic_cli.cli.workflow_controller import create_workflow_manager_from_settings
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
from agentic_cli.settings_persistence import SettingsPersistence
from agentic_cli.workflow.settings import WorkflowSettingsMixin
from agentic_cli.cli.settings import CLISettingsMixin

# Heavy imports - lazy loaded on first access
_lazy_imports = {
    "GoogleADKWorkflowManager": "agentic_cli.workflow.adk.manager",
    "BaseWorkflowManager": "agentic_cli.workflow.base_manager",
    "LangGraphWorkflowManager": "agentic_cli.workflow.langgraph.manager",
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
    "create_workflow_manager_from_settings",
    # Workflow
    "BaseWorkflowManager",  # lazy
    "GoogleADKWorkflowManager",  # lazy (Google ADK)
    "LangGraphWorkflowManager",  # lazy (requires langgraph extra)
    "AgentConfig",
    "WorkflowEvent",
    "EventType",
    # Settings
    "BaseSettings",
    "SettingsContext",
    "SettingsValidationError",
    "SettingsPersistence",
    "get_settings",
    "set_settings",
    "set_context_settings",
    "get_context_settings",
    "validate_settings",
    "reload_settings",
    # Settings Mixins (organized settings by domain)
    "WorkflowSettingsMixin",
    "CLISettingsMixin",
    # Resolvers (SRP-compliant model/path resolution)
    "ModelResolver",
    "PathResolver",
]

__version__ = "0.4.1"
