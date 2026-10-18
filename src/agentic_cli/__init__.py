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

Every name is loaded on first use, so importing one part of the framework
(for example ``agentic_cli.memory``) does not load the CLI and the workflow.
"""

import importlib
import importlib.util
from typing import TYPE_CHECKING

# Each exported name and the module that defines it, imported on first access
# (PEP 562). Python runs this file before any submodule, so importing anything
# here would load it for every ``agentic_cli.*`` import.
_lazy_imports = {
    # CLI
    "BaseCLIApp": "agentic_cli.cli.app",
    "Command": "agentic_cli.cli.commands",
    "CommandRegistry": "agentic_cli.cli.commands",
    "create_workflow_manager_from_settings": "agentic_cli.workflow.factory",
    # Workflow
    "BaseWorkflowManager": "agentic_cli.workflow.base_manager",
    "GoogleADKWorkflowManager": "agentic_cli.workflow.adk.manager",
    "LangGraphWorkflowManager": "agentic_cli.workflow.langgraph.manager",
    "AgentConfig": "agentic_cli.workflow.config",
    "AgentGraphError": "agentic_cli.workflow.config",
    "ModelSettings": "agentic_cli.workflow.model_settings",
    "ThinkingSettings": "agentic_cli.workflow.model_settings",
    "WorkflowEvent": "agentic_cli.workflow.events",
    "EventType": "agentic_cli.workflow.events",
    # Lifecycle / turn contracts
    "SessionRef": "agentic_cli.workflow.sessions",
    "TurnResult": "agentic_cli.cli.message_processor",
    "TurnStatus": "agentic_cli.cli.message_processor",
    "WorkflowState": "agentic_cli.cli.workflow_controller",
    # Settings
    "BaseSettings": "agentic_cli.config",
    "SettingsContext": "agentic_cli.config",
    "SettingsValidationError": "agentic_cli.config",
    "SettingsPersistence": "agentic_cli.settings_persistence",
    "SettingsSaveResult": "agentic_cli.settings_persistence",
    "get_settings": "agentic_cli.config",
    "set_settings": "agentic_cli.config",
    "set_context_settings": "agentic_cli.config",
    "get_context_settings": "agentic_cli.config",
    "validate_settings": "agentic_cli.config",
    "reload_settings": "agentic_cli.config",
    # Settings Mixins (organized settings by domain)
    "WorkflowSettingsMixin": "agentic_cli.workflow.settings",
    "AppSettingsMixin": "agentic_cli.settings_mixins",
    "CLISettingsMixin": "agentic_cli.settings_mixins",
}

if TYPE_CHECKING:
    from agentic_cli.cli.app import BaseCLIApp
    from agentic_cli.cli.commands import Command, CommandRegistry
    from agentic_cli.cli.message_processor import TurnResult, TurnStatus
    from agentic_cli.cli.workflow_controller import WorkflowState
    from agentic_cli.config import (
        BaseSettings,
        SettingsContext,
        SettingsValidationError,
        get_context_settings,
        get_settings,
        reload_settings,
        set_context_settings,
        set_settings,
        validate_settings,
    )
    from agentic_cli.settings_mixins import AppSettingsMixin, CLISettingsMixin
    from agentic_cli.settings_persistence import SettingsPersistence, SettingsSaveResult
    from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager
    from agentic_cli.workflow.base_manager import BaseWorkflowManager
    from agentic_cli.workflow.config import AgentConfig, AgentGraphError
    from agentic_cli.workflow.events import EventType, WorkflowEvent
    from agentic_cli.workflow.factory import create_workflow_manager_from_settings
    from agentic_cli.workflow.langgraph.manager import LangGraphWorkflowManager
    from agentic_cli.workflow.model_settings import ModelSettings, ThinkingSettings
    from agentic_cli.workflow.sessions import SessionRef
    from agentic_cli.workflow.settings import WorkflowSettingsMixin


def __getattr__(name: str):
    """Load an exported name, or a subpackage, on first access."""
    module_name = _lazy_imports.get(name)
    if module_name is not None:
        value = getattr(importlib.import_module(module_name), name)
        globals()[name] = value  # Cache for future access
        return value
    # Subpackages (``agentic_cli.cli``, ``agentic_cli.workflow``, ...) were
    # attributes as soon as this module imported them eagerly; code may still
    # reach them through the root without importing them.
    if not name.startswith("_") and importlib.util.find_spec(f"{__name__}.{name}") is not None:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_lazy_imports))


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
    "AgentGraphError",
    "ModelSettings",
    "ThinkingSettings",
    "WorkflowEvent",
    "EventType",
    # Lifecycle / turn contracts
    "SessionRef",
    "TurnResult",
    "TurnStatus",
    "WorkflowState",
    # Settings
    "BaseSettings",
    "SettingsContext",
    "SettingsValidationError",
    "SettingsPersistence",
    "SettingsSaveResult",
    "get_settings",
    "set_settings",
    "set_context_settings",
    "get_context_settings",
    "validate_settings",
    "reload_settings",
    # Settings Mixins (organized settings by domain)
    "WorkflowSettingsMixin",
    "AppSettingsMixin",
    "CLISettingsMixin",
]

__version__ = "0.6.4"
