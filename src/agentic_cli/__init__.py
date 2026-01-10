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
"""

from agentic_cli.cli.app import BaseCLIApp
from agentic_cli.cli.commands import Command, CommandRegistry
from agentic_cli.workflow.manager import WorkflowManager
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

__all__ = [
    # CLI
    "BaseCLIApp",
    "Command",
    "CommandRegistry",
    # Workflow
    "WorkflowManager",
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
]

__version__ = "0.1.0"
