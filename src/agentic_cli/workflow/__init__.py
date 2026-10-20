"""Workflow management for agentic CLI applications.

This module uses lazy loading for heavy workflow managers to avoid slow
imports at startup. Light types (EventType, AgentConfig, etc.)
are available immediately.

Submodules:
- adk/: the Google ADK backend (the default)
- deepagents/: the Deep Agents backend (the ``deepagents`` extra)
- langgraph/: the LangGraph backend (the ``langgraph`` extra; deprecated, removed
  in 0.8.0)
"""

# Light imports - always available (fast)
from agentic_cli.workflow.events import (
    EventType,
    UserInputRequest,
    UserInputUnavailable,
    WorkflowEvent,
)
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.model_settings import ModelSettings, ThinkingSettings
from agentic_cli.workflow.mcp import MCPServerConfig
from agentic_cli.workflow.factory import create_workflow_manager_from_settings
from agentic_cli.workflow.agent_loader import (
    load_agents_from_yaml,
    create_workflow_manager_from_yaml,
)
from agentic_cli.workflow.adk_config_bridge import (
    load_adk_agent_native,
    translate_adk_yaml,
)
from agentic_cli.workflow.settings import WorkflowSettingsMixin
from agentic_cli.workflow.service_registry import get_service, get_service_registry

# Heavy imports - lazy loaded on first access
_lazy_imports = {
    # ADK manager
    "GoogleADKWorkflowManager": "agentic_cli.workflow.adk.manager",
    # Deep Agents manager (the deepagents extra)
    "DeepAgentsWorkflowManager": "agentic_cli.workflow.deepagents",
    # LangGraph manager (both old and new paths; deprecated, removed in 0.8.0)
    "LangGraphWorkflowManager": "agentic_cli.workflow.langgraph.manager",
    "LangGraphManager": "agentic_cli.workflow.langgraph.manager",
    # LangGraph state types (for backward compatibility)
    "AgentState": "agentic_cli.workflow.langgraph.state",
    "ResearchState": "agentic_cli.workflow.langgraph.state",
    "ApprovalState": "agentic_cli.workflow.langgraph.state",
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
    "UserInputUnavailable",
    # Factory
    "create_workflow_manager_from_settings",
    "create_workflow_manager_from_yaml",
    "load_agents_from_yaml",
    "load_adk_agent_native",
    "translate_adk_yaml",
    # Config
    "AgentConfig",
    "ModelSettings",
    "ThinkingSettings",
    "MCPServerConfig",
    # Settings mixin
    "WorkflowSettingsMixin",
    # Managers (lazy)
    "GoogleADKWorkflowManager",
    "DeepAgentsWorkflowManager",
    "LangGraphWorkflowManager",
    "LangGraphManager",
    # LangGraph state (lazy, for backward compatibility)
    "AgentState",
    "ResearchState",
    "ApprovalState",
    "CheckpointData",
    "add_messages",
    # Service registry
    "get_service",
    "get_service_registry",
]
