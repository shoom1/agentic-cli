"""Workflow manager factory.

Creates the appropriate workflow manager (ADK or LangGraph)
based on settings. This module has no CLI dependencies — it can
be imported by any consumer (web, CLI, scripts).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings
    from agentic_cli.workflow.base_manager import BaseWorkflowManager
    from agentic_cli.workflow.config import AgentConfig

logger = Loggers.workflow()


def _is_claude_model(model: str | None) -> bool:
    """Check if a model string refers to a Claude (Anthropic) model."""
    return model is not None and model.startswith("claude-")


def _resolve_effective_model(
    model: str | None, settings: "BaseSettings"
) -> str | None:
    """Resolve the effective model from explicit override or settings."""
    if model:
        return model
    return getattr(settings, "default_model", None)


def create_workflow_manager_from_settings(
    agent_configs: list["AgentConfig"],
    settings: "BaseSettings",
    app_name: str | None = None,
    model: str | None = None,
    **kwargs,
) -> "BaseWorkflowManager":
    """Factory function to create the appropriate workflow manager based on settings.

    Creates either a GoogleADKWorkflowManager or LangGraphWorkflowManager
    based on the settings.orchestrator configuration. Claude models are
    automatically routed to LangGraph because ADK's LiteLLM adapter has
    critical issues with tool calling, thinking, and streaming.

    Args:
        agent_configs: List of agent configurations.
        settings: Application settings (determines orchestrator type).
        app_name: Application name for services.
        model: Model override.
        **kwargs: Additional arguments passed to the specific manager.

    Returns:
        BaseWorkflowManager instance (ADK or LangGraph based on settings).

    Raises:
        ImportError: If LangGraph is selected but not installed.

    Example:
        settings = MySettings(orchestrator="langgraph")
        configs = [AgentConfig(name="agent", prompt="...")]
        manager = create_workflow_manager_from_settings(configs, settings)
    """
    from agentic_cli.workflow.settings import OrchestratorType

    orchestrator = getattr(settings, "orchestrator", OrchestratorType.ADK)
    effective_model = _resolve_effective_model(model, settings)
    use_langgraph = orchestrator == OrchestratorType.LANGGRAPH or _is_claude_model(
        effective_model
    )

    if use_langgraph and _is_claude_model(effective_model) and orchestrator != OrchestratorType.LANGGRAPH:
        logger.info(
            "auto_switching_to_langgraph",
            model=effective_model,
            reason="Claude models require LangGraph orchestrator (ADK LiteLLM adapter has critical issues)",
        )

    if use_langgraph:
        try:
            from agentic_cli.workflow.langgraph import LangGraphWorkflowManager

            checkpointer = getattr(settings, "langgraph_checkpointer", "memory")
            return LangGraphWorkflowManager(
                agent_configs=agent_configs,
                settings=settings,
                app_name=app_name,
                model=model,
                checkpointer=checkpointer,
                **kwargs,
            )
        except ImportError as e:
            raise ImportError(
                f"LangGraph orchestrator required but dependencies not installed. "
                f"Install with: pip install agentic-cli[langgraph]\n"
                f"Original error: {e}"
            ) from e

    else:  # Default to ADK
        from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager

        return GoogleADKWorkflowManager(
            agent_configs=agent_configs,
            settings=settings,
            app_name=app_name,
            model=model,
            **kwargs,
        )
