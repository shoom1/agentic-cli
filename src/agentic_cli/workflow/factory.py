"""Workflow manager factory.

Creates the workflow manager for ``settings.orchestrator``. Each orchestrator
has one entry in ``_BACKENDS``: the ``"module:Class"`` that implements it and
the extra that installs it. A backend is imported only when it is chosen, so
an application never loads a backend it does not run. This module has no CLI
dependencies — it can be imported by any consumer (web, CLI, scripts).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Callable

from agentic_cli.logging import Loggers
from agentic_cli.workflow.settings import OrchestratorType

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings
    from agentic_cli.workflow.base_manager import BaseWorkflowManager
    from agentic_cli.workflow.config import AgentConfig

logger = Loggers.workflow()

# Orchestrator → ("module:Class", the extra that installs it). ADK is a core
# dependency, so it has no extra.
_BACKENDS: dict[OrchestratorType, tuple[str, str]] = {
    OrchestratorType.ADK: ("agentic_cli.workflow.adk.manager:GoogleADKWorkflowManager", ""),
    OrchestratorType.LANGGRAPH: (
        "agentic_cli.workflow.langgraph:LangGraphWorkflowManager",
        "langgraph",
    ),
}


def _langgraph_kwargs(settings: "BaseSettings") -> dict[str, Any]:
    # Durable sessions: the unified session_store (sqlite by default) selects
    # the checkpointer so LangGraph persists like ADK. "memory" stays ephemeral.
    return {"checkpointer": getattr(settings, "session_store", "memory")}


# Constructor arguments that only one backend takes, by that backend's entry.
_BACKEND_KWARGS: dict[OrchestratorType, Callable[["BaseSettings"], dict[str, Any]]] = {
    OrchestratorType.LANGGRAPH: _langgraph_kwargs,
}


def _orchestrator(settings: "BaseSettings") -> OrchestratorType:
    """The orchestrator ``settings`` names.

    Settings validation allows only ``OrchestratorType`` values; a settings
    object built some other way may hold a plain string, and anything that is
    no orchestrator runs on ADK, the default.
    """
    value = getattr(settings, "orchestrator", OrchestratorType.ADK)
    try:
        return OrchestratorType(value)
    except ValueError:
        return OrchestratorType.ADK


def _load_backend(orchestrator: OrchestratorType) -> type["BaseWorkflowManager"]:
    """Import the manager class that implements ``orchestrator``.

    Only the import is guarded: a backend whose extra is not installed raises
    ``ImportError`` with that extra's install hint. A backend without an extra
    is part of the core install, so its ``ImportError`` is raised as it is.

    Raises:
        ImportError: The backend's module cannot be imported.
    """
    orchestrator = OrchestratorType(orchestrator)
    target, extra = _BACKENDS[orchestrator]
    module_name, _, class_name = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        if not extra:
            raise
        raise ImportError(
            f"The {orchestrator.value} orchestrator needs the '{extra}' extra. "
            f"Install with: pip install 'agentic-cli[{extra}]'\n"
            f"Original error: {e}"
        ) from e
    return getattr(module, class_name)


def _backend_kwargs(orchestrator: OrchestratorType, settings: "BaseSettings") -> dict[str, Any]:
    """The constructor arguments only ``orchestrator``'s backend takes."""
    build = _BACKEND_KWARGS.get(orchestrator)
    return build(settings) if build is not None else {}


def create_workflow_manager_from_settings(
    agent_configs: list["AgentConfig"],
    settings: "BaseSettings",
    app_name: str | None = None,
    model: str | None = None,
    adk_config_path: str | None = None,
    adk_config_mode: str = "native",
    **kwargs,
) -> "BaseWorkflowManager":
    """Factory function to create the appropriate workflow manager based on settings.

    Creates the manager of the backend ``settings.orchestrator`` names. The
    backend is model-agnostic: each backend builds every agent's model from its
    resolved model, so no model forces a backend.

    Args:
        agent_configs: List of agent configurations.
        settings: Application settings (determines orchestrator type).
        app_name: Application name for services.
        model: The manager's own default model, a ``provider:model``
            reference or a model id; it takes the ``default`` role.
        adk_config_path: A native ADK YAML agent config. In ``native`` mode it
            runs on ADK whichever orchestrator is set; in ``translate`` mode it
            becomes ``AgentConfig``s for the configured orchestrator.
        adk_config_mode: ``"native"`` or ``"translate"``.
        **kwargs: Additional arguments passed to the specific manager.

    Returns:
        The orchestrator's ``BaseWorkflowManager``.

    Raises:
        ImportError: If the orchestrator's extra is not installed.

    Example:
        settings = MySettings(orchestrator="langgraph")
        configs = [AgentConfig(name="agent", prompt="...")]
        manager = create_workflow_manager_from_settings(configs, settings)
    """
    orchestrator = _orchestrator(settings)

    # Reuse an existing native ADK config, if provided.
    if adk_config_path:
        if adk_config_mode == "translate":
            from agentic_cli.workflow.adk_config_bridge import translate_adk_yaml

            agent_configs = translate_adk_yaml(adk_config_path)
            # fall through to normal routing with the translated configs
        else:  # native passthrough — ADK backend only
            if orchestrator is not OrchestratorType.ADK:
                logger.warning(
                    "adk_native_config_forces_adk",
                    reason="native ADK config requires the ADK backend",
                )
            manager_cls = _load_backend(OrchestratorType.ADK)
            return manager_cls(
                agent_configs=agent_configs,
                settings=settings,
                app_name=app_name,
                model=model,
                adk_config_path=adk_config_path,
                **kwargs,
            )

    manager_cls = _load_backend(orchestrator)
    return manager_cls(
        agent_configs=agent_configs,
        settings=settings,
        app_name=app_name,
        model=model,
        **_backend_kwargs(orchestrator, settings),
        **kwargs,
    )
