"""Workflow controller for managing workflow lifecycle.

This module provides the WorkflowController class that encapsulates
the complex async lifecycle of workflow manager initialization,
including background init, readiness checking, and reinitialization.

Also exports create_workflow_manager_from_settings() factory function.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings
    from agentic_cli.workflow.base_manager import BaseWorkflowManager
    from agentic_cli.workflow.config import AgentConfig
    from thinking_prompt import ThinkingPromptSession

logger = Loggers.cli()

# Thread pool for background initialization (single worker)
_init_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="workflow-init")


def create_workflow_manager_from_settings(
    agent_configs: list["AgentConfig"],
    settings: "BaseSettings",
    app_name: str | None = None,
    model: str | None = None,
    **kwargs,
) -> "BaseWorkflowManager":
    """Factory function to create the appropriate workflow manager based on settings.

    Creates either a GoogleADKWorkflowManager or LangGraphWorkflowManager
    based on the settings.orchestrator configuration.

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

    if orchestrator == OrchestratorType.LANGGRAPH:
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
                f"LangGraph orchestrator selected but dependencies not installed. "
                f"Install with: pip install agentic-cli[langgraph]\n"
                f"Original error: {e}"
            ) from e

    else:  # Default to ADK
        from agentic_cli.workflow.adk_manager import GoogleADKWorkflowManager

        return GoogleADKWorkflowManager(
            agent_configs=agent_configs,
            settings=settings,
            app_name=app_name,
            model=model,
            **kwargs,
        )


class WorkflowController:
    """Manages the workflow manager lifecycle.

    Encapsulates:
    - Background initialization in ThreadPoolExecutor
    - Readiness checking (blocking and non-blocking)
    - Reinitialization when model/settings change
    - Cleanup of pending init tasks

    Example:
        controller = WorkflowController(
            agent_configs=configs,
            settings=settings,
        )
        await controller.start_background_init()

        # Later, when needed:
        if await controller.ensure_initialized(ui):
            workflow = controller.workflow
            async for event in workflow.process(message, user_id):
                ...
    """

    def __init__(
        self,
        agent_configs: list["AgentConfig"],
        settings: "BaseSettings",
    ) -> None:
        """Initialize the workflow controller.

        Args:
            agent_configs: List of agent configurations for the workflow
            settings: Application settings instance
        """
        self._settings = settings

        # Closure captures agent_configs for lazy creation (used by _background_init)
        app_name = settings.app_name

        def _create_workflow() -> "BaseWorkflowManager":
            return create_workflow_manager_from_settings(
                agent_configs=agent_configs,
                settings=settings,
                app_name=app_name,
            )

        self._create_fn = _create_workflow

        # Workflow state
        self._workflow: "BaseWorkflowManager | None" = None
        self._init_task: asyncio.Task[None] | None = None
        self._init_error: Exception | None = None

    @property
    def workflow(self) -> "BaseWorkflowManager":
        """Get the workflow manager.

        Raises:
            RuntimeError: If workflow is not yet initialized
        """
        if self._workflow is None:
            raise RuntimeError("Workflow not initialized yet")
        return self._workflow

    @property
    def is_ready(self) -> bool:
        """Check if workflow is initialized and ready."""
        return self._workflow is not None

    @property
    def init_error(self) -> Exception | None:
        """Get initialization error, if any."""
        return self._init_error

    @property
    def model(self) -> str | None:
        """Get the current model name, or None if not initialized."""
        if self._workflow is None:
            return None
        return self._workflow.model

    async def start_background_init(self) -> None:
        """Start background initialization of workflow manager.

        Creates an async task that:
        1. Creates workflow manager in ThreadPoolExecutor
        2. Calls initialize_services() to preload LLM, build graph, etc.

        This is non-blocking - the task runs in the background.
        """
        self._init_task = asyncio.create_task(self._background_init())

    async def _background_init(self) -> None:
        """Initialize workflow manager in background.

        Creates the workflow manager and calls initialize_services() to
        preload LLM, build graph, and set up checkpointing. This avoids
        lag on the first user message.
        """
        loop = asyncio.get_running_loop()

        def _create_workflow() -> "BaseWorkflowManager":
            return self._create_fn()

        try:
            logger.debug("background_init_starting")

            # Step 1: Create workflow manager (sync, in thread pool)
            self._workflow = await loop.run_in_executor(
                _init_executor, _create_workflow
            )

            # Step 2: Initialize services (async - builds graph, loads LLM, etc.)
            await self._workflow.initialize_services()

            logger.info("background_init_complete", model=self._workflow.model)

        except Exception as e:
            self._init_error = e
            logger.error("background_init_failed", error=str(e))

    async def ensure_initialized(
        self,
        ui: "ThinkingPromptSession | None" = None,
    ) -> bool:
        """Wait for background initialization to complete.

        Args:
            ui: Optional UI session for showing "waiting" feedback

        Returns:
            True if initialization succeeded, False otherwise
        """
        if self._workflow is not None:
            return True

        if self._init_task is None:
            return False

        if not self._init_task.done():
            # Show user we're waiting for initialization
            if ui is not None:
                ui.start_thinking(lambda: "Waiting for initialization...")
            try:
                await self._init_task
            finally:
                if ui is not None:
                    ui.finish_thinking(add_to_history=False)

        if self._init_error:
            if ui is not None:
                ui.add_error(f"Initialization failed: {self._init_error}")
            return False

        return self._workflow is not None

    async def reinitialize(self, model: str | None = None) -> None:
        """Reinitialize the workflow with optional new model.

        Args:
            model: Optional new model to use

        Raises:
            RuntimeError: If workflow is not initialized
            Exception: If reinitialization fails
        """
        if self._workflow is None:
            raise RuntimeError("Cannot reinitialize - workflow not initialized")

        await self._workflow.reinitialize(model=model, preserve_sessions=True)

    async def cancel_init(self) -> None:
        """Cancel pending initialization task if still running."""
        if self._init_task and not self._init_task.done():
            self._init_task.cancel()
            try:
                await self._init_task
            except asyncio.CancelledError:
                pass

    def update_status_bar(self, ui: "ThinkingPromptSession") -> None:
        """Update UI status bar with current workflow status.

        Args:
            ui: UI session to update
        """
        if self._init_error:
            ui.status_text = "Init failed - check API keys"
        elif self._workflow is not None:
            model = self._workflow.model
            ui.status_text = f"{model} | Ctrl+C: cancel | /help: commands"
        # If still initializing, leave status bar unchanged
