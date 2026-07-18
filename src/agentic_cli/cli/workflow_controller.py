"""Workflow controller for managing workflow lifecycle.

This module provides the WorkflowController class that encapsulates
the complex async lifecycle of workflow manager initialization,
including background init, readiness checking, and reinitialization.

The factory function ``create_workflow_manager_from_settings()`` now
lives in ``workflow.factory``; it is re-exported here for backward
compatibility.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncIterator

from agentic_cli.logging import Loggers

# Re-export factory helpers for backward compatibility
from agentic_cli.workflow.factory import (  # noqa: F401
    _is_claude_model,
    _resolve_effective_model,
    create_workflow_manager_from_settings,
)

if TYPE_CHECKING:
    from agentic_cli.cli.usage_tracker import UsageTracker
    from agentic_cli.config import BaseSettings
    from agentic_cli.workflow.base_manager import BaseWorkflowManager
    from agentic_cli.workflow.config import AgentConfig
    from thinking_prompt import ThinkingPromptSession

logger = Loggers.cli()


class WorkflowController:
    """Manages the workflow manager lifecycle.

    Encapsulates:
    - Background initialization in ThreadPoolExecutor
    - Readiness checking (blocking and non-blocking); the manager is
      published only after initialize_services() succeeds, so is_ready
      implies a fully-initialized manager
    - Reinitialization when model/settings change (an orchestrator swap
      initializes the replacement first, swaps atomically, then cleans up
      the old manager)
    - close(): idempotent shutdown — cancels pending init and cleans up
      the live manager; invoked on app exit via background_init()

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
        self._agent_configs = agent_configs
        self._app_name = settings.app_name

        # Closure captures agent_configs for lazy creation (used by _background_init)
        def _create_workflow() -> "BaseWorkflowManager":
            return create_workflow_manager_from_settings(
                agent_configs=agent_configs,
                settings=settings,
                app_name=self._app_name,
            )

        self._create_fn = _create_workflow

        # Thread pool for background initialization (per-controller, single worker)
        self._init_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="workflow-init"
        )

        # Workflow state
        self._workflow: "BaseWorkflowManager | None" = None
        self._init_task: asyncio.Task[None] | None = None
        self._init_error: Exception | None = None
        self.usage_tracker: "UsageTracker | None" = None
        # Status-bar jobs segment, published by JobMonitor; None when idle.
        self.jobs_status_segment: str | None = None

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

        The manager is published to ``self._workflow`` only after
        initialize_services() succeeds, so ``is_ready`` /
        ``ensure_initialized()`` never report a partially-initialized or
        failed manager as ready.
        """
        loop = asyncio.get_running_loop()

        def _create_workflow() -> "BaseWorkflowManager":
            return self._create_fn()

        manager: "BaseWorkflowManager | None" = None
        try:
            logger.debug("background_init_starting")

            # Step 1: Create workflow manager (sync, in thread pool)
            manager = await loop.run_in_executor(
                self._init_executor, _create_workflow
            )

            # Step 2: Initialize services (async - builds graph, loads LLM, etc.)
            await manager.initialize_services()

            self._workflow = manager
            logger.info("background_init_complete", model=manager.model)

        except asyncio.CancelledError:
            if manager is not None:
                await self._cleanup_manager(manager)
            raise
        except Exception as e:
            self._init_error = e
            if manager is not None:
                await self._cleanup_manager(manager)
            logger.debug("background_init_failed", error=str(e))

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
                ctx = ui.start_thinking(lambda: "Waiting for initialization...", content_format="ansi")
            try:
                await self._init_task
            finally:
                if ui is not None:
                    ctx.finish(add_to_history=False)

        if self._init_error:
            if ui is not None:
                ui.add_error(f"Initialization failed: {self._init_error}")
            return False

        return self._workflow is not None

    def _needs_orchestrator_swap(self, new_model: str | None = None) -> bool:
        """Check if the current manager still matches the orchestrator setting.

        The backend is chosen purely by ``settings.orchestrator`` and is
        model-agnostic (ADK runs Claude natively via ``AnthropicLlm``), so a model
        change alone never forces a swap. A swap is needed when the live manager's
        ``backend_type`` no longer matches the configured orchestrator — e.g. the
        orchestrator setting was changed, leaving a stale manager in place. This
        happens regardless of whether a new model was given.

        Compares by ``backend_type`` string rather than importing a backend class,
        so it never pulls in the optional ``langgraph`` extra on an ADK-only
        install.
        """
        if self._workflow is None:
            return False

        from agentic_cli.workflow.settings import OrchestratorType

        orchestrator = getattr(self._settings, "orchestrator", OrchestratorType.ADK)
        target_backend = getattr(orchestrator, "value", str(orchestrator))
        return getattr(self._workflow, "backend_type", None) != target_backend

    async def reinitialize(self, model: str | None = None) -> None:
        """Reinitialize the workflow with optional new model.

        If the live manager's backend no longer matches ``settings.orchestrator``
        (e.g. the orchestrator setting was changed), the entire workflow manager
        is replaced. Otherwise, the existing manager is reinitialized in place.

        Args:
            model: Optional new model to use

        Raises:
            RuntimeError: If workflow is not initialized
            Exception: If reinitialization fails
        """
        if self._workflow is None:
            raise RuntimeError("Cannot reinitialize - workflow not initialized")

        if self._needs_orchestrator_swap(model):
            logger.info(
                "orchestrator_swap",
                old_model=self._workflow.model,
                new_model=model,
            )
            # Initialize the replacement fully before swapping so a failed
            # init leaves the working manager in place; clean up whichever
            # manager ends up unused.
            new_workflow = create_workflow_manager_from_settings(
                agent_configs=self._agent_configs,
                settings=self._settings,
                app_name=self._app_name,
                model=model,
            )
            try:
                await new_workflow.initialize_services()
            except Exception:
                await self._cleanup_manager(new_workflow)
                raise
            old_workflow, self._workflow = self._workflow, new_workflow
            await self._cleanup_manager(old_workflow)
        else:
            await self._workflow.reinitialize(model=model, preserve_sessions=True)

    async def cancel_init(self) -> None:
        """Cancel pending initialization task if still running."""
        if self._init_task and not self._init_task.done():
            self._init_task.cancel()
            try:
                await self._init_task
            except asyncio.CancelledError:
                pass
        self._init_executor.shutdown(wait=False)

    @staticmethod
    async def _cleanup_manager(manager: "BaseWorkflowManager") -> None:
        """Best-effort manager cleanup; a failing cleanup is logged, not raised."""
        try:
            await manager.cleanup()
        except Exception as e:
            logger.warning("workflow_manager_cleanup_failed", error=str(e))

    async def close(self) -> None:
        """Release the controller: cancel pending init, clean up the manager.

        Idempotent — safe to call multiple times. Invoked from application
        shutdown (the background_init context manager exit).
        """
        await self.cancel_init()
        manager, self._workflow = self._workflow, None
        if manager is not None:
            await self._cleanup_manager(manager)

    def update_status_bar(self, ui: "ThinkingPromptSession") -> None:
        """Update UI status bar with current workflow status.

        Args:
            ui: UI session to update
        """
        if self._init_error:
            ui.set_status("Init failed - check API keys")
        elif self._workflow is not None:
            parts = [self._workflow.model]
            if self.usage_tracker:
                token_summary = self.usage_tracker.format_status_bar()
                if token_summary:
                    parts.append(token_summary)
            if self.jobs_status_segment:
                parts.append(self.jobs_status_segment)
            parts.extend(["Ctrl+C: cancel", "/help: commands"])
            ui.set_status(" | ".join(parts))
        # If still initializing, leave status bar unchanged

    @asynccontextmanager
    async def background_init(
        self,
        ui: "ThinkingPromptSession",
    ) -> AsyncIterator[None]:
        """Context manager for background initialization lifecycle.

        Starts background init on entry, cancels on exit.
        Spawns a task to update status bar when init completes.

        Args:
            ui: UI session for status bar updates

        Usage:
            async with controller.background_init(session):
                await session.run_async()
        """
        # Start background initialization (non-blocking)
        await self.start_background_init()

        # Spawn task to update status bar when init completes
        async def wait_and_update() -> None:
            await self.ensure_initialized()
            self.update_status_bar(ui)

        update_task = asyncio.create_task(wait_and_update())

        try:
            yield
        finally:
            # Cancel status update task if still running
            if not update_task.done():
                update_task.cancel()
                try:
                    await update_task
                except asyncio.CancelledError:
                    pass
            # Cancel pending init and clean up the manager (app shutdown)
            await self.close()
