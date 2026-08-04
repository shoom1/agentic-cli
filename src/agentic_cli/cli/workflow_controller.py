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
from enum import Enum
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


class WorkflowState(str, Enum):
    """Lifecycle state of a :class:`WorkflowController`.

    Derived from the controller's internals rather than stored, so the reported
    state can never drift from reality:

    - ``UNINITIALIZED`` — no init attempted (or none since the last failure was
      cleared).
    - ``INITIALIZING`` — a background init task is in flight.
    - ``READY`` — a manager finished ``initialize_services()`` and is published.
    - ``FAILED`` — the last init attempt raised; ``init_error`` holds it.
      ``start_background_init()`` clears it and retries.
    - ``CLOSED`` — ``close()`` ran; the controller is terminal.
    """

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    FAILED = "failed"
    CLOSED = "closed"


class _Construction:
    """A manager being built in the init executor, settled exactly once.

    The worker thread cannot be cancelled, so whoever stops waiting for it must
    still take responsibility for what it eventually returns. ``claim()`` is the
    single-shot token that decides who does: shutdown if it gets there first,
    otherwise the future's done-callback.
    """

    __slots__ = ("future", "_settled")

    def __init__(self, future: "asyncio.Future") -> None:
        self.future = future
        self._settled = False

    def claim(self) -> bool:
        """Take responsibility for the result. True for exactly one caller."""
        if self._settled:
            return False
        self._settled = True
        return True

    def release(self) -> None:
        """Give the claim back, for a claimer that could not finish.

        A settler cancelled between claiming and releasing the manager would
        otherwise strand it: the claim is single-shot, so nobody else — not
        even the future's own callback — could take over.
        """
        self._settled = False


class WorkflowController:
    """Manages the workflow manager lifecycle.

    Encapsulates:
    - Background initialization in ThreadPoolExecutor, single-flight: repeated
      ``start_background_init()`` calls join the in-flight attempt instead of
      building a second manager
    - Readiness checking (blocking and non-blocking); the manager is published
      only after ``initialize_services()`` succeeds, so ``is_ready`` implies a
      fully-initialized manager and never masks ``init_error``
    - Reinitialization when model/settings change (an orchestrator swap
      initializes the replacement first, swaps atomically, then cleans up the
      old manager)
    - ``close()``: idempotent shutdown — cancels pending init, shuts down the
      init executor, and cleans up the live manager; invoked on app exit via
      ``background_init()``

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
        self._closed = False
        # Serializes every lifecycle transition (start_background_init,
        # reinitialize/swap, close). Each of them read-modify-writes
        # ``_workflow`` across awaits, so two in flight could publish two
        # managers — leaking one — or publish after close(). The background
        # init *task* never takes this lock (close() awaits that task while
        # holding it), it checks ``_closed`` before publishing instead.
        self._lifecycle_lock = asyncio.Lock()
        # A manager currently being constructed in the init executor, and every
        # in-flight cleanup of one the controller ended up owning — whether it
        # finished building after we stopped waiting, or its release outlived
        # the cancelled caller that asked for it. Both exist because neither a
        # worker thread nor an unreachable manager's cleanup can be abandoned.
        self._construction: "_Construction | None" = None
        self._orphan_cleanups: set[asyncio.Task] = set()
        # Single-flight shutdown, owned by the controller so a cancelled caller
        # cannot abandon it half-done (see close()).
        self._close_task: asyncio.Task[None] | None = None
        self.usage_tracker: "UsageTracker | None" = None
        # Status-bar jobs segment, published by JobMonitor; None when idle.
        self.jobs_status_segment: str | None = None

    @property
    def workflow(self) -> "BaseWorkflowManager":
        """Get the workflow manager.

        Only ever a ``READY`` one. A manager can be published but unusable —
        a failed in-place reinitialization leaves it uninitialized, and it is
        deliberately retained so a retry can reuse its (possibly in-memory)
        session store — and handing that out would look like success.

        Raises:
            RuntimeError: If no fully initialized workflow is available.
        """
        if self._workflow is None or self.state is not WorkflowState.READY:
            raise RuntimeError("Workflow not initialized yet")
        return self._workflow

    @property
    def state(self) -> WorkflowState:
        """Current lifecycle state (derived, never stored — cannot drift)."""
        if self._closed:
            return WorkflowState.CLOSED
        if self._workflow is not None:
            # A published manager that failed an in-place reinitialization is
            # no longer usable, whatever the controller last recorded.
            if getattr(self._workflow, "is_initialized", True):
                return WorkflowState.READY
            return WorkflowState.FAILED
        if self._init_task is not None and not self._init_task.done():
            return WorkflowState.INITIALIZING
        if self._init_error is not None:
            return WorkflowState.FAILED
        return WorkflowState.UNINITIALIZED

    @property
    def is_ready(self) -> bool:
        """True only when a fully initialized manager is published."""
        return self.state is WorkflowState.READY

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
        """Start background initialization of the workflow manager.

        Creates an async task that:
        1. Creates workflow manager in ThreadPoolExecutor
        2. Calls initialize_services() to preload LLM, build graph, etc.

        Non-blocking, single-flight and retryable:

        - already ``READY`` → no-op;
        - already ``INITIALIZING`` → no-op (the in-flight attempt is joined by
          ``ensure_initialized()``), so two callers can never build two managers;
        - ``FAILED`` → the recorded error is cleared and a fresh attempt starts.

        A manager that is published but uninitialized (a failed in-place
        reinitialization) is *revived* rather than replaced: it still owns the
        session service that reinitialization preserved, and with
        ``session_store='memory'`` building a replacement would silently throw
        the conversation away. Only if reviving it fails is it released.

        Raises:
            RuntimeError: If the controller has been closed.
        """
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("WorkflowController is closed")
            if self.state is WorkflowState.READY:
                return
            if self._init_task is not None and not self._init_task.done():
                return
            revive, self._workflow = self._workflow, None
            self._init_error = None
            self._init_task = asyncio.create_task(self._background_init(revive))

    async def _background_init(
        self, revive: "BaseWorkflowManager | None" = None
    ) -> None:
        """Initialize workflow manager in background.

        Creates the workflow manager and calls initialize_services() to
        preload LLM, build graph, and set up checkpointing. This avoids
        lag on the first user message.

        The manager is published to ``self._workflow`` only after
        initialize_services() succeeds, so ``is_ready`` / ``ensure_initialized()``
        never report a partially-initialized or failed manager as ready — and
        only if the controller has not been closed in the meantime, so shutdown
        never leaves a live backend behind. A manager that fails (or is
        cancelled) mid-init is cleaned up rather than leaked.

        Args:
            revive: An existing, uninitialized manager to re-initialize instead
                of building a new one (see ``start_background_init``).
        """
        loop = asyncio.get_running_loop()

        def _create_workflow() -> "BaseWorkflowManager":
            return self._create_fn()

        manager: "BaseWorkflowManager | None" = revive
        try:
            logger.debug("background_init_starting", reviving=revive is not None)

            # Step 1: Create workflow manager (sync, in thread pool)
            if manager is None:
                manager = await self._construct_manager(loop, _create_workflow)

            # Step 2: Initialize services (async - builds graph, loads LLM, etc.)
            await manager.initialize_services()

            if self._closed:
                # close() ran while we were initializing; it has already taken
                # its snapshot of _workflow, so publishing now would strand
                # this manager. Release it instead.
                await self._cleanup_manager(manager)
                logger.debug("background_init_discarded_after_close")
                return

            self._workflow = manager
            # A recorded failure must not outlive its recovery: state,
            # ensure_initialized() and the status bar all read this field.
            self._init_error = None
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

    async def _construct_manager(self, loop, create_fn) -> "BaseWorkflowManager":
        """Build a manager in the init executor, keeping ownership of the result.

        The await is **shielded**: cancelling it would cancel the asyncio future
        too, and asyncio then discards whatever the (uncancellable) worker
        thread returns — a fully constructed manager, unreachable and never
        cleaned up. Shielded, the future survives, so shutdown can settle it or
        its done-callback can release it.
        """
        construction = _Construction(loop.run_in_executor(self._init_executor, create_fn))
        self._construction = construction
        try:
            manager = await asyncio.shield(construction.future)
        except BaseException:
            self._abandon_construction(construction)
            raise
        construction.claim()  # the result is ours; nobody else may release it
        if self._construction is construction:
            self._construction = None
        return manager

    def _spawn_cleanup(self, manager: "BaseWorkflowManager") -> "asyncio.Task | None":
        """Release a manager in a task the **controller** owns.

        Cleanup is not the caller's to abandon. By the time it starts, the
        manager is already unreachable — nothing else holds a reference — so a
        cleanup cancelled halfway leaks its backend (session service, sandbox,
        job manager) for the life of the process, with no one left to retry.
        Running it in a tracked task means a cancelled caller only stops
        *waiting*, and a later ``close()`` can join what it left running.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - loop already gone
            logger.warning("orphan_manager_not_cleaned")
            return None
        task = loop.create_task(self._cleanup_manager(manager))
        self._orphan_cleanups.add(task)
        task.add_done_callback(self._orphan_cleanups.discard)
        return task

    def _abandon_construction(self, construction: "_Construction") -> None:
        """Arrange for a manager we no longer want to be released, once.

        Shutdown settles ``_construction`` deterministically; this callback is
        the fallback for a cancellation that is never followed by ``close()``.
        Whichever runs first claims the result, so it is cleaned exactly once
        and never published.
        """

        def _on_done(future: "asyncio.Future") -> None:
            if not construction.claim():
                return  # shutdown got there first
            if future.cancelled() or future.exception() is not None:
                return
            self._spawn_cleanup(future.result())

        construction.future.add_done_callback(_on_done)

    async def _settle_construction(self) -> None:
        """Release anything the init executor is still building.

        Cancellation-safe *through completion*: ``cancel_init()`` is public and
        may be awaited directly, so a caller can be cancelled at either of the
        two waits here.

        - Still waiting on the worker thread: the claim is **handed back** and
          the fallback callback re-armed, so the manager is released exactly
          once by whichever of that callback or a later ``close()`` gets there
          first.
        - Already releasing the manager: the cleanup belongs to the controller
          and keeps running; it stays tracked in ``_orphan_cleanups``, which is
          what a later ``close()`` joins. Cancelling here once consumed the
          claim *and* aborted the cleanup, leaving a half-released manager that
          nothing could finish.
        """
        construction, self._construction = self._construction, None
        if construction is not None and construction.claim():
            try:
                manager = await asyncio.shield(construction.future)
            except asyncio.CancelledError:
                construction.release()
                self._construction = construction
                self._abandon_construction(construction)
                raise
            except Exception as exc:  # noqa: BLE001 - shutdown must not fail
                logger.debug("construction_failed_during_shutdown", error=str(exc))
            else:
                cleanup = self._spawn_cleanup(manager)
                if cleanup is not None:
                    await asyncio.shield(cleanup)
        if self._orphan_cleanups:
            # Shielded for the same reason: these are the controller's tasks,
            # and gather() would otherwise cancel them along with its awaiter.
            await asyncio.shield(
                asyncio.gather(*list(self._orphan_cleanups), return_exceptions=True)
            )

    async def ensure_initialized(
        self,
        ui: "ThinkingPromptSession | None" = None,
    ) -> bool:
        """Wait for background initialization to complete.

        Reports readiness truthfully: it awaits the in-flight attempt (if any)
        and then answers from the resulting state, so a failed initialization
        is never reported as ready.

        Recovers from ``FAILED``: a fresh attempt is started (settings may have
        been corrected since), so a failed reinitialization does not wedge the
        session until restart.

        Args:
            ui: Optional UI session for showing "waiting" feedback

        Returns:
            True if a fully initialized manager is available, False otherwise
        """
        if self._closed:
            return False

        if self.state is WorkflowState.FAILED:
            await self.start_background_init()

        if self._init_task is not None and not self._init_task.done():
            # Show user we're waiting for initialization
            if ui is not None:
                ctx = ui.start_thinking(lambda: "Waiting for initialization...", content_format="ansi")
            try:
                await asyncio.shield(self._init_task)
            except asyncio.CancelledError:
                if self._init_task.cancelled():
                    return False
                raise
            finally:
                if ui is not None:
                    ctx.finish(add_to_history=False)

        # Readiness is the state, not the presence of a past error — the two
        # agreed only as long as every success remembered to clear the error.
        ready = self.state is WorkflowState.READY
        if not ready and self._init_error is not None and ui is not None:
            ui.add_error(f"Initialization failed: {self._init_error}")
        return ready

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

        Either outcome is well-defined: on success a fully initialized manager
        is published; on failure the controller enters ``FAILED`` with
        ``init_error`` set, so nothing can observe a READY controller wrapping
        an uninitialized manager (``workflow`` refuses to hand it out).
        Recovery is a fresh ``start_background_init()``
        (``ensure_initialized()`` triggers one), which revives that same
        manager so its preserved sessions survive.

        Serialized against every other lifecycle transition, so two concurrent
        swaps cannot both publish and leak one of the replacements.

        Args:
            model: Optional new model to use

        Raises:
            RuntimeError: If workflow is not initialized
            Exception: If reinitialization fails
        """
        async with self._lifecycle_lock:
            if self._workflow is None:
                raise RuntimeError("Cannot reinitialize - workflow not initialized")

            if self._needs_orchestrator_swap(model):
                await self._swap_orchestrator(model)
            else:
                await self._reinitialize_in_place(model)

    async def _swap_orchestrator(self, model: str | None) -> None:
        """Replace the manager with one for the configured orchestrator.

        Initializes the replacement fully before swapping, so a failed init
        leaves the working manager in place; whichever manager ends up unused
        is cleaned up. The caller holds ``_lifecycle_lock``.
        """
        logger.info(
            "orchestrator_swap", old_model=self._workflow.model, new_model=model
        )
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
        if self._closed:
            await self._cleanup_manager(new_workflow)
            raise RuntimeError("WorkflowController is closed")
        old_workflow, self._workflow = self._workflow, new_workflow
        self._init_error = None
        await self._cleanup_manager(old_workflow)

    async def _reinitialize_in_place(self, model: str | None) -> None:
        """Reinitialize the live manager. The caller holds ``_lifecycle_lock``.

        On failure the manager is **kept**, not released: it rolled itself back
        to uninitialized (so ``state`` is FAILED and ``workflow`` refuses to
        hand it out), but it still owns the session service its own
        ``reinitialize(preserve_sessions=True)`` restored. Releasing it here
        closed that service — with ``session_store='memory'`` the conversation
        went with it, for a failure the user could correct and retry.
        """
        try:
            await self._workflow.reinitialize(model=model, preserve_sessions=True)
        except Exception as e:
            self._init_error = e
            logger.warning("reinitialize_failed", error=str(e))
            raise
        # Success: drop any error recorded by an earlier attempt, so state,
        # ensure_initialized() and the status bar cannot disagree.
        self._init_error = None

    async def cancel_init(self) -> None:
        """Cancel pending initialization task and shut the init executor down.

        Terminal for the executor: after this, ``start_background_init()`` can
        no longer schedule work, so use it as part of shutdown (see ``close()``)
        rather than to abort one attempt.

        Waits for any manager still under construction in the executor and
        releases it — the worker thread is not cancellable, and a manager it
        returns after we stop waiting would otherwise be unreachable.
        """
        if self._init_task and not self._init_task.done():
            self._init_task.cancel()
            try:
                await self._init_task
            except asyncio.CancelledError:
                pass
        await self._settle_construction()
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

        Idempotent and terminal — repeated calls join the same shutdown and the
        controller stays ``CLOSED``. Invoked from application shutdown (the
        ``background_init()`` context manager exit).

        The teardown runs in a task the **controller** owns, and callers join it
        under a shield: whoever asked for the shutdown may be cancelled (Ctrl+C
        during exit, a cancelled task group) without abandoning a manager
        half-cleaned or a construction still running in the executor. A later
        ``close()`` therefore waits for that work rather than returning because
        the flag is already set.
        """
        # Set synchronously, before any await: nothing may publish from here on.
        self._closed = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_once())
        await asyncio.shield(self._close_task)

    async def _close_once(self) -> None:
        """The actual teardown. Runs exactly once; owned by the controller.

        Takes the lifecycle lock, so it waits for an in-flight reinitialization
        or swap instead of racing it, and nothing can publish afterwards: a
        background init that finishes later sees ``_closed`` and releases its
        manager rather than installing it.
        """
        async with self._lifecycle_lock:
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
            # Cancel the status-update task first so it isn't woken by the
            # init cancellation below, then release everything we own.
            if not update_task.done():
                update_task.cancel()
                try:
                    await update_task
                except asyncio.CancelledError:
                    pass
            await self.close()
