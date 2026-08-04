"""Background harness monitor that keeps long-running jobs visible in the UI.

Milestone 2 of the job-control substrate. The agent loop is **not** involved:
this task runs independently of any LLM turn, so detached jobs advance their
state and stay visible even while the user is idle at the prompt or busy in an
unrelated turn.

Surface choice — the **status bar only**. ``thinking_prompt`` thinking boxes are
turn-oriented (created and finished within a single turn) and the ``add_*``
output methods print straight to the console via ``print_formatted_text``, which
would corrupt the live prompt if called from a background coroutine.
``ThinkingPromptSession.set_status`` merely updates the status text and
invalidates the app, which is safe to call from any coroutine. So the monitor
renders job state into a status-bar segment and never prints.

The monitor owns the *jobs* part of the status line; ``WorkflowController``
remains the single composer of the full bar (model | tokens | jobs | hints) and
reads the segment the monitor publishes via ``jobs_status_segment``.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator

from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from thinking_prompt import ThinkingPromptSession

    from agentic_cli.cli.workflow_controller import WorkflowController

logger = Loggers.cli()

# Glyphs for the transient "just finished" note in the status bar.
_DONE_GLYPH = {
    "succeeded": "✓",
    "failed": "✗",
    "cancelled": "⊘",
    "unknown": "?",
}

_DEFAULT_INTERVAL = 2.0  # seconds between reconcile/render ticks
_DEFAULT_NOTE_TICKS = 5  # how many ticks a finished-job note lingers


class JobMonitor:
    """Periodically reconcile jobs and render their state to the status bar.

    Args:
        ui: The active ``ThinkingPromptSession`` (only ``set_status`` is used,
            indirectly, via the controller).
        controller: The ``WorkflowController``; provides the lazily-created
            ``job_manager`` and composes the full status bar.
        interval: Seconds between ticks.
        note_ticks: How many ticks a finished-job note stays in the segment.
    """

    def __init__(
        self,
        ui: "ThinkingPromptSession",
        controller: "WorkflowController",
        *,
        settings: Any = None,
        interval: float = _DEFAULT_INTERVAL,
        note_ticks: int = _DEFAULT_NOTE_TICKS,
    ) -> None:
        self._ui = ui
        self._controller = controller
        self._settings = settings
        self._interval = interval
        self._note_ticks = note_ticks
        self._task: asyncio.Task[None] | None = None
        # Last seen state value per job id, to detect terminal transitions.
        self._states: dict[str, str] = {}
        # Recently-finished notes: ``[label, ticks_remaining]`` entries.
        self._notes: list[list[Any]] = []
        # Last segment we published, to avoid redundant status-bar redraws.
        self._last_segment: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background poll loop (idempotent)."""
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Cancel the poll loop and wait for it to unwind."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    @asynccontextmanager
    async def running(self) -> AsyncIterator["JobMonitor"]:
        """Run the monitor for the duration of the ``async with`` block."""
        self.start()
        try:
            yield self
        finally:
            await self.stop()

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _job_manager(self) -> Any | None:
        """Return the JobManager if the workflow is ready and has one."""
        if not self._controller.is_ready:
            return None
        try:
            return self._controller.workflow.job_manager
        except (RuntimeError, AttributeError):
            return None

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            try:
                self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a transient error must not kill the loop
                logger.debug("job_monitor_poll_failed", exc_info=True)

    def poll_once(self) -> str | None:
        """Reconcile jobs, publish the segment, and refresh the bar on change.

        Returns the segment string (or ``None``). Exposed for tests so the tick
        logic can be exercised without the asyncio loop.
        """
        jm = self._job_manager()
        if jm is None:
            return self._last_segment
        # ``list()`` reconciles non-terminal jobs and returns all records.
        recs = jm.list()
        segment = self._build_segment(recs)
        self._controller.jobs_status_segment = segment
        if segment != self._last_segment:
            self._last_segment = segment
            self._controller.update_status_bar(self._ui)
        return segment

    # ------------------------------------------------------------------
    # Segment construction
    # ------------------------------------------------------------------

    def _build_segment(self, recs: list[Any]) -> str | None:
        """Build the status-bar jobs segment from the current job records."""
        from agentic_cli.tools.jobs.backends import TERMINAL_STATES

        terminal_vals = {s.value for s in TERMINAL_STATES}
        running = queued = pending_resume = 0
        seen: set[str] = set()
        auto_resume = bool(getattr(self._settings, "job_auto_resume", False))

        for rec in recs:
            seen.add(rec.job_id)
            state = rec.state.value
            if state == "running":
                running += 1
            elif state == "queued":
                queued += 1
            if (
                auto_resume
                and getattr(rec, "resume_on_complete", False)
                and not getattr(rec, "resumed", False)
                and state in terminal_vals
            ):
                pending_resume += 1
            prev = self._states.get(rec.job_id)
            # Announce a job that we previously saw active and is now terminal.
            if state in terminal_vals and prev is not None and prev not in terminal_vals:
                glyph = _DONE_GLYPH.get(state, "•")
                self._notes.append([f"{glyph} {self._short(rec.name)}", self._note_ticks])
            self._states[rec.job_id] = state

        # Forget jobs that were cleaned away so bookkeeping doesn't grow.
        for job_id in [j for j in self._states if j not in seen]:
            del self._states[job_id]

        # Age out transient notes.
        for note in self._notes:
            note[1] -= 1
        self._notes = [n for n in self._notes if n[1] > 0]

        parts: list[str] = []
        active: list[str] = []
        if running:
            active.append(f"{running} running")
        if queued:
            active.append(f"{queued} queued")
        if active:
            parts.append("jobs: " + ", ".join(active))
        parts.extend(label for label, _ in self._notes)
        if pending_resume:
            parts.append(f"↻{pending_resume} to resume")
        return " · ".join(parts) if parts else None

    @staticmethod
    def _short(name: str, limit: int = 24) -> str:
        name = (name or "").strip() or "job"
        return name if len(name) <= limit else name[: limit - 1] + "…"
