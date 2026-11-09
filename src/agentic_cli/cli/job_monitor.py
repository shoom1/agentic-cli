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
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator

from agentic_cli.cli.waiting_jobs import waiting_here
from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from collections.abc import Callable

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
# How long the answer to "can this job's conversation be opened" is kept: a
# lookup may read the whole stored conversation, and the tick runs every 2 s.
_CONVERSATION_RECHECK_S = 30.0


class JobMonitor:
    """Periodically reconcile jobs and render their state to the status bar.

    Args:
        ui: The active ``ThinkingPromptSession`` (only ``set_status`` is used,
            indirectly, via the controller).
        controller: The ``WorkflowController``; provides the lazily-created
            ``job_manager`` and composes the full status bar.
        interval: Seconds between ticks.
        note_ticks: How many ticks a finished-job note stays in the segment.
        conversation: The current conversation's ``(user, session)``: the
            resume cue counts the finished jobs that wait for it, as
            ``/jobs resume`` tells them (``cli/waiting_jobs.py``; another
            conversation's are told when it is opened). The cue shows
            whatever ``job_auto_resume`` says, naming ``/jobs resume`` when
            nothing resumes them on its own.
    """

    def __init__(
        self,
        ui: "ThinkingPromptSession",
        controller: "WorkflowController",
        *,
        settings: Any = None,
        interval: float = _DEFAULT_INTERVAL,
        note_ticks: int = _DEFAULT_NOTE_TICKS,
        conversation: "Callable[[], tuple[str | None, str | None]] | None" = None,
    ) -> None:
        self._ui = ui
        self._controller = controller
        self._settings = settings
        # (user, session) of the conversation shown: the resume cue counts its
        # jobs only (None: every job).
        self._conversation = conversation
        self._interval = interval
        self._note_ticks = note_ticks
        self._task: asyncio.Task[None] | None = None
        # Last seen state value per job id, to detect terminal transitions.
        self._states: dict[str, str] = {}
        # Recently-finished notes: ``[label, ticks_remaining]`` entries.
        self._notes: list[list[Any]] = []
        # Last segment we published, to avoid redundant status-bar redraws.
        self._last_segment: str | None = None
        # Per job id: whether its conversation can be opened, and when that
        # was asked (``_conversation_opens``), of the workflow ``_opens_of``.
        self._opens: dict[str, tuple[bool, float]] = {}
        self._opens_of: Any = None

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

    def forget_conversations(self) -> None:
        """Forget what it was told of jobs' conversations: the next tick asks again.

        For when an answer kept may be wrong: a conversation was deleted
        (``/sessions --delete``). A workflow replaced (a new conversation on
        another backend) is noticed on its own.
        """
        self._opens.clear()

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
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a transient error must not kill the loop
                logger.debug("job_monitor_poll_failed", exc_info=True)

    async def poll_once(self) -> str | None:
        """Reconcile jobs, publish the segment, and refresh the bar on change.

        Returns the segment string (or ``None``). Exposed for tests so one
        tick runs without the poll loop.

        ``list()`` runs on a worker thread: it takes the jobs directory's
        lock, which another CLI of the app may hold, and the tick must not
        freeze the prompt meanwhile. A stop during it (``stop()``) ends the
        tick; the thread's ``list()`` finishes on its own.
        """
        jm = self._job_manager()
        if jm is None:
            return self._last_segment
        # ``list()`` reconciles non-terminal jobs and returns all records.
        recs = await asyncio.to_thread(jm.list)
        segment = self._build_segment(recs, len(await self._waiting(recs)))
        self._controller.jobs_status_segment = segment
        if segment != self._last_segment:
            self._last_segment = segment
            self._controller.update_status_bar(self._ui)
        return segment

    async def _waiting(self, recs: list[Any]) -> list[Any]:
        """The finished jobs among ``recs`` that wait for the current conversation."""
        here = self._conversation() if self._conversation is not None else None
        awaiting = [rec for rec in recs if rec.awaiting_resume]
        seen = {rec.job_id for rec in awaiting}
        for job_id in [j for j in self._opens if j not in seen]:
            del self._opens[job_id]
        return await waiting_here(awaiting, here, self._conversation_opens)

    async def _conversation_opens(self, record: Any) -> bool:
        """``workflow.can_resume(record)``, its answer kept ``_CONVERSATION_RECHECK_S``.

        Kept for the workflow that gave it: another one (a new conversation
        on another backend, which holds other conversations) is asked
        afresh. A
        failed lookup is not kept: the rule takes it as "it can be opened",
        and the next tick asks again.
        """
        workflow = self._controller.workflow
        if workflow is not self._opens_of:
            self._opens.clear()
            self._opens_of = workflow
        kept = self._opens.get(record.job_id)
        now = time.monotonic()
        if kept is not None and now - kept[1] < _CONVERSATION_RECHECK_S:
            return kept[0]
        can_resume = getattr(workflow, "can_resume", None)
        if can_resume is None:
            return True  # it cannot say: the job waits for its conversation
        opens = bool(await can_resume(record))
        self._opens[record.job_id] = (opens, now)
        return opens

    # ------------------------------------------------------------------
    # Segment construction
    # ------------------------------------------------------------------

    def _build_segment(self, recs: list[Any], waiting: int = 0) -> str | None:
        """Build the status-bar jobs segment from the current job records.

        ``waiting``: how many finished jobs wait for the current
        conversation (``_waiting``), the ``↻N`` cue.
        """
        from agentic_cli.tools.jobs.backends import TERMINAL_STATES

        terminal_vals = {s.value for s in TERMINAL_STATES}
        running = queued = 0
        seen: set[str] = set()
        auto_resume = bool(getattr(self._settings, "job_auto_resume", False))

        for rec in recs:
            seen.add(rec.job_id)
            state = rec.state.value
            if state == "running":
                running += 1
            elif state == "queued":
                queued += 1
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
        if waiting:
            # Off, nothing resumes them on its own: the cue says how to.
            parts.append(
                f"↻{waiting} to resume" if auto_resume else f"↻{waiting}: /jobs resume"
            )
        return " · ".join(parts) if parts else None

    @staticmethod
    def _short(name: str, limit: int = 24) -> str:
        name = (name or "").strip() or "job"
        return name if len(name) <= limit else name[: limit - 1] + "…"

