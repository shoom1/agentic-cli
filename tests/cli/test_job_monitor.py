"""Tests for the harness JobMonitor (milestone 2: jobs UI monitor).

The monitor is the background task that keeps long-running jobs visible in the
status bar. These tests drive it against a real ``JobManager`` running real
subprocess jobs, plus a couple of focused unit tests for the segment logic.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_cli.cli.job_monitor import JobMonitor
from agentic_cli.tools.jobs import JobManager, JobState


# --- Lightweight fakes ---------------------------------------------------------


class FakeUI:
    """Captures every status string set on the (fake) session."""

    def __init__(self) -> None:
        self.statuses: list[str] = []

    def set_status(self, text) -> None:
        self.statuses.append(str(text))


class FakeController:
    """Minimal stand-in for WorkflowController as the monitor sees it.

    Composes the status bar the same way the real controller does
    (model | jobs | hints) so the integration assertions are meaningful.
    """

    def __init__(self, ui: FakeUI, job_manager) -> None:
        self.is_ready = True
        self.workflow = SimpleNamespace(model="test-model", job_manager=job_manager)
        self.usage_tracker = None
        self.jobs_status_segment: str | None = None
        self._ui = ui
        self.status_updates = 0

    def update_status_bar(self, ui) -> None:
        self.status_updates += 1
        parts = [self.workflow.model]
        if self.jobs_status_segment:
            parts.append(self.jobs_status_segment)
        parts.extend(["Ctrl+C: cancel", "/help: commands"])
        ui.set_status(" | ".join(parts))


@pytest.fixture
def jm(tmp_path: Path) -> JobManager:
    return JobManager(base_dir=tmp_path / "jobs", max_concurrent=2)


async def _await_until(predicate, timeout: float = 5.0, interval: float = 0.02) -> bool:
    """Async poll that yields to the loop so background tasks can run."""
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return predicate()


# --- Segment logic (no asyncio) ------------------------------------------------


class TestSegmentLogic:
    def test_no_jobs_is_none(self, jm: JobManager):
        ui = FakeUI()
        mon = JobMonitor(ui, FakeController(ui, jm))
        assert mon._build_segment([]) is None

    def test_running_and_queued_counts(self, jm: JobManager):
        ui = FakeUI()
        mon = JobMonitor(ui, FakeController(ui, jm))
        for _ in range(3):
            jm.submit(tool="run_shell_job", backend="subprocess", spec={"command": "sleep 1"})
        jm.reconcile()  # 2 start (cap=2), 1 stays queued
        seg = mon._build_segment(jm.list())
        assert seg is not None
        assert "2 running" in seg
        assert "1 queued" in seg

    def test_finished_job_note_then_ages_out(self, jm: JobManager):
        ui = FakeUI()
        mon = JobMonitor(ui, FakeController(ui, jm), note_ticks=2)
        rec = jm.submit(
            tool="run_shell_job", backend="subprocess",
            spec={"command": "echo hi"}, name="greet",
        )
        # First observe it active so the terminal transition is detectable.
        rec.state = JobState.RUNNING
        mon._build_segment([rec])
        # Now mark terminal and rebuild → a ✓ note appears.
        rec.state = JobState.SUCCEEDED
        seg = mon._build_segment([rec])
        assert "✓ greet" in seg
        # The note lingers for note_ticks builds, then disappears.
        mon._build_segment([rec])  # tick 2
        seg_gone = mon._build_segment([rec])  # aged out
        assert seg_gone is None

    def test_no_note_for_jobs_already_terminal_on_first_sight(self, jm: JobManager):
        ui = FakeUI()
        mon = JobMonitor(ui, FakeController(ui, jm))
        rec = jm.submit(tool="t", backend="subprocess", spec={"command": "echo x"}, name="old")
        rec.state = JobState.SUCCEEDED  # first time we ever see it: already done
        assert mon._build_segment([rec]) is None

    def test_short_truncates_long_names(self):
        assert JobMonitor._short("x" * 40, limit=10) == "x" * 9 + "…"
        assert JobMonitor._short("", ) == "job"


# --- poll_once publishes + refreshes -------------------------------------------


class TestPollOnce:
    def test_publishes_segment_and_refreshes_on_change(self, jm: JobManager):
        ui = FakeUI()
        ctrl = FakeController(ui, jm)
        mon = JobMonitor(ui, ctrl)

        jm.submit(tool="run_shell_job", backend="subprocess", spec={"command": "sleep 2"})
        seg = mon.poll_once()
        assert seg is not None and "running" in seg
        assert ctrl.jobs_status_segment == seg
        assert ui.statuses and "running" in ui.statuses[-1]

    def test_no_redraw_when_segment_unchanged(self, jm: JobManager):
        ui = FakeUI()
        ctrl = FakeController(ui, jm)
        mon = JobMonitor(ui, ctrl)
        # No jobs → segment None on every tick → no status updates at all.
        mon.poll_once()
        mon.poll_once()
        assert ctrl.status_updates == 0

    def test_no_manager_is_safe(self, tmp_path: Path):
        ui = FakeUI()
        ctrl = FakeController(ui, None)  # workflow has no job_manager
        mon = JobMonitor(ui, ctrl)
        assert mon.poll_once() is None
        assert ctrl.status_updates == 0


# --- Full background loop against real subprocess jobs -------------------------


class TestBackgroundLoop:
    async def test_live_job_appears_then_completes(self, jm: JobManager):
        ui = FakeUI()
        ctrl = FakeController(ui, jm)
        mon = JobMonitor(ui, ctrl, interval=0.05, note_ticks=3)

        async with mon.running():
            jm.submit(
                tool="run_shell_job", backend="subprocess",
                spec={"command": "sleep 0.3"}, name="sleeper",
            )
            # The running job shows up in the status bar.
            saw_running = await _await_until(lambda: any("running" in s for s in ui.statuses))
            assert saw_running, ui.statuses
            # And once it finishes, a ✓ note for it appears.
            saw_done = await _await_until(lambda: any("✓ sleeper" in s for s in ui.statuses))
            assert saw_done, ui.statuses
