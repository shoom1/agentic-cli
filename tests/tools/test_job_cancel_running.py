"""A cancelled job keeps its slot until it has actually stopped.

``JobManager.cancel`` recorded a job CANCELLED whatever the backend managed,
and started the next queued job. A running in-process job cannot be stopped
(Python cannot interrupt a thread), so with ``max_concurrent=1`` cancelling it
started the second job while the first kept running: two at once. A
subprocess that outlived SIGTERM did the same.

A backend now says whether its cancel stopped the job. One still running stays
RUNNING with ``cancel_requested``, holding its slot, and is recorded CANCELLED
when it ends, as any cancelled job is; only then does the next one start.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

import agentic_cli.tools.jobs.backends as backends_mod
from agentic_cli.tools.jobs import JobManager, JobState
from agentic_cli.tools.jobs.tools import job_cancel
from agentic_cli.workflow.service_registry import set_service_registry


def _until(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


class _Gate:
    """In-process job bodies that run until released, recording who runs."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.running: list[str] = []
        self._lock = threading.Lock()

    def work(self, name: str) -> str:
        with self._lock:
            self.running.append(name)
        self.release.wait(10)
        with self._lock:
            self.running.remove(name)
        return name


@pytest.fixture
def jobs(tmp_path: Path):
    manager = JobManager(base_dir=tmp_path / "jobs", max_concurrent=1)
    yield manager
    manager.close()


def _submit(jobs: JobManager, gate: _Gate, name: str):
    def work():  # a function, not the bound method: the spec is deep-copied
        return gate.work(name)

    return jobs.submit(tool="t", backend="inprocess", spec={"target": work})


class TestInProcess:
    def test_cancelling_a_running_job_keeps_its_slot_until_it_returns(self, jobs):
        gate = _Gate()
        first = _submit(jobs, gate, "first")
        second = _submit(jobs, gate, "second")
        _until(lambda: gate.running == ["first"])

        cancelled = jobs.cancel(first.job_id)

        assert cancelled.state is JobState.RUNNING
        assert cancelled.cancel_requested is True
        jobs.reconcile()
        time.sleep(0.2)
        assert gate.running == ["first"]  # the second has not started
        assert jobs.get(second.job_id).state is JobState.QUEUED

        gate.release.set()
        _until(lambda: (jobs.reconcile(), jobs.get(first.job_id).state)[1] is JobState.CANCELLED)
        _until(lambda: (jobs.reconcile(), jobs.get(second.job_id).state)[1] is JobState.SUCCEEDED)

    def test_a_queued_job_is_cancelled_at_once(self, jobs):
        gate = _Gate()
        _submit(jobs, gate, "first")
        second = _submit(jobs, gate, "second")

        cancelled = jobs.cancel(second.job_id)

        assert cancelled.state is JobState.CANCELLED
        gate.release.set()

    def test_the_tool_says_the_job_is_still_running(self, jobs):
        gate = _Gate()
        first = _submit(jobs, gate, "first")
        _until(lambda: gate.running == ["first"])

        token = set_service_registry({"job_manager": jobs})
        try:
            result = job_cancel(first.job_id)
        finally:
            token.var.reset(token)
        gate.release.set()

        assert result["success"] is True
        assert result["state"] == "running"
        assert result["cancel_requested"] is True
        assert "still running" in result["message"]

    def test_a_cancel_request_is_kept_in_the_record(self, jobs, tmp_path):
        gate = _Gate()
        first = _submit(jobs, gate, "first")
        _until(lambda: gate.running == ["first"])
        jobs.cancel(first.job_id)

        assert jobs.get(first.job_id).summary()["cancel_requested"] is True
        gate.release.set()


class TestSubprocess:
    def test_a_process_that_outlives_sigterm_keeps_its_slot(self, jobs, monkeypatch):
        monkeypatch.setattr(backends_mod, "_CANCEL_WAIT_S", 0.3)
        stubborn = jobs.submit(
            tool="run_shell_job", backend="subprocess",
            spec={"command": "trap '' TERM; sleep 1.5"},
        )
        after = jobs.submit(tool="run_shell_job", backend="subprocess", spec={"command": "true"})
        time.sleep(0.3)

        cancelled = jobs.cancel(stubborn.job_id)

        assert cancelled.state is JobState.RUNNING
        assert cancelled.cancel_requested is True
        jobs.reconcile()
        assert jobs.get(after.job_id).state is JobState.QUEUED
        _until(lambda: (jobs.reconcile(), jobs.get(stubborn.job_id).state)[1] is JobState.CANCELLED)
        _until(lambda: (jobs.reconcile(), jobs.get(after.job_id).state)[1] is JobState.SUCCEEDED)

    def test_a_process_that_stops_on_sigterm_is_cancelled_at_once(self, jobs):
        rec = jobs.submit(tool="run_shell_job", backend="subprocess", spec={"command": "sleep 30"})
        time.sleep(0.3)

        cancelled = jobs.cancel(rec.job_id)

        assert cancelled.state is JobState.CANCELLED
