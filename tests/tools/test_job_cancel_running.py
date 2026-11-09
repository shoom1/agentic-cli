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

import json
import shlex
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
    def work():
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


class TestACancelFromAnotherManager:
    """A cancel another CLI requests reaches the manager that runs the job.

    The second manager cannot stop a process that ignores SIGTERM, so it
    records the request and leaves the job RUNNING. The owner read the state
    and its own lifecycle fields back from disk but not the request: its next
    poll found the shell around the command killed (exit -15), recorded the
    job FAILED while the command still ran, started the next job in its slot,
    and wrote ``cancel_requested`` false over the other manager's.
    """

    def test_the_owner_holds_the_slot_and_records_the_job_cancelled(
        self, jobs, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(backends_mod, "_CANCEL_WAIT_S", 0.3)
        stubborn = jobs.submit(
            tool="run_shell_job", backend="subprocess",
            spec={"command": "trap '' TERM; echo ready; sleep 2"},
        )
        after = jobs.submit(tool="run_shell_job", backend="subprocess", spec={"command": "true"})
        _until(lambda: jobs.tail(stubborn.job_id) == ["ready"])

        other = JobManager(base_dir=tmp_path / "jobs", max_concurrent=1)
        try:
            cancelled = other.cancel(stubborn.job_id)
        finally:
            other.close()
        assert cancelled.state is JobState.RUNNING
        assert cancelled.cancel_requested is True

        jobs.reconcile()
        assert jobs.get(stubborn.job_id).state is JobState.RUNNING
        assert jobs.get(after.job_id).state is JobState.QUEUED
        _until(lambda: (jobs.reconcile(), jobs.get(stubborn.job_id).state)[1] is JobState.CANCELLED)
        _until(lambda: (jobs.reconcile(), jobs.get(after.job_id).state)[1] is JobState.SUCCEEDED)

        restarted = JobManager(base_dir=tmp_path / "jobs", max_concurrent=1)
        try:
            assert restarted.get(stubborn.job_id).cancel_requested is True
        finally:
            restarted.close()


class TestAJobThatHasAlreadyEnded:
    """A cancel records a job that ended before it as the job ended.

    ``cancel`` reloaded the record but never polled the job, so one that had
    ended since anybody last looked was still RUNNING to it. Its own manager
    recorded it CANCELLED, without its exit code; another manager waited out
    the cancel on the owner's unreaped process and recorded a cancel request,
    which the owner then turned into CANCELLED.
    """

    def test_its_own_manager_records_how_it_ended(self, jobs):
        rec = jobs.submit(tool="run_shell_job", backend="subprocess", spec={"command": "true"})
        _until((jobs.base_dir / rec.job_id / "exit_code").exists)  # ended; nothing polled it

        cancelled = jobs.cancel(rec.job_id)

        assert (cancelled.state, cancelled.exit_code) == (JobState.SUCCEEDED, 0)
        on_disk = json.loads((jobs.base_dir / rec.job_id / "meta.json").read_text())
        assert (on_disk["state"], on_disk["exit_code"]) == ("succeeded", 0)

    def test_another_manager_records_how_it_ended_without_waiting(self, jobs, tmp_path):
        gate = tmp_path / "gate"
        rec = jobs.submit(
            tool="run_shell_job", backend="subprocess",
            spec={"command": f"while [ ! -e {shlex.quote(str(gate))} ]; do sleep 0.02; done"},
        )
        other = JobManager(base_dir=tmp_path / "jobs", max_concurrent=1)
        try:
            assert other.get(rec.job_id).state is JobState.RUNNING
            gate.write_text("go")
            _until((jobs.base_dir / rec.job_id / "exit_code").exists)  # its owner never polled

            started = time.monotonic()
            cancelled = other.cancel(rec.job_id)
            took = time.monotonic() - started
        finally:
            other.close()

        assert (cancelled.state, cancelled.exit_code) == (JobState.SUCCEEDED, 0)
        assert cancelled.cancel_requested is False
        assert took < backends_mod._CANCEL_WAIT_S, "the cancel waited on a job that had ended"
        on_disk = json.loads((jobs.base_dir / rec.job_id / "meta.json").read_text())
        assert (on_disk["state"], on_disk["cancel_requested"]) == ("succeeded", False)
        owner_view = jobs.get(rec.job_id)
        assert (owner_view.state, owner_view.exit_code) == (JobState.SUCCEEDED, 0)


def _cancel_with(manager: JobManager, job_id: str) -> dict:
    """``job_cancel`` as the model calls it, against ``manager``."""
    token = set_service_registry({"job_manager": manager})
    try:
        return job_cancel(job_id)
    finally:
        token.var.reset(token)


class TestTheToolSaysWhatHappened:
    """``job_cancel`` reported success for a job it had not cancelled.

    A job running in another live manager whose backend cannot cancel it from
    here, an unreadable record, the lock unavailable: ``cancel`` returns the
    record as it is, still running, and the tool said ``success`` with
    ``state: running``, so a model told the user the job was stopped. And a
    job that had already ended was reported as if this call had acted on it.
    """

    def test_a_job_it_could_not_reach_was_not_cancelled(self, jobs, tmp_path):
        gate = _Gate()
        first = _submit(jobs, gate, "first")
        _until(lambda: gate.running == ["first"])
        other = JobManager(base_dir=tmp_path / "jobs", max_concurrent=1)
        try:
            result = _cancel_with(other, first.job_id)  # a thread elsewhere: out of reach
        finally:
            other.close()
            gate.release.set()

        assert result["success"] is False
        assert result["state"] == "running"
        assert "was not cancelled" in result["error"]
        assert "still running" in result["error"]
        assert "cancel_requested" not in result

    def test_a_job_that_had_ended_was_not_cancelled(self, jobs):
        rec = jobs.submit(tool="run_shell_job", backend="subprocess", spec={"command": "true"})
        _until(lambda: jobs.get(rec.job_id).state is JobState.SUCCEEDED)

        result = _cancel_with(jobs, rec.job_id)

        assert result["success"] is True
        assert result["state"] == "succeeded"
        assert "already finished" in result.get("message", "")
        assert "nothing was cancelled" in result.get("message", "")

    def test_a_queued_job_it_could_not_reach_is_still_queued(self, jobs, tmp_path):
        gate = _Gate()
        _submit(jobs, gate, "first")
        second = _submit(jobs, gate, "second")  # queued, for its submitter only
        other = JobManager(base_dir=tmp_path / "jobs", max_concurrent=1)
        try:
            result = _cancel_with(other, second.job_id)
        finally:
            other.close()
            gate.release.set()

        assert (result["success"], result["state"]) == (False, "queued")
        assert "still queued" in result["error"]

    def test_a_job_on_a_backend_this_app_has_not_registered(self, tmp_path):
        class Remote(backends_mod.JobBackend):
            name = "remote"
            survives_restart = True
            cancels_foreign_jobs = True

            def start(self, record, job_dir):
                record.backend_handle = "remote-1"

            def poll(self, record, job_dir):
                return JobState.RUNNING

            def cancel(self, record, job_dir):
                return True

        owner = JobManager(base_dir=tmp_path / "jobs")
        owner.register_backend(Remote())
        rec = owner.submit(tool="t", backend="remote", spec={})
        owner.close()
        other = JobManager(base_dir=tmp_path / "jobs")  # the app no longer registers it
        try:
            result = _cancel_with(other, rec.job_id)
        finally:
            other.close()

        assert (result["success"], result["state"]) == (False, "running")
        assert "runs on the 'remote' backend, which this app has not registered" in (
            result["error"]
        )

    def test_a_cancelled_job_is_reported_cancelled(self, jobs):
        gate = _Gate()
        _submit(jobs, gate, "first")
        second = _submit(jobs, gate, "second")  # queued: cancelled at once

        result = _cancel_with(jobs, second.job_id)
        gate.release.set()

        assert result == {"success": True, "state": "cancelled"}


class _Watched(backends_mod.SubprocessBackend):
    """The subprocess backend, saying when a cancel starts waiting for the job."""

    def __init__(self) -> None:
        super().__init__()
        self.waiting = threading.Event()

    def cancel(self, record, job_dir):
        self.waiting.set()
        return super().cancel(record, job_dir)


class TestTheWaitHoldsNoLock:
    """A cancel waits for the job's processes outside the jobs directory's lock.

    The subprocess backend waits up to ``_CANCEL_WAIT_S`` for a job's process
    group to exit, and ``cancel`` waited inside the cross-process lock: every
    other CLI of the app (its status bar's tick, its ``/jobs``) waited with
    it, and the manager running the job could not record the job's end
    meanwhile. The request is now recorded under the lock, the backend waits
    without it, and how the job ended is recorded under it again, a terminal
    state recorded meanwhile winning.
    """

    @pytest.fixture
    def stubborn(self, jobs, tmp_path):
        """A job that ignores SIGTERM and runs until its gate opens."""
        gate = tmp_path / "gate"
        command = (
            f"trap '' TERM; echo ready; "
            f"while [ ! -e {shlex.quote(str(gate))} ]; do sleep 0.02; done"
        )
        job = jobs.submit(tool="run_shell_job", backend="subprocess", spec={"command": command})
        _until(lambda: jobs.tail(job.job_id) == ["ready"])
        yield job, gate
        gate.write_text("go")

    def _cancelling(self, tmp_path, job_id: str):
        """Another CLI's manager cancelling ``job_id`` on a thread of its own."""
        watched = _Watched()
        other = JobManager(base_dir=tmp_path / "jobs", backends={"subprocess": watched})
        returned: list = []
        thread = threading.Thread(target=lambda: returned.append(other.cancel(job_id)))
        thread.start()
        assert watched.waiting.wait(5), "the cancel never reached the backend"
        return other, thread, returned

    def test_another_managers_list_is_not_held_for_the_wait(
        self, jobs, stubborn, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(backends_mod, "_CANCEL_WAIT_S", 1.5)
        job, gate = stubborn
        other, thread, returned = self._cancelling(tmp_path, job.job_id)
        try:
            started = time.monotonic()
            jobs.list()
            took = time.monotonic() - started
        finally:
            thread.join(10)
            other.close()

        assert took < 0.5, f"list() waited {took:.2f} s for the cancel's wait"
        (cancelled,) = returned
        assert (cancelled.state, cancelled.cancel_requested) == (JobState.RUNNING, True)

    def test_a_job_that_ends_during_the_wait_is_recorded_cancelled(
        self, jobs, stubborn, tmp_path, monkeypatch
    ):
        """Its manager records how it ended while the cancel waits: cancelled,
        since the cancel was requested first; the cancel returns that."""
        monkeypatch.setattr(backends_mod, "_CANCEL_WAIT_S", 3.0)
        job, gate = stubborn
        other, thread, returned = self._cancelling(tmp_path, job.job_id)
        try:
            gate.write_text("go")  # the job ends while the cancel waits
            _until(lambda: (jobs.reconcile(), jobs.get(job.job_id).state)[1] in (
                JobState.CANCELLED, JobState.FAILED, JobState.SUCCEEDED,
            ))
        finally:
            thread.join(10)
            other.close()

        (cancelled,) = returned
        assert cancelled.state is JobState.CANCELLED
        on_disk = json.loads((jobs.base_dir / job.job_id / "meta.json").read_text())
        assert (on_disk["state"], on_disk["cancel_requested"]) == ("cancelled", True)
        assert jobs.get(job.job_id).state is JobState.CANCELLED


class _Unreachable(backends_mod.SubprocessBackend):
    """A subprocess backend whose cancel fails before it reaches the job (a
    remote backend's API call that times out, say)."""

    def cancel(self, record, job_dir):
        raise ConnectionError("cluster API unreachable")


class TestABackendCancelThatRaises:
    """A cancel whose backend raises changes nothing, as before the request
    was recorded ahead of the backend's wait.

    The request (``cancel_requested``) is written before the backend is
    asked, so a manager that sees the job end meanwhile records it
    CANCELLED. When the backend raised, the request stayed: the caller saw
    the error, and a job that then ran to its end was recorded CANCELLED. The
    request this cancel made is withdrawn now, and the error re-raised.
    """

    @pytest.fixture
    def unreachable(self, tmp_path):
        manager = JobManager(
            base_dir=tmp_path / "jobs", max_concurrent=1,
            backends={"subprocess": _Unreachable()},
        )
        yield manager
        manager.close()

    def _on_disk(self, jobs: JobManager, job_id: str) -> dict:
        return json.loads((jobs.base_dir / job_id / "meta.json").read_text())

    def test_a_running_job_ends_as_it_ends(self, unreachable):
        job = unreachable.submit(
            tool="run_shell_job", backend="subprocess", spec={"command": "sleep 0.5"},
        )
        time.sleep(0.1)

        with pytest.raises(ConnectionError, match="cluster API unreachable"):
            unreachable.cancel(job.job_id)

        assert self._on_disk(unreachable, job.job_id)["cancel_requested"] is False
        _until(lambda: unreachable.get(job.job_id).state not in (
            JobState.QUEUED, JobState.RUNNING,
        ))
        ended = unreachable.get(job.job_id)
        assert (ended.state, ended.exit_code, ended.cancel_requested) == (
            JobState.SUCCEEDED, 0, False,
        )

    def test_a_queued_job_stays_queued_with_no_request(self, unreachable):
        first = unreachable.submit(
            tool="run_shell_job", backend="subprocess", spec={"command": "sleep 0.5"},
        )
        queued = unreachable.submit(
            tool="run_shell_job", backend="subprocess", spec={"command": "true"},
        )

        with pytest.raises(ConnectionError):
            unreachable.cancel(queued.job_id)

        on_disk = self._on_disk(unreachable, queued.job_id)
        assert (on_disk["state"], on_disk["cancel_requested"]) == ("queued", False)
        _until(lambda: (unreachable.reconcile(), unreachable.get(queued.job_id).state)[1]
               is JobState.SUCCEEDED)
        assert unreachable.get(first.job_id).state is JobState.SUCCEEDED

    def test_a_request_made_before_is_kept(self, jobs, tmp_path, monkeypatch):
        """Only the request this cancel made is withdrawn: one an earlier
        cancel recorded, its job still running, stays."""
        monkeypatch.setattr(backends_mod, "_CANCEL_WAIT_S", 0.2)
        job = jobs.submit(
            tool="run_shell_job", backend="subprocess",
            spec={"command": "trap '' TERM; echo ready; sleep 1"},
        )
        _until(lambda: jobs.tail(job.job_id) == ["ready"])
        assert jobs.cancel(job.job_id).cancel_requested is True  # it outlives SIGTERM
        other = JobManager(base_dir=tmp_path / "jobs", backends={"subprocess": _Unreachable()})
        try:
            with pytest.raises(ConnectionError):
                other.cancel(job.job_id)
        finally:
            other.close()

        assert self._on_disk(jobs, job.job_id)["cancel_requested"] is True
        _until(lambda: (jobs.reconcile(), jobs.get(job.job_id).state)[1] is JobState.CANCELLED)
