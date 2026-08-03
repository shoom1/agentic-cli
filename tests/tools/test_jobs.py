"""Tests for the long-running job substrate (JobManager + backends + tools)."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from agentic_cli.tools.jobs import JobManager, JobRecord, JobState
from agentic_cli.tools.jobs.backends import JobBackend, default_backends


def _wait(jm: JobManager, job_id: str, timeout: float = 5.0) -> JobRecord:
    """Poll a job until it reaches a terminal state (or timeout)."""
    end = time.time() + timeout
    terminal = {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED, JobState.UNKNOWN}
    while time.time() < end:
        rec = jm.get(job_id)
        assert rec is not None
        if rec.state in terminal:
            return rec
        time.sleep(0.05)
    return jm.get(job_id)  # type: ignore[return-value]


def _this_host() -> str:
    import socket

    return socket.gethostname()


def _dead_pid() -> int:
    """A pid that is certainly not running: a child we started and reaped."""
    import sys

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


@pytest.fixture
def jm(tmp_path: Path) -> JobManager:
    return JobManager(base_dir=tmp_path / "jobs", max_concurrent=2)


class TestSubprocessBackend:
    def test_success(self, jm: JobManager):
        rec = jm.submit(tool="run_shell_job", backend="subprocess", spec={"command": "echo hi"})
        rec = _wait(jm, rec.job_id)
        assert rec.state is JobState.SUCCEEDED
        assert rec.exit_code == 0
        assert "hi" in "\n".join(jm.tail(rec.job_id, 5, "stdout"))

    def test_failure_via_subshell_sentinel(self, jm: JobManager):
        # `exit 3` must not skip the sentinel write (subshell isolation).
        rec = jm.submit(tool="run_shell_job", backend="subprocess", spec={"command": "exit 3"})
        rec = _wait(jm, rec.job_id)
        assert rec.state is JobState.FAILED
        assert rec.exit_code == 3

    def test_cwd_is_respected(self, jm: JobManager, tmp_path: Path):
        workdir = tmp_path / "work"
        workdir.mkdir()
        rec = jm.submit(
            tool="run_shell_job",
            backend="subprocess",
            spec={"command": "pwd", "cwd": str(workdir)},
        )
        rec = _wait(jm, rec.job_id)
        out = "\n".join(jm.tail(rec.job_id, 5, "stdout"))
        assert str(workdir) in out

    def test_cancel(self, jm: JobManager):
        rec = jm.submit(tool="run_shell_job", backend="subprocess", spec={"command": "sleep 5"})
        time.sleep(0.2)
        cancelled = jm.cancel(rec.job_id)
        assert cancelled is not None and cancelled.state is JobState.CANCELLED


class TestConcurrency:
    def test_cap_and_queue(self, jm: JobManager):
        ids = [
            jm.submit(tool="run_shell_job", backend="subprocess", spec={"command": "sleep 0.4"}).job_id
            for _ in range(3)
        ]
        jm.reconcile()
        states = [jm.get(i).state for i in ids]  # type: ignore[union-attr]
        assert states.count(JobState.RUNNING) == 2
        assert states.count(JobState.QUEUED) == 1
        for i in ids:
            _wait(jm, i)
        jm.reconcile()
        assert all(jm.get(i).state is JobState.SUCCEEDED for i in ids)  # type: ignore[union-attr]


class TestInProcessBackend:
    def test_returns_result(self, jm: JobManager):
        rec = jm.submit(
            tool="calc", backend="inprocess",
            spec={"target": lambda a, b: a + b, "args": (2, 3)},
        )
        rec = _wait(jm, rec.job_id)
        assert rec.state is JobState.SUCCEEDED
        assert jm.result(rec.job_id) == 5

    def test_exception_marks_failed(self, jm: JobManager):
        def boom():
            raise RuntimeError("nope")

        rec = jm.submit(tool="calc", backend="inprocess", spec={"target": boom})
        rec = _wait(jm, rec.job_id)
        assert rec.state is JobState.FAILED


class TestPersistenceAndReconcile:
    def test_reload_terminal_job(self, tmp_path: Path):
        base = tmp_path / "jobs"
        jm = JobManager(base_dir=base, max_concurrent=2)
        rec = _wait(jm, jm.submit(tool="t", backend="subprocess", spec={"command": "echo x"}).job_id)
        assert rec.state is JobState.SUCCEEDED
        # Fresh manager over the same dir reloads the record.
        jm2 = JobManager(base_dir=base)
        reloaded = jm2.get(rec.job_id)
        assert reloaded is not None and reloaded.state is JobState.SUCCEEDED

    def test_vanished_process_reconciles_to_unknown(self, tmp_path: Path):
        base = tmp_path / "jobs"
        base.mkdir(parents=True)
        # A reliably-dead PID: start a process and reap it.
        p = subprocess.Popen(["true"])
        p.wait()
        dead_pid = p.pid

        job_id = "deadbeef0001"
        (base / job_id).mkdir()
        rec = JobRecord(
            job_id=job_id, tool="t", backend="subprocess", name="t",
            state=JobState.RUNNING, spec={"command": "sleep 999"}, pid=dead_pid,
        )
        (base / job_id / "meta.json").write_text(json.dumps(rec.to_dict()))

        # New manager: no live Popen handle, no sentinel, pid dead → UNKNOWN.
        jm = JobManager(base_dir=base)
        reloaded = jm.get(job_id)
        assert reloaded is not None and reloaded.state is JobState.UNKNOWN

    def test_clean_removes_terminal(self, jm: JobManager):
        rec = _wait(jm, jm.submit(tool="t", backend="subprocess", spec={"command": "echo x"}).job_id)
        assert (jm.base_dir / rec.job_id).exists()
        removed = jm.clean()
        assert removed >= 1
        assert not (jm.base_dir / rec.job_id).exists()
        assert jm.get(rec.job_id) is None


class TestManagerGuards:
    def test_unknown_backend_raises(self, jm: JobManager):
        with pytest.raises(ValueError):
            jm.submit(tool="t", backend="does-not-exist", spec={})

    def test_default_backends_present(self):
        b = default_backends()
        assert set(b) == {"subprocess", "inprocess"}
        assert b["subprocess"].survives_restart is True
        assert b["inprocess"].survives_restart is False


class TestRegistryAndTools:
    def test_long_running_flag(self):
        # The framework no longer ships a long-running *starter* tool (those are
        # app-provided), so register a throwaway one to prove the flag threads
        # through @register_tool. Observe-only tools default to long_running=False.
        from agentic_cli.tools.registry import ToolCategory, get_registry, register_tool
        from agentic_cli.workflow.permissions import EXEMPT

        @register_tool(
            category=ToolCategory.OTHER,
            capabilities=EXEMPT,
            long_running=True,
            description="probe long-running tool for tests",
        )
        def _jobs_test_long_running_probe() -> dict:
            return {"success": True}

        reg = get_registry()
        assert reg.get("_jobs_test_long_running_probe").long_running is True
        assert reg.get("job_status").long_running is False

    def test_tools_via_service_registry(self, tmp_path: Path):
        # Exercise the observe-only tools against a job submitted directly via
        # the JobManager (the starter tool is app-provided, not imported here).
        from agentic_cli.tools.jobs import job_list, job_status
        from agentic_cli.workflow.service_registry import JOB_MANAGER, set_service_registry

        jm = JobManager(base_dir=tmp_path / "jobs", max_concurrent=2)
        token = set_service_registry({JOB_MANAGER: jm})
        try:
            rec = jm.submit(
                tool="run_shell_job", backend="subprocess",
                spec={"command": "echo hi"}, name="greet",
            )
            job_id = rec.job_id

            # Poll the public tool until terminal.
            end = time.time() + 5
            while time.time() < end:
                st = job_status(job_id)
                if st["state"] in ("succeeded", "failed"):
                    break
                time.sleep(0.05)
            assert st["success"] is True
            assert st["state"] == "succeeded"
            # job_status returns the result once finished, so the agent needs no
            # separate job_result call (minimal tool surface).
            assert "result" in st
            assert st["result"]["exit_code"] == 0

            listing = job_list()
            assert listing["success"] is True
            assert any(j["job_id"] == job_id for j in listing["jobs"])
        finally:
            token.var.reset(token)

    def test_minimal_bundle_is_just_job_status(self):
        from agentic_cli.tools import JOB_MANAGEMENT_TOOLS, JOB_TOOLS, job_status

        assert JOB_TOOLS == [job_status]
        assert len(JOB_MANAGEMENT_TOOLS) == 5

    def test_tools_error_without_manager(self):
        from agentic_cli.tools.jobs import job_status
        from agentic_cli.workflow.service_registry import clear_service_registry

        token = clear_service_registry()
        try:
            res = job_status("whatever")
            assert res["success"] is False
            assert "not available" in res["error"]
        finally:
            token.var.reset(token)


class TestResumeMetadata:
    """Phase-2 association layer: resume metadata on jobs (no behavior change)."""

    def test_defaults_are_fire_and_forget(self, jm: JobManager):
        rec = jm.submit(tool="t", backend="subprocess", spec={"command": "echo x"})
        assert rec.resume_on_complete is False
        assert rec.session_id is None and rec.user_id is None
        assert rec.call_id is None and rec.resumed is False
        _wait(jm, rec.job_id)
        # A non-resume job never shows up as awaiting resume.
        assert jm.awaiting_resume() == []

    def test_submit_stores_explicit_metadata_and_persists(self, tmp_path: Path):
        base = tmp_path / "jobs"
        jm = JobManager(base_dir=base, max_concurrent=2)
        rec = jm.submit(
            tool="run_shell_job", backend="subprocess", spec={"command": "echo x"},
            resume_on_complete=True, call_id="fc-123", call_name="run_shell_job",
            session_id="sess-1", user_id="user-1",
        )
        assert rec.resume_on_complete is True
        assert rec.call_id == "fc-123"
        assert rec.call_name == "run_shell_job"
        assert rec.session_id == "sess-1" and rec.user_id == "user-1"
        # Survives a reload (persisted in meta.json).
        jm2 = JobManager(base_dir=base)
        reloaded = jm2.get(rec.job_id)
        assert reloaded is not None
        assert reloaded.resume_on_complete is True
        assert reloaded.call_id == "fc-123"
        assert reloaded.session_id == "sess-1"

    def test_autofills_session_user_from_workflow_service(self, jm: JobManager):
        from types import SimpleNamespace

        from agentic_cli.workflow.service_registry import WORKFLOW, set_service_registry

        fake_wf = SimpleNamespace(active_session_id="sess-A", active_user_id="user-A")
        token = set_service_registry({WORKFLOW: fake_wf})
        try:
            rec = jm.submit(
                tool="run_shell_job", backend="subprocess", spec={"command": "echo x"},
                resume_on_complete=True, call_id="fc-1",
            )
        finally:
            token.var.reset(token)
        assert rec.session_id == "sess-A"
        assert rec.user_id == "user-A"
        # call_name defaults to the tool name when resume is requested.
        assert rec.call_name == "run_shell_job"

    def test_missing_context_is_nonfatal(self, jm: JobManager):
        from agentic_cli.workflow.service_registry import clear_service_registry

        token = clear_service_registry()
        try:
            # No workflow service → can't fill session/user, but the job still runs.
            rec = jm.submit(
                tool="t", backend="subprocess", spec={"command": "echo x"},
                resume_on_complete=True, call_id="fc-1",
            )
        finally:
            token.var.reset(token)
        assert rec.resume_on_complete is True
        assert rec.session_id is None and rec.user_id is None

    def test_awaiting_resume_and_mark_resumed(self, tmp_path: Path):
        base = tmp_path / "jobs"
        jm = JobManager(base_dir=base, max_concurrent=2)
        rec = jm.submit(
            tool="run_shell_job", backend="subprocess", spec={"command": "echo x"},
            resume_on_complete=True, call_id="fc-1", session_id="s", user_id="u",
        )
        # Also submit a fire-and-forget job that must never appear.
        other = jm.submit(tool="t", backend="subprocess", spec={"command": "echo y"})
        _wait(jm, rec.job_id)
        _wait(jm, other.job_id)

        awaiting = jm.awaiting_resume()
        assert [r.job_id for r in awaiting] == [rec.job_id]

        # Marking it resumed (durably) drops it from the list.
        jm.mark_resumed(rec.job_id)
        assert jm.awaiting_resume() == []
        jm_reloaded = JobManager(base_dir=base)
        assert jm_reloaded.get(rec.job_id).resumed is True  # type: ignore[union-attr]
        assert jm_reloaded.awaiting_resume() == []


class TestResumeLifecycle:
    """pending → resuming → delivered/failed, with crash recovery."""

    def _jm(self, tmp_path):
        from agentic_cli.tools.jobs import JobManager

        return JobManager(base_dir=tmp_path / "jobs")

    def _finished_job(self, jm, job_id="j1"):
        from agentic_cli.tools.jobs import JobRecord
        from agentic_cli.tools.jobs.backends import JobState

        rec = JobRecord(
            job_id=job_id, tool="run_shell_job", backend="subprocess", name="build",
            state=JobState.SUCCEEDED, resume_on_complete=True, call_id="c1",
            session_id="s", user_id="u", finished_at=1.0,
        )
        jm._records[rec.job_id] = rec
        jm._persist(rec)
        return rec

    def test_new_job_is_pending(self, tmp_path):
        from agentic_cli.tools.jobs.manager import ResumeState

        jm = self._jm(tmp_path)
        rec = self._finished_job(jm)
        assert rec.resume_state == ResumeState.PENDING.value
        assert [r.job_id for r in jm.awaiting_resume()] == ["j1"]

    def test_claim_is_exclusive_and_hides_from_awaiting(self, tmp_path):
        from agentic_cli.tools.jobs.manager import ResumeState

        jm = self._jm(tmp_path)
        rec = self._finished_job(jm)

        assert jm.begin_resume("j1") is True
        assert jm.begin_resume("j1") is False  # already claimed
        assert rec.resume_state == ResumeState.RESUMING.value
        assert jm.awaiting_resume() == []

    def test_claim_of_unknown_job_is_false(self, tmp_path):
        assert self._jm(tmp_path).begin_resume("nope") is False

    def test_complete_records_delivered_and_failed(self, tmp_path):
        from agentic_cli.tools.jobs.manager import ResumeState

        jm = self._jm(tmp_path)
        rec = self._finished_job(jm)
        jm.begin_resume("j1")
        jm.complete_resume("j1", delivered=True)
        assert rec.resume_state == ResumeState.DELIVERED.value
        assert rec.resumed is True

        rec2 = self._finished_job(jm, "j2")
        jm.begin_resume("j2")
        jm.complete_resume("j2", delivered=False, error="turn failed")
        assert rec2.resume_state == ResumeState.FAILED.value
        assert rec2.resume_error == "turn failed"
        assert jm.awaiting_resume() == []  # terminal either way

    def test_interrupted_resume_is_recovered_as_failed_not_replayed(self, tmp_path):
        """A crash between claim and delivery must not silently re-deliver."""
        import json

        from agentic_cli.tools.jobs import JobManager
        from agentic_cli.tools.jobs.manager import ResumeState

        jm = self._jm(tmp_path)
        self._finished_job(jm)
        jm.begin_resume("j1")  # process dies here

        # The claim records *who* holds it. Rewrite it to a process that is
        # genuinely gone, which is what "the CLI crashed" looks like on disk.
        meta = tmp_path / "jobs" / "j1" / "meta.json"
        data = json.loads(meta.read_text())
        data["resume_owner"] = f"{_this_host()}:{_dead_pid()}"
        meta.write_text(json.dumps(data))

        reloaded = JobManager(base_dir=tmp_path / "jobs")
        rec = reloaded._records["j1"]
        assert rec.resume_state == ResumeState.FAILED.value
        assert "interrupted" in (rec.resume_error or "")
        assert reloaded.awaiting_resume() == []  # no automatic replay

    def test_ownerless_interrupted_resume_is_recovered(self, tmp_path):
        """Records written before claims were owned still recover."""
        import json

        from agentic_cli.tools.jobs import JobManager
        from agentic_cli.tools.jobs.manager import ResumeState

        jm = self._jm(tmp_path)
        self._finished_job(jm)
        jm.begin_resume("j1")

        meta = tmp_path / "jobs" / "j1" / "meta.json"
        data = json.loads(meta.read_text())
        data.pop("resume_owner", None)
        meta.write_text(json.dumps(data))

        reloaded = JobManager(base_dir=tmp_path / "jobs")
        assert reloaded._records["j1"].resume_state == ResumeState.FAILED.value

    def test_legacy_resumed_flag_migrates(self, tmp_path):
        """Records written before the lifecycle carried a bool ``resumed``."""
        import json

        from agentic_cli.tools.jobs import JobManager
        from agentic_cli.tools.jobs.manager import ResumeState

        jm = self._jm(tmp_path)
        self._finished_job(jm)
        meta = tmp_path / "jobs" / "j1" / "meta.json"
        data = json.loads(meta.read_text())
        data.pop("resume_state", None)
        data["resumed"] = True
        meta.write_text(json.dumps(data))

        reloaded = JobManager(base_dir=tmp_path / "jobs")
        assert reloaded._records["j1"].resume_state == ResumeState.DELIVERED.value
        assert reloaded.awaiting_resume() == []


class TestCrossProcessResumeClaims:
    """Two JobManagers over one jobs directory = two CLI processes.

    The claim used to consult only in-memory records, so both would see
    ``PENDING`` and both would deliver the same job result into their own
    conversation. And each manager's startup recovery flipped the *other's*
    live ``RESUMING`` claim to FAILED behind its back.
    """

    @staticmethod
    def _seed_finished_job(base: Path, job_id: str = "j1") -> None:
        """Persist a terminal, resume-flagged job without holding a manager."""
        seeder = JobManager(base_dir=base)
        rec = JobRecord(
            job_id=job_id, tool="run_shell_job", backend="subprocess", name="build",
            state=JobState.SUCCEEDED, resume_on_complete=True, call_id="c1",
            session_id="s", user_id="u", finished_at=1.0,
        )
        seeder._records[rec.job_id] = rec
        seeder._job_dir(rec.job_id).mkdir(parents=True, exist_ok=True)
        seeder._persist(rec)
        seeder.close()

    def test_exactly_one_of_two_managers_claims(self, tmp_path: Path):
        base = tmp_path / "jobs"
        self._seed_finished_job(base)

        first = JobManager(base_dir=base)
        second = JobManager(base_dir=base)
        try:
            assert [r.job_id for r in first.awaiting_resume()] == ["j1"]
            assert [r.job_id for r in second.awaiting_resume()] == ["j1"]

            claims = [first.begin_resume("j1"), second.begin_resume("j1")]
            assert claims.count(True) == 1, (
                f"both managers claimed the same job result: {claims}"
            )
        finally:
            first.close()
            second.close()

    def test_loser_sees_the_claim_after_reloading(self, tmp_path: Path):
        from agentic_cli.tools.jobs.manager import ResumeState

        base = tmp_path / "jobs"
        self._seed_finished_job(base)

        first = JobManager(base_dir=base)
        second = JobManager(base_dir=base)
        try:
            assert first.begin_resume("j1") is True
            assert second.begin_resume("j1") is False
            assert second._records["j1"].resume_state == ResumeState.RESUMING.value
            assert second.awaiting_resume() == []
        finally:
            first.close()
            second.close()

    def test_startup_does_not_fail_a_live_claim(self, tmp_path: Path):
        from agentic_cli.tools.jobs.manager import ResumeState

        base = tmp_path / "jobs"
        self._seed_finished_job(base)

        holder = JobManager(base_dir=base)
        try:
            assert holder.begin_resume("j1") is True

            # A second process starts while the first is mid-delivery.
            newcomer = JobManager(base_dir=base)
            try:
                assert (
                    newcomer._records["j1"].resume_state
                    == ResumeState.RESUMING.value
                ), "a live claim was recovered as failed"
                assert holder._records["j1"].resume_state == ResumeState.RESUMING.value

                holder.complete_resume("j1", delivered=True)
                assert (
                    holder._records["j1"].resume_state == ResumeState.DELIVERED.value
                )
            finally:
                newcomer.close()
        finally:
            holder.close()

    def test_completing_another_processes_claim_raises(self, tmp_path: Path):
        """Only the claiming process closes a claim out."""
        from agentic_cli.tools.jobs.manager import ResumeState, ResumeStateError

        base = tmp_path / "jobs"
        self._seed_finished_job(base)

        mine = JobManager(base_dir=base)
        try:
            assert mine.begin_resume("j1") is True

            # Another (live) process took the claim between our claim and our
            # completion — on disk, that is what it looks like.
            meta = base / "j1" / "meta.json"
            data = json.loads(meta.read_text())
            data["resume_owner"] = f"{_this_host()}:{os.getpid() + 1}"
            meta.write_text(json.dumps(data))

            with pytest.raises(ResumeStateError, match="being delivered by"):
                mine.complete_resume("j1", delivered=True)

            # And the other process's claim is intact.
            assert (
                json.loads(meta.read_text())["resume_state"]
                == ResumeState.RESUMING.value
            )
        finally:
            mine.close()


class _GatedBackend(JobBackend):
    """Jobs stay RUNNING until the shared gate is opened, then SUCCEEDED.

    Two managers share one gate, so the test controls exactly when each of
    them *observes* the finish (and therefore when each writes metadata).
    """

    name = "gated"
    survives_restart = True

    def __init__(self, gate: dict) -> None:
        self._gate = gate

    def start(self, record, job_dir):  # pragma: no cover - never started here
        record.state = JobState.RUNNING

    def poll(self, record, job_dir):
        return JobState.SUCCEEDED if self._gate.get("done") else JobState.RUNNING

    def cancel(self, record, job_dir):  # pragma: no cover
        return None


class TestResumeStateIsAtomicAcrossProcesses:
    """A normal metadata write must not clobber another process's claim.

    Only ``begin_resume``/``complete_resume`` were lock-and-reload aware.
    Every *other* persist (``_refresh`` after a poll, ``cancel``, ``submit``)
    wrote the whole record from memory — including a stale ``resume_state`` —
    so a second CLI merely *observing* a job erased the first one's claim and
    could then claim it too, delivering the same result into two conversations.
    """

    @staticmethod
    def _manager(base: Path, gate: dict) -> JobManager:
        # The backend must be known before _load_existing reconciles, or the
        # record is written off as UNKNOWN on the way in.
        return JobManager(
            base_dir=base,
            backends={**default_backends(), "gated": _GatedBackend(gate)},
        )

    @classmethod
    def _seed_running_job(cls, base: Path, gate: dict, job_id: str = "j1") -> None:
        seeder = cls._manager(base, gate)
        rec = JobRecord(
            job_id=job_id, tool="run_shell_job", backend="gated", name="build",
            state=JobState.RUNNING, resume_on_complete=True, call_id="c1",
            session_id="s", user_id="u",
        )
        seeder._records[rec.job_id] = rec
        seeder._job_dir(rec.job_id).mkdir(parents=True, exist_ok=True)
        seeder._persist(rec)
        seeder.close()

    def test_observer_persist_does_not_erase_a_claim(self, tmp_path: Path):
        from agentic_cli.tools.jobs.manager import ResumeState

        base = tmp_path / "jobs"
        gate: dict = {}
        self._seed_running_job(base, gate)

        first = self._manager(base, gate)
        second = self._manager(base, gate)  # both still see it RUNNING
        try:
            gate["done"] = True

            # A observes the finish, then claims it.
            assert first.get("j1").state is JobState.SUCCEEDED
            assert first.begin_resume("j1") is True

            # B observes the same finish — a plain metadata write, from a
            # record whose resume fields predate A's claim.
            assert second.get("j1").state is JobState.SUCCEEDED

            on_disk = json.loads((base / "j1" / "meta.json").read_text())
            assert on_disk["resume_state"] == ResumeState.RESUMING.value, (
                "an observer's persist erased the other process's claim"
            )
            assert second.begin_resume("j1") is False, "the job was claimed twice"
        finally:
            first.close()
            second.close()

    def test_observer_persist_does_not_erase_a_completed_delivery(
        self, tmp_path: Path
    ):
        from agentic_cli.tools.jobs.manager import ResumeState

        base = tmp_path / "jobs"
        gate: dict = {}
        self._seed_running_job(base, gate)

        first = self._manager(base, gate)
        second = self._manager(base, gate)
        try:
            gate["done"] = True
            first.get("j1")
            assert first.begin_resume("j1") is True
            first.complete_resume("j1", delivered=True)

            second.get("j1")  # observer write
            on_disk = json.loads((base / "j1" / "meta.json").read_text())
            assert on_disk["resume_state"] == ResumeState.DELIVERED.value
            assert second.begin_resume("j1") is False
        finally:
            first.close()
            second.close()

    def test_stale_recovery_cannot_overwrite_delivered(self, tmp_path: Path):
        """Startup recovery must re-read under the lock before deciding."""
        from agentic_cli.tools.jobs.manager import ResumeState

        base = tmp_path / "jobs"
        gate: dict = {}
        self._seed_running_job(base, gate)

        holder = self._manager(base, gate)
        stale = self._manager(base, gate)
        try:
            gate["done"] = True
            holder.get("j1")
            assert holder.begin_resume("j1") is True

            # ``stale`` loaded the record before the claim; give it the view a
            # crashed-owner claim would have, then complete the real delivery.
            rec = stale._records["j1"]
            rec.resume_state = ResumeState.RESUMING.value
            rec.resume_owner = f"{_this_host()}:{_dead_pid()}"
            holder.complete_resume("j1", delivered=True)

            stale._recover_interrupted_resumes()

            on_disk = json.loads((base / "j1" / "meta.json").read_text())
            assert on_disk["resume_state"] == ResumeState.DELIVERED.value, (
                "stale recovery overwrote a completed delivery"
            )
            assert stale._records["j1"].resume_state == ResumeState.DELIVERED.value
        finally:
            holder.close()
            stale.close()

    def test_unavailable_lock_fails_the_claim_closed(self, tmp_path: Path, monkeypatch):
        """No lock, no claim — a duplicate delivery is worse than a late one."""
        from agentic_cli.tools.jobs import manager as jobs_manager
        from agentic_cli.tools.jobs.manager import ResumeState

        base = tmp_path / "jobs"
        TestCrossProcessResumeClaims._seed_finished_job(base)
        jm = JobManager(base_dir=base)
        try:
            def _no_lock(fd):
                raise OSError("flock unavailable")

            monkeypatch.setattr(jobs_manager, "_acquire_lock", _no_lock)

            assert jm.begin_resume("j1") is False
            assert jm._records["j1"].resume_state == ResumeState.PENDING.value
        finally:
            jm.close()

    def test_unavailable_lock_blocks_startup_recovery(
        self, tmp_path: Path, monkeypatch
    ):
        from agentic_cli.tools.jobs import manager as jobs_manager
        from agentic_cli.tools.jobs.manager import ResumeState

        base = tmp_path / "jobs"
        TestCrossProcessResumeClaims._seed_finished_job(base)
        claimer = JobManager(base_dir=base)
        try:
            assert claimer.begin_resume("j1") is True
            meta = base / "j1" / "meta.json"
            data = json.loads(meta.read_text())
            data["resume_owner"] = f"{_this_host()}:{_dead_pid()}"
            meta.write_text(json.dumps(data))

            def _no_lock(fd):
                raise OSError("flock unavailable")

            monkeypatch.setattr(jobs_manager, "_acquire_lock", _no_lock)

            reloaded = JobManager(base_dir=base)
            try:
                assert (
                    reloaded._records["j1"].resume_state
                    == ResumeState.RESUMING.value
                ), "recovery rewrote state it could not lock"
            finally:
                reloaded.close()
        finally:
            claimer.close()


class _InProcessLikeBackend(JobBackend):
    """Only the process that started a job can see it running.

    Mirrors the in-process backend: the handle lives in memory, so another
    process polling the same record can only answer UNKNOWN.
    """

    name = "inproc"
    survives_restart = False

    def __init__(self) -> None:
        self._handles: set[str] = set()

    def start(self, record, job_dir):
        self._handles.add(record.job_id)

    def poll(self, record, job_dir):
        return JobState.RUNNING if record.job_id in self._handles else JobState.UNKNOWN

    def cancel(self, record, job_dir):  # pragma: no cover
        self._handles.discard(record.job_id)


class _CountingBackend(JobBackend):
    """Records every launch, so a double start is visible."""

    name = "counting"
    survives_restart = True

    def __init__(self, log: list[str]) -> None:
        self._log = log

    def start(self, record, job_dir):
        self._log.append(record.job_id)

    def poll(self, record, job_dir):
        return JobState.RUNNING

    def cancel(self, record, job_dir):  # pragma: no cover
        return None


class TestExecutionOwnership:
    """Who is *running* a job is shared state, just like who is delivering it.

    Two CLIs share the jobs directory. Without a recorded execution owner, a
    second one polled a job whose handle lives in the first one's memory, got
    UNKNOWN, and marked a perfectly healthy job terminal — making it resumable
    and its result deliverable. And two managers that both saw a job QUEUED
    both launched it.
    """

    @staticmethod
    def _manager(base: Path, backend: JobBackend) -> JobManager:
        return JobManager(
            base_dir=base, backends={**default_backends(), backend.name: backend}
        )

    def test_second_manager_leaves_a_live_in_process_job_alone(self, tmp_path: Path):
        base = tmp_path / "jobs"
        first = self._manager(base, _InProcessLikeBackend())
        try:
            rec = first.submit(
                tool="t", backend="inproc", spec={}, resume_on_complete=True,
                session_id="s", user_id="u",
            )
            assert rec.state is JobState.RUNNING

            second = self._manager(base, _InProcessLikeBackend())
            try:
                mine = second._records[rec.job_id]
                assert mine.state is JobState.RUNNING, (
                    "another process's running job was written off as UNKNOWN"
                )

                second.reconcile()
                assert second._records[rec.job_id].state is JobState.RUNNING
                assert second.awaiting_resume() == [], (
                    "a live job became deliverable to a second process"
                )

                on_disk = json.loads((base / rec.job_id / "meta.json").read_text())
                assert on_disk["state"] == JobState.RUNNING.value
            finally:
                second.close()
        finally:
            first.close()

    def test_a_dead_owners_job_is_still_reconciled(self, tmp_path: Path):
        """Ownership defers to a *live* process only."""
        base = tmp_path / "jobs"
        first = self._manager(base, _InProcessLikeBackend())
        try:
            rec = first.submit(tool="t", backend="inproc", spec={})
            meta = base / rec.job_id / "meta.json"
            data = json.loads(meta.read_text())
            data["exec_owner"] = f"{_this_host()}:{_dead_pid()}"
            meta.write_text(json.dumps(data))

            second = self._manager(base, _InProcessLikeBackend())
            try:
                assert second._records[rec.job_id].state is JobState.UNKNOWN
            finally:
                second.close()
        finally:
            first.close()

    def test_two_managers_launch_a_queued_job_once(self, tmp_path: Path):
        base = tmp_path / "jobs"
        starts: list[str] = []
        first = self._manager(base, _CountingBackend(starts))
        second = self._manager(base, _CountingBackend(starts))
        try:
            queued = JobRecord(
                job_id="q1", tool="t", backend="counting", name="n",
                state=JobState.QUEUED,
            )
            first._records["q1"] = queued
            first._job_dir("q1").mkdir(parents=True, exist_ok=True)
            first._persist(queued)
            # The second process loaded its own copy while it was still queued.
            second._records["q1"] = JobRecord.from_dict(queued.to_dict())

            first._maybe_start_queued()
            second._maybe_start_queued()

            assert starts == ["q1"], f"the job was launched {len(starts)} times"
        finally:
            first.close()
            second.close()

    def test_launch_claim_is_recorded_durably(self, tmp_path: Path):
        base = tmp_path / "jobs"
        starts: list[str] = []
        jm = self._manager(base, _CountingBackend(starts))
        try:
            rec = jm.submit(tool="t", backend="counting", spec={})
            on_disk = json.loads((base / rec.job_id / "meta.json").read_text())
            # <host>:<pid>:<manager instance> — the handle lives in a specific
            # manager, so a sibling in the same process is a different owner.
            assert on_disk["exec_owner"].startswith(f"{_this_host()}:{os.getpid()}:")
        finally:
            jm.close()

    def test_interrupted_launch_is_failed_not_replayed(self, tmp_path: Path):
        base = tmp_path / "jobs"
        starts: list[str] = []
        seeder = self._manager(base, _CountingBackend(starts))
        queued = JobRecord(
            job_id="q1", tool="t", backend="counting", name="n",
            state=JobState.QUEUED, exec_owner=f"{_this_host()}:{_dead_pid()}",
        )
        seeder._records["q1"] = queued
        seeder._job_dir("q1").mkdir(parents=True, exist_ok=True)
        seeder._persist(queued, owns_shared_state=True)
        seeder.close()

        reloaded = self._manager(base, _CountingBackend(starts))
        try:
            rec = reloaded._records["q1"]
            assert rec.state is JobState.FAILED
            assert "interrupted" in (rec.error or "")
            assert starts == [], "an interrupted launch was replayed"
        finally:
            reloaded.close()

    def test_unlocked_launch_is_skipped(self, tmp_path: Path, monkeypatch):
        """Fail closed: a late launch beats a double launch."""
        from agentic_cli.tools.jobs import manager as jobs_manager

        base = tmp_path / "jobs"
        starts: list[str] = []
        jm = self._manager(base, _CountingBackend(starts))
        try:
            queued = JobRecord(
                job_id="q1", tool="t", backend="counting", name="n",
                state=JobState.QUEUED,
            )
            jm._records["q1"] = queued
            jm._job_dir("q1").mkdir(parents=True, exist_ok=True)
            jm._persist(queued)

            def _boom(fd):
                raise OSError("flock unavailable")

            monkeypatch.setattr(jobs_manager, "_acquire_lock", _boom)
            jm._maybe_start_queued()

            assert starts == []
            assert jm._records["q1"].state is JobState.QUEUED
        finally:
            jm.close()


class _GatedInProcessBackend(JobBackend):
    """In-process semantics: only the starter can poll, gated completion."""

    name = "gated_inproc"
    survives_restart = False

    def __init__(self, gate: dict) -> None:
        self._gate = gate
        self._handles: set[str] = set()

    def start(self, record, job_dir):
        self._handles.add(record.job_id)

    def poll(self, record, job_dir):
        if record.job_id not in self._handles:
            return JobState.UNKNOWN  # no handle here — cannot judge
        return JobState.SUCCEEDED if self._gate.get("done") else JobState.RUNNING

    def cancel(self, record, job_dir):
        self._cancelled = True
        self._handles.discard(record.job_id)


class _SentinelBackend(JobBackend):
    """Restart-safe *and* remotely cancellable (a durable handle, like a pid)."""

    name = "sentinel"
    survives_restart = True
    cancels_foreign_jobs = True

    def __init__(self, gate: dict) -> None:
        self._gate = gate
        self.cancelled: list[str] = []

    def start(self, record, job_dir):
        return None

    def poll(self, record, job_dir):
        return JobState.SUCCEEDED if self._gate.get("done") else JobState.RUNNING

    def cancel(self, record, job_dir):
        self.cancelled.append(record.job_id)


class TestJobRecordLifecycleCoherence:
    """The whole persisted record is shared state, not just its owner fields.

    Execution ownership stopped a second manager from mangling a running job,
    but it also stopped it from ever learning the job had *finished*: the
    observer skipped the record entirely, so a completed job stayed RUNNING in
    its view forever. And every other mutator (cancel, clean, recovery) still
    acted on whatever it had in memory, so a stale copy could overwrite a
    terminal outcome another manager had already recorded.
    """

    @staticmethod
    def _manager(base: Path, backend: JobBackend) -> JobManager:
        return JobManager(
            base_dir=base, backends={**default_backends(), backend.name: backend}
        )

    def test_observer_sees_the_owners_terminal_state(self, tmp_path: Path):
        base = tmp_path / "jobs"
        gate: dict = {}
        owner = self._manager(base, _GatedInProcessBackend(gate))
        try:
            rec = owner.submit(tool="t", backend="gated_inproc", spec={})
            assert rec.state is JobState.RUNNING

            observer = self._manager(base, _GatedInProcessBackend(gate))
            try:
                assert observer._records[rec.job_id].state is JobState.RUNNING

                gate["done"] = True
                owner.reconcile()  # the owner records the outcome durably
                assert owner._records[rec.job_id].state is JobState.SUCCEEDED

                observer.reconcile()
                assert observer._records[rec.job_id].state is JobState.SUCCEEDED, (
                    "a completed job stayed RUNNING for the observer"
                )
            finally:
                observer.close()
        finally:
            owner.close()

    def test_observer_reads_a_restart_safe_sentinel_itself(self, tmp_path: Path):
        """A shared sentinel needs no owner to report it."""
        base = tmp_path / "jobs"
        gate: dict = {}
        owner = self._manager(base, _SentinelBackend(gate))
        try:
            rec = owner.submit(tool="t", backend="sentinel", spec={})
            observer = self._manager(base, _SentinelBackend(gate))
            try:
                gate["done"] = True
                observer.reconcile()
                assert observer._records[rec.job_id].state is JobState.SUCCEEDED
            finally:
                observer.close()
        finally:
            owner.close()

    def test_stale_cancel_does_not_overwrite_a_completed_job(self, tmp_path: Path):
        base = tmp_path / "jobs"
        gate: dict = {}
        owner = self._manager(base, _SentinelBackend(gate))
        observer_backend = _SentinelBackend(gate)
        try:
            rec = owner.submit(tool="t", backend="sentinel", spec={})
            observer = self._manager(base, observer_backend)
            try:
                gate["done"] = True
                owner.reconcile()
                assert owner._records[rec.job_id].state is JobState.SUCCEEDED

                # The observer still believes it is running.
                assert observer._records[rec.job_id].state is JobState.RUNNING
                cancelled = observer.cancel(rec.job_id)

                assert cancelled.state is JobState.SUCCEEDED, (
                    "a stale cancel overwrote a completed job"
                )
                on_disk = json.loads((base / rec.job_id / "meta.json").read_text())
                assert on_disk["state"] == JobState.SUCCEEDED.value
                assert observer_backend.cancelled == []
            finally:
                observer.close()
        finally:
            owner.close()

    def test_foreign_in_process_job_is_not_reported_cancelled(self, tmp_path: Path):
        """This backend instance holds no handle — it cannot cancel anything."""
        base = tmp_path / "jobs"
        gate: dict = {}
        owner = self._manager(base, _GatedInProcessBackend(gate))
        try:
            rec = owner.submit(tool="t", backend="gated_inproc", spec={})
            observer = self._manager(base, _GatedInProcessBackend(gate))
            try:
                result = observer.cancel(rec.job_id)

                assert result.state is JobState.RUNNING, (
                    "a job this manager cannot cancel was reported cancelled"
                )
                on_disk = json.loads((base / rec.job_id / "meta.json").read_text())
                assert on_disk["state"] == JobState.RUNNING.value
                assert owner._records[rec.job_id].state is JobState.RUNNING
            finally:
                observer.close()
        finally:
            owner.close()

    def test_a_restart_safe_foreign_job_can_still_be_cancelled(self, tmp_path: Path):
        base = tmp_path / "jobs"
        gate: dict = {}
        owner = self._manager(base, _SentinelBackend(gate))
        observer_backend = _SentinelBackend(gate)
        try:
            rec = owner.submit(tool="t", backend="sentinel", spec={})
            observer = self._manager(base, observer_backend)
            try:
                result = observer.cancel(rec.job_id)
                assert result.state is JobState.CANCELLED
                assert observer_backend.cancelled == [rec.job_id]
            finally:
                observer.close()
        finally:
            owner.close()

    def test_terminal_state_is_monotonic_on_write(self, tmp_path: Path):
        base = tmp_path / "jobs"
        TestCrossProcessResumeClaims._seed_finished_job(base)
        jm = JobManager(base_dir=base)
        try:
            rec = jm._records["j1"]
            rec.state = JobState.CANCELLED  # a stale view, about to be written

            jm._persist(rec)

            on_disk = json.loads((base / "j1" / "meta.json").read_text())
            assert on_disk["state"] == JobState.SUCCEEDED.value
            assert rec.state is JobState.SUCCEEDED, "the stale view was not corrected"
        finally:
            jm.close()

    def test_clean_reloads_before_removing(self, tmp_path: Path):
        """A stale terminal view must not delete a job that is still running."""
        base = tmp_path / "jobs"
        gate: dict = {}
        owner = self._manager(base, _GatedInProcessBackend(gate))
        try:
            rec = owner.submit(tool="t", backend="gated_inproc", spec={})
            observer = self._manager(base, _GatedInProcessBackend(gate))
            try:
                observer._records[rec.job_id].state = JobState.UNKNOWN  # stale

                assert observer.clean() == 0, "a live job was cleaned away"
                assert (base / rec.job_id / "meta.json").exists()
            finally:
                observer.close()
        finally:
            owner.close()

    def test_launch_recovery_respects_a_durable_outcome(self, tmp_path: Path):
        """A queued+dead-owner view must not overwrite a recorded success."""
        base = tmp_path / "jobs"
        gate: dict = {"done": True}
        owner = self._manager(base, _SentinelBackend(gate))
        try:
            rec = owner.submit(tool="t", backend="sentinel", spec={})
            owner.reconcile()
            assert owner._records[rec.job_id].state is JobState.SUCCEEDED

            stale = self._manager(base, _SentinelBackend(gate))
            try:
                mine = stale._records[rec.job_id]
                mine.state = JobState.QUEUED
                mine.exec_owner = f"{_this_host()}:{_dead_pid()}"

                stale._recover_interrupted_launches()

                assert mine.state is JobState.SUCCEEDED
                on_disk = json.loads((base / rec.job_id / "meta.json").read_text())
                assert on_disk["state"] == JobState.SUCCEEDED.value
            finally:
                stale.close()
        finally:
            owner.close()

    def test_unlocked_reconcile_is_skipped(self, tmp_path: Path, monkeypatch):
        from agentic_cli.tools.jobs import manager as jobs_manager

        base = tmp_path / "jobs"
        gate: dict = {}
        jm = self._manager(base, _SentinelBackend(gate))
        try:
            rec = jm.submit(tool="t", backend="sentinel", spec={})
            gate["done"] = True

            def _boom(fd):
                raise OSError("flock unavailable")

            monkeypatch.setattr(jobs_manager, "_acquire_lock", _boom)
            jm.reconcile()

            assert jm._records[rec.job_id].state is JobState.RUNNING
        finally:
            jm.close()


def _wait_for_gate(path: str, timeout: float = 5.0) -> str:
    """Block until ``path`` appears. Module level so the spec stays copyable."""
    import os

    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            return "done"
        time.sleep(0.02)
    return "timeout"


class TestObserverSeesRealBackendCompletion:
    """The shipped in-process backend, observed from a second manager.

    ``InProcessBackend.poll`` reads the on-disk ``exit_code`` sentinel *before*
    consulting its in-memory future, so a second manager can read the outcome —
    it just answers UNKNOWN while the sentinel is absent. Skipping the poll
    entirely (because the record is foreign-owned) meant the observer never saw
    the job finish; polling and taking UNKNOWN at face value would have marked
    a live job terminal. Poll, and ignore only the "I cannot tell" answer.
    """

    @staticmethod
    def _manager(base: Path) -> JobManager:
        return JobManager(base_dir=base)

    def test_observer_sees_the_sentinel_after_the_owner_stops_polling(
        self, tmp_path: Path
    ):
        base = tmp_path / "jobs"
        gate_file = tmp_path / "gate"
        owner = self._manager(base)
        observer = None
        try:
            rec = owner.submit(
                tool="t",
                backend="inprocess",
                spec={"target": _wait_for_gate, "kwargs": {"path": str(gate_file)}},
                resume_on_complete=True,
                session_id="s",
                user_id="u",
            )
            assert rec.state is JobState.RUNNING

            observer = self._manager(base)
            observer.reconcile()
            assert observer._records[rec.job_id].state is JobState.RUNNING, (
                "an UNKNOWN poll from a foreign backend marked a live job terminal"
            )

            gate_file.write_text("go")
            owner.close()  # the owner stops polling; its process stays alive

            deadline = time.time() + 5
            while time.time() < deadline:
                observer.reconcile()
                if observer._records[rec.job_id].state is JobState.SUCCEEDED:
                    break
                time.sleep(0.05)

            assert observer._records[rec.job_id].state is JobState.SUCCEEDED, (
                "the observer never saw the durable sentinel"
            )
            assert [r.job_id for r in observer.awaiting_resume()] == [rec.job_id]
        finally:
            if observer is not None:
                observer.close()
            owner.close()

    def test_foreign_cancellation_is_a_backend_capability(self, tmp_path: Path):
        """Not inferred from restart-safety — the two are different questions."""
        from agentic_cli.tools.jobs.backends import (
            InProcessBackend,
            SubprocessBackend,
        )

        assert InProcessBackend.cancels_foreign_jobs is False
        assert SubprocessBackend.cancels_foreign_jobs is True


class TestTerminalRecordsAreReloaded:
    """A terminal record still has shared state: its delivery lifecycle."""

    @staticmethod
    def _pair(base: Path):
        TestCrossProcessResumeClaims._seed_finished_job(base)
        return JobManager(base_dir=base), JobManager(base_dir=base)

    def test_get_list_and_awaiting_reflect_a_completed_delivery(
        self, tmp_path: Path
    ):
        from agentic_cli.tools.jobs.manager import ResumeState

        base = tmp_path / "jobs"
        owner, observer = self._pair(base)
        try:
            assert [r.job_id for r in observer.awaiting_resume()] == ["j1"]

            assert owner.begin_resume("j1") is True
            owner.complete_resume("j1", delivered=True)

            assert observer.get("j1").resume_state == ResumeState.DELIVERED.value
            listed = {r.job_id: r for r in observer.list()}
            assert listed["j1"].resume_state == ResumeState.DELIVERED.value
            assert observer.awaiting_resume() == []
        finally:
            owner.close()
            observer.close()

    def test_begin_resume_reloads_before_judging_terminality(self, tmp_path: Path):
        base = tmp_path / "jobs"
        owner, observer = self._pair(base)
        try:
            # The observer's snapshot predates the job finishing.
            observer._records["j1"].state = JobState.RUNNING

            assert observer.begin_resume("j1") is True, (
                "a stale non-terminal snapshot refused a deliverable job"
            )
        finally:
            owner.close()
            observer.close()


class TestDeletedAndUnreadableRecords:
    """A record another manager removed must stay removed."""

    @staticmethod
    def _pair(base: Path):
        TestCrossProcessResumeClaims._seed_finished_job(base)
        return JobManager(base_dir=base), JobManager(base_dir=base)

    def _assert_gone(self, base: Path) -> None:
        assert not (base / "j1").exists(), "a deleted job record was recreated"

    def test_get_does_not_resurrect_a_cleaned_job(self, tmp_path: Path):
        base = tmp_path / "jobs"
        cleaner, stale = self._pair(base)
        try:
            assert cleaner.clean() == 1
            assert stale.get("j1") is None
            assert "j1" not in stale._records
            self._assert_gone(base)
        finally:
            cleaner.close()
            stale.close()

    def test_cancel_does_not_resurrect_a_cleaned_job(self, tmp_path: Path):
        base = tmp_path / "jobs"
        cleaner, stale = self._pair(base)
        try:
            stale._records["j1"].state = JobState.RUNNING  # stale, looks cancellable
            assert cleaner.clean() == 1

            assert stale.cancel("j1") is None
            self._assert_gone(base)
        finally:
            cleaner.close()
            stale.close()

    def test_begin_resume_does_not_resurrect_a_cleaned_job(self, tmp_path: Path):
        base = tmp_path / "jobs"
        cleaner, stale = self._pair(base)
        try:
            assert cleaner.clean() == 1
            assert stale.begin_resume("j1") is False
            self._assert_gone(base)
        finally:
            cleaner.close()
            stale.close()

    def test_complete_resume_does_not_resurrect_a_cleaned_job(self, tmp_path: Path):
        import shutil

        base = tmp_path / "jobs"
        cleaner, stale = self._pair(base)
        try:
            assert stale.begin_resume("j1") is True
            # An active delivery is protected from clean() (see
            # test_clean_leaves_an_active_delivery_alone), so model the record
            # being removed out from under us directly.
            shutil.rmtree(base / "j1")

            stale.complete_resume("j1", delivered=True)  # must not raise
            self._assert_gone(base)
        finally:
            cleaner.close()
            stale.close()

    def test_unreadable_state_fails_closed(self, tmp_path: Path):
        """Corrupt metadata is not permission to act — or to overwrite."""
        base = tmp_path / "jobs"
        owner, observer = self._pair(base)
        try:
            meta = base / "j1" / "meta.json"
            meta.write_text("{ this is not json")

            assert observer.begin_resume("j1") is False
            assert "j1" in observer._records, "a corrupt record was forgotten"
            assert meta.read_text() == "{ this is not json", (
                "a corrupt record was overwritten"
            )
        finally:
            owner.close()
            observer.close()

    def test_clean_leaves_an_active_delivery_alone(self, tmp_path: Path):
        base = tmp_path / "jobs"
        deliverer, cleaner = self._pair(base)
        try:
            assert deliverer.begin_resume("j1") is True

            assert cleaner.clean() == 0, "a job being delivered was cleaned away"
            assert (base / "j1" / "meta.json").exists()
        finally:
            deliverer.close()
            cleaner.close()


class TestNoUnsafeUnlockedWrites:
    """Without the lock there is no safe read-modify-write of shared fields.

    The fallback path read ``meta.json`` and then rewrote it whole — which is
    precisely the race the lock exists to prevent. A delivery completed between
    that read and that write was silently regressed from DELIVERED, and the
    job could then be claimed and delivered a second time.
    """

    @staticmethod
    def _no_lock(monkeypatch):
        from agentic_cli.tools.jobs import manager as jobs_manager

        def _boom(fd):
            raise OSError("flock unavailable")

        monkeypatch.setattr(jobs_manager, "_acquire_lock", _boom)

    @staticmethod
    def _write_state(base: Path, job_id: str, **fields) -> None:
        """Stand in for another process's write, straight to disk."""
        meta = base / job_id / "meta.json"
        data = json.loads(meta.read_text())
        data.update(fields)
        meta.write_text(json.dumps(data))

    def test_unlocked_write_leaves_an_existing_record_untouched(
        self, tmp_path: Path, monkeypatch
    ):
        """No lock, no rewrite.

        The read-then-rewrite fallback preserved the shared fields *only* when
        nothing changed between its read and its write — which is the race the
        lock exists to prevent, so it was never a safe fallback. The property
        under test is therefore the absence of the write itself: an unlocked
        persist must not touch a record another process may be inside.
        """
        base = tmp_path / "jobs"
        TestCrossProcessResumeClaims._seed_finished_job(base)
        jm = JobManager(base_dir=base)
        try:
            assert jm.begin_resume("j1") is True
            rec = jm._records["j1"]
            meta = base / "j1" / "meta.json"
            before = meta.read_text()

            self._no_lock(monkeypatch)
            rec.state = JobState.CANCELLED  # an ordinary change we would persist
            jm._persist(rec)

            assert meta.read_text() == before, (
                "an unlocked persist rewrote a record it could not lock"
            )
        finally:
            jm.close()

    def test_write_after_a_concurrent_complete_does_not_regress_it(
        self, tmp_path: Path, monkeypatch
    ):
        """read → (another process completes) → write, with no lock held."""
        from agentic_cli.tools.jobs.manager import ResumeState

        base = tmp_path / "jobs"
        TestCrossProcessResumeClaims._seed_finished_job(base)
        jm = JobManager(base_dir=base)
        try:
            assert jm.begin_resume("j1") is True
            rec = jm._records["j1"]

            self._no_lock(monkeypatch)
            # Between our read and our write, the delivery completes elsewhere.
            self._write_state(
                base,
                "j1",
                resume_state=ResumeState.DELIVERED.value,
                resume_owner=None,
            )

            jm._persist(rec)  # an ordinary metadata write

            on_disk = json.loads((base / "j1" / "meta.json").read_text())
            assert on_disk["resume_state"] == ResumeState.DELIVERED.value, (
                "an unlocked write regressed a completed delivery"
            )
        finally:
            jm.close()

    def test_unlocked_write_creates_a_record_that_has_none(
        self, tmp_path: Path, monkeypatch
    ):
        """Nothing on disk means nothing to clobber — the write must happen."""
        base = tmp_path / "jobs"
        jm = JobManager(base_dir=base)
        try:
            self._no_lock(monkeypatch)
            rec = JobRecord(
                job_id="new", tool="t", backend="subprocess", name="n",
                state=JobState.QUEUED,
            )
            jm._job_dir("new").mkdir(parents=True, exist_ok=True)
            jm._persist(rec)

            assert (base / "new" / "meta.json").exists()
        finally:
            jm.close()

    def test_unlocked_complete_resume_does_not_regress_delivered(
        self, tmp_path: Path, monkeypatch
    ):
        from agentic_cli.tools.jobs.manager import ResumeState

        base = tmp_path / "jobs"
        TestCrossProcessResumeClaims._seed_finished_job(base)
        jm = JobManager(base_dir=base)
        try:
            assert jm.begin_resume("j1") is True

            self._no_lock(monkeypatch)
            self._write_state(
                base,
                "j1",
                resume_state=ResumeState.DELIVERED.value,
                resume_owner=None,
            )

            jm.complete_resume("j1", delivered=False, error="turn failed")

            on_disk = json.loads((base / "j1" / "meta.json").read_text())
            assert on_disk["resume_state"] == ResumeState.DELIVERED.value
        finally:
            jm.close()

    def test_unlocked_complete_resume_leaves_the_record_claimed(
        self, tmp_path: Path, monkeypatch
    ):
        """Failed-safe: still RESUMING on disk, so it is never re-delivered."""
        from agentic_cli.tools.jobs.manager import ResumeState

        base = tmp_path / "jobs"
        TestCrossProcessResumeClaims._seed_finished_job(base)
        jm = JobManager(base_dir=base)
        try:
            assert jm.begin_resume("j1") is True
            self._no_lock(monkeypatch)

            jm.complete_resume("j1", delivered=True)  # must not raise

            on_disk = json.loads((base / "j1" / "meta.json").read_text())
            assert on_disk["resume_state"] == ResumeState.RESUMING.value
            assert jm.awaiting_resume() == [], "the job became deliverable again"
        finally:
            jm.close()


class TestBackendClose:
    """Owned executors must not outlive the manager."""

    def test_inprocess_backend_shuts_pool_down(self):
        from agentic_cli.tools.jobs.backends import InProcessBackend

        backend = InProcessBackend(max_workers=2)
        backend.close()
        assert backend._pool._shutdown is True

    def test_close_is_idempotent(self):
        from agentic_cli.tools.jobs.backends import InProcessBackend

        backend = InProcessBackend(max_workers=1)
        backend.close()
        backend.close()  # must not raise

    def test_subprocess_backend_close_is_noop(self):
        from agentic_cli.tools.jobs.backends import SubprocessBackend

        SubprocessBackend().close()

    def test_manager_close_closes_every_backend(self, tmp_path):
        from agentic_cli.tools.jobs import JobManager

        jm = JobManager(base_dir=tmp_path / "jobs")
        jm.close()
        jm.close()  # idempotent
        assert jm._backends["inprocess"]._pool._shutdown is True


class TestResumeTransitionGuards:
    """Only a deliverable job can be claimed; only a claim can be completed."""

    def _jm(self, tmp_path):
        from agentic_cli.tools.jobs import JobManager

        return JobManager(base_dir=tmp_path / "jobs")

    def _record(self, jm, **over):
        from agentic_cli.tools.jobs import JobRecord
        from agentic_cli.tools.jobs.backends import JobState

        fields = dict(
            job_id="j1", tool="run_shell_job", backend="subprocess", name="build",
            state=JobState.SUCCEEDED, resume_on_complete=True, call_id="c1",
            session_id="s", user_id="u", finished_at=1.0,
        )
        fields.update(over)
        rec = JobRecord(**fields)
        jm._records[rec.job_id] = rec
        jm._persist(rec)
        return rec

    def test_running_job_cannot_be_claimed(self, tmp_path):
        from agentic_cli.tools.jobs.backends import JobState

        jm = self._jm(tmp_path)
        self._record(jm, state=JobState.RUNNING)
        assert jm.begin_resume("j1") is False

    def test_job_without_resume_flag_cannot_be_claimed(self, tmp_path):
        jm = self._jm(tmp_path)
        self._record(jm, resume_on_complete=False)
        assert jm.begin_resume("j1") is False

    def test_completing_an_unclaimed_job_raises(self, tmp_path):
        from agentic_cli.tools.jobs.manager import ResumeStateError

        jm = self._jm(tmp_path)
        rec = self._record(jm)
        with pytest.raises(ResumeStateError, match="pending"):
            jm.complete_resume("j1", delivered=True)
        assert rec.resume_state == "pending"  # state untouched

    def test_completing_twice_raises(self, tmp_path):
        from agentic_cli.tools.jobs.manager import ResumeStateError

        jm = self._jm(tmp_path)
        self._record(jm)
        jm.begin_resume("j1")
        jm.complete_resume("j1", delivered=True)
        with pytest.raises(ResumeStateError):
            jm.complete_resume("j1", delivered=False, error="late")

    def test_completing_an_unknown_job_raises(self, tmp_path):
        from agentic_cli.tools.jobs.manager import ResumeStateError

        with pytest.raises(ResumeStateError, match="Unknown job"):
            self._jm(tmp_path).complete_resume("nope", delivered=True)

    def test_mark_resumed_shim_claims_then_completes(self, tmp_path):
        from agentic_cli.tools.jobs.manager import ResumeState

        jm = self._jm(tmp_path)
        rec = self._record(jm)
        jm.mark_resumed("j1")
        assert rec.resume_state == ResumeState.DELIVERED.value
        jm.mark_resumed("j1")  # already terminal: no-op, no raise
        assert rec.resume_state == ResumeState.DELIVERED.value


class TestStartupRecoveryHonorsDurableReads:
    """Recovery reloads durable state — and must act on what the read *said*.

    ``_recover_interrupted_launches`` / ``_recover_interrupted_resumes`` called
    ``_reload_durable()`` and threw the answer away. A record another manager
    deleted between the initial load and recovery was therefore rewritten from
    the stale in-memory copy (``owns_shared_state=True`` writes unconditionally),
    resurrecting a job that had been cleaned away; a record that could not be
    parsed was overwritten with a recovery verdict decided on state we had
    failed to read.
    """

    @staticmethod
    def _seed(base: Path, **fields) -> JobManager:
        """A manager holding one loaded record, matching what is on disk."""
        jm = JobManager(base_dir=base)
        rec = JobRecord(
            job_id="j1", tool="run_shell_job", backend="subprocess", name="build",
            session_id="s", user_id="u", **fields,
        )
        jm._records["j1"] = rec
        jm._job_dir("j1").mkdir(parents=True, exist_ok=True)
        jm._persist(rec)
        return jm

    @staticmethod
    def _interrupted_launch(base: Path) -> JobManager:
        return TestStartupRecoveryHonorsDurableReads._seed(
            base,
            state=JobState.QUEUED,
            exec_owner=f"{_this_host()}:{_dead_pid()}:abcd1234",
        )

    @staticmethod
    def _interrupted_resume(base: Path) -> JobManager:
        from agentic_cli.tools.jobs.manager import ResumeState

        jm = TestStartupRecoveryHonorsDurableReads._seed(
            base, state=JobState.SUCCEEDED, resume_on_complete=True, finished_at=1.0
        )
        rec = jm._records["j1"]
        rec.resume_state = ResumeState.RESUMING.value
        rec.resume_owner = f"{_this_host()}:{_dead_pid()}"
        jm._persist(rec, owns_shared_state=True)
        return jm

    @pytest.mark.parametrize("kind", ["launches", "resumes"])
    def test_deleted_record_is_not_resurrected(self, tmp_path: Path, kind: str):
        base = tmp_path / "jobs"
        jm = (
            self._interrupted_launch(base)
            if kind == "launches"
            else self._interrupted_resume(base)
        )
        try:
            meta = base / "j1" / "meta.json"
            assert meta.exists()
            meta.unlink()  # another manager cleaned the job away

            getattr(jm, f"_recover_interrupted_{kind}")()

            assert not meta.exists(), "recovery recreated a deleted record"
            assert "j1" not in jm._records, "a deleted record was not forgotten"
        finally:
            jm.close()

    @pytest.mark.parametrize("kind", ["launches", "resumes"])
    def test_unreadable_record_is_left_untouched(self, tmp_path: Path, kind: str):
        base = tmp_path / "jobs"
        jm = (
            self._interrupted_launch(base)
            if kind == "launches"
            else self._interrupted_resume(base)
        )
        try:
            meta = base / "j1" / "meta.json"
            meta.write_text("{not json")
            before = meta.read_bytes()

            getattr(jm, f"_recover_interrupted_{kind}")()

            assert meta.read_bytes() == before, "recovery overwrote unreadable state"
        finally:
            jm.close()


def _touch_marker(marker: str) -> str:
    Path(marker).write_text("ran")
    return "ran"


class TestInProcessShutdownDoesNotStrandQueuedJobs:
    """``close()`` promises jobs are not cancelled — the pool broke that promise.

    ``shutdown(cancel_futures=True)`` dropped work that was already submitted
    and already recorded RUNNING. Nothing then wrote the ``exit_code`` sentinel,
    so another manager polling the record got UNKNOWN, saw a live foreign
    execution owner, correctly refused to believe the UNKNOWN — and left the
    job RUNNING forever, undeliverable.
    """

    def test_queued_job_still_reaches_a_terminal_state(self, tmp_path: Path):
        from agentic_cli.tools.jobs.backends import InProcessBackend

        base = tmp_path / "jobs"
        gate, marker = tmp_path / "gate", tmp_path / "marker"

        # One worker, two jobs: the second is submitted (and RUNNING) but its
        # future is still queued behind the first.
        owner = JobManager(
            base_dir=base,
            max_concurrent=4,
            backends={"inprocess": InProcessBackend(max_workers=1)},
        )
        try:
            blocking = owner.submit(
                tool="wait", backend="inprocess",
                spec={"target": _wait_for_gate, "kwargs": {"path": str(gate), "timeout": 10.0}},
            )
            queued = owner.submit(
                tool="mark", backend="inprocess",
                spec={"target": _touch_marker, "args": (str(marker),)},
                resume_on_complete=True, call_id="c1", session_id="s", user_id="u",
            )
            assert owner._records[blocking.job_id].state is JobState.RUNNING
            assert owner._records[queued.job_id].state is JobState.RUNNING
        finally:
            owner.close()

        gate.write_text("go")  # let the running job finish and the queue drain

        observer = JobManager(base_dir=base)
        try:
            rec = _wait(observer, queued.job_id, timeout=10.0)
            assert rec.state is JobState.SUCCEEDED, (
                f"a submitted job was stranded as {rec.state.value}"
            )
            assert marker.exists(), "the queued job never ran"
            assert observer.begin_resume(queued.job_id) is True
        finally:
            observer.close()
