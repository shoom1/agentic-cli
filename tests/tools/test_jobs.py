"""Tests for the long-running job substrate (JobManager + backends + tools)."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest

from agentic_cli.tools.jobs import JobManager, JobRecord, JobState
from agentic_cli.tools.jobs.backends import default_backends


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
