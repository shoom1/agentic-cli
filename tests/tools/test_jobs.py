"""Tests for the long-running job substrate (JobManager + backends + tools)."""

from __future__ import annotations

import contextlib
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


class TestTheWorkingDirectory:
    """A subprocess job runs where it was submitted. One left queued is
    launched by whichever manager opens the jobs directory next: another CLI,
    perhaps started in another project, where ``make clean`` would act on the
    wrong tree."""

    @pytest.mark.parametrize(("cwd", "runs_in"), [(None, "."), ("build", "build")])
    def test_a_queued_job_runs_where_it_was_submitted(
        self, tmp_path: Path, monkeypatch, cwd, runs_in
    ):
        base = tmp_path / "jobs"
        here, there = tmp_path / "project_x", tmp_path / "project_y"
        for project in (here, there):
            (project / "build").mkdir(parents=True)
        monkeypatch.chdir(here)
        first = JobManager(base_dir=base, max_concurrent=1)
        busy = first.submit(tool="t", backend="subprocess", spec={"command": "sleep 30"})
        second = None
        try:
            queued = first.submit(
                tool="t", backend="subprocess", spec={"command": "pwd", "cwd": cwd}
            )
            assert queued.state is JobState.QUEUED
            first.close()  # the first CLI exits with the job still queued

            monkeypatch.chdir(there)
            second = JobManager(base_dir=base, max_concurrent=2)
            rec = _wait(second, queued.job_id)
            assert rec.state is JobState.SUCCEEDED
            ran_in = "".join(second.tail(queued.job_id, 1, "stdout")).strip()
            assert Path(ran_in).resolve() == (here / runs_in).resolve()
        finally:
            first.cancel(busy.job_id)  # its starter reaps it at once
            if second is not None:
                second.close()


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


def _greet(name: str) -> str:
    return f"hi {name}"


class TestSavingARecord:
    """Saving a record never copies its spec, and a failed submit leaves nothing.

    ``to_dict`` deep-copied the whole record (``dataclasses.asdict``) before it
    replaced what JSON cannot hold, so a job whose target was a bound method of
    an object holding a lock could not be submitted. And ``submit`` kept the
    record before writing it: the stray QUEUED record was launched, or raised
    again, at the next submit, after that one had saved its own job.
    """

    def test_a_bound_method_of_an_object_holding_a_lock_runs(self, jm: JobManager):
        import threading

        class Service:
            def __init__(self) -> None:
                self._lock = threading.Lock()

            def work(self) -> str:
                with self._lock:
                    return "done"

        rec = jm.submit(tool="t", backend="inprocess", spec={"target": Service().work})

        assert _wait(jm, rec.job_id).state is JobState.SUCCEEDED
        assert jm.result(rec.job_id) == "done"

    def test_a_submit_whose_write_fails_leaves_nothing_behind(
        self, jm: JobManager, monkeypatch
    ):
        from agentic_cli.tools.jobs import manager as jobs_manager

        real_write = jobs_manager.atomic_write_json
        failures = [OSError("No space left on device")]

        def write_once_failing(path, data, **kwargs):
            if failures:
                raise failures.pop()
            return real_write(path, data, **kwargs)

        monkeypatch.setattr(jobs_manager, "atomic_write_json", write_once_failing)

        with pytest.raises(OSError):
            jm.submit(tool="t", backend="subprocess", spec={"command": "true"})
        assert [p.name for p in jm.base_dir.iterdir() if p.is_dir()] == []

        good = jm.submit(tool="t", backend="subprocess", spec={"command": "true"})
        assert [r.job_id for r in jm.list()] == [good.job_id]
        assert [p.name for p in jm.base_dir.iterdir() if p.is_dir()] == [good.job_id]

    def test_the_saved_json_is_what_it_was(self):
        """Field by field, the same keys and values ``asdict`` gave."""
        import dataclasses

        rec = JobRecord(
            job_id="j1", tool="t", backend="inprocess", name="n",
            state=JobState.SUCCEEDED,
            spec={"target": _greet, "args": ("x",), "kwargs": {"k": [1, {"a": None}]},
                  "cwd": "/tmp"},
            backend_handle="thread", pid=42, exit_code=0, tags=["a", "b"],
            submitted_at=1.0, started_at=2.0, finished_at=3.0, error="e",
            session_id="s", user_id="u", resume_on_complete=True, call_id="c",
            call_name="t", resume_state="resuming", resume_error="r",
            resume_owner="h:1", exec_owner="h:1:abc", cancel_requested=True,
        )
        expected = dataclasses.asdict(rec)  # what to_dict wrote, for a spec it could copy
        expected["state"] = "succeeded"
        expected["spec"] = {**expected["spec"], "target": "<non-serializable: function>"}

        assert json.dumps(rec.to_dict(), indent=2) == json.dumps(expected, indent=2)

    def test_the_dict_does_not_share_the_record_s_lists(self):
        rec = JobRecord(
            job_id="j1", tool="t", backend="subprocess", name="n",
            state=JobState.QUEUED, spec={"command": ["echo", "hi"]}, tags=["a"],
        )
        saved = rec.to_dict()
        saved["tags"].append("b")
        saved["spec"]["command"].append("there")

        assert rec.tags == ["a"]
        assert rec.spec == {"command": ["echo", "hi"]}


class TestTheExitSentinel:
    """A poll that lands while the sentinel is being written waits for it.

    The writer once truncated ``exit_code`` before writing the code, and the
    reader took an empty file as exit 1: under load about 1 job in 70 that
    succeeded, or exited 3, was recorded FAILED with exit 1 (the flaky CI tests).
    """

    def test_an_empty_sentinel_is_not_yet_an_exit_code(self, tmp_path: Path):
        from agentic_cli.tools.jobs.backends import _read_exit_code

        sentinel = tmp_path / "exit_code"
        sentinel.write_text("")

        assert _read_exit_code(sentinel) is None

    def test_a_subprocess_job_polled_mid_write_keeps_running(self, tmp_path: Path):
        from agentic_cli.tools.jobs.backends import SubprocessBackend

        backend = SubprocessBackend()
        record = JobRecord(job_id="j", tool="t", backend="subprocess", name="t",
                           state=JobState.QUEUED, spec={"command": "sleep 0.5"})
        backend.start(record, tmp_path)
        (tmp_path / "exit_code").write_text("")  # opened by the writer, not yet written

        assert backend.poll(record, tmp_path) is JobState.RUNNING
        deadline = time.time() + 5
        while (state := backend.poll(record, tmp_path)) is JobState.RUNNING and time.time() < deadline:
            time.sleep(0.02)
        assert (state, record.exit_code) == (JobState.SUCCEEDED, 0)

    def test_an_inprocess_job_polled_mid_write_keeps_running(self, tmp_path: Path):
        import threading

        from agentic_cli.tools.jobs.backends import InProcessBackend

        release = threading.Event()
        backend = InProcessBackend(max_workers=1)
        record = JobRecord(job_id="j", tool="t", backend="inprocess", name="t",
                           state=JobState.QUEUED, spec={"target": release.wait, "args": (5,)})
        try:
            backend.start(record, tmp_path)
            (tmp_path / "exit_code").write_text("")  # opened by the writer, not yet written

            assert backend.poll(record, tmp_path) is JobState.RUNNING
            release.set()
            deadline = time.time() + 5
            while (state := backend.poll(record, tmp_path)) is JobState.RUNNING and time.time() < deadline:
                time.sleep(0.02)
            assert (state, record.exit_code) == (JobState.SUCCEEDED, 0)
        finally:
            release.set()
            backend.close()


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

    def test_awaiting_resume_and_delivery(self, tmp_path: Path):
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

        # Delivering it (durably) drops it from the list.
        assert jm.begin_resume(rec.job_id)
        jm.complete_resume(rec.job_id, delivered=True)
        assert jm.awaiting_resume() == []
        jm_reloaded = JobManager(base_dir=base)
        assert jm_reloaded.get(rec.job_id).resumed is True  # type: ignore[union-attr]
        assert jm_reloaded.awaiting_resume() == []


@contextlib.contextmanager
def _active_tool_call(call_id: str, name: str):
    """Run the block as Deep Agents runs a tool: inside its own tool call."""
    from agentic_cli.workflow.sessions import (
        ToolCallRef,
        reset_active_tool_call,
        set_active_tool_call,
    )

    token = set_active_tool_call(ToolCallRef(call_id, name))
    try:
        yield
    finally:
        reset_active_tool_call(token)


class TestTheActiveToolCall:
    """A resumable job is linked to the tool call that started it.

    On Deep Agents a tool written for ADK gets ``tool_context=None``, so the
    call id it would read there is None; the backend's tool middleware makes
    each call the active tool call instead, and ``submit`` reads it.
    """

    def test_a_resumable_job_takes_the_active_call(self, jm: JobManager):
        with _active_tool_call("call-7", "run_shell_job"):
            rec = jm.submit(
                tool="run_shell_job", backend="subprocess", spec={"command": "echo x"},
                resume_on_complete=True, session_id="s", user_id="u",
            )

        assert (rec.call_id, rec.call_name) == ("call-7", "run_shell_job")
        reloaded = JobManager(base_dir=jm.base_dir).get(rec.job_id)
        assert (reloaded.call_id, reloaded.call_name) == ("call-7", "run_shell_job")

    def test_an_explicit_call_id_wins(self, jm: JobManager):
        with _active_tool_call("call-7", "other_tool"):
            rec = jm.submit(
                tool="t", backend="subprocess", spec={"command": "echo x"},
                resume_on_complete=True, call_id="fc-1", session_id="s", user_id="u",
            )

        # The tool named its call: nothing of the active one is taken.
        assert (rec.call_id, rec.call_name) == ("fc-1", "t")

    def test_an_explicit_call_name_is_kept(self, jm: JobManager):
        """``examples/jobs_demo.py``'s shape: its own name, the id from ``tool_context``."""
        with _active_tool_call("call-7", "run_shell_job"):
            rec = jm.submit(
                tool="run_shell_job", backend="subprocess", spec={"command": "echo x"},
                resume_on_complete=True, call_id=None, call_name="shell",
                session_id="s", user_id="u",
            )

        assert (rec.call_id, rec.call_name) == ("call-7", "shell")

    def test_without_an_active_call_nothing_is_filled(self, jm: JobManager):
        rec = jm.submit(
            tool="t", backend="subprocess", spec={"command": "echo x"},
            resume_on_complete=True, session_id="s", user_id="u",
        )

        assert (rec.call_id, rec.call_name) == (None, "t")

    def test_a_fire_and_forget_job_is_not_linked(self, jm: JobManager):
        with _active_tool_call("call-7", "run_shell_job"):
            rec = jm.submit(tool="t", backend="subprocess", spec={"command": "echo x"})

        assert (rec.call_id, rec.call_name) == (None, None)


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
        jm._create(rec)
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
        seeder._create(rec)
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
        seeder._create(rec)
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
            first._create(queued)
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
        seeder._create(queued)
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
            jm._create(queued)

            def _boom(fd):
                raise OSError("flock unavailable")

            monkeypatch.setattr(jobs_manager, "_acquire_lock", _boom)
            jm._maybe_start_queued()

            assert starts == []
            assert jm._records["q1"].state is JobState.QUEUED
        finally:
            jm.close()


class _RemoteBackend(JobBackend):
    """A durable remote job: any manager that has the backend can reach it."""

    name = "remote"
    survives_restart = True
    cancels_foreign_jobs = True

    def __init__(self, outcome: JobState = JobState.RUNNING) -> None:
        self.outcome = outcome
        self.polled: list[str] = []

    def start(self, record, job_dir):
        record.backend_handle = f"remote-{record.job_id}"

    def poll(self, record, job_dir):
        self.polled.append(record.job_id)
        return self.outcome

    def cancel(self, record, job_dir):
        return True


class TestAJobWhoseBackendIsNotRegistered:
    """A manager that lacks a job's backend leaves the job as it is.

    The framework builds every manager with the default backends, and an app
    registers its own afterwards, once the constructor has reconciled. The
    manager wrote UNKNOWN, a terminal state, over a job whose backend it did
    not have, whoever was running it: a custom backend's running job ended
    whenever another CLI started, or its own restarted, and was offered for
    resume as finished. A queued one was claimed and its launch raised.
    """

    @staticmethod
    def _owner(base: Path, **kwargs) -> tuple[JobManager, _RemoteBackend]:
        manager = JobManager(base_dir=base, **kwargs)
        backend = _RemoteBackend()
        manager.register_backend(backend)
        return manager, backend

    def test_another_manager_leaves_a_running_job_alone(self, tmp_path: Path):
        base = tmp_path / "jobs"
        owner, _ = self._owner(base)
        try:
            rec = owner.submit(
                tool="t", backend="remote", spec={}, resume_on_complete=True,
                session_id="s", user_id="u",
            )
            assert owner.get(rec.job_id).state is JobState.RUNNING

            other = JobManager(base_dir=base)  # the app registers "remote" later
            try:
                on_disk = json.loads((base / rec.job_id / "meta.json").read_text())
                assert on_disk["state"] == JobState.RUNNING.value
                assert other.awaiting_resume() == []
                assert other.get(rec.job_id).state is JobState.RUNNING
            finally:
                other.close()
            assert owner.get(rec.job_id).state is JobState.RUNNING
            assert owner.awaiting_resume() == []
        finally:
            owner.close()

    def test_a_restarted_manager_polls_it_once_the_backend_is_registered(
        self, tmp_path: Path
    ):
        base = tmp_path / "jobs"
        owner, _ = self._owner(base)
        rec = owner.submit(tool="t", backend="remote", spec={})
        owner.close()

        restarted = JobManager(base_dir=base)
        try:
            assert restarted.get(rec.job_id).state is JobState.RUNNING
            backend = _RemoteBackend(outcome=JobState.SUCCEEDED)
            restarted.register_backend(backend)

            assert restarted.get(rec.job_id).state is JobState.SUCCEEDED
            assert backend.polled == [rec.job_id]
        finally:
            restarted.close()

    def test_a_queued_job_is_neither_claimed_nor_launched(self, tmp_path: Path):
        base = tmp_path / "jobs"
        owner, _ = self._owner(base, max_concurrent=1)
        try:
            running = owner.submit(tool="t", backend="remote", spec={})
            queued = owner.submit(tool="t", backend="remote", spec={})
            assert queued.state is JobState.QUEUED

            other = JobManager(base_dir=base)
            try:
                other.reconcile()
                on_disk = json.loads((base / queued.job_id / "meta.json").read_text())
                assert (on_disk["state"], on_disk["exec_owner"]) == ("queued", None)
            finally:
                other.close()

            owner.cancel(running.job_id)  # its slot frees: the owner launches it
            assert owner.get(queued.job_id).state is JobState.RUNNING
        finally:
            owner.close()

    def test_cancel_tail_and_result_leave_it_alone(self, tmp_path: Path):
        base = tmp_path / "jobs"
        owner, backend = self._owner(base, max_concurrent=1)
        other = None
        try:
            running = owner.submit(tool="t", backend="remote", spec={})
            queued = owner.submit(tool="t", backend="remote", spec={})
            finished = owner.submit(tool="t", backend="remote", spec={})
            owner.cancel(finished.job_id)  # queued: cancelled at once

            other = JobManager(base_dir=base)
            before = {
                job_id: (base / job_id / "meta.json").read_bytes()
                for job_id in (running.job_id, queued.job_id)
            }
            assert other.cancel(running.job_id).state is JobState.RUNNING
            assert other.cancel(queued.job_id).state is JobState.QUEUED
            for job_id, data in before.items():
                assert (base / job_id / "meta.json").read_bytes() == data
            assert other.tail(running.job_id) == []
            assert other.result(finished.job_id) is None
        finally:
            if other is not None:
                other.close()
            owner.close()

    def test_it_holds_no_slot_of_the_cap(self, tmp_path: Path):
        """It is never judged here, so it never ends here: counted, it starved the cap."""
        base = tmp_path / "jobs"
        owner, _ = self._owner(base)
        stale = owner.submit(tool="t", backend="remote", spec={})
        owner.close()

        other = JobManager(base_dir=base, max_concurrent=1)  # no "remote" here
        try:
            job = other.submit(tool="t", backend="subprocess", spec={"command": "true"})

            assert job.state is JobState.RUNNING, "the cap was full of a job it cannot judge"
            assert other.get(stale.job_id).state is JobState.RUNNING
        finally:
            other.close()

    def test_it_is_logged_at_warning_only_when_asked_for(
        self, tmp_path: Path, monkeypatch
    ):
        """An app registers its backends after the constructor has reconciled,
        so every start and orchestrator switch passes such a job: those
        passes log at debug. A cancel, tail or result asks for the job by
        name, and the first of them warns."""
        from agentic_cli.tools.jobs import manager as jobs_manager

        logged: list[tuple[str, str, dict]] = []

        class _Recorder:
            def __getattr__(self, level):
                return lambda event, **fields: logged.append((level, event, fields))

        def levels() -> list[str]:
            return [
                level
                for level, event, fields in logged
                if event == "job_backend_not_registered"
                and fields == {"job_id": rec.job_id, "backend": "remote"}
            ]

        base = tmp_path / "jobs"
        owner, _ = self._owner(base, max_concurrent=1)
        try:
            rec = owner.submit(tool="t", backend="remote", spec={})
            queued = owner.submit(tool="t", backend="remote", spec={})
            monkeypatch.setattr(jobs_manager, "logger", _Recorder())
            other = JobManager(base_dir=base)
            try:
                other.reconcile()
                other.get(rec.job_id)
                other.get(queued.job_id)
                assert levels() and set(levels()) == {"debug"}
                assert all(
                    level == "debug"
                    for level, event, _ in logged
                    if event == "job_backend_not_registered"
                ), "the queued job's scheduling warned"

                passive = len(levels())
                other.cancel(rec.job_id)
                other.tail(rec.job_id)
                assert levels()[passive:].count("warning") == 1
                assert levels()[-1] == "debug"  # the tail: warned once already
            finally:
                other.close()
        finally:
            owner.close()


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
            with jm._mutate(rec):
                rec.state = JobState.CANCELLED  # a stale decision, about to be written

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
    """Block until ``path`` appears."""
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
        from agentic_cli.tools.jobs.manager import ClaimLockUnavailable

        base = tmp_path / "jobs"
        TestCrossProcessResumeClaims._seed_finished_job(base)
        jm = JobManager(base_dir=base)
        try:
            assert jm.begin_resume("j1") is True
            rec = jm._records["j1"]
            meta = base / "j1" / "meta.json"
            before = meta.read_text()

            self._no_lock(monkeypatch)
            with pytest.raises(ClaimLockUnavailable):
                with jm._mutate(rec):
                    rec.state = JobState.CANCELLED  # an ordinary change to write

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

            self._no_lock(monkeypatch)
            # Between our read and our write, the delivery completes elsewhere.
            self._write_state(
                base,
                "j1",
                resume_state=ResumeState.DELIVERED.value,
                resume_owner=None,
            )

            # Every call that writes a record, without the lock.
            jm.get("j1")
            jm.list()
            jm.result("j1")
            jm.cancel("j1")
            jm.begin_resume("j1")

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
            rec = jm.submit(tool="t", backend="subprocess", spec={"command": "true"})

            assert (base / rec.job_id / "meta.json").exists()
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
        jm._create(rec)
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
        jm._create(rec)
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
        with jm._mutate(rec):
            rec.resume_state = ResumeState.RESUMING.value
            rec.resume_owner = f"{_this_host()}:{_dead_pid()}"
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


class TestAnInProcessJobRunsOnlyWhereItWasSubmitted:
    """A queued in-process job is launched by the manager that submitted it.

    Its callable cannot be saved (``"<non-serializable: function>"`` on disk),
    yet any manager that loaded the record claimed and launched it: another
    CLI, or the manager an orchestrator switch builds while the old one is
    still open. It failed, "launch failed: inprocess job requires a callable
    spec['target']". Such a job is now claimed by its submitter at submit,
    and one still queued when that manager closes is recorded FAILED.
    """

    @staticmethod
    def _busy_then_queued(owner: JobManager, gate: Path, marker: Path, **resume):
        busy = owner.submit(
            tool="wait", backend="inprocess",
            spec={"target": _wait_for_gate, "kwargs": {"path": str(gate), "timeout": 10.0}},
        )
        queued = owner.submit(
            tool="mark", backend="inprocess",
            spec={"target": _touch_marker, "args": (str(marker),)}, **resume,
        )
        assert (busy.state, queued.state) == (JobState.RUNNING, JobState.QUEUED)
        return busy, queued

    def test_another_manager_leaves_it_to_its_submitter(self, tmp_path: Path):
        base = tmp_path / "jobs"
        gate, marker = tmp_path / "gate", tmp_path / "marker"
        owner = JobManager(base_dir=base, max_concurrent=1)
        other = None
        try:
            _, queued = self._busy_then_queued(owner, gate, marker)

            other = JobManager(base_dir=base)  # another CLI, or a switch's new manager
            other.reconcile()
            seen = other.get(queued.job_id)
            assert seen.state is JobState.QUEUED, f"launched elsewhere: {seen.error}"

            gate.write_text("go")
            deadline = time.time() + 10
            while owner.get(queued.job_id).state is not JobState.SUCCEEDED:
                assert time.time() < deadline, owner.get(queued.job_id)
                owner.reconcile()  # the busy job ends: its slot frees
                time.sleep(0.05)
            assert marker.read_text() == "ran"
        finally:
            gate.write_text("go")
            if other is not None:
                other.close()
            owner.close()

    def test_close_fails_one_still_queued(self, tmp_path: Path):
        base = tmp_path / "jobs"
        gate, marker = tmp_path / "gate", tmp_path / "marker"
        owner = JobManager(base_dir=base, max_concurrent=1)
        later = None
        try:
            _, queued = self._busy_then_queued(
                owner, gate, marker,
                resume_on_complete=True, call_id="c1", session_id="s", user_id="u",
            )
            owner.close()
            owner.close()  # idempotent

            on_disk = json.loads((base / queued.job_id / "meta.json").read_text())
            assert on_disk["state"] == JobState.FAILED.value
            assert "closed before the job started" in on_disk["error"]
            assert on_disk["finished_at"] is not None

            later = JobManager(base_dir=base)
            rec = later.get(queued.job_id)
            assert rec.state is JobState.FAILED
            assert "closed before the job started" in rec.error
            assert [r.job_id for r in later.awaiting_resume()] == [queued.job_id]
            assert not marker.exists()
        finally:
            gate.write_text("go")
            if later is not None:
                later.close()
            owner.close()

    def test_close_without_the_lock_leaves_it_queued(self, tmp_path: Path, monkeypatch):
        """Startup recovery fails it once this process is gone. The backends
        are released all the same."""
        from agentic_cli.tools.jobs import manager as jobs_manager

        class _Released(JobBackend):
            name = "released"
            closed = False

            def start(self, record, job_dir):
                return None

            def poll(self, record, job_dir):
                return JobState.RUNNING

            def cancel(self, record, job_dir):
                return True

            def close(self):
                self.closed = True

        base = tmp_path / "jobs"
        gate, marker = tmp_path / "gate", tmp_path / "marker"
        owner = JobManager(base_dir=base, max_concurrent=1)
        released = _Released()
        owner.register_backend(released)
        try:
            _, queued = self._busy_then_queued(owner, gate, marker)

            def _boom(fd):
                raise OSError("flock unavailable")

            monkeypatch.setattr(jobs_manager, "_acquire_lock", _boom)
            owner.close()

            on_disk = json.loads((base / queued.job_id / "meta.json").read_text())
            assert on_disk["state"] == JobState.QUEUED.value
            assert released.closed, "close() released no backend"
        finally:
            gate.write_text("go")
            monkeypatch.undo()
            owner.close()

    def test_close_with_nothing_queued_does_not_wait_for_the_lock(
        self, tmp_path: Path
    ):
        """Another CLI's transaction holds the lock; this one has nothing to fail."""
        import fcntl
        import threading

        base = tmp_path / "jobs"
        jm = JobManager(base_dir=base)
        jm.submit(tool="t", backend="subprocess", spec={"command": "true"})
        fd = os.open(base / ".resume.lock", os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        closing = threading.Thread(target=jm.close, daemon=True)
        try:
            closing.start()
            closing.join(2.0)
            assert not closing.is_alive(), "close() waited for another CLI's lock"
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            closing.join(5.0)

    def test_close_with_nothing_queued_does_not_warn_of_the_lock(
        self, tmp_path: Path, monkeypatch
    ):
        """The jobs directory is gone, so no lock can be taken: nothing to say."""
        import shutil

        from agentic_cli.tools.jobs import manager as jobs_manager

        logged: list[str] = []

        class _Recorder:
            def __getattr__(self, level):
                return lambda event, **fields: logged.append(event)

        base = tmp_path / "jobs"
        jm = JobManager(base_dir=base)
        shutil.rmtree(base)
        monkeypatch.setattr(jobs_manager, "logger", _Recorder())

        jm.close()

        assert "job_close_skipped_unlocked" not in logged
