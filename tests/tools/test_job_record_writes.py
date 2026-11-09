"""Every write of a job record keeps what it did not change.

A job's ``meta.json`` is shared by every manager over the jobs directory:
another CLI of the app, or the manager an orchestrator switch builds while
the old one is open. A manager kept its own copy of each record and merged it
back by rules spread over several methods, each listing fields by hand, so a
field one of them missed was written over: a launch reloaded only the state
and the execution owner, and wrote another manager's cancel request or
delivery state away. Every write wrote the whole record, erasing the keys a
newer version had added. ``result()`` reloaded and decided outside the
cross-process lock, so a cancel another manager requested meanwhile was
recorded but not acted on. And ``complete_resume`` without the lock changed
the record in memory only, until the next reload put it back.

Each field now has a declared rule (``_FIELD_RULES``), and every write goes
through one transaction that reloads the record by those rules and writes it
onto the dict it read.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import socket
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from agentic_cli.tools.jobs import JobManager, JobRecord, JobState
from agentic_cli.tools.jobs.backends import JobBackend
from agentic_cli.tools.jobs.manager import ResumeState, ResumeStateError


@pytest.fixture(scope="module")
def dead_pid() -> int:
    """A pid that is certainly not running: a child we started and reaped."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _meta(base: Path, job_id: str) -> Path:
    return base / job_id / "meta.json"


def _read(base: Path, job_id: str) -> dict:
    return json.loads(_meta(base, job_id).read_text())


def _write_keys(base: Path, job_id: str, **values: Any) -> None:
    """Stand in for another manager's write, made while no manager is inside."""
    data = _read(base, job_id)
    data.update(values)
    _meta(base, job_id).write_text(json.dumps(data))


class _Scripted(JobBackend):
    """A job runs until the test ends it; a cancel stops it unless told not to."""

    name = "scripted"
    survives_restart = True
    cancels_foreign_jobs = True

    def __init__(
        self, *, survives_restart: bool = True, stoppable: bool = True
    ) -> None:
        self.survives_restart = survives_restart
        self.stoppable = stoppable
        self.ended: set[str] = set()

    def start(self, record, job_dir):
        record.backend_handle = f"handle-{record.job_id}"

    def poll(self, record, job_dir):
        if record.job_id in self.ended:
            record.exit_code = 0
            return JobState.SUCCEEDED
        return JobState.RUNNING

    def cancel(self, record, job_dir):
        return self.stoppable


_RESUMABLE = {"resume_on_complete": True, "session_id": "s", "user_id": "u"}


def _manager(base: Path, backend: JobBackend, **kwargs: Any) -> JobManager:
    return JobManager(base_dir=base, backends={backend.name: backend}, **kwargs)


def _submit(jm: JobManager) -> JobRecord:
    return jm.submit(tool="t", backend="scripted", spec={}, **_RESUMABLE)


def _seed(base: Path, **fields: Any) -> str:
    """A record on disk that no manager has loaded yet."""
    rec = JobRecord(job_id="j1", tool="t", backend="scripted", name="n", **fields)
    _meta(base, rec.job_id).parent.mkdir(parents=True)
    _meta(base, rec.job_id).write_text(json.dumps(rec.to_dict()))
    return rec.job_id


Inject = Callable[[str], None]


def _poll_that_ends(call: Callable[[JobManager, str], Any]):
    """A running job ends; ``call`` is what polls it."""

    def run(base: Path, dead: int, inject: Inject) -> str:
        backend = _Scripted()
        owner = _manager(base, backend)
        try:
            job = _submit(owner)
            inject(job.job_id)
            backend.ended.add(job.job_id)
            call(owner, job.job_id)
            return job.job_id
        finally:
            owner.close()

    return run


def _cancel(*, stoppable: bool):
    def run(base: Path, dead: int, inject: Inject) -> str:
        owner = _manager(base, _Scripted(stoppable=stoppable))
        try:
            job = _submit(owner)
            inject(job.job_id)
            owner.cancel(job.job_id)
            return job.job_id
        finally:
            owner.close()

    return run


def _launch(base: Path, dead: int, inject: Inject) -> str:
    """The running job ends, and its slot goes to the queued one."""
    backend = _Scripted()
    owner = _manager(base, backend, max_concurrent=1)
    try:
        first = _submit(owner)
        queued = _submit(owner)
        inject(queued.job_id)
        backend.ended.add(first.job_id)
        owner.get(first.job_id)
        return queued.job_id
    finally:
        owner.close()


def _finished(base: Path) -> tuple[JobManager, str]:
    backend = _Scripted()
    owner = _manager(base, backend)
    job = _submit(owner)
    backend.ended.add(job.job_id)
    owner.get(job.job_id)
    return owner, job.job_id


def _begin_resume(base: Path, dead: int, inject: Inject) -> str:
    owner, job_id = _finished(base)
    try:
        inject(job_id)
        owner.begin_resume(job_id)
        return job_id
    finally:
        owner.close()


def _complete_resume(base: Path, dead: int, inject: Inject) -> str:
    owner, job_id = _finished(base)
    try:
        assert owner.begin_resume(job_id)
        inject(job_id)
        with contextlib.suppress(ResumeStateError):  # another delivery's claim
            owner.complete_resume(job_id, delivered=False, error="turn failed")
        return job_id
    finally:
        owner.close()


def _recover_launches(base: Path, dead: int, inject: Inject) -> str:
    job_id = _seed(
        base,
        state=JobState.QUEUED,
        exec_owner=f"{socket.gethostname()}:{dead}:abcd1234",
    )
    inject(job_id)
    _manager(base, _Scripted()).close()
    return job_id


def _recover_resumes(base: Path, dead: int, inject: Inject) -> str:
    job_id = _seed(
        base, state=JobState.SUCCEEDED, finished_at=1.0, **_RESUMABLE,
        resume_state=ResumeState.RESUMING.value,
        resume_owner=f"{socket.gethostname()}:{dead}",
    )
    inject(job_id)
    _manager(base, _Scripted()).close()
    return job_id


def _close(base: Path, dead: int, inject: Inject) -> str:
    """A job claimed at submit is still queued when its manager closes."""
    owner = _manager(base, _Scripted(survives_restart=False), max_concurrent=1)
    try:
        _submit(owner)
        queued = _submit(owner)
        inject(queued.job_id)
    finally:
        owner.close()
    return queued.job_id


@dataclass(frozen=True)
class _WritePath:
    """One way a manager writes a record."""

    run: Callable[[Path, int, Inject], str]
    # The fields this path writes by design: another manager's value of any
    # other field must survive it.
    changes: frozenset[str]
    # What it writes when nobody else has written anything.
    writes: dict[str, Any]


_PATHS: dict[str, _WritePath] = {
    # A poll that finds the job ended records how it ended.
    "get": _WritePath(
        _poll_that_ends(lambda jm, job_id: jm.get(job_id)),
        frozenset({"state", "exit_code"}),
        {"state": "succeeded", "exit_code": 0},
    ),
    "list": _WritePath(
        _poll_that_ends(lambda jm, job_id: jm.list()),
        frozenset({"state", "exit_code"}),
        {"state": "succeeded", "exit_code": 0},
    ),
    "result": _WritePath(
        _poll_that_ends(lambda jm, job_id: jm.result(job_id)),
        frozenset({"state", "exit_code"}),
        {"state": "succeeded", "exit_code": 0},
    ),
    # A cancel that stops the job ends it now.
    "cancel": _WritePath(
        _cancel(stoppable=True),
        frozenset({"state", "finished_at"}),
        {"state": "cancelled"},
    ),
    # A cancel that cannot stop it records the request.
    "cancel_requested": _WritePath(
        _cancel(stoppable=False),
        frozenset({"cancel_requested"}),
        {"state": "running", "cancel_requested": True},
    ),
    # A launch claims the job, then starts it.
    "launch": _WritePath(
        _launch,
        frozenset({"state", "exec_owner", "started_at", "backend_handle"}),
        {"state": "running"},
    ),
    # The delivery transitions move the claim.
    "begin_resume": _WritePath(
        _begin_resume,
        frozenset({"resume_state", "resume_owner", "resume_error"}),
        {"resume_state": "resuming"},
    ),
    "complete_resume": _WritePath(
        _complete_resume,
        frozenset({"resume_state", "resume_owner", "resume_error"}),
        {"resume_state": "failed", "resume_error": "turn failed"},
    ),
    # Startup recovery fails what a dead process left halfway.
    "recover_launches": _WritePath(
        _recover_launches,
        frozenset({"state", "error", "finished_at"}),
        {"state": "failed"},
    ),
    "recover_resumes": _WritePath(
        _recover_resumes,
        frozenset({"resume_state", "resume_owner", "resume_error"}),
        {"resume_state": "failed"},
    ),
    # A close fails its own claims that never started.
    "close": _WritePath(
        _close,
        frozenset({"state", "error", "finished_at"}),
        {"state": "failed"},
    ),
}

# A value of each field a manager may change after submit, as another
# manager would have written it.
_ELSEWHERE: dict[str, Any] = {
    "state": "cancelled",
    "backend_handle": "handle-elsewhere",
    "pid": 4242,
    "exit_code": 7,
    "started_at": 2.0,
    "finished_at": 3.0,
    "error": "written elsewhere",
    "exec_owner": "otherhost:1:abc",
    "resume_state": "failed",
    "resume_error": "written elsewhere",
    "resume_owner": "otherhost:9",
    "cancel_requested": True,
}


class TestTheFieldTable:
    def test_every_field_of_a_record_has_a_rule(self):
        """A field added to ``JobRecord`` fails here until it has one."""
        from agentic_cli.tools.jobs.manager import _FIELD_RULES

        assert set(_FIELD_RULES) == {f.name for f in dataclasses.fields(JobRecord)}

    def test_another_managers_values_cover_every_field_that_changes(self):
        from agentic_cli.tools.jobs.manager import _FIELD_RULES, _FieldRule

        assert set(_ELSEWHERE) == {
            name
            for name, rule in _FIELD_RULES.items()
            if rule is not _FieldRule.IMMUTABLE
        }


class TestAnotherManagersWritesSurvive:
    """For each field a manager may change, another manager writes a value of
    its own; each write path then runs in the first. The value survives unless
    that path writes the field by design (``_WritePath.changes``)."""

    @pytest.mark.parametrize(
        ("path", "field"),
        [
            pytest.param(path, field, id=f"{path}-{field}")
            for path, case in _PATHS.items()
            for field in _ELSEWHERE
            if field not in case.changes
        ],
    )
    def test_its_value_survives(
        self, tmp_path: Path, dead_pid: int, path: str, field: str
    ):
        base = tmp_path / "jobs"
        value = _ELSEWHERE[field]

        job_id = _PATHS[path].run(
            base, dead_pid, lambda job_id: _write_keys(base, job_id, **{field: value})
        )

        assert _read(base, job_id)[field] == value, f"{path} wrote {field} over"


class TestKeysThisVersionDoesNotDeclare:
    """A later version's keys survive every write this version makes."""

    @pytest.mark.parametrize("path", list(_PATHS))
    def test_they_survive(self, tmp_path: Path, dead_pid: int, path: str):
        base = tmp_path / "jobs"
        case = _PATHS[path]

        job_id = case.run(
            base,
            dead_pid,
            lambda job_id: _write_keys(
                base, job_id, owner_lease="written by a newer CLI"
            ),
        )

        on_disk = _read(base, job_id)
        assert {key: on_disk[key] for key in case.writes} == case.writes  # it wrote
        assert on_disk.get("owner_lease") == "written by a newer CLI"

    def test_the_legacy_resumed_key_is_not_written_back(self, tmp_path: Path):
        """``resume_state`` replaces it; a reader of either version reads that."""
        base = tmp_path / "jobs"
        job_id = _seed(base, state=JobState.SUCCEEDED, finished_at=1.0, **_RESUMABLE)
        data = _read(base, job_id)
        del data["resume_state"]
        data["resumed"] = False
        _meta(base, job_id).write_text(json.dumps(data))

        jm = _manager(base, _Scripted())
        try:
            assert jm.begin_resume(job_id) is True
        finally:
            jm.close()

        on_disk = _read(base, job_id)
        assert on_disk["resume_state"] == ResumeState.RESUMING.value
        assert "resumed" not in on_disk


class TestResultDecidesUnderTheLock:
    """A cancel another manager requests while ``result()`` reads the job is
    acted on: the job ends CANCELLED, as ``job_cancel`` told the model."""

    class _Gated(_Scripted):
        def __init__(self) -> None:
            super().__init__(stoppable=False)
            self.on_poll: Callable[[], None] | None = None
            self.on_cancel: Callable[[], None] | None = None

        def poll(self, record, job_dir):
            if self.on_poll is None:
                return super().poll(record, job_dir)
            self.on_poll()
            record.exit_code = 1
            return JobState.FAILED  # it ended (the cancel's signal, say)

        def cancel(self, record, job_dir):
            if self.on_cancel is not None:
                self.on_cancel()
            return False  # cannot stop it now

    def test_a_cancel_requested_meanwhile_ends_it_cancelled(self, tmp_path: Path):
        base = tmp_path / "jobs"
        mine, theirs = self._Gated(), self._Gated()
        owner = _manager(base, mine)
        job = _submit(owner)
        other = _manager(base, theirs)  # another CLI, started after the submit
        in_cancel, polled = threading.Event(), threading.Event()
        told: dict[str, JobRecord] = {}
        try:
            # The other manager's cancel is under way when result() starts,
            # and waits (a little) for the owner's poll before it records
            # the request.
            theirs.on_cancel = lambda: (in_cancel.set(), polled.wait(0.5))
            mine.on_poll = polled.set
            canceller = threading.Thread(
                target=lambda: told.update(record=other.cancel(job.job_id)), daemon=True
            )
            canceller.start()
            assert in_cancel.wait(5)

            owner.result(job.job_id)
            canceller.join(5)

            assert told["record"].cancel_requested is True  # "cancelled when it ends"
            assert _read(base, job.job_id)["state"] == JobState.CANCELLED.value
            assert owner.get(job.job_id).state is JobState.CANCELLED
        finally:
            other.close()
            owner.close()


class TestCompleteResumeWithoutTheLock:
    """It cannot write, so it changes nothing: the record stays as the disk
    has it, claimed by this process, until startup recovery fails it."""

    def test_the_record_stays_as_the_disk_has_it(self, tmp_path: Path, monkeypatch):
        from agentic_cli.tools.jobs import manager as jobs_manager

        base = tmp_path / "jobs"
        owner, job_id = _finished(base)
        try:
            assert owner.begin_resume(job_id)

            def _no_lock(fd):
                raise OSError("flock unavailable")

            monkeypatch.setattr(jobs_manager, "_acquire_lock", _no_lock)
            owner.complete_resume(job_id, delivered=True)

            rec = owner.get(job_id)  # no lock: the record as this manager has it
            on_disk = _read(base, job_id)
            assert (rec.resume_state, rec.resume_owner) == (
                on_disk["resume_state"], on_disk["resume_owner"],
            )
            assert rec.resume_state == ResumeState.RESUMING.value
        finally:
            monkeypatch.undo()
            owner.close()


class TestAReadThatChangesNothingWritesNothing:
    """The job monitor lists every job every two seconds: a refresh that
    finds nothing new leaves each record's file as it is."""

    def test_the_file_is_not_rewritten(self, tmp_path: Path):
        base = tmp_path / "jobs"
        owner = _manager(base, _Scripted())
        try:
            job = _submit(owner)
            before = os.stat(_meta(base, job.job_id)).st_ino

            owner.get(job.job_id)
            owner.list()
            owner.result(job.job_id)
            owner.awaiting_resume()

            assert os.stat(_meta(base, job.job_id)).st_ino == before
        finally:
            owner.close()
