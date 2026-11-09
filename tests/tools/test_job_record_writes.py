"""Every write of a job record keeps what it did not change.

A job's ``meta.json`` is shared by every manager over the jobs directory:
another CLI of the app, or the manager an orchestrator switch builds while
the old one is open. Each field of a record has a declared rule
(``_FIELD_RULES``) saying how a manager reads it back before it writes.
``result()`` reloaded and decided outside the cross-process lock, so a cancel
another manager requested meanwhile was recorded but not acted on.
"""

from __future__ import annotations

import dataclasses
import json
import threading
from pathlib import Path
from typing import Any, Callable

from agentic_cli.tools.jobs import JobManager, JobRecord, JobState
from agentic_cli.tools.jobs.backends import JobBackend


def _meta(base: Path, job_id: str) -> Path:
    return base / job_id / "meta.json"


def _read(base: Path, job_id: str) -> dict:
    return json.loads(_meta(base, job_id).read_text())


class _Scripted(JobBackend):
    """A job runs until the test ends it; a cancel stops it unless told not to."""

    name = "scripted"
    survives_restart = True
    cancels_foreign_jobs = True

    def __init__(self, *, survives_restart: bool = True, stoppable: bool = True) -> None:
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
            name for name, rule in _FIELD_RULES.items() if rule is not _FieldRule.IMMUTABLE
        }


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
