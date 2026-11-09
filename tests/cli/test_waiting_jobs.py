"""Which finished jobs wait to be told in the open conversation: one rule.

Three places say it: the line an opened conversation says ("A finished job
waits for this conversation…"), the status bar's ``↻N``, and the deliveries
(``/jobs resume``, a turn's end). They decided it three ways: this
conversation's jobs only; this one's and those with no conversation; and
those, plus the jobs whose conversation is gone. So one start said one job
waited, the status bar counted two and ``/jobs resume`` told three. And the
status bar counted a job whose result was being delivered (``RESUMING``), for
the life of the process when that delivery could not be recorded.

The rule (``cli/waiting_jobs.py``): a finished job flagged
``resume_on_complete`` whose delivery is still ``PENDING`` is told here when
it is this conversation's, has no conversation, or its conversation is gone
(``can_resume`` says so).
"""

from __future__ import annotations

import pytest

pytest.importorskip("google.adk")

from agentic_cli.cli.job_monitor import JobMonitor  # noqa: E402
from agentic_cli.tools.jobs import JobRecord, JobState  # noqa: E402
from agentic_cli.workflow.events import WorkflowEvent  # noqa: E402
from tests.cli.test_compact_and_new import (  # noqa: E402,F401
    Workflow,
    app_on,
    fake_settings,
    project,
    quiet_app,
    said,
)


def finished(job_id: str, session_id: str | None, user_id: str | None, *,
             resume_state: str = "pending") -> JobRecord:
    return JobRecord(
        job_id=job_id, tool="run_shell_job", backend="subprocess", name=job_id,
        state=JobState.SUCCEEDED, resume_on_complete=True, call_id=f"c-{job_id}",
        session_id=session_id, user_id=user_id, resume_state=resume_state,
        finished_at=1.0,
    )


class Jobs:
    """A job manager holding these records: ``awaiting_resume`` returns the
    finished, flagged, ``PENDING`` ones, as the real one does."""

    def __init__(self, records: list[JobRecord]) -> None:
        self.records = records
        self.claimed: list[str] = []

    def list(self, **_) -> list[JobRecord]:
        return list(self.records)

    def awaiting_resume(self) -> list[JobRecord]:
        return [
            r for r in self.records
            if r.resume_on_complete and r.resume_state == "pending"
            and r.state.value in ("succeeded", "failed", "cancelled", "unknown")
        ]

    def begin_resume(self, job_id: str) -> bool:
        self.claimed.append(job_id)
        return True

    def complete_resume(self, job_id: str, *, delivered: bool, error=None) -> None:
        pass


class JobsWorkflow(Workflow):
    """Conversations: the open one, ``other`` (stored), ``deleted`` (gone)."""

    supports_job_resume = True

    def __init__(self, jobs: Jobs) -> None:
        super().__init__()
        self.job_manager = jobs

    async def can_resume(self, record) -> bool:
        return record.session_id not in (None, "deleted")

    async def resume_with_job_result(self, record):
        yield WorkflowEvent.text("The build finished.")


def monitor_of(app) -> JobMonitor:
    return JobMonitor(
        app.session, app._workflow_controller, settings=app.settings,
        conversation=lambda: (app.settings.default_user, app.session_id),
    )


async def test_the_line_the_status_bar_and_jobs_resume_agree(project):
    jobs = Jobs([])
    app = app_on(JobsWorkflow(jobs), fake_settings())
    me, here = app.settings.default_user, app.session_id
    jobs.records += [
        finished("here", here, me),
        finished("orphan", None, None),
        finished("gone", "deleted", me),
        finished("elsewhere", "other", me),
        finished("someone-elses", here, "them"),
    ]

    await app._name_its_waiting_jobs()
    cue = await monitor_of(app).poll_once()
    await app.process_input("/jobs resume")

    assert said(app.session, "add_message")[0] == (
        "3 finished jobs wait for this conversation: /jobs resume tells them now."
    )
    assert cue == "↻3: /jobs resume"
    assert jobs.claimed == ["here", "orphan", "gone"]


async def test_a_job_being_delivered_is_not_counted(project):
    """Its delivery is claimed (``RESUMING``): it waits no more, here or anywhere."""
    jobs = Jobs([])
    app = app_on(JobsWorkflow(jobs), fake_settings())
    me, here = app.settings.default_user, app.session_id
    jobs.records += [
        finished("waits", here, me),
        finished("delivering", here, me, resume_state="resuming"),
    ]

    assert await monitor_of(app).poll_once() == "↻1: /jobs resume"


async def test_the_status_bar_asks_about_a_conversation_a_while_after_it_asked(project):
    """A conversation's lookup may read all of it, and the tick runs every
    2 seconds: its answer is kept a while."""
    jobs = Jobs([])
    workflow = JobsWorkflow(jobs)
    asked: list[str] = []
    looked_up = workflow.can_resume

    async def can_resume(record) -> bool:
        asked.append(record.job_id)
        return await looked_up(record)

    workflow.can_resume = can_resume
    app = app_on(workflow, fake_settings())
    jobs.records.append(finished("elsewhere", "other", app.settings.default_user))
    monitor = monitor_of(app)

    for _ in range(3):
        assert await monitor.poll_once() is None

    assert asked == ["elsewhere"]


def test_a_record_says_whether_it_waits_for_delivery():
    """What ``JobManager.awaiting_resume`` returns, judged on one record."""
    assert finished("j", "s", "u").awaiting_resume
    assert not finished("j", "s", "u", resume_state="resuming").awaiting_resume
    assert not finished("j", "s", "u", resume_state="delivered").awaiting_resume
    running = finished("j", "s", "u")
    running.state = JobState.RUNNING
    assert not running.awaiting_resume
    unflagged = finished("j", "s", "u")
    unflagged.resume_on_complete = False
    assert not unflagged.awaiting_resume


class TestTheStatusBarAsksAgain:
    """The answers the status bar keeps are dropped when they may be wrong:
    a conversation deleted (``/sessions --delete``), or the workflow replaced
    (an orchestrator switch: another backend's conversations)."""

    def _setup(self, project):
        jobs = Jobs([])
        workflow = JobsWorkflow(jobs)
        stored = {"other"}

        async def can_resume(record) -> bool:
            return record.session_id in stored

        async def delete_session(session_id, user_id=None) -> bool:
            stored.discard(session_id)
            return True

        workflow.can_resume = can_resume
        workflow.delete_session = delete_session
        app = app_on(workflow, fake_settings())
        jobs.records.append(finished("elsewhere", "other", app.settings.default_user))
        monitor = monitor_of(app)
        app._job_monitor = monitor  # as run() keeps it
        return app, monitor, jobs

    async def test_after_a_conversation_is_deleted(self, project):
        app, monitor, _ = self._setup(project)
        assert await monitor.poll_once() is None  # it waits for its conversation

        await app.process_input("/sessions --delete=other")

        assert await monitor.poll_once() == "↻1: /jobs resume"  # gone: told here

    async def test_after_an_orchestrator_switch(self, project):
        app, monitor, jobs = self._setup(project)
        assert await monitor.poll_once() is None

        class Swapped(JobsWorkflow):
            async def can_resume(self, record) -> bool:
                return False  # sessions do not move between backends

        app._workflow_controller._workflow = Swapped(jobs)

        assert await monitor.poll_once() == "↻1: /jobs resume"
