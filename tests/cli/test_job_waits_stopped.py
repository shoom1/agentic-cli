"""Ctrl+C while a job command waits for the job manager says the request stopped.

The CLI calls the job manager on a worker thread, so Ctrl+C now stops the
wait for it (another CLI holding the jobs directory, a cancel waiting for
the job's processes). thinking-prompt says "Operation cancelled..." only
when a box ended, and these waits open none, so the request stopped
without a word. Each such wait now says it, once: ``/jobs`` and its
subcommands, and the resume of finished jobs (``/jobs resume``, a turn's
end) while it lists them or looks a job's conversation up.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_cli.cli.app import BaseCLIApp
from agentic_cli.cli.builtin_commands import JobsCommand
from agentic_cli.cli.message_processor import CANCELLED_LINE, MessageProcessor
from agentic_cli.tools.jobs import JobManager, JobRecord, JobState
from tests.cli.dialog_fakes import autospec_session
from tests.event_replay import ReplayController
from tests.test_message_processor_cancel import Workflow

SETTINGS = SimpleNamespace(
    default_user="u", job_auto_resume=True, verbose_thinking=False, dialog_placement="box",
)


def said(session) -> list[str]:
    return [c.args[1] for c in session.add_message.call_args_list]


class Waiting:
    """A job manager each of whose calls waits: another CLI holds the jobs directory."""

    def __init__(self, awaiting: list | None = None) -> None:
        self.waiting = threading.Event()
        self.release = threading.Event()
        self._awaiting = awaiting

    def _wait(self) -> None:
        self.waiting.set()
        self.release.wait(5)

    def cancel(self, job_id):
        self._wait()

    def clean(self):
        self._wait()
        return 0

    def get(self, job_id):
        self._wait()

    def list(self, **_):
        self._wait()
        return []

    def awaiting_resume(self):
        if self._awaiting is not None:
            return list(self._awaiting)
        self._wait()
        return []

    def has_backend(self, name):
        return True


async def _stop_while_it_waits(task: asyncio.Task, waiting: threading.Event) -> None:
    assert await asyncio.to_thread(waiting.wait, 5), "it never reached the job manager"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


class TestTheJobsCommand:
    async def test_a_cancel_waiting_for_its_job_says_so(self, tmp_path: Path):
        """``/jobs cancel`` waits for a job that ignores SIGTERM (up to 2 s)."""
        gate = tmp_path / "gate"
        jobs = JobManager(base_dir=tmp_path / "jobs")
        try:
            job = jobs.submit(tool="t", backend="subprocess", spec={
                "command": f"trap '' TERM; while [ ! -e {gate} ]; do sleep 0.02; done",
            })
            time.sleep(0.3)
            app = SimpleNamespace(workflow=SimpleNamespace(job_manager=jobs),
                                  session=autospec_session())
            handler = asyncio.create_task(JobsCommand().execute(f"cancel {job.job_id}", app))
            await asyncio.sleep(0.5)  # the backend waits for the job
            handler.cancel()
            with pytest.raises(asyncio.CancelledError):
                await handler
        finally:
            gate.write_text("go")
            await asyncio.sleep(2.0)  # the thread's cancel finishes on its own
            jobs.close()

        assert said(app.session) == [CANCELLED_LINE]

    @pytest.mark.parametrize("line", ["cancel j1", "clean", "j1", "", "all"])
    async def test_each_wait_says_so(self, line):
        jobs = Waiting()
        app = SimpleNamespace(workflow=SimpleNamespace(job_manager=jobs),
                              session=autospec_session())

        try:
            await _stop_while_it_waits(
                asyncio.create_task(JobsCommand().execute(line, app)), jobs.waiting,
            )
        finally:
            jobs.release.set()

        assert said(app.session) == [CANCELLED_LINE]


class LookingUp(Workflow):
    """Its conversation lookup (``can_resume``) waits."""

    def __init__(self) -> None:
        super().__init__(hang=False)
        self.looking = threading.Event()

    async def can_resume(self, record) -> bool:
        self.looking.set()
        await asyncio.Event().wait()
        return True  # pragma: no cover


def app_on(workflow, jobs) -> BaseCLIApp:
    workflow.job_manager = jobs
    app = BaseCLIApp.__new__(BaseCLIApp)
    app._workflow_controller = ReplayController(workflow)
    app._message_processor = MessageProcessor()
    app.session = autospec_session()
    app._settings = SETTINGS
    app._usage_tracker = None
    app._session_id = "s"
    return app


class TestTheResumeOfFinishedJobs:
    async def test_stopped_while_it_lists_them(self):
        jobs = Waiting()
        app = app_on(Workflow(hang=False), jobs)

        try:
            await _stop_while_it_waits(
                asyncio.create_task(app.resume_finished_jobs()), jobs.waiting,
            )
        finally:
            jobs.release.set()

        assert said(app.session) == [CANCELLED_LINE]

    async def test_stopped_while_it_looks_a_conversation_up(self):
        elsewhere = JobRecord(
            job_id="j1", tool="t", backend="subprocess", name="build",
            state=JobState.SUCCEEDED, resume_on_complete=True,
            session_id="other", user_id="u",
        )
        workflow = LookingUp()
        app = app_on(workflow, Waiting(awaiting=[elsewhere]))

        await _stop_while_it_waits(
            asyncio.create_task(app.resume_finished_jobs()), workflow.looking,
        )

        assert said(app.session) == [CANCELLED_LINE]
