"""Every call the CLI makes to the job manager runs off the event loop.

``JobManager`` takes the jobs directory's cross-process lock (a blocking
``flock``) for nearly every call, and a cancel waits for the job's processes
to exit. The jobs directory is the user's, shared by every CLI of the app, so
a call made on the event loop's thread froze the prompt, its redraws and
Ctrl+C for as long as another CLI held the lock. Each entry point a user
reaches runs here under the loop guard (``tests/loop_guard.py``), against a
real ``JobManager``: the line an opened conversation says, a turn's end,
``/jobs`` and each of its subcommands, the status bar's tick, a turn stopped
after it started a job, and ADK answering a stopped turn's calls.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("google.adk")

from agentic_cli.cli.app import BaseCLIApp  # noqa: E402
from agentic_cli.cli.job_monitor import JobMonitor  # noqa: E402
from agentic_cli.cli.message_processor import CANCELLED_LINE, MessageProcessor  # noqa: E402
from agentic_cli.tools.jobs import JobManager, JobState  # noqa: E402
from agentic_cli.tools.jobs.backends import TERMINAL_STATES  # noqa: E402
from agentic_cli.workflow.events import WorkflowEvent  # noqa: E402
from tests.cli.dialog_fakes import autospec_session, opening_boxes  # noqa: E402
from tests.cli.test_compact_and_new import (  # noqa: E402,F401
    app_on,
    fake_settings,
    project,
    quiet_app,
    said,
)
from tests.cli.test_job_monitor import FakeController, FakeUI  # noqa: E402
from tests.cli.test_resume_session import Workflow as StoredWorkflow, conversation  # noqa: E402
from tests.event_replay import ReplayController  # noqa: E402
from tests.loop_guard import no_blocking_calls  # noqa: E402
from tests.test_message_processor_cancel import Workflow as HangingWorkflow  # noqa: E402

SETTINGS = SimpleNamespace(default_user="u", verbose_thinking=False, dialog_placement="box")


class JobsWorkflow(StoredWorkflow):
    """A ready workflow with stored conversations, a real job manager, and
    resume turns that say the job's result."""

    def __init__(self, jobs: JobManager, *stored) -> None:
        super().__init__(*stored)
        self.job_manager = jobs
        self.resumed: list[str] = []

    async def can_resume(self, record) -> bool:
        return True

    async def resume_with_job_result(self, record):
        self.resumed.append(record.job_id)
        yield WorkflowEvent.text("The build finished.")


@pytest.fixture
def jobs(tmp_path: Path):
    manager = JobManager(base_dir=tmp_path / "jobs")
    yield manager
    manager.close()


def _until(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def finished(jobs: JobManager, session_id: str, user_id: str):
    """A job of that conversation that has ended and waits to be told there."""
    record = jobs.submit(
        tool="run_shell_job", backend="subprocess", spec={"command": "true"},
        name="build", resume_on_complete=True, call_id="c1",
        session_id=session_id, user_id=user_id,
    )
    _until(lambda: jobs.get(record.job_id).state in TERMINAL_STATES)
    return record


def running(jobs: JobManager):
    return jobs.submit(
        tool="run_shell_job", backend="subprocess", spec={"command": "sleep 30"}, name="train",
    )


class TestTheApp:
    async def test_the_line_an_opened_conversation_says(self, project, jobs):
        """``--continue`` / ``--resume <id>`` name the jobs that wait there."""
        workflow = JobsWorkflow(jobs, conversation("abc", "Plan the trip", 2))
        app = app_on(workflow, fake_settings())
        finished(jobs, "abc", app.settings.default_user)

        with no_blocking_calls():
            assert await app._resume("abc")

        assert "A finished job waits for this conversation: /jobs resume tells it now." in (
            said(app.session, "add_message")
        )

    async def test_a_turns_end_tells_its_finished_jobs(self, project, jobs):
        workflow = JobsWorkflow(jobs)
        app = app_on(workflow, fake_settings(job_auto_resume=True))
        record = finished(jobs, app.session_id, app.settings.default_user)

        with no_blocking_calls():
            await app.process_input("Hello.")

        assert workflow.resumed == [record.job_id]
        assert jobs.get(record.job_id).resume_state == "delivered"


class TestTheJobsCommand:
    @pytest.mark.parametrize("line", [
        "/jobs", "/jobs all", "/jobs {running}", "/jobs cancel {running}", "/jobs clean",
        "/jobs resume",
    ])
    async def test_each_subcommand(self, project, jobs, line):
        workflow = JobsWorkflow(jobs)
        app = app_on(workflow, fake_settings())
        job = running(jobs)
        finished(jobs, app.session_id, app.settings.default_user)

        with no_blocking_calls():
            await app.process_input(line.format(running=job.job_id))

        assert said(app.session, "add_error") == []
        if "cancel" in line:
            assert jobs.get(job.job_id).state is JobState.CANCELLED
        else:
            jobs.cancel(job.job_id)


class TestTheStatusBar:
    async def test_its_tick(self, jobs):
        job = running(jobs)
        ui = FakeUI()
        controller = FakeController(ui, jobs)
        monitor = JobMonitor(ui, controller, interval=0.02)

        with no_blocking_calls():
            async with monitor.running():
                for _ in range(250):
                    if controller.jobs_status_segment:
                        break
                    await asyncio.sleep(0.02)

        assert controller.jobs_status_segment == "jobs: 1 running"
        jobs.cancel(job.job_id)


class TestAStoppedTurn:
    async def test_it_names_the_jobs_it_started(self, jobs):
        ui = opening_boxes(autospec_session())
        workflow = HangingWorkflow()
        workflow.job_manager = jobs
        turn = asyncio.create_task(
            MessageProcessor().process("hi", ReplayController(workflow), ui, SETTINGS)
        )
        await asyncio.wait_for(workflow.streaming.wait(), timeout=2)
        job = await asyncio.to_thread(running, jobs)  # the turn's tool submitted it

        with no_blocking_calls():
            turn.cancel()
            with pytest.raises(asyncio.CancelledError):
                await turn

        assert said(ui, "add_message") == [
            CANCELLED_LINE,
            f"Still running in the background: train ({job.job_id}). "
            "Stop one with /jobs cancel <id>.",
        ]
        jobs.cancel(job.job_id)


class TestADKAnsweringAStoppedTurnsCalls:
    async def test_a_call_whose_job_was_submitted(self, jobs):
        from google.adk.events import Event
        from google.genai import types

        from agentic_cli.workflow.adk.interrupted_calls_plugin import (
            ABORTED,
            InterruptedCallsPlugin,
        )
        from agentic_cli.workflow.interrupts import OpenCalls
        from agentic_cli.workflow.service_registry import JOB_MANAGER, set_service_registry
        from agentic_cli.workflow.sessions import (
            SessionRef,
            reset_active_turn,
            set_active_turn,
        )

        job = jobs.submit(
            tool="ic_start_job", backend="subprocess", spec={"command": "sleep 30"},
            call_id="toolu_j", session_id="s", user_id="u",
        )
        seal = Event(
            invocation_id="i", author="solo", error_code=ABORTED,
            content=types.Content(role="user", parts=[types.Part(
                function_response=types.FunctionResponse(
                    id="toolu_j", name="ic_start_job", response={"error": "aborted"},
                ),
            )]),
        )
        plugin = InterruptedCallsPlugin(OpenCalls())
        turn = set_active_turn(SessionRef("app", "u", "s"))
        registry = set_service_registry({JOB_MANAGER: jobs})
        try:
            with no_blocking_calls():
                answered = await plugin.on_event_callback(invocation_context=None, event=seal)
        finally:
            registry.var.reset(registry)
            reset_active_turn(turn)

        (part,) = answered.content.parts
        assert part.function_response.response["job_id"] == job.job_id
        jobs.cancel(job.job_id)


class TestAStopWhileTheClaimIsTaken:
    async def test_the_claim_its_thread_takes_is_handed_back(self, tmp_path):
        """Ctrl+C stops the wait for the claim, not the claim: its thread
        takes it all the same, and it is handed back (``release_resume``).
        Nothing reached the agent, so the job waits for a later delivery."""
        taking, release, taken = threading.Event(), threading.Event(), threading.Event()

        class SlowToClaim(JobManager):
            def begin_resume(self, job_id: str) -> bool:
                taking.set()
                release.wait(5)  # another CLI holds the jobs directory's lock
                try:
                    return super().begin_resume(job_id)
                finally:
                    taken.set()

        jobs = SlowToClaim(base_dir=tmp_path / "jobs")
        try:
            record = finished(jobs, "s", "u")
            app = BaseCLIApp.__new__(BaseCLIApp)
            app._workflow_controller = ReplayController(HangingWorkflow())
            app._turn_lock = asyncio.Lock()
            app._message_processor = MessageProcessor()
            app.session = opening_boxes(autospec_session())
            app._settings = SETTINGS
            app._usage_tracker = None
            app._session_id = "s"

            delivery = asyncio.create_task(app._deliver_resume(jobs, record))
            assert await asyncio.to_thread(taking.wait, 5)
            delivery.cancel()
            with pytest.raises(asyncio.CancelledError):
                await delivery
            release.set()
            assert await asyncio.to_thread(taken.wait, 5)
            for _ in range(250):
                if jobs.get(record.job_id).resume_state != "resuming":
                    break
                await asyncio.sleep(0.02)

            after = jobs.get(record.job_id)
            assert (after.resume_state, after.resume_owner) == ("pending", None)
            assert [r.job_id for r in jobs.awaiting_resume()] == [record.job_id]
        finally:
            jobs.close()
