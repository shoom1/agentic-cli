"""A job's delivery is claimed only once its result can reach the agent.

The coordinator claimed a finished job (``begin_resume``: ``PENDING`` →
``RESUMING``) before ``process_resume`` checked that the workflow was ready
and looked the job's conversation up (``can_resume``). When the workflow had
not started, or Ctrl+C landed during the lookup, the claim was closed
``FAILED``, which is final: nothing had reached the agent, and no later
``/jobs resume`` told the result. The claim now follows the checks: a job
whose delivery stops before it stays ``PENDING`` for a later one. A Ctrl+C
during the turn itself, once the notice may have been written, still closes
the claim ``FAILED``.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call

import pytest

from agentic_cli.cli.app import BaseCLIApp
from agentic_cli.cli.message_processor import CANCELLED_LINE, MessageProcessor
from agentic_cli.tools.jobs import JobManager
from agentic_cli.tools.jobs.backends import TERMINAL_STATES
from tests.cli.dialog_fakes import autospec_session, opening_boxes
from tests.event_replay import ReplayController
from tests.test_message_processor_cancel import Workflow

SETTINGS = SimpleNamespace(
    default_user="u", job_auto_resume=True, verbose_thinking=False, dialog_placement="box",
)


class Starting(ReplayController):
    """A controller whose workflow has not started (``ensure_initialized``
    answers no) until ``started`` is set."""

    def __init__(self, workflow) -> None:
        super().__init__(workflow)
        self.started = False

    async def ensure_initialized(self, ui=None) -> bool:
        return self.started


class Looking(Workflow):
    """Its conversation lookup (``can_resume``) waits: a slow session store."""

    def __init__(self) -> None:
        super().__init__(hang=False)
        self.looking = asyncio.Event()

    async def can_resume(self, record) -> bool:
        self.looking.set()
        await asyncio.Event().wait()
        return True  # pragma: no cover


@pytest.fixture
def jobs(tmp_path: Path):
    manager = JobManager(base_dir=tmp_path / "jobs")
    yield manager
    manager.close()


def finished(jobs: JobManager):
    """A job of the app's conversation that has ended, its result to tell."""
    record = jobs.submit(
        tool="run_shell_job", backend="subprocess", spec={"command": "true"}, name="build",
        resume_on_complete=True, call_id="c1", session_id="s", user_id="u",
    )
    deadline = time.monotonic() + 10
    while jobs.get(record.job_id).state not in TERMINAL_STATES:
        assert time.monotonic() < deadline, "the job did not end"
        time.sleep(0.02)
    return record


def app_with(controller, jobs: JobManager) -> BaseCLIApp:
    controller.workflow.job_manager = jobs
    app = BaseCLIApp.__new__(BaseCLIApp)
    app._workflow_controller = controller
    app._turn_lock = asyncio.Lock()
    app._message_processor = MessageProcessor()
    app.session = opening_boxes(autospec_session())
    app._settings = SETTINGS
    app._usage_tracker = None
    app._session_id = "s"
    return app


async def test_a_workflow_that_has_not_started_leaves_the_job_for_a_later_delivery(jobs):
    record = finished(jobs)
    controller = Starting(Workflow(hang=False))
    app = app_with(controller, jobs)

    assert await app.resume_finished_jobs() == 0

    assert jobs.get(record.job_id).resume_state == "pending"
    controller.started = True
    assert await app.resume_finished_jobs() == 1
    assert jobs.get(record.job_id).resume_state == "delivered"


async def test_a_stop_while_its_conversation_is_looked_up_leaves_the_job_pending(jobs):
    record = finished(jobs)
    workflow = Looking()
    app = app_with(ReplayController(workflow), jobs)

    delivery = asyncio.create_task(app.resume_finished_jobs())
    await asyncio.wait_for(workflow.looking.wait(), timeout=5)
    delivery.cancel()
    with pytest.raises(asyncio.CancelledError):
        await delivery

    assert jobs.get(record.job_id).resume_state == "pending"
    assert [c for c in app.session.add_message.call_args_list
            if c == call("system", CANCELLED_LINE)] == [call("system", CANCELLED_LINE)]
    assert [r.job_id for r in jobs.awaiting_resume()] == [record.job_id]


async def test_a_stop_during_the_turn_closes_the_claim_failed(jobs):
    """Once the turn runs, the notice may already be in the conversation: a
    later delivery would tell it twice."""
    record = finished(jobs)
    workflow = Workflow()  # its turn runs until it is stopped
    app = app_with(ReplayController(workflow), jobs)

    delivery = asyncio.create_task(app.resume_finished_jobs())
    await asyncio.wait_for(workflow.streaming.wait(), timeout=5)
    delivery.cancel()
    with pytest.raises(asyncio.CancelledError):
        await delivery

    after = jobs.get(record.job_id)
    assert (after.resume_state, after.resume_error) == ("failed", "resume cancelled")
