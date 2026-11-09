"""A finished job whose conversation could not be looked up is said once, plainly.

When ``can_resume`` fails (the session store unreachable, say), the job's
delivery stops before it is claimed: nothing reached the agent, and the job
waits for a later delivery. The user is told ("Background job 'build' could
not be resumed now: …"). ``/jobs resume`` then also said "No finished
background jobs to resume.", about the job it had just named, and every turn
end with ``job_auto_resume`` said the same failure again. ``/jobs resume``
now says the failure each time it is asked, and nothing contradicting it;
the deliveries the CLI makes on its own (a turn's end, a conversation
opened) say it once per job and conversation.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agentic_cli.cli.app import BaseCLIApp
from agentic_cli.cli.builtin_commands import JobsCommand
from agentic_cli.cli.message_processor import MessageProcessor
from agentic_cli.tools.jobs import JobRecord, JobState
from tests.cli.dialog_fakes import autospec_session
from tests.event_replay import ReplayController
from tests.test_message_processor_cancel import Workflow

FAILURE = "Background job 'build' could not be resumed now: session store unreachable"


class Jobs:
    """Holds one finished job with no conversation: it is told wherever the user is."""

    def __init__(self) -> None:
        self.record = JobRecord(
            job_id="j1", tool="t", backend="subprocess", name="build",
            state=JobState.SUCCEEDED, resume_on_complete=True,
        )

    def awaiting_resume(self):
        return [self.record]

    def begin_resume(self, job_id):  # pragma: no cover - never reached
        raise AssertionError("claimed though its conversation could not be looked up")


class Unreachable(Workflow):
    """Its session store cannot be reached."""

    def __init__(self) -> None:
        super().__init__(hang=False)

    async def can_resume(self, record) -> bool:
        raise ConnectionError("session store unreachable")


def app_with() -> BaseCLIApp:
    workflow = Unreachable()
    workflow.job_manager = Jobs()
    app = BaseCLIApp.__new__(BaseCLIApp)
    app._workflow_controller = ReplayController(workflow)
    app._turn_lock = asyncio.Lock()
    app._message_processor = MessageProcessor()
    app.session = autospec_session()
    app._settings = SimpleNamespace(
        default_user="u", job_auto_resume=True, verbose_thinking=False,
        dialog_placement="box",
    )
    app._usage_tracker = None
    app._session_id = "s"
    return app


def errors(app) -> list[str]:
    return [c.args[0] for c in app.session.add_error.call_args_list]


def messages(app) -> list[str]:
    return [c.args[1] for c in app.session.add_message.call_args_list]


async def test_jobs_resume_says_the_failure_and_nothing_that_contradicts_it():
    app = app_with()

    await JobsCommand().execute("resume", app)
    await JobsCommand().execute("resume", app)

    assert errors(app) == [FAILURE, FAILURE]  # asked twice, told twice
    assert "No finished background jobs to resume." not in messages(app)


async def test_a_turns_end_says_it_once_per_conversation():
    app = app_with()

    await app._handle_message("Hello.")
    await app._handle_message("And again.")
    assert errors(app) == [FAILURE]

    app._session_id = "t"  # another conversation: said there once too
    await app._handle_message("Hello there.")
    await app._handle_message("And again there.")
    assert errors(app) == [FAILURE, FAILURE]
