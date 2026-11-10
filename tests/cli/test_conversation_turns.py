"""What writes to the open conversation runs in its core (``cli/conversation.py``).

``ConversationCore`` owns the turn lock, which every write to the
conversation and every change of it takes. ``BaseCLIApp`` reads it through
``_turn_lock``; an app built with ``__new__`` (tests, an application's own
construction) has one too. Under it the core runs a turn (``run_turn``),
``/compact`` (``compact``) and a finished job's notice (``run_job_resume``,
for each job that waits for the conversation: ``deliver_finished_jobs``,
by the one rule ``waiting_jobs`` names them by), on the app's parts as they
are (``TurnParts``), and the app forwards to it: a line typed, its public
``compact()`` and ``resume_finished_jobs()``, and the line that names the
jobs waiting.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call

import pytest

from agentic_cli.cli import AppInfo
from agentic_cli.cli import app as app_module
from agentic_cli.cli.app import BaseCLIApp
from agentic_cli.cli.conversation import ConversationCore, Place, TurnParts
from agentic_cli.cli.conversation_lock import ConversationLock
from agentic_cli.cli.message_processor import (
    CANCELLED_LINE,
    MessageProcessor,
    TurnResult,
    TurnStatus,
)
from agentic_cli.config import BaseSettings
from agentic_cli.workflow.config import AgentConfig

from tests.cli.dialog_fakes import autospec_session


@pytest.fixture(autouse=True)
def quiet_app(monkeypatch):
    """No global logging configuration or warning hook left behind."""
    monkeypatch.setattr(app_module, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "install_log_filters", lambda: None)


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def build(cls: type[BaseCLIApp] = BaseCLIApp, **settings) -> BaseCLIApp:
    return cls(
        app_info=AppInfo(name="Turns test", version="0"),
        agent_configs=[AgentConfig(name="assistant", prompt="Help.")],
        settings=BaseSettings(session_store="sqlite", **settings),
    )


def place(folder: Path) -> Place:
    return Place(backend="adk", store="sqlite", app_name="app", user="me", locks=folder)


def held_elsewhere(where: Place, session_id: str) -> int | None:
    """Whether another process could open it: the holder's pid, or None (and
    it is let go again)."""
    other = ConversationLock(where.locks)
    holder = other.take(where.key(session_id))
    other.release()
    return holder


def new_core(folder: Path) -> ConversationCore:
    """A conversation this run made: held from its first write."""
    return ConversationCore("a", place(folder), opened_empty=True, deferred=True)


class Processor:
    """Runs a turn or ``/compact`` as the app's processor does, recording
    each with the conversation it named and whether the turn lock was held."""

    def __init__(self, core: ConversationCore, status: TurnStatus = TurnStatus.COMPLETED):
        self.core, self.status = core, status
        self.ran: list[tuple] = []

    async def process(
        self, *, message, workflow_controller, ui, settings, usage_tracker, session_id
    ) -> TurnResult:
        self.ran.append(("turn", message, session_id, self.core.turn_lock.locked()))
        return TurnResult(self.status)

    async def process_compact(
        self, focus, *, workflow_controller, ui, settings, usage_tracker, session_id
    ) -> str:
        self.ran.append(("compact", focus, session_id, self.core.turn_lock.locked()))
        return "summarized"


def parts_for(processor: Processor) -> TurnParts:
    return TurnParts(
        controller=SimpleNamespace(is_ready=True),
        processor=processor,
        ui=autospec_session(),
        settings=SimpleNamespace(default_user="me"),
        tracker=None,
    )


class TestTheTurnLock:
    def test_the_core_owns_it_and_the_app_reads_it(self, home):
        app = build()

        assert isinstance(app._conversation.turn_lock, asyncio.Lock)
        assert app._turn_lock is app._conversation.turn_lock

    def test_an_app_built_with_new_has_one(self):
        app = BaseCLIApp.__new__(BaseCLIApp)

        assert app._turn_lock is app._conversation.turn_lock

    async def test_a_change_of_conversation_waits_for_it(self, home):
        """``/new`` takes the core's lock: it waits for a turn holding it."""
        app = build()

        async with app._conversation.turn_lock:
            new = asyncio.create_task(app.start_new_session())
            await asyncio.sleep(0.05)
            assert not new.done()
        await new


class TestATurn:
    """``ConversationCore.run_turn``: the user's line, as ``BaseCLIApp`` runs it."""

    async def test_it_runs_under_the_lock_in_the_conversation_held_first(self, tmp_path):
        core = new_core(tmp_path)
        processor = Processor(core)
        parts = parts_for(processor)

        assert await core.run_turn("Hello.", parts) is True

        assert processor.ran == [("turn", "Hello.", "a", True)]
        assert held_elsewhere(place(tmp_path), "a") == os.getpid()  # held at its first write
        assert (core.turned, core.written) == (True, True)
        assert parts.ui.add_message.call_args_list == [call("user", "Hello.")]
        assert not core.turn_lock.locked()

    async def test_one_the_workflow_was_not_up_for_stores_nothing(self, tmp_path):
        core = new_core(tmp_path)

        await core.run_turn("Hello.", parts_for(Processor(core, TurnStatus.UNAVAILABLE)))

        assert (core.turned, core.written) == (True, False)
        assert core.fresh

    async def test_a_processor_that_returns_nothing_still_ran_it(self, tmp_path):
        """Whether the turn ran is the core's to say, not the processor's
        result: an app's processor whose ``process()`` returns nothing ran
        the turn all the same."""

        class ReturnsNothing(Processor):
            async def process(self, **kwargs) -> None:
                await super().process(**kwargs)

        core = new_core(tmp_path)

        assert await core.run_turn("Hello.", parts_for(ReturnsNothing(core))) is True
        assert (core.turned, core.written) == (True, True)

    async def test_one_open_in_another_process_does_not_run(self, tmp_path):
        other = ConversationLock(tmp_path)
        other.take(place(tmp_path).key("a"))
        core = new_core(tmp_path)
        processor = Processor(core)
        parts = parts_for(processor)

        try:
            assert await core.run_turn("Hello.", parts) is False
        finally:
            other.release()

        assert processor.ran == [] and not core.turned
        assert parts.ui.add_error.call_args_list == [
            call(f"Conversation 'a' is open in another process ({os.getpid()}).")
        ]

    async def test_a_line_stopped_while_it_waits_is_shown_and_said_cancelled(self, tmp_path):
        core = new_core(tmp_path)
        processor = Processor(core)
        parts = parts_for(processor)

        async with core.turn_lock:
            turn = asyncio.create_task(core.run_turn("Hello.", parts))
            await asyncio.sleep(0.05)
            turn.cancel()
            with pytest.raises(asyncio.CancelledError):
                await turn

        assert processor.ran == []
        assert parts.ui.add_message.call_args_list == [
            call("user", "Hello."), call("system", CANCELLED_LINE),
        ]


class TestCompact:
    """``ConversationCore.compact``: ``/compact`` as a turn."""

    async def test_it_runs_under_the_lock_in_the_conversation_held_first(self, tmp_path):
        core = new_core(tmp_path)
        processor = Processor(core)

        assert await core.compact("the plan", parts_for(processor)) == "summarized"

        assert processor.ran == [("compact", "the plan", "a", True)]
        assert held_elsewhere(place(tmp_path), "a") == os.getpid()
        assert not core.turn_lock.locked()

    async def test_one_open_in_another_process_is_not_summarized(self, tmp_path):
        other = ConversationLock(tmp_path)
        other.take(place(tmp_path).key("a"))
        core = new_core(tmp_path)
        processor = Processor(core)

        try:
            assert await core.compact(None, parts_for(processor)) is None
        finally:
            other.release()

        assert processor.ran == []


class TestTheAppForwards:
    """A line typed and ``compact()`` run in the app's core, on its parts as
    they are; what the app does on its own after a turn calls its public
    methods, so an override of them applies."""

    @staticmethod
    def the_apps_parts(app: BaseCLIApp, parts: TurnParts) -> bool:
        return (
            parts.controller is app._workflow_controller
            and parts.processor is app._message_processor
            and parts.ui is app.session
            and parts.settings is app.settings
            and parts.tracker is app.usage_tracker
        )

    async def test_a_line_typed_is_the_cores_turn(self, home, monkeypatch):
        app = build()
        seen = []

        async def run_turn(core, message, parts):
            seen.append((core, message, parts))
            return True

        monkeypatch.setattr(ConversationCore, "run_turn", run_turn)

        await app.process_input("Hello.")

        ((core, message, parts),) = seen
        assert core is app._conversation and message == "Hello."
        assert self.the_apps_parts(app, parts)

    async def test_compact_is_the_cores(self, home, monkeypatch):
        app = build()
        seen = []

        async def compact(core, focus, parts):
            seen.append((core, focus, parts))
            return "summarized"

        monkeypatch.setattr(ConversationCore, "compact", compact)

        assert await app.compact("the plan") == "summarized"

        ((core, focus, parts),) = seen
        assert core is app._conversation and focus == "the plan"
        assert self.the_apps_parts(app, parts)

    async def test_a_turns_end_tells_jobs_through_the_apps_own_method(self, home, monkeypatch):
        class Telling(BaseCLIApp):
            told = 0

            async def resume_finished_jobs(self) -> int:
                self.told += 1
                return 0

        app = build(Telling, job_auto_resume=True)

        async def run_turn(core, message, parts):
            return True

        monkeypatch.setattr(ConversationCore, "run_turn", run_turn)

        await app.process_input("Hello.")

        assert app.told == 1

    async def test_one_whose_processor_returns_nothing_tells_them_too(self, home):
        """An app's ``MessageProcessor`` whose ``process()`` forgets to
        return its result: the turn ran, and its end tells the jobs."""

        class Telling(BaseCLIApp):
            told = 0

            async def resume_finished_jobs(self) -> int:
                self.told += 1
                return 0

        class ReturnsNothing(MessageProcessor):
            async def process(self, **kwargs) -> None:
                await asyncio.sleep(0)

        app = build(Telling, job_auto_resume=True)
        app._message_processor = ReturnsNothing()

        await app.process_input("Hello.")

        assert app.told == 1

    async def test_a_turn_that_did_not_run_tells_no_jobs(self, home, monkeypatch):
        class Telling(BaseCLIApp):
            told = 0

            async def resume_finished_jobs(self) -> int:
                self.told += 1
                return 0

        app = build(Telling, job_auto_resume=True)

        async def run_turn(core, message, parts):
            return False  # open in another process

        monkeypatch.setattr(ConversationCore, "run_turn", run_turn)

        await app.process_input("Hello.")

        assert app.told == 0


class Jobs:
    """A job manager's resume lifecycle: what it claimed, and how each ended."""

    def __init__(self, *records) -> None:
        self.records = list(records)
        self.claimed: list[str] = []
        self.completed: list[tuple[str, bool, str | None]] = []

    def awaiting_resume(self) -> list:
        return [r for r in self.records if r.job_id not in self.claimed]

    def begin_resume(self, job_id: str) -> bool:
        if job_id in self.claimed:
            return False
        self.claimed.append(job_id)
        return True

    def complete_resume(self, job_id: str, *, delivered: bool, error=None) -> None:
        self.completed.append((job_id, delivered, error))


def finished(job_id: str, session_id: str | None = None, user_id: str | None = "me"):
    return SimpleNamespace(job_id=job_id, name="build", session_id=session_id, user_id=user_id)


class Resumer(Processor):
    """Tells a job as the app's processor does: it claims the job first."""

    def __init__(self, core: ConversationCore, *, hang: bool = False) -> None:
        super().__init__(core)
        self.hang = hang
        self.claimed = asyncio.Event()

    async def process_resume(
        self, *, record, workflow_controller, ui, settings, usage_tracker, claim
    ) -> TurnResult:
        if not await claim():
            return TurnResult(TurnStatus.UNAVAILABLE, error="another delivery has the job")
        self.claimed.set()
        if self.hang:
            await asyncio.Event().wait()
        self.ran.append(("job", record.job_id, self.core.turn_lock.locked()))
        return TurnResult(TurnStatus.COMPLETED)


def job_parts(processor, jobs: Jobs, *, gone=("deleted",), kept_elsewhere=None) -> TurnParts:
    """The app's parts, its workflow ready: ``gone`` conversations can be
    resumed no more (``can_resume``)."""

    async def can_resume(record) -> bool:
        return record.session_id not in gone

    return TurnParts(
        controller=SimpleNamespace(
            is_ready=True, workflow=SimpleNamespace(job_manager=jobs, can_resume=can_resume)
        ),
        processor=processor,
        ui=autospec_session(),
        settings=SimpleNamespace(default_user="me"),
        tracker=None,
        kept_elsewhere=kept_elsewhere,
    )


WAITING = (
    finished("here", "a"),
    finished("orphan", None, None),
    finished("gone", "deleted"),
    finished("elsewhere", "other"),
    finished("someone-elses", "a", "them"),
)


class TestFinishedJobs:
    """``ConversationCore.deliver_finished_jobs``: each job that waits for the
    conversation (``waiting_jobs``: ``cli/waiting_jobs.py``'s rule) told as a
    turn of it (``run_job_resume``), claimed only once it can reach the agent."""

    async def test_it_tells_the_jobs_that_wait_here_each_under_the_lock(self, tmp_path):
        core = new_core(tmp_path)
        processor, jobs = Resumer(core), Jobs(*WAITING)

        assert await core.deliver_finished_jobs(job_parts(processor, jobs)) == 3

        assert processor.ran == [("job", "here", True), ("job", "orphan", True), ("job", "gone", True)]
        assert jobs.completed == [("here", True, None), ("orphan", True, None), ("gone", True, None)]
        assert (core.turned, core.written) == (True, True)  # its own job adds to it
        assert held_elsewhere(place(tmp_path), "a") == os.getpid()  # held to write its notice
        assert not core.turn_lock.locked()

    async def test_the_jobs_waiting_are_named_by_the_same_rule(self, tmp_path):
        core = new_core(tmp_path)

        waiting = await core.waiting_jobs(job_parts(Resumer(core), Jobs(*WAITING)))

        assert [record.job_id for record in waiting] == ["here", "orphan", "gone"]

    async def test_one_another_backend_keeps_waits_for_it(self, tmp_path):
        core = new_core(tmp_path)
        processor, jobs = Resumer(core), Jobs(finished("gone", "deleted"))

        async def kept_elsewhere(record) -> bool:
            return True

        parts = job_parts(processor, jobs, kept_elsewhere=kept_elsewhere)

        assert await core.deliver_finished_jobs(parts) == 0
        assert await core.waiting_jobs(parts) == []
        assert jobs.claimed == []

    async def test_nothing_is_told_before_the_workflow_runs(self, tmp_path):
        core = new_core(tmp_path)
        jobs = Jobs(*WAITING)
        parts = job_parts(Resumer(core), jobs)
        parts.controller.is_ready = False

        assert await core.deliver_finished_jobs(parts) == 0
        assert jobs.claimed == []

    async def test_a_cancel_once_it_is_claimed_records_it_failed(self, tmp_path):
        core = new_core(tmp_path)
        processor, jobs = Resumer(core, hang=True), Jobs(finished("here", "a"))
        parts = job_parts(processor, jobs)

        notice = asyncio.create_task(core.run_job_resume(jobs, jobs.records[0], parts))
        await asyncio.wait_for(processor.claimed.wait(), timeout=2)
        notice.cancel()
        with pytest.raises(asyncio.CancelledError):
            await notice

        assert jobs.completed == [("here", False, "resume cancelled")]
        assert not core.turn_lock.locked()

    async def test_a_failure_before_the_claim_is_said_once_when_told_on_its_own(self, tmp_path):
        from agentic_cli.cli.conversation import telling_on_its_own

        class Unreachable(Resumer):
            async def process_resume(self, **kwargs) -> TurnResult:
                raise ConnectionError("session store unreachable")

        core = new_core(tmp_path)
        jobs = Jobs(finished("orphan", None, None))
        parts = job_parts(Unreachable(core), jobs)

        with telling_on_its_own():  # a turn's end, twice
            assert await core.deliver_finished_jobs(parts) == 1
            assert await core.deliver_finished_jobs(parts) == 1
        assert await core.deliver_finished_jobs(parts) == 1  # /jobs resume: asked, told

        failure = "Background job 'build' could not be resumed now: session store unreachable"
        assert parts.ui.add_error.call_args_list == [call(failure), call(failure)]
        assert jobs.claimed == []


class TestTheAppForwardsItsJobs:
    async def test_resume_finished_jobs_is_the_cores(self, home, monkeypatch):
        app = build()
        seen = []

        async def deliver_finished_jobs(core, parts):
            seen.append((core, parts))
            return 2

        monkeypatch.setattr(ConversationCore, "deliver_finished_jobs", deliver_finished_jobs)

        assert await app.resume_finished_jobs() == 2

        ((core, parts),) = seen
        assert core is app._conversation
        assert TestTheAppForwards.the_apps_parts(app, parts)
        assert parts.kept_elsewhere == app._kept_elsewhere  # the other backends' look

    async def test_the_line_naming_the_jobs_waiting_counts_the_cores(self, home, monkeypatch):
        app = build()
        app.session = autospec_session()
        app._workflow_controller = SimpleNamespace(
            workflow=SimpleNamespace(job_manager=Jobs()), is_ready=True
        )

        async def waiting_jobs(core, parts):
            return [finished("here", core.session_id), finished("orphan")]

        monkeypatch.setattr(ConversationCore, "waiting_jobs", waiting_jobs)

        await app._name_its_waiting_jobs()

        assert app.session.add_message.call_args_list == [call(
            "system",
            "2 finished jobs wait for this conversation: /jobs resume tells them now.",
        )]
