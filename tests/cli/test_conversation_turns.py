"""What writes to the open conversation runs in its core (``cli/conversation.py``).

``ConversationCore`` owns the turn lock, which every write to the
conversation and every change of it takes. ``BaseCLIApp`` reads it through
``_turn_lock``; an app built with ``__new__`` (tests, an application's own
construction) has one too. Under it the core runs a turn (``run_turn``) and
``/compact`` (``compact``), on the app's parts as they are (``TurnParts``),
and the app forwards to it: a line typed, and its public ``compact()``.
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
from agentic_cli.cli.message_processor import CANCELLED_LINE, TurnResult, TurnStatus
from agentic_cli.config import BaseSettings
from agentic_cli.workflow.config import AgentConfig

from tests.cli.dialog_fakes import autospec_session


@pytest.fixture(autouse=True)
def quiet_app(monkeypatch):
    """No global logging configuration or warning hook left behind."""
    monkeypatch.setattr(app_module, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "_install_adk_notice_hook", lambda: None)


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

        result = await core.run_turn("Hello.", parts)

        assert result.status is TurnStatus.COMPLETED
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

    async def test_one_open_in_another_process_does_not_run(self, tmp_path):
        other = ConversationLock(tmp_path)
        other.take(place(tmp_path).key("a"))
        core = new_core(tmp_path)
        processor = Processor(core)
        parts = parts_for(processor)

        try:
            assert await core.run_turn("Hello.", parts) is None
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
            return TurnResult(TurnStatus.COMPLETED)

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
            return TurnResult(TurnStatus.COMPLETED)

        monkeypatch.setattr(ConversationCore, "run_turn", run_turn)

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
            return None  # open in another process

        monkeypatch.setattr(ConversationCore, "run_turn", run_turn)

        await app.process_input("Hello.")

        assert app.told == 0
