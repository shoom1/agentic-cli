"""The conversation the turns write to is the one held and loaded.

A conversation lives in one CLI process at a time (``cli/conversation_lock.py``),
so the lock must follow the conversation the app is in, however it changes:
``/new``, ``/resume``, the conversation a run starts in (``--continue``,
``--resume``, a new one), an orchestrator switch in ``/settings``, and the
``session_id`` setter. Each goes through one protocol
(``ConversationCore.switch``): the target's lock is taken (one another
process holds is refused), the memories of the conversation left are
proposed, the workflow loads the target, then the switch is made (the lock
kept, the one before let go, the task list and the usage counters reset)
or abandoned, and the status line is drawn. A Ctrl+C at any step leaves the
app in exactly one conversation, holding exactly its lock.

The place a conversation is held at (its backend and store) is recorded
when the lock is taken: a ``/settings`` change of ``session_store`` takes
effect at the next start, and the lock keeps the store the workflow runs.

The app is a real ``BaseCLIApp`` whose session is a fake and whose
controller holds a fake workflow (``tests/cli/test_resume_session.py``).
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("google.adk")

from agentic_cli.cli.conversation_lock import ConversationLock  # noqa: E402
from agentic_cli.config import BaseSettings  # noqa: E402
from tests.cli.test_compact_and_new import (  # noqa: E402,F401
    app_on,
    fake_settings,
    project,
    quiet_app,
    said,
)
from agentic_cli.cli.session_args import SessionStart  # noqa: E402
from tests.cli.console_app import ScriptedManager  # noqa: E402
from tests.cli.test_resume_at_startup import (  # noqa: E402,F401
    Session,
    app_with,
    manager,
    resumed,
    store,
)
from tests.cli.test_resume_session import Workflow, conversation  # noqa: E402


@pytest.fixture
def home(project, monkeypatch):
    """HOME is the test's: the app's conversation locks live in its workspace."""
    monkeypatch.setenv("HOME", str(project))
    return project


def open_elsewhere(app, key: tuple) -> ConversationLock:
    """Another process that has the conversation ``key`` names open."""
    other = ConversationLock(app.settings.sessions_dir / "locks")
    assert other.take(key) is None
    return other


def free(app, key: tuple) -> bool:
    """Whether another process could open the conversation ``key`` names now."""
    other = ConversationLock(app.settings.sessions_dir / "locks")
    taken = other.take(key) is None
    other.release()
    return taken


class TestASessionStoreChange:
    """``session_store`` takes effect at the next start: until then the
    workflow keeps its store, and so does the lock."""

    async def test_the_lock_keeps_the_store_the_workflow_runs(self, home):
        workflow = Workflow(conversation("abc", "Plan the trip", 2))
        app = app_on(workflow, BaseSettings(session_store="sqlite"))
        assert app._hold(app.session_id) is None
        sqlite_key = app._conversation_key("abc")

        assert await app.apply_settings({"session_store": "memory"})

        assert app._conversation_key("abc") == sqlite_key
        assert not free(app, app._conversation_key(app.session_id))

    async def test_one_open_elsewhere_is_still_refused(self, home):
        workflow = Workflow(conversation("abc", "Plan the trip", 2))
        app = app_on(workflow, BaseSettings(session_store="sqlite"))
        other = open_elsewhere(app, app._conversation_key("abc"))
        before = app.session_id

        assert await app.apply_settings({"session_store": "memory"})
        resumed = await app.resume_session("abc")
        other.release()

        assert resumed is False and app.session_id == before
        assert said(app.session, "add_error") == [
            f"Could not open 'abc': it is open in another process ({os.getpid()})."
        ]


class TestACtrlCWhileTheTargetIsLookedUpAgain:
    """The workflow took the target and found it empty, and the switch asks
    whether it is still listed (deleted by another process as it opened, or
    a first turn killed before its input was stored): a Ctrl+C there left the
    workflow on the target while the app stayed where it was, so ``/status``
    named the wrong conversation."""

    async def test_the_workflow_is_given_back_the_conversation_the_app_stays_in(
        self, project
    ):
        workflow = Workflow(conversation("abc", "Plan the trip", 2))
        app = app_on(workflow, fake_settings())
        old = app.session_id
        load, info = workflow.load_session, workflow.session_info
        looking, lookups = asyncio.Event(), []

        async def deleted_first(session_id, user_id=None):
            workflow.stored.pop("abc", None)
            return await load(session_id, user_id)

        async def session_info(session_id, *, user_id=None):
            lookups.append(session_id)
            if len(lookups) == 2:  # asked again, after the load
                looking.set()
                await asyncio.Event().wait()
            return await info(session_id, user_id=user_id)

        workflow.load_session, workflow.session_info = deleted_first, session_info
        resuming = asyncio.create_task(app.resume_session("abc"))
        await asyncio.wait_for(looking.wait(), 5)
        resuming.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(resuming, 5)

        assert app.session_id == old
        assert [c for c in workflow.calls if c.startswith("load_session")] == [
            "load_session abc", f"load_session {old}",
        ]


class TestTheConversationARunStartsIn:
    """Held from the start, whatever the start does: the turns write to it."""

    @staticmethod
    def looked_at(app_ref: list, *session_ids: str):
        """An ``until`` for the session: once the prompt is up (the startup's
        open done when ``resumed``), who holds each conversation named."""
        seen: list = []

        def looked(session) -> bool:
            if not seen:
                app = app_ref[0]
                ids = session_ids or (app.session_id,)
                seen.append([
                    ConversationLock(app.settings.sessions_dir / "locks").take(
                        app._conversation_key(session_id)
                    )
                    for session_id in ids
                ])
            return True

        return looked, seen

    @pytest.mark.parametrize("start", [SessionStart(), SessionStart("resume", "abc")])
    async def test_it_is_held_when_the_workflow_did_not_start(
        self, manager, tmp_path, monkeypatch, start
    ):
        """The user fixes the start with ``/settings`` and goes on in it: a
        second terminal's ``--continue`` must not open it meanwhile."""
        (tmp_path / "init-fails").touch()
        (tmp_path / "ready").touch()  # a later retry would start
        monkeypatch.setattr(ScriptedManager, "initializations", 0)
        store(manager, "abc", "Plan the trip", "Sure.", hours=2)
        app_ref: list = []
        looked, seen = self.looked_at(app_ref)
        app = app_with(Session(until=looked), start)
        app_ref.append(app)

        await app.run()

        assert app.session_id != "abc"  # it starts in a new one
        assert seen == [[os.getpid()]]

    async def test_it_is_held_while_the_one_asked_for_opens_then_let_go(self, manager):
        store(manager, "abc", "Plan the trip", "Sure.", hours=2)
        app_ref: list = []
        looked, seen = self.looked_at(app_ref, "abc")
        session = Session(until=lambda s: resumed(s) and looked(s))
        app = app_with(session, SessionStart("resume", "abc"))
        fresh = app.session_id
        app_ref.append(app)

        await app.run()

        assert app.session_id == "abc" and seen == [[os.getpid()]]
        assert free(app, app._conversation_key(fresh))


class TestATurnTypedBeforeTheStartupOpen:
    """``--continue`` and ``--resume`` open their conversation once the prompt
    is up; a line typed or pasted ahead (keys pressed while the workflow
    started) went to the run's new conversation, while the lock held the one
    asked for. It now waits for the open, and runs there."""

    async def test_it_waits_and_runs_in_the_conversation_opened(self, project, monkeypatch):
        workflow = Workflow(conversation("T", "Earlier work", 2))
        app = app_on(workflow, fake_settings())
        app.session.app = SimpleNamespace(is_running=False)  # the prompt is not up yet

        async def no_proposals():
            return None

        monkeypatch.setattr(app, "_name_waiting_proposals", no_proposals)
        app._pending_resume = "T"  # what --resume T / --continue chose
        fresh = app.session_id

        opening = asyncio.create_task(app._open_on_startup())
        await asyncio.sleep(0.05)
        typed = asyncio.create_task(app._handle_message("typed ahead"))
        await asyncio.sleep(0.05)
        app.session.app.is_running = True
        await asyncio.wait_for(opening, 5)
        await asyncio.wait_for(typed, 5)

        assert app.session_id == "T" != fresh
        assert workflow.turns == ["T"]


class TestAnOrchestratorSwitch:
    """``/settings`` swaps the backend: the conversation goes on there, from
    empty, under the same id, through the same switch. Its lock moved to the
    new backend's key (it stayed on the old one's, so another process could
    open the conversation this one writes to); the old backend's task list
    and the memories it holds stay behind (the next turn redrew its tasks in
    the new, empty conversation, and the exit proposed memories from it)."""

    @staticmethod
    def swapping(app_settings):
        workflow = Workflow()
        workflow.backend_type = "adk"
        app = app_on(workflow, app_settings)
        replacement = Workflow()
        replacement.backend_type = "deepagents"
        replacement.initialize_services = AsyncMock()
        return app, replacement

    @staticmethod
    async def swap(app, replacement) -> None:
        with patch(
            "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings",
            return_value=replacement,
        ):
            assert await app.apply_settings({"orchestrator": "deepagents"})
        assert app._workflow_controller.workflow is replacement

    async def test_the_lock_follows_it_to_the_new_backend(self, home):
        app, replacement = self.swapping(BaseSettings(session_store="sqlite"))
        assert app._hold(app.session_id) is None  # as the start does
        old_key = app._conversation_key(app.session_id)

        await self.swap(app, replacement)

        key = app._conversation_key(app.session_id)
        assert key[0] == "deepagents" and key[1:] == old_key[1:]
        assert not free(app, key) and free(app, old_key)
        assert replacement.calls == [f"load_session {app.session_id}"]

    async def test_the_old_backends_tasks_and_memories_stay_behind(self, home):
        app, replacement = self.swapping(fake_settings(auto_extract_session_facts=True))
        app._message_processor._last_task_progress = "- [x] the old backend's task"
        app._turned = True  # a turn ran on the old backend

        await self.swap(app, replacement)
        await app._extract_session_facts_on_exit()

        assert app._message_processor._last_task_progress is None
        assert app._turned is False and replacement.ended == []

    async def test_one_open_on_the_new_backend_elsewhere_starts_a_new_one_there(self, home):
        app, replacement = self.swapping(BaseSettings(session_store="sqlite"))
        old = app.session_id
        assert app._hold(old) is None
        there = ("deepagents", *app._conversation_key(old)[1:])
        other = open_elsewhere(app, there)

        await self.swap(app, replacement)
        other.release()

        new = app.session_id
        assert new != old and app._conversation_key(new)[0] == "deepagents"
        assert not free(app, app._conversation_key(new))
        assert said(app.session, "add_message")[-1] == (
            f"This conversation is open in another process ({os.getpid()}) on deepagents: "
            f"a new one starts here, {new}. Its history stays with the previous "
            "backend: switch back to continue it."
        )


class TestTheSessionIdSetter:
    """``app.session_id = <id>`` moved the turns to another conversation with
    no lock, no load and no reset: the next turn wrote to a conversation
    another process could have open, under the old one's task list. A
    setter cannot run the switch (it is async): it does what can be done at
    once, holding the conversation and starting the task list and the usage
    counters over, and warns that ``resume_session`` is the way."""

    async def test_it_holds_the_conversation_and_starts_over_with_a_warning(self, home):
        workflow = Workflow()
        app = app_on(workflow, BaseSettings(session_store="sqlite"))
        old = app.session_id
        assert app._hold(old) is None
        app._message_processor._last_task_progress = "- [ ] a task"
        app._turned = True

        with pytest.warns(DeprecationWarning, match="resume_session"):
            app.session_id = "theirs"

        assert app.session_id == "theirs"
        assert not free(app, app._conversation_key("theirs"))
        assert free(app, app._conversation_key(old))
        assert app._message_processor._last_task_progress is None
        assert app._turned is False

    async def test_one_open_elsewhere_is_refused_and_nothing_changes(self, home):
        app = app_on(Workflow(), BaseSettings(session_store="sqlite"))
        old = app.session_id
        assert app._hold(old) is None
        other = open_elsewhere(app, app._conversation_key("theirs"))

        with pytest.warns(DeprecationWarning), pytest.raises(
            RuntimeError, match=rf"'theirs' is open in another process \({os.getpid()}\)"
        ):
            app.session_id = "theirs"
        other.release()

        assert app.session_id == old and not free(app, app._conversation_key(old))

    async def test_the_next_turn_runs_in_it(self, project):
        workflow = Workflow()
        app = app_on(workflow, fake_settings())

        with pytest.warns(DeprecationWarning):
            app.session_id = "theirs"
        await app.process_input("Hello.")

        assert workflow.turns == ["theirs"]
