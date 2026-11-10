"""The conversation the turns write to is the one held and loaded.

A conversation lives in one CLI process at a time (``cli/conversation_lock.py``),
so the lock must follow the conversation the app is in, however it changes:
``/new`` (on any backend: a conversation never changes backend, so another
backend is a new conversation's), ``/resume``, the conversation a run starts
in (``--continue``, ``--resume``, a new one), a ``/settings`` change of
``orchestrator`` or ``session_store`` while the workflow is down, and the
``session_id`` setter. Each goes through one protocol
(``ConversationCore.switch``): the target's lock is taken (one another
process holds is refused), the memories of the conversation left are
proposed, the workflow loads the target, then the switch is made (the lock
kept, the one before let go, the task list and the usage counters reset)
or abandoned, and the status line is drawn. A Ctrl+C at any step leaves the
app in exactly one conversation, holding exactly its lock.

The place a conversation is held at (its backend and store) is the one the
workflow runs it in: recorded as the app starts, and for a new conversation
on another backend, and moved by a ``/settings`` change of the orchestrator
or ``session_store`` while the workflow is down (its next start builds on
the new values). A change of ``session_store`` while the workflow runs takes
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
    """The one the app was given is held from the start, whatever the start
    does; the run's own new one from its first write (``TestARunThatWritesNothing``)."""

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
    async def test_its_own_is_held_from_its_first_write_when_the_workflow_did_not_start(
        self, manager, tmp_path, monkeypatch, start
    ):
        """The user fixes the start with ``/settings`` and goes on in it. No
        store lists it before its first turn, so a second terminal's
        ``--continue`` cannot open it meanwhile: nothing holds it yet, and
        its first turn does (``TestARunThatWritesNothing``)."""
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
        assert seen == [[None]]  # free: nothing holds it before its first write

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

    async def test_it_is_shown_under_the_conversation_it_runs_in(self, project, monkeypatch):
        """Echoed when it runs: after "Resumed:" and the replay, not above them."""
        workflow = Workflow(conversation("T", "Earlier work", 2))
        app = app_on(workflow, fake_settings())
        app.session.app = SimpleNamespace(is_running=False)

        async def no_proposals():
            return None

        monkeypatch.setattr(app, "_name_waiting_proposals", no_proposals)
        app._pending_resume = "T"

        opening = asyncio.create_task(app._open_on_startup())
        await asyncio.sleep(0.05)
        typed = asyncio.create_task(app._handle_message("typed ahead"))
        await asyncio.sleep(0.05)
        app.session.app.is_running = True
        await asyncio.wait_for(opening, 5)
        await asyncio.wait_for(typed, 5)

        shown = [args[1] for name, args, _ in app.session.mock_calls if name == "add_message"]
        assert shown.index("typed ahead") > max(
            shown.index(text) for text in shown if text.startswith(("Resumed:", "T message"))
        )


class TestATypedAheadLineCancelledWhileItWaits:
    """Ctrl+C on a line whose turn waits for the startup open: it never ran,
    and it is shown and said cancelled, not dropped without a word (the pty
    test is ``test_console_turns``'s)."""

    async def test_it_is_shown_and_said_cancelled(self, project, monkeypatch):
        from agentic_cli.cli.message_processor import CANCELLED_LINE

        workflow = Workflow(conversation("T", "Earlier work", 2))
        app = app_on(workflow, fake_settings())
        gate = asyncio.Event()

        async def opening():
            async with app._turn_lock:
                await gate.wait()

        holder = asyncio.create_task(opening())
        await asyncio.sleep(0)
        typed = asyncio.create_task(app._handle_message("typed ahead"))
        await asyncio.sleep(0.05)
        typed.cancel()
        with pytest.raises(asyncio.CancelledError):
            await typed
        gate.set()
        await holder

        assert said(app.session, "add_message") == ["typed ahead", CANCELLED_LINE]
        assert workflow.turns == [] and not app._turn_lock.locked()


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

    async def test_none_keeps_the_conversation_held(self, home):
        """The turns then name no conversation, and the workflow writes to
        the one it last loaded (``session_ref``'s fallback): the one held."""
        app = app_on(Workflow(), BaseSettings(session_store="sqlite"))
        old = app.session_id
        assert app._hold(old) is None

        with pytest.warns(DeprecationWarning):
            app.session_id = None

        assert app.session_id is None
        assert not free(app, app._conversation_key(old))

    async def test_none_holds_the_conversation_the_workflow_writes_to(self, home):
        """The turns then name none, and the workflow writes to the one it
        loaded last (``session_ref``'s fallback): that one is held, whatever
        an earlier setter left held."""
        workflow = Workflow()
        app = app_on(workflow, BaseSettings(session_store="sqlite"))
        first = app.session_id
        assert app._hold(first) is None
        workflow.session_id = first  # the conversation the workflow loaded
        with pytest.warns(DeprecationWarning):
            app.session_id = "theirs"  # the workflow is not told

        with pytest.warns(DeprecationWarning):
            app.session_id = None

        assert not free(app, app._conversation_key(first))
        assert free(app, app._conversation_key("theirs"))

    async def test_the_next_turn_runs_in_it(self, project):
        workflow = Workflow()
        app = app_on(workflow, fake_settings())

        with pytest.warns(DeprecationWarning):
            app.session_id = "theirs"
        await app.process_input("Hello.")

        assert workflow.turns == ["theirs"]


class TestAPlaceChangeWhileTheWorkflowIsDown:
    """The first start fails (``orchestrator=deepagents`` without its extra,
    say), and ``/settings`` changes the orchestrator or the session store,
    as the warning invites. The workflow is not running, so nothing is
    rebuilt: the next message's start builds a manager on the orchestrator
    the settings name then, on the run's store. The store is fixed for the
    run (a change of it moves nothing, and no lock), so only an orchestrator
    change moves the conversation, and only one with nothing stored
    (``fresh``): it moves with the change, through the switch every change of
    conversation makes. Its own new one is held from its first write, so
    there is no hold to move; one a line typed while the workflow was down
    held (``typed_while_down``) is held where the turns will write."""

    class Manager(Workflow):
        """A manager of one backend; ``fails`` makes its start fail."""

        def __init__(self, backend: str, fails: bool) -> None:
            super().__init__()
            self.backend_type = backend
            self.fails = fails
            self.is_initialized = False

        async def initialize_services(self) -> None:
            if self.fails:
                raise RuntimeError(f"{self.backend_type} cannot start (its extra is missing)")
            self.is_initialized = True

        async def cleanup(self) -> None:
            pass

    @classmethod
    def builds(cls, monkeypatch, fails) -> list:
        """The managers the controller builds, each failing when ``fails(n,
        backend)`` says so (n: how many were built before)."""
        built: list = []

        def create(agent_configs, settings, app_name=None, model=None, orchestrator=None,
                   session_store=None):
            backend = orchestrator or getattr(settings.orchestrator, "value", settings.orchestrator)
            manager = cls.Manager(backend, fails(len(built), backend))
            manager.session_store = session_store  # the store it was built for
            built.append(manager)
            return manager

        monkeypatch.setattr(
            "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings", create
        )
        return built

    @staticmethod
    async def started(app_settings):
        """An app whose start ran, as ``run()`` runs it, and failed."""
        from agentic_cli.cli import AppInfo
        from agentic_cli.cli.app import BaseCLIApp
        from agentic_cli.cli.conversation import Place
        from agentic_cli.workflow.config import AgentConfig
        from tests.cli.dialog_fakes import autospec_session, opening_boxes

        app = BaseCLIApp(
            app_info=AppInfo(name="Place test", version="0"),
            agent_configs=[AgentConfig(name="solo", prompt="Help.")],
            settings=app_settings,
        )
        app.session = opening_boxes(autospec_session())
        app._conversation.place = Place.of(app_settings)  # as run() records it
        await app._workflow_controller.start_background_init()
        assert await app._adopt_session_on_startup()
        assert not app._workflow_controller.is_ready
        return app

    @staticmethod
    def key_where(app, backend: str, store: str) -> tuple:
        return (backend, store, *app._conversation_key(app.session_id)[2:])

    @staticmethod
    async def typed_while_down(app) -> None:
        """A line typed while the workflow is down: its turn retries the
        start, which fails, so nothing is written; it held the conversation
        first, since the start it tried might have written it."""
        key = app._conversation_key(app.session_id)
        await app.process_input("hello")
        assert not app._workflow_controller.is_ready
        assert app._conversation.fresh and not free(app, key)

    async def test_an_orchestrator_change_moves_the_lock_where_the_turns_run(
        self, home, monkeypatch
    ):
        built = self.builds(monkeypatch, lambda n, backend: backend == "deepagents")
        app = await self.started(BaseSettings(session_store="sqlite", orchestrator="deepagents"))

        assert await app.apply_settings({"orchestrator": "adk"})
        await app.process_input("hello")
        await app._workflow_controller.close()

        assert [m.backend_type for m in built] == ["deepagents", "adk"]
        assert built[-1].turns == [app.session_id]
        assert not free(app, self.key_where(app, "adk", "sqlite"))
        assert free(app, self.key_where(app, "deepagents", "sqlite"))

    async def test_a_store_change_takes_effect_at_the_next_app_start(self, home, monkeypatch):
        """The session store is fixed for the run: the retry builds on the
        store the run started with, the turns run there, and the lock stays."""
        built = self.builds(monkeypatch, lambda n, backend: n == 0)
        app = await self.started(BaseSettings(session_store="sqlite", orchestrator="adk"))
        key = app._conversation_key(app.session_id)

        assert await app.apply_settings({"session_store": "memory"})
        await app.process_input("hello")
        await app._workflow_controller.close()

        assert said(app.session, "add_success") == [
            "Session Store: memory (takes effect at the next start)"
        ]
        assert [m.session_store for m in built] == ["sqlite", "sqlite"]
        assert built[-1].turns == [app.session_id]
        assert app._conversation_key(app.session_id) == key and not free(app, key)

    async def test_one_open_there_elsewhere_refuses_the_change(self, home, monkeypatch):
        self.builds(monkeypatch, lambda n, backend: backend == "deepagents")
        app = await self.started(BaseSettings(session_store="sqlite", orchestrator="deepagents"))
        await self.typed_while_down(app)
        other = open_elsewhere(app, self.key_where(app, "adk", "sqlite"))

        applied = await app.apply_settings({"orchestrator": "adk"})
        other.release()
        await app._workflow_controller.close()

        assert applied is False
        assert app.settings.orchestrator.value == "deepagents"
        assert said(app.session, "add_error")[-1] == (  # after the start's own failure
            f"Could not set Orchestrator: conversation '{app.session_id}' is open in "
            f"another process ({os.getpid()}) on adk (sqlite). Settings were not changed."
        )
        assert not free(app, self.key_where(app, "deepagents", "sqlite"))

    async def test_its_own_unwritten_one_moves_and_its_first_turn_holds_it(
        self, home, monkeypatch
    ):
        """Nothing holds the run's own new one yet: the change moves it, and
        its first turn holds it where it runs (refused here: another process
        has that id open there, which only a collision of ids could do)."""
        built = self.builds(monkeypatch, lambda n, backend: backend == "deepagents")
        app = await self.started(BaseSettings(session_store="sqlite", orchestrator="deepagents"))
        other = open_elsewhere(app, self.key_where(app, "adk", "sqlite"))

        applied = await app.apply_settings({"orchestrator": "adk"})
        await app.process_input("hello")
        other.release()
        await app._workflow_controller.close()

        assert applied and app._conversation_key(app.session_id)[0] == "adk"
        assert built[-1].turns == []
        assert said(app.session, "add_error")[-1] == (
            f"Conversation '{app.session_id}' is open in another process ({os.getpid()})."
        )

    async def test_a_manager_kept_to_revive_keeps_the_lock_on_its_backend(
        self, home, monkeypatch
    ):
        """A failed reinitialization keeps its manager, and the next start
        revives it on the backend and the store it had: the turns go on
        there, and so does the lock (the change waits for /new)."""
        built = self.builds(monkeypatch, lambda n, backend: False)
        app = app_on(self.Manager("adk", fails=False), BaseSettings(session_store="sqlite"))
        kept = app._workflow_controller._workflow
        assert app._hold(app.session_id) is None
        kept.is_initialized = False  # a failed reinitialization left it down

        assert await app.apply_settings({"orchestrator": "deepagents"})
        await app.process_input("hello")

        assert built == [] and kept.turns == [app.session_id]
        assert not free(app, self.key_where(app, "adk", "sqlite"))


class TestANewConversationOnAnotherBackend:
    """``/new [backend]``: a conversation belongs to one backend for its whole
    life, so another backend is a new conversation's. ``/new deepagents``
    starts one there (the setting stays; nothing is saved); plain ``/new``
    starts one on the settings' orchestrator, the backend of new
    conversations. An unknown backend, or one whose extra is not installed,
    is refused, and nothing changes."""

    Manager = TestAPlaceChangeWhileTheWorkflowIsDown.Manager

    @classmethod
    def on_adk(cls, monkeypatch, app_settings=None):
        """An app running ADK; the controller builds any other backend's fake."""
        built = TestAPlaceChangeWhileTheWorkflowIsDown.builds(monkeypatch, lambda n, b: False)
        adk = cls.Manager("adk", fails=False)
        adk.is_initialized = True
        adk.session_store = "sqlite"  # the store it was built for
        app = app_on(adk, app_settings or BaseSettings(session_store="sqlite"))
        assert app._hold(app.session_id) is None  # as the start does
        return app, adk, built

    @staticmethod
    def running(app) -> str:
        return app._workflow_controller.workflow.backend_type

    async def test_it_starts_one_there_and_lets_the_old_one_go(self, home, monkeypatch):
        app, adk, built = self.on_adk(monkeypatch)
        old, old_key = app.session_id, app._conversation_key(app.session_id)
        app._message_processor._last_task_progress = "- [ ] an adk task"
        app.usage_tracker.record({"prompt_tokens": 100, "completion_tokens": 5})

        await app.process_input("/new deepagents")

        new = app.session_id
        assert new != old and self.running(app) == "deepagents"
        assert [m.backend_type for m in built] == ["deepagents"]
        assert built[0].calls == [f"load_session {new}"]
        assert app._conversation_key(new)[0] == "deepagents"
        assert free(app, app._conversation_key(new))  # held from its first write
        assert free(app, old_key)
        adk.cleanup  # released by the controller
        assert app._message_processor._last_task_progress is None
        assert app.usage_tracker.invocation_count == 0
        assert said(app.session, "add_message")[-1] == f"New conversation {new} on deepagents."
        assert app.settings.orchestrator.value == "adk"  # the setting stays

    async def test_it_keeps_the_store_the_run_started_with(self, home, monkeypatch):
        """A ``session_store`` change takes effect at the next app start, not
        at a new conversation on another backend."""
        app, _, built = self.on_adk(monkeypatch)
        assert await app.apply_settings({"session_store": "memory"})

        await app.process_input("/new deepagents")

        assert self.running(app) == "deepagents" and built[0].session_store == "sqlite"
        assert app._conversation.place.store == "sqlite"
        await app.process_input("Hello.")  # its first write holds it there
        assert not free(app, ("deepagents", "sqlite", *app._conversation_key(app.session_id)[2:]))

    async def test_a_model_switch_then_keeps_that_backend(self, home, monkeypatch):
        app, _, built = self.on_adk(monkeypatch)
        await app.process_input("/new deepagents")
        conversation = app.session_id

        assert await app.apply_settings({"model": "google:gemini-2.5-pro"})

        assert self.running(app) == "deepagents" and app._workflow_controller.backend == "deepagents"
        assert built[0].reinitialize.await_count == 1 and len(built) == 1
        assert app.session_id == conversation

    async def test_plain_new_runs_on_the_settings_orchestrator(self, home, monkeypatch):
        app, _, built = self.on_adk(monkeypatch)
        await app.process_input("/new deepagents")

        await app.process_input("/new")

        assert self.running(app) == "adk"
        assert [m.backend_type for m in built] == ["deepagents", "adk"]
        assert said(app.session, "add_message")[-1] == f"New conversation {app.session_id} on adk."

    async def test_an_unknown_backend_is_refused(self, home, monkeypatch):
        app, adk, built = self.on_adk(monkeypatch)
        old = app.session_id

        await app.process_input("/new nope")

        assert app.session_id == old and built == [] and self.running(app) == "adk"
        assert said(app.session, "add_error") == [
            "Unknown backend 'nope': adk, deepagents or langgraph."
        ]

    async def test_one_whose_extra_is_missing_is_refused(self, home, monkeypatch):
        import sys

        app, adk, built = self.on_adk(monkeypatch)
        old = app.session_id
        monkeypatch.setitem(sys.modules, "agentic_cli.workflow.deepagents", None)

        await app.process_input("/new deepagents")

        assert app.session_id == old and built == [] and self.running(app) == "adk"
        assert said(app.session, "add_error") == [
            "The deepagents orchestrator needs the 'deepagents' extra. "
            "Install with: pip install 'agentic-cli[deepagents]'"
        ]

    async def test_langgraph_says_it_is_deprecated(self, home, monkeypatch):
        from agentic_cli.cli.app import _LANGGRAPH_DEPRECATION

        app, _, _ = self.on_adk(monkeypatch)

        await app.process_input("/new langgraph")

        assert self.running(app) == "langgraph"
        assert said(app.session, "add_warning") == [_LANGGRAPH_DEPRECATION]

    async def test_an_app_overriding_start_new_session_keeps_its_new(self, home, monkeypatch):
        """``start_new_session()`` overridden without the ``backend`` keyword:
        plain ``/new`` calls it as before."""
        from agentic_cli.cli.app import BaseCLIApp

        calls: list[str] = []

        class App(BaseCLIApp):
            async def start_new_session(self) -> str:
                calls.append("theirs")
                return await super().start_new_session()

        app, _, _ = self.on_adk(monkeypatch)
        app.__class__ = App
        old = app.session_id

        await app.process_input("/new")

        assert calls == ["theirs"] and app.session_id != old

    async def test_ctrl_c_before_the_new_backend_runs_changes_nothing(self, home, monkeypatch):
        app, adk, built = self.on_adk(monkeypatch)
        old = app.session_id
        starting = asyncio.Event()

        async def initialize_services():
            starting.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(self.Manager, "initialize_services", lambda self: initialize_services())
        new = asyncio.create_task(app.process_input("/new deepagents"))
        await asyncio.wait_for(starting.wait(), 5)
        new.cancel()
        with pytest.raises(asyncio.CancelledError):
            await new

        assert app.session_id == old and self.running(app) == "adk"
        assert not free(app, app._conversation_key(old))

    async def test_ctrl_c_once_the_new_backend_runs_opens_its_conversation(
        self, home, monkeypatch
    ):
        """Too late to undo once the new backend is published (the old one is
        being released): the new conversation is opened there, then the
        cancel goes on. The app never runs one backend in another's
        conversation."""
        app, adk, built = self.on_adk(monkeypatch)
        old = app.session_id
        releasing, released = asyncio.Event(), asyncio.Event()

        async def cleanup():
            releasing.set()
            await released.wait()

        adk.cleanup = cleanup
        new = asyncio.create_task(app.process_input("/new deepagents"))
        await asyncio.wait_for(releasing.wait(), 5)
        new.cancel()
        with pytest.raises(asyncio.CancelledError):
            await new
        released.set()

        assert app.session_id != old and self.running(app) == "deepagents"
        assert app._conversation_key(app.session_id)[0] == "deepagents"
        await app.process_input("Hello.")
        assert built[0].turns == [app.session_id]  # its turns write there, held there
        assert not free(app, app._conversation_key(app.session_id))
        await app._workflow_controller.close()


class TestAnOrchestratorChange:
    """``/settings`` changes ``orchestrator``, the backend of new
    conversations, while the workflow runs. A conversation never changes
    backend (its history is in its backend's store alone), so the open one
    stays where it is: when nothing of it is stored yet (no turn ran in it,
    and it was not opened from the store) a new conversation starts on the
    new backend at once, as ``/new <backend>`` starts one; else the change
    applies to the next ``/new``, and nothing is rebuilt. (It moved the open
    conversation's id to the new backend, from empty, and its lock and task
    list with it or not.)"""

    on_adk = TestANewConversationOnAnotherBackend.on_adk
    running = staticmethod(TestANewConversationOnAnotherBackend.running)

    @staticmethod
    async def saved_by_settings(app, edits: dict) -> AsyncMock:
        """``/settings`` with ``edits`` saved: what it saved."""
        from thinking_prompt import TextField

        from agentic_cli.cli.settings_command import SettingsCommand
        from tests.cli.dialog_fakes import edited_dialog

        app._build_ui_items = lambda: [TextField(key, key) for key in edits]
        app.session.show_dialog.return_value = True  # Save
        app.save_settings = AsyncMock(return_value=SimpleNamespace(
            project_path=None, user_path=None, user_scoped_keys=()))
        with patch("agentic_cli.cli.settings_command.Dialog", edited_dialog(edits)):
            await SettingsCommand().execute("", app)
        return app.save_settings

    async def test_a_used_conversation_stays_and_the_next_new_runs_there(
        self, home, monkeypatch
    ):
        app, adk, built = self.on_adk(monkeypatch)
        await app.process_input("Hello.")
        conversation, key = app.session_id, app._conversation_key(app.session_id)

        saved = await self.saved_by_settings(app, {"orchestrator": "deepagents"})

        assert built == [] and adk.reinitialize.await_count == 0
        assert app.session_id == conversation and self.running(app) == "adk"
        assert app._conversation_key(conversation) == key and not free(app, key)
        assert app.settings.orchestrator.value == "deepagents"
        saved.assert_awaited_once_with(keys={"orchestrator"})
        assert said(app.session, "add_success")[0] == (
            "Orchestrator: deepagents (applies to the next /new)"
        )

        await app.process_input("/new")
        assert self.running(app) == "deepagents"

    async def test_one_opened_from_the_store_stays_too(self, home, monkeypatch):
        app, adk, built = self.on_adk(monkeypatch)
        adk.stored = {"abc"}
        adk.session_info = lambda session_id, *, user_id=None: _row_of(session_id)
        adk.recent_messages = _no_messages
        assert await app.resume_session("abc")

        assert await app.apply_settings({"orchestrator": "deepagents"})

        assert built == [] and app.session_id == "abc" and self.running(app) == "adk"

    async def test_a_fresh_conversation_starts_a_new_one_there(self, home, monkeypatch):
        app, adk, built = self.on_adk(monkeypatch)
        old, old_key = app.session_id, app._conversation_key(app.session_id)

        saved = await self.saved_by_settings(app, {"orchestrator": "deepagents"})

        new = app.session_id
        assert new != old and self.running(app) == "deepagents"
        assert [m.backend_type for m in built] == ["deepagents"]
        assert app._conversation_key(new)[0] == "deepagents"
        assert free(app, app._conversation_key(new))  # held from its first write
        assert free(app, old_key)
        saved.assert_awaited_once_with(keys={"orchestrator"})
        assert said(app.session, "add_message")[-1] == f"New conversation {new} on deepagents."
        assert said(app.session, "add_success")[0] == "Orchestrator: deepagents"

    async def test_a_backend_that_cannot_start_changes_nothing(self, home, monkeypatch):
        app, adk, built = self.on_adk(monkeypatch)
        TestAPlaceChangeWhileTheWorkflowIsDown.builds(monkeypatch, lambda n, b: True)
        old = app.session_id

        assert await app.apply_settings({"orchestrator": "deepagents"}) is False

        assert app.session_id == old and self.running(app) == "adk"
        assert app.settings.orchestrator.value == "adk"
        assert said(app.session, "add_error") == [
            "Could not start a new conversation on deepagents: deepagents cannot start "
            "(its extra is missing). Settings were not changed."
        ]

    async def test_ctrl_c_before_the_new_backend_runs_changes_nothing(self, home, monkeypatch):
        app, adk, built = self.on_adk(monkeypatch)
        old = app.session_id
        starting = asyncio.Event()

        async def initialize_services():
            starting.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(self.Manager, "initialize_services", lambda self: initialize_services())
        applying = asyncio.create_task(app.apply_settings({"orchestrator": "deepagents"}))
        await asyncio.wait_for(starting.wait(), 5)
        applying.cancel()
        with pytest.raises(asyncio.CancelledError):
            await applying

        assert app.session_id == old and self.running(app) == "adk"
        assert app.settings.orchestrator.value == "adk"

    async def test_ctrl_c_once_the_new_backend_runs_keeps_the_change_and_saves_it(
        self, home, monkeypatch
    ):
        """Too late to undo: the new conversation runs there, the setting
        names the backend that runs, and ``/settings`` saves it."""
        app, adk, built = self.on_adk(monkeypatch)
        old = app.session_id
        releasing, released = asyncio.Event(), asyncio.Event()

        async def cleanup():
            releasing.set()
            await released.wait()

        adk.cleanup = cleanup
        saving = asyncio.create_task(self.saved_by_settings(app, {"orchestrator": "deepagents"}))
        await asyncio.wait_for(releasing.wait(), 5)
        saving.cancel()
        with pytest.raises(asyncio.CancelledError):
            await saving
        released.set()

        assert app.session_id != old and self.running(app) == "deepagents"
        assert app.settings.orchestrator.value == "deepagents"
        app.save_settings.assert_awaited_once_with(keys={"orchestrator"})
        assert "Orchestrator: deepagents" in said(app.session, "add_success")
        await app._workflow_controller.close()

    Manager = TestAPlaceChangeWhileTheWorkflowIsDown.Manager


def _row_of(session_id):
    from agentic_cli.workflow.sessions import SessionInfo

    async def row():
        return SessionInfo(session_id, "Earlier", 1, None)

    return row()


async def _no_messages(session_id, limit=20, *, user_id=None):
    return []


class TestWhetherAConversationMayMove:
    """A conversation never changes backend once anything of it is stored.
    It may (an orchestrator change then starts a new one there, or, while
    the workflow is down, the next start runs it there) only when it began in
    this run as a new one, no turn of it ran on a ready workflow (a turn the
    workflow was not up for stores nothing), and an id the app was given was
    found empty by a load. Decided under the turn lock: after a startup resume
    or an app's task that holds it."""

    Down = TestAPlaceChangeWhileTheWorkflowIsDown
    New = TestANewConversationOnAnotherBackend

    async def test_a_used_one_stays_when_the_workflow_went_down(self, home, monkeypatch):
        """Turns stored on adk; a failed reinitialization kept the manager, and
        its revive failed: the next start builds afresh. The conversation stays
        on adk, and the change waits for /new."""
        app, adk, built = self.New.on_adk(monkeypatch)
        await app.process_input("Hello.")
        conversation, key = app.session_id, app._conversation_key(app.session_id)
        controller = app._workflow_controller
        adk.is_initialized, adk.fails = False, True
        await controller.start_background_init()
        assert await controller.ensure_initialized() is False
        assert controller.next_start_builds

        assert await app.apply_settings({"orchestrator": "deepagents"})
        await app.process_input("Again.")
        await controller.close()

        assert said(app.session, "add_success") == [
            "Orchestrator: deepagents (applies to the next /new)"
        ]
        assert controller.backend == "adk" and [m.backend_type for m in built] == ["adk"]
        assert built[0].turns == [conversation] and app.session_id == conversation
        assert app._conversation_key(conversation) == key and not free(app, key)

    async def test_an_id_given_to_the_app_is_not_moved_unread(self, home, monkeypatch):
        """``session_id=`` names a stored conversation, and the first start
        failed: nothing said it is empty, so it stays where it is."""
        from agentic_cli.cli import AppInfo
        from agentic_cli.cli.app import BaseCLIApp
        from agentic_cli.cli.conversation import Place
        from agentic_cli.workflow.config import AgentConfig
        from tests.cli.dialog_fakes import autospec_session, opening_boxes

        self.Down.builds(monkeypatch, lambda n, backend: n == 0)
        settings = BaseSettings(session_store="sqlite", orchestrator="adk")
        app = BaseCLIApp(
            app_info=AppInfo(name="Given id", version="0"),
            agent_configs=[AgentConfig(name="solo", prompt="Help.")],
            settings=settings, session_id="abc",
        )
        app.session = opening_boxes(autospec_session())
        app._conversation.place = Place.of(settings)
        await app._workflow_controller.start_background_init()
        assert await app._adopt_session_on_startup()

        assert await app.apply_settings({"orchestrator": "deepagents"})
        await app._workflow_controller.close()

        assert app._workflow_controller.backend == "adk"
        assert said(app.session, "add_success")[-1] == (
            "Orchestrator: deepagents (applies to the next /new)"
        )
        assert not free(app, ("adk", *app._conversation_key("abc")[1:]))

    async def test_a_turn_the_workflow_was_not_up_for_stores_nothing(self, home, monkeypatch):
        built = self.Down.builds(monkeypatch, lambda n, backend: backend == "deepagents")
        app = await self.Down.started(
            BaseSettings(session_store="sqlite", orchestrator="deepagents")
        )
        await app.process_input("hello")  # its start fails again: nothing stored

        assert await app.apply_settings({"orchestrator": "adk"})
        await app.process_input("hello again")
        await app._workflow_controller.close()

        assert built[-1].backend_type == "adk" and built[-1].turns == [app.session_id]
        assert not free(app, self.Down.key_where(app, "adk", "sqlite"))

    async def test_it_is_decided_after_the_startup_resume(self, home, monkeypatch):
        """``/settings`` saved while ``--resume`` opens its conversation (it
        holds the turn lock): the run's empty conversation is no longer the
        one open, and the stored one it opens stays on its backend."""
        app, adk, built = self.New.on_adk(monkeypatch)
        adk.stored = {"abc"}
        adk.session_info = lambda session_id, *, user_id=None: _row_of(session_id)
        adk.recent_messages = _no_messages
        gate = asyncio.Event()

        async def opening():
            async with app._turn_lock:
                await gate.wait()
                await app._resume("abc")

        task = asyncio.create_task(opening())
        await asyncio.sleep(0)
        applying = asyncio.create_task(app.apply_settings({"orchestrator": "deepagents"}))
        await asyncio.sleep(0.05)
        gate.set()
        await asyncio.wait_for(task, 5)
        assert await asyncio.wait_for(applying, 5)

        assert app.session_id == "abc" and built == [] and self.New.running(app) == "adk"
        assert said(app.session, "add_success")[-1] == (
            "Orchestrator: deepagents (applies to the next /new)"
        )


class TestAConversationAMoveWaitedOn:
    """What makes a conversation stay on its backend, at the last moment: a
    turn the ready workflow ran (stored, cancelled or failed: it may have
    stored the input), and a turn queued behind ``/settings`` that the move
    waits for."""

    Down = TestAPlaceChangeWhileTheWorkflowIsDown
    New = TestANewConversationOnAnotherBackend

    async def test_a_cancelled_turn_keeps_it_on_its_backend(self, home, monkeypatch):
        app, adk, built = self.New.on_adk(monkeypatch)
        started = asyncio.Event()

        async def process(message, user_id, session_id=None):
            adk.turns.append(session_id)  # the input is stored as the turn starts
            started.set()
            await asyncio.Event().wait()
            yield None

        adk.process = process
        turn = asyncio.create_task(app.process_input("Hello."))
        await asyncio.wait_for(started.wait(), 5)
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn
        conversation = app.session_id

        assert await app.apply_settings({"orchestrator": "deepagents"})

        assert not app._conversation.fresh and app.session_id == conversation
        assert said(app.session, "add_success") == [
            "Orchestrator: deepagents (applies to the next /new)"
        ]
        assert built == []

    async def test_a_failed_turn_keeps_it_on_its_backend(self, home, monkeypatch):
        app, adk, built = self.New.on_adk(monkeypatch)

        async def process(message, user_id, session_id=None):
            adk.turns.append(session_id)
            raise RuntimeError("provider down")
            yield None

        adk.process = process
        await app.process_input("Hello.")
        conversation = app.session_id

        assert await app.apply_settings({"orchestrator": "deepagents"})

        assert not app._conversation.fresh and app.session_id == conversation
        assert said(app.session, "add_success") == [
            "Orchestrator: deepagents (applies to the next /new)"
        ]

    async def test_a_turn_queued_behind_settings_that_stores_one_keeps_it(
        self, home, monkeypatch
    ):
        """Down; /settings decides the move; an app's turn queued on the turn
        lock behind it starts the workflow and stores a turn; then the move
        takes the lock: the conversation stays where its turn was stored."""
        built = self.Down.builds(monkeypatch, lambda n, b: n == 0)
        app = await self.Down.started(
            BaseSettings(session_store="sqlite", orchestrator="deepagents")
        )
        controller = app._workflow_controller
        conversation = app.session_id
        gate = asyncio.Event()

        async def an_apps_task():
            async with app._turn_lock:
                await gate.wait()

        holder = asyncio.create_task(an_apps_task())
        await asyncio.sleep(0)
        applying = asyncio.create_task(app.apply_settings({"orchestrator": "adk"}))
        await asyncio.sleep(0.01)
        turn = asyncio.create_task(app._handle_message("an app's turn"))  # behind /settings
        await asyncio.sleep(0.01)
        gate.set()
        assert await asyncio.wait_for(applying, 5)
        await asyncio.wait_for(asyncio.gather(holder, turn), 5)

        assert said(app.session, "add_success") == ["Orchestrator: adk (applies to the next /new)"]
        assert controller.backend == "deepagents" and app.session_id == conversation
        assert [(m.backend_type, m.turns) for m in built] == [
            ("deepagents", []), ("deepagents", [conversation])
        ]
        assert not free(app, app._conversation_key(conversation))
        assert app._conversation_key(conversation)[0] == "deepagents"
        await controller.close()


class TestABackendThatCannotRun:
    """No backend that cannot run is accepted: an ``orchestrator`` change to
    one whose extra is not installed is refused, whatever the conversation
    (it was saved for the next /new, which then failed, as did the next
    start), and plain ``/new`` on such a setting says why in one line."""

    New = TestANewConversationOnAnotherBackend
    MISSING = (
        "The deepagents orchestrator needs the 'deepagents' extra. "
        "Install with: pip install 'agentic-cli[deepagents]'"
    )

    async def test_the_setting_is_refused_in_a_used_conversation(self, home, monkeypatch):
        import sys

        app, adk, built = self.New.on_adk(monkeypatch)
        await app.process_input("Hello.")
        monkeypatch.setitem(sys.modules, "agentic_cli.workflow.deepagents", None)

        assert await app.apply_settings({"orchestrator": "deepagents"}) is False

        assert app.settings.orchestrator.value == "adk" and built == []
        assert said(app.session, "add_error") == [f"Failed to set Orchestrator: {self.MISSING}"]

    async def test_plain_new_on_it_says_why_and_changes_nothing(self, home, monkeypatch):
        """Checked before anything changes: the conversation left proposes
        no memories for a /new that does not happen."""
        import sys

        app, adk, built = self.New.on_adk(
            monkeypatch, BaseSettings(session_store="sqlite", auto_extract_session_facts=True)
        )
        await app.process_input("Hello.")
        old = app.session_id
        app.settings.update_setting("orchestrator", "deepagents")  # as a file left it
        monkeypatch.setitem(sys.modules, "agentic_cli.workflow.deepagents", None)

        await app.process_input("/new")

        assert app.session_id == old and built == [] and "on_session_end" not in adk.calls
        assert said(app.session, "add_error") == [self.MISSING]


class TestAnAppsOwnStartNewSession:
    """The flows go through the app's public ``start_new_session()``, so an
    override applies: an orchestrator change in a conversation with nothing
    stored calls it with no keyword (it runs on the settings' orchestrator,
    which names the new backend). ``/new <backend>`` on an app whose override
    takes no ``backend`` is refused plainly."""

    New = TestANewConversationOnAnotherBackend

    async def test_the_orchestrator_change_calls_it(self, home, monkeypatch):
        from agentic_cli.cli.app import BaseCLIApp

        calls: list = []

        class App(BaseCLIApp):
            async def start_new_session(self) -> str:
                calls.append("theirs")
                return await super().start_new_session()

        app, _, _ = self.New.on_adk(monkeypatch)
        app.__class__ = App
        old = app.session_id

        assert await app.apply_settings({"orchestrator": "deepagents"})

        assert calls == ["theirs"] and app.session_id != old
        assert self.New.running(app) == "deepagents"

    async def test_new_with_a_backend_is_refused_when_it_takes_none(self, home, monkeypatch):
        from agentic_cli.cli.app import BaseCLIApp

        class App(BaseCLIApp):
            async def start_new_session(self) -> str:
                return await super().start_new_session()

        app, _, built = self.New.on_adk(monkeypatch)
        app.__class__ = App
        old = app.session_id

        await app.process_input("/new deepagents")

        assert app.session_id == old and built == []
        assert said(app.session, "add_error") == ["This app's /new takes no backend."]


class TestACtrlCInAMoveOfBackend:
    """Ctrl+C at each await of a new conversation on another backend, and of
    the move of an empty conversation while the workflow is down: the app
    ends in one conversation, held where its turns write, the settings
    naming the backend that runs, and the user is told where it is."""

    New = TestANewConversationOnAnotherBackend
    Down = TestAPlaceChangeWhileTheWorkflowIsDown

    async def test_at_the_new_backends_load_its_conversation_is_said(self, home, monkeypatch):
        app, adk, built = self.New.on_adk(monkeypatch)
        old = app.session_id
        loading = asyncio.Event()
        load = self.New.Manager.load_session

        async def load_session(self, session_id, user_id=None):
            if self.backend_type == "deepagents":
                loading.set()
                await asyncio.Event().wait()
            return await load(self, session_id, user_id)

        monkeypatch.setattr(self.New.Manager, "load_session", load_session)
        new = asyncio.create_task(app.process_input("/new deepagents"))
        await asyncio.wait_for(loading.wait(), 5)
        new.cancel()
        with pytest.raises(asyncio.CancelledError):
            await new

        assert app.session_id != old and self.New.running(app) == "deepagents"
        assert app._conversation_key(app.session_id)[0] == "deepagents"
        assert f"New conversation {app.session_id} on deepagents." in said(
            app.session, "add_message"
        )

    @staticmethod
    async def down_on_deepagents(monkeypatch):
        """An app whose start fails on deepagents (it would start on adk)."""
        TestAPlaceChangeWhileTheWorkflowIsDown.builds(monkeypatch, lambda n, b: b == "deepagents")
        return await TestAPlaceChangeWhileTheWorkflowIsDown.started(
            BaseSettings(session_store="sqlite", orchestrator="deepagents")
        )

    async def test_while_the_turn_lock_is_held_elsewhere_nothing_changes(self, home, monkeypatch):
        app = await self.down_on_deepagents(monkeypatch)
        await TestAPlaceChangeWhileTheWorkflowIsDown.typed_while_down(app)
        key = app._conversation_key(app.session_id)
        gate = asyncio.Event()

        async def an_apps_task():
            async with app._turn_lock:
                await gate.wait()

        holder = asyncio.create_task(an_apps_task())
        await asyncio.sleep(0)
        applying = asyncio.create_task(app.apply_settings({"orchestrator": "adk"}))
        await asyncio.sleep(0.05)
        applying.cancel()
        with pytest.raises(asyncio.CancelledError):
            await applying
        gate.set()
        await holder

        assert app.settings.orchestrator.value == "deepagents"
        assert app._workflow_controller.backend == "deepagents"
        assert app._conversation_key(app.session_id) == key and not free(app, key)
        await app._workflow_controller.close()

    async def test_while_the_controller_waits_for_a_release_nothing_changes(
        self, home, monkeypatch
    ):
        """A release still running makes the controller wait before it takes
        the new backend: the lock taken there is let go again."""
        app = await self.down_on_deepagents(monkeypatch)
        await TestAPlaceChangeWhileTheWorkflowIsDown.typed_while_down(app)
        controller = app._workflow_controller
        key = app._conversation_key(app.session_id)
        releasing = asyncio.Event()
        release = asyncio.create_task(releasing.wait())  # a release still running
        controller._orphan_cleanups.add(release)
        release.add_done_callback(controller._orphan_cleanups.discard)

        applying = asyncio.create_task(app.apply_settings({"orchestrator": "adk"}))
        await asyncio.sleep(0.05)
        applying.cancel()
        with pytest.raises(asyncio.CancelledError):
            await applying
        releasing.set()
        await release

        assert app.settings.orchestrator.value == "deepagents"
        assert controller.backend == "deepagents"
        assert app._conversation_key(app.session_id) == key and not free(app, key)
        assert free(app, ("adk", *key[1:]))
        await controller.close()


class TestAWorkflowThatCameUpWhileTheChangeWaited:
    """``/settings`` decided to move the conversation (the workflow down, its
    next start building afresh), then waited for the turn lock while an app's
    task started the workflow without a turn. The move re-decides under the
    lock, as for a running workflow: a new conversation on the new backend
    (``start_new_session``), so a Ctrl+C there ends as ``/new``'s does, the
    app in one conversation on the backend that runs, which the settings
    name. Before, the move went on: a Ctrl+C once the controller ran the new
    backend left the conversation and the settings on the old one."""

    Down = TestAPlaceChangeWhileTheWorkflowIsDown
    New = TestANewConversationOnAnotherBackend

    async def came_up_meanwhile(self, monkeypatch, *, slow_release: bool, gate_load: bool):
        """The reviewer's W12 (``slow_release``: Ctrl+C while the old manager
        is released) and W12b (``gate_load``: Ctrl+C at the new backend's
        load); returns the app, its first id and the managers built."""
        built = self.Down.builds(monkeypatch, lambda n, b: n == 0)
        app = await self.Down.started(
            BaseSettings(session_store="sqlite", orchestrator="deepagents")
        )
        controller = app._workflow_controller
        first = app.session_id
        gate_a, b_done = asyncio.Event(), asyncio.Event()
        releasing, release_gate, loading = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def task_a():
            async with app._turn_lock:
                await gate_a.wait()

        async def task_b():  # an app's task: starts the workflow, writes nothing
            async with app._turn_lock:
                assert await controller.ensure_initialized()
                if slow_release:
                    async def slow():
                        releasing.set()
                        await release_gate.wait()
                    controller.workflow.cleanup = slow
                b_done.set()

        if gate_load:
            load = self.New.Manager.load_session

            async def load_session(manager, session_id, user_id=None):
                if manager.backend_type == "adk":
                    loading.set()
                    await asyncio.Event().wait()
                return await load(manager, session_id, user_id)

            monkeypatch.setattr(self.New.Manager, "load_session", load_session)

        a = asyncio.create_task(task_a())
        await asyncio.sleep(0)
        applying = asyncio.create_task(app.apply_settings({"orchestrator": "adk"}))
        await asyncio.sleep(0.01)
        b = asyncio.create_task(task_b())
        await asyncio.sleep(0.01)
        gate_a.set()
        await asyncio.wait_for(b_done.wait(), 5)
        await asyncio.wait_for((releasing if slow_release else loading).wait(), 5)
        applying.cancel()
        with pytest.raises(asyncio.CancelledError):
            await applying
        release_gate.set()
        await asyncio.gather(a, b)
        return app, first, built

    def ends_in_one_place(self, app, first: str, built: list) -> None:
        controller = app._workflow_controller
        assert app.session_id != first  # a new conversation, as on a running workflow
        assert app.settings.orchestrator.value == "adk" == controller.backend
        assert app._conversation.place.backend == "adk"
        assert f"New conversation {app.session_id} on adk." in said(app.session, "add_message")

    async def test_ctrl_c_while_the_old_backend_is_released(self, home, monkeypatch):
        app, first, built = await self.came_up_meanwhile(
            monkeypatch, slow_release=True, gate_load=False
        )

        self.ends_in_one_place(app, first, built)
        await app.process_input("hello")
        assert built[-1].backend_type == "adk" and built[-1].turns == [app.session_id]
        assert not free(app, app._conversation_key(app.session_id))
        assert said(app.session, "add_success") == ["Orchestrator: adk"]
        await app._workflow_controller.close()

    async def test_ctrl_c_at_the_new_backends_load(self, home, monkeypatch):
        app, first, built = await self.came_up_meanwhile(
            monkeypatch, slow_release=False, gate_load=True
        )

        self.ends_in_one_place(app, first, built)
        await app._workflow_controller.close()


class TestTheExitHintOnAnotherBackend:
    """``--resume`` opens a conversation on its own backend, whatever the
    settings' orchestrator names (``tests/cli/test_resume_on_its_backend.py``):
    one ``/new <backend>`` started elsewhere needs no setting first. The
    session store is the run's, which a ``session_store`` change in the run
    leaves for the next start: the hint names it then."""

    New = TestANewConversationOnAnotherBackend

    async def test_on_another_backend_it_needs_no_setting(self, home, monkeypatch):
        app, _, _ = self.New.on_adk(monkeypatch)
        await app.process_input("/new deepagents")
        app._command = "my-app"

        assert app._resume_hint() == (
            f"Resume this conversation with: my-app --resume {app.session_id}"
        )

    async def test_on_the_settings_backend_it_is_as_before(self, home, monkeypatch):
        app, _, _ = self.New.on_adk(monkeypatch)
        await app.process_input("/new")

        assert app._resume_hint() == f"Resume this conversation with /resume {app.session_id}"

    async def test_after_a_store_change_it_names_the_runs_store(self, home, monkeypatch):
        """The next start opens the new store, where the conversation is not."""
        app, _, _ = self.New.on_adk(monkeypatch)
        assert await app.apply_settings({"session_store": "memory"})

        assert app._resume_hint() == (
            f"Resume this conversation with /resume {app.session_id} "
            "(set session_store to sqlite first: it is stored there)"
        )

    async def test_after_both_changes_it_names_the_store_alone(self, home, monkeypatch):
        app, _, _ = self.New.on_adk(monkeypatch)
        await app.process_input("/new deepagents")
        assert await app.apply_settings(
            {"session_store": "postgres", "postgres_uri": "postgresql://h/db"}
        )

        assert app._resume_hint() == (
            f"Resume this conversation with /resume {app.session_id} "
            "(set session_store to sqlite first: it is stored there)"
        )


class TestARunThatWritesNothing:
    """An id this run generated (the start's new conversation, ``/new``, an
    orchestrator change's new one) is held from its first write, under the
    turn lock before a turn runs (a line typed while the workflow is down
    takes it before the start its turn retries): no store lists it before
    then, so no other process can open it, and a run in which no turn runs
    makes no lock file in HOME. An id the app was given is held from the
    start."""

    async def test_a_start_new_and_exit_leave_no_lock_file(self, manager):
        session = Session(inputs=["/new"])
        app = app_with(session)

        await app.run()

        assert session.prompted
        assert not (app.settings.sessions_dir / "locks").exists()

    async def test_the_runs_own_is_held_at_its_first_turn(self, home):
        workflow = Workflow()
        app = app_on(workflow, BaseSettings(session_store="sqlite"))
        key = app._conversation_key(app.session_id)

        assert free(app, key)
        await app.process_input("Hello.")
        assert workflow.turns == [app.session_id] and not free(app, key)

    async def test_its_first_turn_is_refused_if_another_process_has_it(self, home):
        """Its id could only be another's by a collision of random ids."""
        workflow = Workflow()
        app = app_on(workflow, BaseSettings(session_store="sqlite"))
        other = open_elsewhere(app, app._conversation_key(app.session_id))

        await app.process_input("Hello.")
        other.release()

        assert workflow.turns == []
        assert said(app.session, "add_error") == [
            f"Conversation '{app.session_id}' is open in another process ({os.getpid()})."
        ]

    async def test_a_compact_holds_it_before_it_writes(self, home):
        workflow = Workflow()
        app = app_on(workflow, BaseSettings(session_store="sqlite"))
        other = open_elsewhere(app, app._conversation_key(app.session_id))

        await app.process_input("/compact")
        other.release()
        await app.process_input("/compact")

        assert len(workflow.compacted) == 1  # the first refused
        assert not free(app, app._conversation_key(app.session_id))

    @pytest.mark.parametrize("line", ["Hello.", "/compact"])
    async def test_a_locks_folder_it_cannot_write_lets_the_write_go_on_unheld(
        self, home, line
    ):
        """A read-only sessions folder (a postgres store, say): the run says
        once that it could not hold the conversation, and why, and its turns
        and /compact run unheld, as before."""
        workflow = Workflow()
        app = app_on(workflow, BaseSettings(session_store="sqlite"))
        locks = app.settings.sessions_dir / "locks"
        locks.mkdir(parents=True)
        os.chmod(locks, 0o500)
        try:
            await app.process_input(line)
            await app.process_input(line)
        finally:
            os.chmod(locks, 0o700)

        done = workflow.turns if line == "Hello." else workflow.compacted
        assert len(done) == 2
        assert said(app.session, "add_error") == []
        warnings = said(app.session, "add_warning")
        assert len(warnings) == 1
        assert warnings[0].startswith(
            f"Could not hold conversation '{app.session_id}' for this process: "
        )
        assert "Permission denied" in warnings[0]

    async def test_a_new_conversation_is_held_at_its_first_turn(self, home, monkeypatch):
        app, _, _ = TestANewConversationOnAnotherBackend.on_adk(monkeypatch)
        await app.process_input("/new")
        new = app.session_id

        assert free(app, app._conversation_key(new))
        await app.process_input("Hello.")
        assert not free(app, app._conversation_key(new))
