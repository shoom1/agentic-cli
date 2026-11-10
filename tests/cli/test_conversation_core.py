"""The conversation the app is in (``cli/conversation.py``).

``ConversationCore`` keeps what makes the conversation the CLI's turns run
in: its id, whether a turn ran in it since it was opened (``turned``), and
the lock this process holds on it, at its ``Place`` (the backend, the store,
the app and the user: the same id in another backend or store is another
conversation). ``BaseCLIApp`` holds one, and its ``_session_id`` and
``_turned`` read and write it; an app built with ``__new__`` (tests, an
application's own construction) has one too.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_cli.cli import AppInfo
from agentic_cli.cli import app as app_module
from agentic_cli.cli.app import BaseCLIApp
from agentic_cli.cli.conversation import ConversationCore, Place
from agentic_cli.cli.conversation_lock import ConversationLock
from agentic_cli.config import BaseSettings
from agentic_cli.workflow.config import AgentConfig


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


def place(folder: Path, store: str = "sqlite", backend: str = "adk") -> Place:
    return Place(backend=backend, store=store, app_name="app", user="me", locks=folder)


def held_elsewhere(where: Place, session_id: str) -> int | None:
    """Whether another process could open it: the holder's pid, or None (and
    it is let go again)."""
    other = ConversationLock(where.locks)
    holder = other.take(where.key(session_id))
    other.release()
    return holder


def build(session_id: str | None = None) -> BaseCLIApp:
    return BaseCLIApp(
        app_info=AppInfo(name="Core test", version="0"),
        agent_configs=[AgentConfig(name="assistant", prompt="Help.")],
        settings=BaseSettings(session_store="sqlite"),
        session_id=session_id,
    )


class TestThePlace:
    def test_it_is_read_from_the_settings(self, home):
        settings = BaseSettings(session_store="sqlite", orchestrator="deepagents")

        assert Place.of(settings) == Place(
            backend="deepagents", store="sqlite", app_name=settings.app_name,
            user=settings.default_user, locks=settings.sessions_dir / "locks",
        )

    def test_a_conversation_is_keyed_by_its_place_and_id(self, tmp_path):
        assert place(tmp_path).key("abc") == ("adk", "sqlite", "app", "me", "abc")


class TestTheCore:
    def test_a_new_core_is_in_its_conversation_unused_and_holding_nothing(self, tmp_path):
        core = ConversationCore("abc", place(tmp_path))

        assert (core.session_id, core.turned) == ("abc", False)
        assert held_elsewhere(place(tmp_path), "abc") is None

    def test_what_it_holds_another_process_is_refused(self, tmp_path):
        core = ConversationCore("abc", place(tmp_path))

        assert core.hold("abc") is None
        assert held_elsewhere(place(tmp_path), "abc") == os.getpid()

    def test_one_held_elsewhere_is_refused_with_its_holder(self, tmp_path):
        other = ConversationLock(tmp_path)
        other.take(place(tmp_path).key("abc"))

        assert ConversationCore("xyz", place(tmp_path)).hold("abc") == os.getpid()

    def test_a_hold_taken_beside_is_kept_or_let_go(self, tmp_path):
        core = ConversationCore("abc", place(tmp_path))
        core.hold("abc")

        kept, dropped = core.acquire("kept"), core.acquire("dropped")
        assert held_elsewhere(place(tmp_path), "abc") == os.getpid()  # until the commit
        core.commit(kept)
        core.abandon(dropped)

        assert held_elsewhere(place(tmp_path), "abc") is None
        assert held_elsewhere(place(tmp_path), "kept") == os.getpid()
        assert held_elsewhere(place(tmp_path), "dropped") is None

    def test_nothing_is_locked_on_the_memory_store(self, tmp_path):
        core = ConversationCore("abc", place(tmp_path, store="memory"))

        assert core.hold("abc") is None
        assert list(tmp_path.iterdir()) == []  # no lock file made

    def test_release_lets_it_go(self, tmp_path):
        core = ConversationCore("abc", place(tmp_path))
        core.hold("abc")

        core.release()

        assert held_elsewhere(place(tmp_path), "abc") is None


class FakeWorkflow:
    """A ready workflow: what it holds, the conversation it adopted last, and
    a gate per call to hold one until it is set (``gates["load b"]``)."""

    def __init__(self, *stored: str, failing: str | None = None) -> None:
        self.stored = set(stored)
        self.failing = failing
        self.adopted: str | None = None
        self.calls: list[str] = []
        self.gates: dict[str, asyncio.Event] = {}
        self.waiting = asyncio.Event()

    async def _at(self, call: str) -> None:
        self.calls.append(call)
        gate = self.gates.get(call)
        if gate is not None:
            self.waiting.set()
            await gate.wait()

    async def load_session(self, session_id, user_id=None) -> bool:
        await self._at(f"load {session_id}")
        if session_id == self.failing:
            raise RuntimeError("database is locked")
        self.adopted = session_id
        return session_id in self.stored

    async def session_info(self, session_id, *, user_id=None):
        await self._at(f"info {session_id}")
        return object() if session_id in self.stored else None


class Around:
    """What a switch acts on besides the conversation, recording what it does."""

    def __init__(self, workflow=None) -> None:
        self.workflow = workflow
        self.done: list[str] = []
        self.proposing: asyncio.Event | None = None

    def parts(self):
        from agentic_cli.cli.conversation import SwitchParts

        async def propose(session_id: str) -> None:
            self.done.append(f"propose {session_id}")
            if self.proposing is not None:
                await self.proposing.wait()

        return SwitchParts(
            workflow=self.workflow,
            processor=SimpleNamespace(reset=lambda: self.done.append("tasks reset")),
            tracker=SimpleNamespace(
                reset=lambda: self.done.append("usage reset"),
                reset_context=lambda: self.done.append("context reset"),
            ),
            propose=propose,
            status=lambda: self.done.append("status"),
        )


class TestTheSwitch:
    """``ConversationCore.switch``: the one protocol every change of
    conversation goes through."""

    @staticmethod
    def core_in(tmp_path, session_id: str = "a") -> ConversationCore:
        core = ConversationCore(session_id, place(tmp_path))
        assert core.hold(session_id) is None
        return core

    async def test_it_holds_the_target_and_lets_the_one_left_go(self, tmp_path):
        core, around = self.core_in(tmp_path), Around(FakeWorkflow("b"))

        switched = await core.switch("b", around.parts(), stored=True)

        assert (switched.made, switched.loaded) == (True, True)
        assert (core.session_id, around.workflow.adopted) == ("b", "b")
        assert held_elsewhere(place(tmp_path), "b") == os.getpid()
        assert held_elsewhere(place(tmp_path), "a") is None
        assert around.done == ["tasks reset", "usage reset", "status"]

    async def test_one_open_elsewhere_is_refused_and_nothing_changes(self, tmp_path):
        core, around = self.core_in(tmp_path), Around(FakeWorkflow("b"))
        other = ConversationLock(tmp_path)
        other.take(place(tmp_path).key("b"))

        switched = await core.switch("b", around.parts(), stored=True)

        assert (switched.made, switched.holder) == (False, os.getpid())
        assert core.session_id == "a" and around.workflow.calls == []
        assert held_elsewhere(place(tmp_path), "a") == os.getpid()

    async def test_a_conversation_used_has_its_memories_proposed_before_the_load(
        self, tmp_path
    ):
        core, around = self.core_in(tmp_path), Around(FakeWorkflow("b"))
        core.turned = True

        await core.switch("b", around.parts())

        assert around.done[0] == "propose a" and core.turned is False

    @pytest.mark.parametrize("turned, propose", [(False, True), (True, False)])
    async def test_one_unused_or_a_switch_told_not_to_proposes_none(
        self, tmp_path, turned, propose
    ):
        core, around = self.core_in(tmp_path), Around(FakeWorkflow())
        core.turned = turned

        await core.switch("b", around.parts(), propose=propose)

        assert not [d for d in around.done if d.startswith("propose")]
        assert core.turned is False

    async def test_a_stored_one_deleted_as_it_opens_is_gone_and_nothing_changes(self, tmp_path):
        """Not held by the load, nor listed any more: the workflow is given
        back the conversation the app stays in."""
        core, around = self.core_in(tmp_path), Around(FakeWorkflow())

        switched = await core.switch("b", around.parts(), stored=True)

        assert (switched.made, switched.gone) == (False, True)
        assert core.session_id == "a" and around.workflow.adopted == "a"
        assert around.workflow.calls == ["load b", "info b", "load a"]
        assert held_elsewhere(place(tmp_path), "a") == os.getpid()
        assert held_elsewhere(place(tmp_path), "b") is None
        assert around.done == []

    async def test_a_stored_one_the_load_finds_empty_is_still_opened(self, tmp_path):
        """Listed, yet holding no message (a first turn killed before its
        input was stored): it is there to open."""
        core, workflow = self.core_in(tmp_path), FakeWorkflow()
        workflow.session_info = lambda session_id, user_id=None: _row()
        around = Around(workflow)

        switched = await core.switch("b", around.parts(), stored=True)

        assert (switched.made, switched.loaded) == (True, False)

    async def test_a_new_conversation_the_load_finds_empty_is_opened(self, tmp_path):
        core, around = self.core_in(tmp_path), Around(FakeWorkflow())

        switched = await core.switch("b", around.parts())

        assert (switched.made, switched.loaded) == (True, False)
        assert around.workflow.calls == ["load b"]

    async def test_a_load_that_fails_changes_nothing(self, tmp_path):
        core, around = self.core_in(tmp_path), Around(FakeWorkflow("b", failing="b"))

        with pytest.raises(RuntimeError, match="database is locked"):
            await core.switch("b", around.parts(), stored=True)

        assert core.session_id == "a" and around.done == []
        assert held_elsewhere(place(tmp_path), "a") == os.getpid()
        assert held_elsewhere(place(tmp_path), "b") is None

    async def test_without_a_ready_workflow_nothing_is_loaded(self, tmp_path):
        core, around = self.core_in(tmp_path), Around(workflow=None)

        switched = await core.switch("b", around.parts())

        assert (switched.made, switched.loaded) == (True, None)
        assert held_elsewhere(place(tmp_path), "b") == os.getpid()

    async def test_the_conversation_it_is_in_is_held_whatever_its_load_says(self, tmp_path):
        """The one a run starts in, or the same one in another backend: there
        is no other to stay in, so it is held before the workflow loads it."""
        core = ConversationCore("a", place(tmp_path))
        around = Around(FakeWorkflow(failing="a"))

        with pytest.raises(RuntimeError, match="database is locked"):
            await core.switch("a", around.parts())

        assert held_elsewhere(place(tmp_path), "a") == os.getpid()
        assert around.done == ["tasks reset", "usage reset"]

    async def test_another_place_keys_the_lock_there_and_resets_the_context_only(
        self, tmp_path
    ):
        """The same id in another place (a backend or store the workflow's
        next start builds, while it is down). The usage counters keep the
        session's totals; the context is another one's."""
        core, around = self.core_in(tmp_path), Around(FakeWorkflow())
        core.turned = True
        there = place(tmp_path, backend="deepagents")

        switched = await core.switch("a", around.parts(), place=there, propose=False)

        assert switched.made and core.place == there and core.turned is False
        assert held_elsewhere(there, "a") == os.getpid()
        assert held_elsewhere(place(tmp_path), "a") is None
        assert around.done == ["tasks reset", "context reset", "status"]

    async def test_a_stored_one_on_another_backend_is_opened_before_the_load(self, tmp_path):
        """``/resume`` of a conversation another backend keeps: that backend
        runs already, and the one left is not on it, so there is nowhere
        else to stay: the switch is made before the load, a fresh start for
        the task list and every usage counter."""
        core, around = self.core_in(tmp_path), Around(FakeWorkflow("b", failing="b"))
        there = place(tmp_path, backend="deepagents")

        with pytest.raises(RuntimeError, match="database is locked"):
            await core.switch("b", around.parts(), place=there, stored=True)

        assert (core.session_id, core.place) == ("b", there)
        assert held_elsewhere(there, "b") == os.getpid()
        assert held_elsewhere(place(tmp_path), "a") is None
        assert around.done == ["tasks reset", "usage reset"]
        assert "load a" not in around.workflow.calls  # not given back: not there

    async def test_one_on_another_backend_deleted_as_it_opens_is_made_and_starts_empty(
        self, tmp_path
    ):
        core, around = self.core_in(tmp_path), Around(FakeWorkflow())
        there = place(tmp_path, backend="deepagents")

        switched = await core.switch("b", around.parts(), place=there, stored=True)

        assert (switched.made, switched.gone, switched.loaded) == (True, True, False)
        assert (core.session_id, core.place, core.fresh) == ("b", there, True)
        assert around.workflow.calls == ["load b", "info b"]
        assert held_elsewhere(there, "b") == os.getpid()
        assert around.done == ["tasks reset", "usage reset", "status"]

    async def test_the_same_id_stored_on_another_backend_is_another_conversation(
        self, tmp_path
    ):
        """A twin (an earlier build switched the backend under an open
        conversation): not the one the app is in, moved. Its usage starts
        over, and so do the flags of the one left (it was written)."""
        core, around = self.core_in(tmp_path), Around(FakeWorkflow("a"))
        core.written, core.opened_empty = True, True
        there = place(tmp_path, backend="deepagents")

        switched = await core.switch("a", around.parts(), place=there, stored=True)

        assert switched.made and core.place == there
        assert around.done == ["tasks reset", "usage reset", "status"]
        assert (core.written, core.opened_empty, core.deferred) == (False, False, False)
        assert held_elsewhere(there, "a") == os.getpid()
        assert held_elsewhere(place(tmp_path), "a") is None

    async def test_a_hold_taken_before_is_the_one_kept(self, tmp_path):
        """At startup the target is held before the prompt, and opened once it is up."""
        core, around = self.core_in(tmp_path), Around(FakeWorkflow("b"))
        hold = core.acquire("b")

        assert (await core.switch("b", around.parts(), stored=True, hold=hold)).made
        assert held_elsewhere(place(tmp_path), "b") == os.getpid()
        assert held_elsewhere(place(tmp_path), "a") is None

    @pytest.mark.parametrize("step", ["propose", "load b", "info b", "load a"])
    async def test_a_ctrl_c_at_any_step_leaves_it_in_one_conversation_holding_it(
        self, tmp_path, step
    ):
        """Whatever await a Ctrl+C lands on, the app stays in the conversation
        it was in, holding it alone, and the workflow ends on it too, given
        back even when the cancel lands while it is given back."""
        core, workflow = self.core_in(tmp_path), FakeWorkflow()  # b is gone
        around = Around(workflow)
        core.turned = step == "propose"
        gate = asyncio.Event()
        if step == "propose":
            around.proposing = gate
        else:
            workflow.gates[step] = gate
        waiting = (lambda: "propose a" in around.done) if step == "propose" else (
            lambda: workflow.waiting.is_set()
        )

        switch = asyncio.create_task(core.switch("b", around.parts(), stored=True))
        for _ in range(200):
            if waiting():
                break
            await asyncio.sleep(0.01)
        switch.cancel()
        gate.set()  # what the cancel left running ends now
        with pytest.raises(asyncio.CancelledError):
            await switch
        # Once the workflow took b, it is given a back; before, it took nothing.
        given_back = "a" if step in ("info b", "load a") else None
        for _ in range(200):
            if workflow.adopted == given_back:
                break
            await asyncio.sleep(0.01)

        assert core.session_id == "a" and workflow.adopted == given_back
        assert held_elsewhere(place(tmp_path), "a") == os.getpid()
        assert held_elsewhere(place(tmp_path), "b") is None
        assert "tasks reset" not in around.done


class TestHeldFromItsFirstWrite:
    """An id this run generated is in no store until its first write, so no
    other process can open it before then: nothing holds it (no lock file is
    made) until ``hold_to_write``, which the app calls under the turn lock
    before a turn, a job's notice or ``/compact`` writes it."""

    async def test_a_generated_id_is_switched_to_unheld(self, tmp_path):
        core, around = TestTheSwitch.core_in(tmp_path), Around(FakeWorkflow())

        switched = await core.switch("b", around.parts(), generated=True)

        assert switched.made and core.session_id == "b" and core.deferred
        assert held_elsewhere(place(tmp_path), "a") is None  # the one left let go
        assert held_elsewhere(place(tmp_path), "b") is None

    def test_it_is_held_at_its_first_write(self, tmp_path):
        core = ConversationCore("a", place(tmp_path), opened_empty=True, deferred=True)

        assert core.hold_to_write() is None

        assert not core.deferred
        assert held_elsewhere(place(tmp_path), "a") == os.getpid()
        assert core.hold_to_write() is None  # held already: nothing more

    def test_one_open_elsewhere_by_then_is_refused(self, tmp_path):
        ConversationLock(tmp_path).take(place(tmp_path).key("a"))
        core = ConversationCore("a", place(tmp_path), opened_empty=True, deferred=True)

        assert core.hold_to_write() == os.getpid()
        assert core.deferred

    async def test_a_run_that_writes_nothing_makes_no_lock_file(self, tmp_path):
        locks = tmp_path / "locks"
        core = ConversationCore("a", place(locks), opened_empty=True, deferred=True)

        await core.switch("a", Around(FakeWorkflow()).parts())  # the start adopts it
        await core.switch("b", Around(FakeWorkflow()).parts(), generated=True)  # /new
        core.release()

        assert not locks.exists()

    async def test_a_given_id_is_held_from_the_start(self, tmp_path):
        core = ConversationCore("a", place(tmp_path))

        await core.switch("a", Around(FakeWorkflow()).parts())

        assert not core.deferred and held_elsewhere(place(tmp_path), "a") == os.getpid()


class TestNothingStored:
    """``fresh``: nothing of the conversation is stored. It began in this run
    as a new one (or a load found it empty), and no turn of it ran on a ready
    workflow (``written``). ``turned`` (its memories to propose) is apart."""

    def test_a_new_id_is_fresh_and_a_given_one_is_not(self, tmp_path):
        assert ConversationCore("a", place(tmp_path), opened_empty=True).fresh
        assert not ConversationCore("a", place(tmp_path)).fresh

    def test_a_turn_written_ends_it_and_a_turn_proposed_alone_does_not(self, tmp_path):
        core = ConversationCore("a", place(tmp_path), opened_empty=True)

        core.turned = True
        assert core.fresh
        core.written = True
        assert not core.fresh

    async def test_a_load_that_finds_it_stored_ends_it(self, tmp_path):
        core = ConversationCore("a", place(tmp_path), opened_empty=True)

        await core.switch("a", Around(FakeWorkflow("a")).parts())

        assert not core.fresh

    async def test_a_load_that_finds_a_given_id_empty_makes_it(self, tmp_path):
        core = ConversationCore("a", place(tmp_path))

        await core.switch("a", Around(FakeWorkflow()).parts())

        assert core.fresh

    async def test_a_new_conversation_is_and_an_opened_one_is_not(self, tmp_path):
        core = TestTheSwitch.core_in(tmp_path)
        core.written = True

        await core.switch("b", Around(FakeWorkflow()).parts())
        assert core.fresh and not core.written
        await core.switch("c", Around(FakeWorkflow("c")).parts(), stored=True)
        assert not core.fresh


class TestTheSwitchAtOnce:
    """``switch_now``: what the switch does without an await (the deprecated
    ``session_id`` setter)."""

    def test_it_holds_the_target_and_starts_over_without_the_workflow(self, tmp_path):
        core, around = TestTheSwitch.core_in(tmp_path), Around(FakeWorkflow())
        core.turned = True

        assert core.switch_now("b", around.parts()) is None

        assert (core.session_id, core.turned) == ("b", False)
        assert held_elsewhere(place(tmp_path), "b") == os.getpid()
        assert held_elsewhere(place(tmp_path), "a") is None
        assert around.workflow.calls == []
        assert around.done == ["tasks reset", "usage reset", "status"]

    def test_one_open_elsewhere_is_refused_and_nothing_changes(self, tmp_path):
        core, around = TestTheSwitch.core_in(tmp_path), Around()
        ConversationLock(tmp_path).take(place(tmp_path).key("b"))

        assert core.switch_now("b", around.parts()) == os.getpid()
        assert core.session_id == "a" and around.done == []

    def test_none_keeps_the_conversation_held(self, tmp_path):
        """No conversation named: the workflow's own, the one it loaded last."""
        core, around = TestTheSwitch.core_in(tmp_path), Around()
        core.turned = True

        assert core.switch_now(None, around.parts()) is None

        assert (core.session_id, core.turned) == (None, True)
        assert held_elsewhere(place(tmp_path), "a") == os.getpid()
        assert around.done == []


def _row():
    async def row():
        return object()

    return row()


class TestTheAppsConversation:
    def test_the_apps_id_and_turned_are_the_cores(self, home):
        app = build("abc")

        assert app._conversation.session_id == "abc" == app.session_id
        app._turned = True
        app._session_id = "xyz"

        assert (app._conversation.session_id, app._conversation.turned) == ("xyz", True)

    def test_its_conversations_are_keyed_by_the_settings_place(self, home):
        app = build()

        assert app._conversation_key("abc") == Place.of(app.settings).key("abc")

    def test_an_app_built_with_new_has_one_too(self):
        app = BaseCLIApp.__new__(BaseCLIApp)

        assert app._turned is False
        app._session_id = "s"
        app._turned = True

        assert (app._conversation.session_id, app._conversation.turned) == ("s", True)
