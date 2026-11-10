"""``session_store=memory``: the CLI keeps one conversation at a time.

The memory store writes no conversation to disk (privacy, demos, tests), so
nothing of a conversation outlives it: ``/new`` drops the conversation left
from the running backend's store (its ``delete_session``), after its
memories are proposed, which read it, and says so ("Previous: <id>, not kept
(session_store is memory)."); one never written has nothing to drop. A
``/new <backend>`` releases the manager that keeps it, store and all; a
backend that fails to start drops nothing. A finished job of a conversation
dropped is told as unavailable, as a deleted conversation's is. One left
while the workflow was down (the manager a failed reinitialization kept,
its store with it), or whose delete raised, is dropped once a workflow is
ready, before it is used: a turn, a delivery of finished jobs, ``/compact``
or the next ``/new``.

Nothing is kept to go back to, so ``/resume`` (with an id or the picker),
the app's ``resume_session`` and ``pick_session``, ``--resume <id>`` and
``--continue`` are refused with one line (``NOT_KEPT``), and nothing changes:
no conversation is looked up, loaded or held, and no backend is started.
``--resume <id>`` then exits with status 1, as for an id that is not stored;
``--continue`` starts a new conversation and says why, once. ``/sessions``
says the same, with no table (the one row it could show is the conversation
the app is in, which ``/status`` names), and ``--delete`` is refused.

A persistent store (sqlite, the default; postgres) keeps every conversation,
and the run never switches the store by itself.

The app is a real ``BaseCLIApp`` whose session is a fake and whose controller
holds a fake workflow over an in-memory store (``tests/cli/test_resume_session.py``),
or the real ADK and Deep Agents managers on the memory store.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("google.adk")

from agentic_cli.cli.message_processor import CANCELLED_LINE  # noqa: E402
from agentic_cli.cli.session_args import SessionStart  # noqa: E402
from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from tests.cli.test_compact_and_new import (  # noqa: E402,F401
    Claude,
    app_on,
    project,
    quiet_app,
    said,
    says,
    serve,
    solo,
)
from tests.cli import test_conversation_switch as switch  # noqa: E402
from tests.cli import test_resume_at_startup as startup  # noqa: E402
from tests.cli.console_app import ScriptedManager  # noqa: E402
from tests.cli.test_resume_at_startup import manager  # noqa: E402,F401
from tests.cli.test_resume_session import Workflow as StoredWorkflow  # noqa: E402
from tests.cli.test_resume_session import conversation  # noqa: E402
from tests.cli.test_waiting_jobs import Jobs, finished  # noqa: E402

DROPPED = "not kept (session_store is memory)"
NOT_KEPT = (
    "session_store is memory: conversations are not kept. "
    "Set it to sqlite to keep and resume them (applies at the next start)."
)


class Workflow(StoredWorkflow):
    """A workflow over an in-memory store: a turn stores its conversation,
    ``delete_session`` drops it, and a job's conversation can be resumed
    while it is stored."""

    supports_job_resume = True

    def __init__(self, *stored, **kwargs) -> None:
        super().__init__(*stored, **kwargs)
        #: Set to hold ``delete_session`` until it is set.
        self.delete_gate: asyncio.Event | None = None
        self.deleting = asyncio.Event()

    async def process(self, message, user_id, session_id=None):
        self.stored.setdefault(session_id, conversation(session_id, message, 2))
        async for event in super().process(message, user_id, session_id):
            yield event

    async def session_exists(self, session_id, *, user_id=None) -> bool:
        return session_id in self.stored

    async def delete_session(self, session_id, *, user_id=None) -> bool:
        self.deleting.set()
        if self.delete_gate is not None:
            await self.delete_gate.wait()
        return await super().delete_session(session_id, user_id=user_id)

    async def can_resume(self, record) -> bool:
        return record.session_id in self.stored


def on_memory(*stored, **values):
    """An app on the memory store, running ``Workflow`` over ``stored``."""
    workflow = Workflow(*stored)
    app = app_on(workflow, BaseSettings(session_store="memory", **values))
    return app, workflow


class TestNewDropsTheConversationLeft:
    async def test_it_is_dropped_and_the_line_says_so(self, project):
        app, workflow = on_memory()
        old = app.session_id
        await app.process_input("Hello.")

        await app.process_input("/new")

        new = app.session_id
        assert new != old and old not in workflow.stored
        assert workflow.calls[-2:] == [f"load_session {new}", f"delete_session {old}"]
        assert said(app.session, "add_message")[-1] == (
            f"New conversation {new}. Previous: {old}, {DROPPED}."
        )

    async def test_its_memories_are_proposed_first(self, project):
        """The proposals read the conversation: it is dropped after them."""
        app, workflow = on_memory(auto_extract_session_facts=True)
        old = app.session_id
        await app.process_input("Hello.")

        new = await app.start_new_session()

        assert workflow.calls == [
            "on_session_end", f"load_session {new}", f"delete_session {old}",
        ]
        assert workflow.ended == [workflow.session_ref(old, app.settings.default_user)]

    async def test_one_never_written_has_nothing_to_drop(self, project):
        app, workflow = on_memory()

        new = await app.start_new_session()

        assert workflow.calls == [f"load_session {new}"]
        assert said(app.session, "add_message") == [f"New conversation {new}."]

    async def test_the_next_one_is_dropped_in_its_turn(self, project):
        app, workflow = on_memory()
        await app.process_input("One.")
        await app.process_input("/new")
        second = app.session_id
        await app.process_input("Two.")

        await app.process_input("/new")

        assert list(workflow.stored) == []
        assert workflow.calls[-1] == f"delete_session {second}"

    async def test_ctrl_c_before_the_switch_is_made_drops_nothing(self, project):
        app, workflow = on_memory()
        workflow.load_gate = asyncio.Event()
        await app.process_input("Hello.")
        old = app.session_id

        starting = asyncio.create_task(app.process_input("/new"))
        await asyncio.wait_for(workflow.loading.wait(), 5)
        starting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(starting, 5)

        assert app.session_id == old and old in workflow.stored
        assert not any(c.startswith("delete_session") for c in workflow.calls)
        assert said(app.session, "add_message").count(CANCELLED_LINE) == 1

    async def test_ctrl_c_while_it_is_dropped_drops_it_all_the_same(self, project):
        """The switch is made: the drop runs to its end, and the line says
        where the app is."""
        app, workflow = on_memory()
        workflow.delete_gate = asyncio.Event()
        await app.process_input("Hello.")
        old = app.session_id

        starting = asyncio.create_task(app.process_input("/new"))
        await asyncio.wait_for(workflow.deleting.wait(), 5)
        starting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(starting, 5)
        workflow.delete_gate.set()
        for _ in range(100):
            if old not in workflow.stored:
                break
            await asyncio.sleep(0.01)

        new = app.session_id
        assert new != old and old not in workflow.stored
        lines = said(app.session, "add_message")
        assert f"New conversation {new}. Previous: {old}, {DROPPED}." in lines
        assert lines.count(CANCELLED_LINE) == 1

    async def test_a_drop_that_fails_never_stops_it(self, project):
        app, workflow = on_memory()
        await app.process_input("Hello.")
        old = app.session_id

        async def broken(session_id, *, user_id=None):
            raise RuntimeError("the store is gone")

        workflow.delete_session = broken

        new = await app.start_new_session()

        assert app.session_id == new != old
        assert said(app.session, "add_error") == []
        assert said(app.session, "add_message")[-1] == (
            f"New conversation {new}. Previous: {old}, {DROPPED}."
        )

    async def test_the_status_bar_then_asks_again_where_jobs_are(self, project):
        """It keeps a conversation's "can be opened" for a while: the one
        dropped can be opened no more."""
        from types import SimpleNamespace

        app, _ = on_memory()
        forgotten: list[bool] = []
        app._job_monitor = SimpleNamespace(forget_conversations=lambda: forgotten.append(True))
        await app.process_input("Hello.")

        await app.process_input("/new")

        assert forgotten == [True]

    async def test_a_persistent_store_keeps_it(self, project):
        """sqlite, the default: the conversation left stays in /sessions."""
        workflow = Workflow()
        app = app_on(workflow, BaseSettings(session_store="sqlite"))
        old = app.session_id
        await app.process_input("Hello.")

        new = await app.start_new_session()

        assert old in workflow.stored
        assert not any(c.startswith("delete_session") for c in workflow.calls)
        assert said(app.session, "add_message")[-1] == (
            f"New conversation {new}. Previous: {old}, listed in /sessions."
        )


class TestResumeIsRefused:
    """The conversation the store holds (``abc``, as an app's own code may
    leave one) is not opened either: nothing is kept to go back to."""

    @staticmethod
    def started(monkeypatch):
        built = switch.TestAPlaceChangeWhileTheWorkflowIsDown.builds(
            monkeypatch, lambda n, b: False
        )
        app, workflow = on_memory(conversation("abc", "Plan the trip", 2))
        return app, workflow, built

    async def test_resume_by_id(self, project, monkeypatch):
        app, workflow, built = self.started(monkeypatch)
        before = app.session_id

        await app.process_input("/resume abc")

        assert said(app.session, "add_error") == [NOT_KEPT]
        assert app.session_id == before and workflow.calls == [] and built == []

    async def test_the_picker(self, project, monkeypatch):
        app, workflow, built = self.started(monkeypatch)
        before = app.session_id

        await app.process_input("/resume")

        assert said(app.session, "add_error") == [NOT_KEPT]
        assert app.session.show_dialog.call_count == 0
        assert app.session_id == before and workflow.calls == [] and built == []

    async def test_the_apps_resume_session(self, project, monkeypatch):
        app, workflow, _ = self.started(monkeypatch)
        before = app.session_id

        assert await app.resume_session("abc") is False

        assert said(app.session, "add_error") == [NOT_KEPT]
        assert app.session_id == before and workflow.calls == []

    async def test_the_apps_pick_session(self, project, monkeypatch):
        app, workflow, _ = self.started(monkeypatch)

        assert await app.pick_session() is None

        assert said(app.session, "add_error") == [NOT_KEPT]
        assert app.session.show_dialog.call_count == 0 and workflow.calls == []

    async def test_while_the_workflow_is_down_it_says_why_all_the_same(
        self, project, monkeypatch
    ):
        """No workflow is needed to say so (``pick_session`` raises no
        "not initialized")."""
        app, _, _ = self.started(monkeypatch)
        controller = app._workflow_controller
        controller._workflow, controller._init_error = None, RuntimeError("no API key")

        await app.process_input("/resume")
        picked = await app.pick_session()

        assert picked is None
        assert said(app.session, "add_error") == [NOT_KEPT, NOT_KEPT]
        assert said(app.session, "add_warning") == []


class TestSessions:
    async def test_it_says_they_are_not_kept(self, project):
        app, _ = on_memory(conversation("abc", "Plan the trip", 2))
        await app.process_input("Hello.")

        await app.process_input("/sessions")

        assert said(app.session, "add_message")[-1] == NOT_KEPT
        assert app.session.add_rich.call_count == 0
        assert said(app.session, "add_error") == []

    async def test_delete_is_refused(self, project):
        app, workflow = on_memory(conversation("abc", "Plan the trip", 2))

        await app.process_input("/sessions --delete=abc")

        assert said(app.session, "add_error") == [NOT_KEPT]
        assert "abc" in workflow.stored and workflow.calls == []

    async def test_status_names_the_conversation_as_not_kept(self, project):
        """``/sessions`` lists none: ``/status`` names the one the app is in,
        and says it is not kept."""
        app, _ = on_memory()

        await app.process_input("/status")

        (panel,) = app.session.add_rich.call_args.args
        labels, values = (list(column.cells) for column in panel.renderable.columns)
        assert dict(zip(labels, values))["Session"] == (
            f"{app.session_id} (not kept: session_store is memory)"
        )

    async def test_while_the_workflow_is_down_it_says_why_all_the_same(self, project):
        app, _ = on_memory()
        controller = app._workflow_controller
        controller._workflow, controller._init_error = None, RuntimeError("no API key")

        await app.process_input("/sessions")

        assert said(app.session, "add_message")[-1] == NOT_KEPT
        assert said(app.session, "add_warning") == []


class TestAtStartup:
    """``--resume <id>`` exits with status 1, as for an id that is not
    stored: the user asked for that conversation, and a script sees that it
    did not open. ``--continue`` starts a new conversation, as when nothing
    is stored, and says why once. Neither looks anything up, nor waits for
    the workflow to say so."""

    @pytest.fixture
    def looked(self, monkeypatch) -> list[str]:
        """The backends whose stored conversations were looked at."""
        from agentic_cli.cli import stored_conversations

        looked: list[str] = []
        opener = stored_conversations.open_session_reader

        async def recording(backend, settings, *, store=None):
            looked.append(backend)
            return await opener(backend, settings, store=store)

        monkeypatch.setattr(stored_conversations, "open_session_reader", recording)
        return looked

    async def test_resume_by_id_is_refused_with_status_1(self, manager, looked):
        startup.store(manager, "abc", "Plan the trip", "Sure.", hours=2)
        session = startup.Session()
        app = startup.app_with(session, SessionStart("resume", "abc"), store="memory")
        app._command = "my-app"

        await app.run()

        assert app.exit_status == 1 and not session.prompted
        assert session.said("error") == [NOT_KEPT]
        assert not startup.resumed(session) and "Goodbye!" not in session.said("system")
        assert looked == []

    async def test_continue_starts_a_new_one_and_says_why(self, manager, looked):
        startup.store(manager, "abc", "Plan the trip", "Sure.", hours=2)
        session = startup.Session()
        app = startup.app_with(session, SessionStart("continue"), store="memory")
        fresh = app.session_id

        await app.run()

        assert app.session_id == fresh and session.prompted and app.exit_status == 0
        assert session.said("warning") == [f"{NOT_KEPT} Starting a new conversation."]
        assert not startup.resumed(session)
        assert not any(line.startswith("No conversation") for line in session.said("system"))
        assert looked == []

    async def test_continue_with_a_workflow_that_cannot_start_points_at_no_resume(
        self, manager, tmp_path, monkeypatch
    ):
        (tmp_path / "init-fails").touch()
        (tmp_path / "ready").touch()  # a later retry would start
        monkeypatch.setattr(ScriptedManager, "initializations", 0)
        session = startup.Session()
        app = startup.app_with(session, SessionStart("continue"), store="memory")

        await app.run()

        assert session.prompted and app.exit_status == 0
        assert session.said("warning") == [
            f"{NOT_KEPT} Starting a new conversation.",
            "The workflow did not start. Fix it with /settings; your next message "
            "starts it again.",
        ]


class TestNewOnAnotherBackend:
    """The conversation left is in the store of the manager the switch
    releases: it goes with that manager, and nothing else deletes it."""

    class Manager(switch.TestAPlaceChangeWhileTheWorkflowIsDown.Manager):
        """A backend's manager over an in-memory store."""

        async def process(self, message, user_id, session_id=None):
            self.stored.setdefault(session_id, conversation(session_id, message, 2))
            async for event in super().process(message, user_id, session_id):
                yield event

        async def session_exists(self, session_id, *, user_id=None) -> bool:
            return session_id in self.stored

    @classmethod
    def on_adk(cls, monkeypatch, *, fails: bool = False):
        built = switch.TestAPlaceChangeWhileTheWorkflowIsDown.builds(
            monkeypatch, lambda n, b: fails
        )
        adk = cls.Manager("adk", fails=False)
        adk.is_initialized = True
        adk.session_store = "memory"  # the store it was built for
        app = app_on(adk, BaseSettings(session_store="memory"))
        return app, adk, built

    async def test_it_goes_with_the_manager_released(self, project, monkeypatch):
        app, adk, built = self.on_adk(monkeypatch)
        old = app.session_id
        await app.process_input("Hello.")

        await app.process_input("/new deepagents")

        new = app.session_id
        assert app._workflow_controller.workflow is built[0]
        assert built[0].session_store == "memory"
        assert adk.cleanup.await_count == 1  # released, its store with it
        assert not any(c.startswith("delete_session") for c in adk.calls + built[0].calls)
        assert said(app.session, "add_message")[-1] == (
            f"New conversation {new} on deepagents. Previous: {old}, {DROPPED}."
        )

    async def test_a_backend_that_fails_to_start_drops_nothing(self, project, monkeypatch):
        app, adk, built = self.on_adk(monkeypatch, fails=True)
        old = app.session_id
        await app.process_input("Hello.")

        await app.process_input("/new deepagents")

        assert app.session_id == old and old in adk.stored
        assert app._workflow_controller.workflow is adk and adk.cleanup.await_count == 0
        assert not any(c.startswith("delete_session") for c in adk.calls)


class TestAJobOfADroppedConversation:
    """It is told as unavailable, as a deleted conversation's is: no store
    keeps its conversation, so it waits for none."""

    @staticmethod
    async def finished_in_the_dropped():
        jobs = Jobs([])
        app, workflow = on_memory()
        workflow.job_manager = jobs
        old = app.session_id
        await app.process_input("Start the build.")
        jobs.records.append(finished("build", old, app.settings.default_user))
        await app.process_input("/new")
        return app, workflow, jobs

    async def test_jobs_resume_tells_it_here_as_unavailable(self, project):
        app, _, jobs = await self.finished_in_the_dropped()

        await app.process_input("/jobs resume")

        assert jobs.claimed == ["build"]
        assert said(app.session, "add_message")[-1] == (
            "✓ Background job 'build' finished (succeeded) while its conversation was "
            "unavailable — fetch the result with /jobs build."
        )

    async def test_the_status_bar_counts_it_here(self, project):
        app, _, _ = await self.finished_in_the_dropped()

        assert await app._new_job_monitor().poll_once() == "↻1: /jobs resume"


UNAVAILABLE = (
    "✓ Background job 'build' finished (succeeded) while its conversation was "
    "unavailable — fetch the result with /jobs build."
)


class Deliveries(Jobs):
    """Jobs whose delivery moves as the real job manager's does: claimed
    (``resuming``) once, then delivered or failed (one told as unavailable
    fails), so a job is told once."""

    def _record(self, job_id: str):
        return next(record for record in self.records if record.job_id == job_id)

    def begin_resume(self, job_id: str) -> bool:
        record = self._record(job_id)
        if record.resume_state != "pending":
            return False
        self.claimed.append(job_id)
        record.resume_state = "resuming"
        return True

    def complete_resume(self, job_id: str, *, delivered: bool, error=None) -> None:
        self._record(job_id).resume_state = "delivered" if delivered else "failed"


class Recorded(Workflow):
    """``Workflow``, whose turns, summaries and starts are in ``calls`` too,
    in order with its loads and deletes; one a failed reinitialization kept
    starts again in place (``initialize_services``)."""

    async def process(self, message, user_id, session_id=None):
        self.calls.append(f"turn {session_id}")
        async for event in super().process(message, user_id, session_id):
            yield event

    async def compact(self, session, focus=None):
        self.calls.append(f"compact {session.session_id}")
        return await super().compact(session, focus)

    async def initialize_services(self, services=None) -> None:
        self.calls.append("initialize_services")
        self.is_initialized = True


class Deletes:
    """A workflow's ``delete_session``, counted, that raises while ``failing``."""

    def __init__(self, workflow: Workflow) -> None:
        self.delete = workflow.delete_session
        self.failing = False
        self.asked: list[str] = []
        workflow.delete_session = self

    async def __call__(self, session_id, *, user_id=None) -> bool:
        self.asked.append(session_id)
        if self.failing:
            raise RuntimeError("the store is busy")
        return await self.delete(session_id, user_id=user_id)


async def with_a_finished_job(store: str = "memory", **values):
    """An app running ``Recorded``, whose conversation had a turn that
    started a job, finished since."""
    jobs = Deliveries([])
    workflow = Recorded()
    workflow.job_manager = jobs
    app = app_on(workflow, BaseSettings(session_store=store, **values))
    old = app.session_id
    await app.process_input("Start the build.")
    jobs.records.append(finished("build", old, app.settings.default_user))
    return app, workflow, jobs, old


async def left_while_down(store: str = "memory", **values):
    """``with_a_finished_job``, then ``/new`` while the workflow is down: the
    manager a failed reinitialization kept (a ``/settings`` rebuild whose
    restore failed too), its store with it, so nothing could drop the
    conversation left."""
    app, workflow, jobs, old = await with_a_finished_job(store, **values)
    workflow.is_initialized = False  # FAILED: kept, its store with it
    assert not app._workflow_controller.is_ready
    await app.process_input("/new")
    assert app.session_id != old and old in workflow.stored
    return app, workflow, jobs, old


class TestDroppedOnceTheWorkflowIsBack:
    """``/new`` while the workflow was down dropped nothing: the kept manager
    comes back with the conversation left in its store, where nothing lists
    or opens it, and its job would never be told. The app drops it before
    the workflow is used again (a turn, a job's delivery, ``/compact``, the
    next ``/new``), and its job is told as unavailable, once."""

    async def test_the_next_turn_drops_it_and_its_job_is_told_unavailable(self, project):
        app, workflow, jobs, old = await left_while_down(job_auto_resume=True)
        new = app.session_id
        workflow.is_initialized = True  # revived, its store with it

        await app.process_input("Hello.")
        await app.process_input("Again.")

        assert not await workflow.session_exists(old)
        assert jobs.records[0].resume_state == "failed"
        assert workflow.calls[-3:] == [f"delete_session {old}", f"turn {new}", f"turn {new}"]
        assert said(app.session, "add_message").count(UNAVAILABLE) == 1

    async def test_a_workflow_that_starts_in_the_turn_drops_it_as_the_turn_ends(
        self, project
    ):
        """The turn starts it (``ensure_initialized`` revives the kept
        manager): it is dropped once the turn is done, so the status bar
        counts its job at once (``job_auto_resume`` off: no delivery follows)."""
        app, workflow, _, old = await left_while_down()
        new = app.session_id

        await app.process_input("Hello.")

        assert workflow.calls[-3:] == [
            "initialize_services", f"turn {new}", f"delete_session {old}",
        ]
        assert await app._new_job_monitor().poll_once() == "↻1: /jobs resume"

    async def test_jobs_resume_drops_it_first(self, project):
        app, workflow, jobs, old = await left_while_down()
        workflow.is_initialized = True

        await app.process_input("/jobs resume")

        assert not await workflow.session_exists(old)
        assert jobs.claimed == ["build"] and jobs.records[0].resume_state == "failed"
        assert said(app.session, "add_message")[-1] == UNAVAILABLE

    async def test_compact_drops_it_first(self, project):
        app, workflow, _, old = await left_while_down()
        new = app.session_id
        workflow.is_initialized = True

        await app.process_input("/compact")

        assert workflow.calls[-2:] == [f"delete_session {old}", f"compact {new}"]

    async def test_the_next_new_drops_it_first(self, project):
        app, workflow, _, old = await left_while_down()
        workflow.is_initialized = True

        await app.process_input("/new")

        newer = app.session_id
        assert workflow.calls[-2:] == [f"delete_session {old}", f"load_session {newer}"]
        assert list(workflow.stored) == []

    async def test_a_drop_that_keeps_failing_is_tried_once_at_each_point(self, project):
        """No loop: it waits for the next point; once dropped, never again."""
        app, workflow, _, old = await left_while_down()
        deletes = Deletes(workflow)
        deletes.failing = True
        workflow.is_initialized = True

        await app.process_input("One.")
        await app.process_input("Two.")

        assert deletes.asked == [old, old] and old in workflow.stored
        assert said(app.session, "add_error") == []

        deletes.failing = False
        await app.process_input("Three.")
        await app.process_input("Four.")

        assert deletes.asked == [old, old, old] and old not in workflow.stored

    async def test_one_the_store_has_no_more_is_done(self, project):
        """``delete_session`` says False (the manager that kept it was
        released): nothing to drop, and it is not asked again."""
        app, workflow, _, old = await left_while_down()
        deletes = Deletes(workflow)
        del workflow.stored[old]
        workflow.is_initialized = True

        await app.process_input("One.")
        await app.process_input("Two.")

        assert deletes.asked == [old]

    async def test_the_conversation_open_now_is_never_dropped(self, project):
        """The deprecated ``session_id`` setter can go back to a conversation
        ``/new`` left: it is not dropped under the app, and the next ``/new``
        drops it once."""
        import warnings

        app, workflow, _, old = await left_while_down()
        workflow.is_initialized = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            app.session_id = old

        await app.process_input("Back in the old one.")

        assert app.session_id == old and old in workflow.stored
        assert not any(c.startswith("delete_session") for c in workflow.calls)

        await app.process_input("/new")

        assert old not in workflow.stored and app._conversation.to_drop == []
        assert [c for c in workflow.calls if c.startswith("delete_session")] == [
            f"delete_session {old}"
        ]

    async def test_a_persistent_store_drops_nothing(self, project):
        """sqlite keeps the conversation left: nothing is dropped later either."""
        app, workflow, jobs, old = await left_while_down("sqlite", job_auto_resume=True)
        workflow.is_initialized = True

        await app.process_input("Hello.")
        await app.process_input("/jobs resume")
        await app.process_input("/new")

        assert old in workflow.stored
        assert not any(c.startswith("delete_session") for c in workflow.calls)
        assert jobs.claimed == [] and jobs.records[0].resume_state == "pending"

    async def test_on_adk_with_a_real_job_manager(self, project, monkeypatch):
        """The kept manager is a real ADK manager on the memory store, and the
        job a real one."""
        import time

        from agentic_cli.tools.jobs.manager import TERMINAL_STATES
        from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager
        from tests.cli.test_compact_and_new import (
            PROMPTS,
            new_session_probe_start_job,
            settings,
        )

        def finished_job(jm, session_id, user_id):
            record = jm.submit(
                tool="run_shell_job", backend="subprocess", spec={"command": "true"},
                name="build", resume_on_complete=True, call_id="c1", call_name="build",
                session_id=session_id, user_id=user_id,
            )
            deadline = time.monotonic() + 10
            while jm.get(record.job_id).state not in TERMINAL_STATES:
                assert time.monotonic() < deadline
                time.sleep(0.02)
            return record

        serve(monkeypatch, Claude(solo=[says("Hi."), says("Hi again.")]))
        manager = GoogleADKWorkflowManager(
            agent_configs=[AgentConfig(name="solo", prompt=PROMPTS["solo"],
                                       tools=[new_session_probe_start_job],
                                       include_state_tools=False)],
            settings=settings(session_store="memory", job_auto_resume=True),
        )
        await manager.initialize_services()
        app = app_on(manager, manager._settings)
        try:
            old = app.session_id
            await app.process_input("Hello.")
            jm = manager.job_manager
            record = await asyncio.to_thread(
                finished_job, jm, old, app.settings.default_user
            )
            manager._initialized = False  # FAILED: kept, its store with it
            await app.process_input("/new")
            manager._initialized = True  # revived

            await app.process_input("Again.")
            stored = await manager.session_exists(old)
            state = jm.get(record.job_id).resume_state
            await app.process_input("/jobs resume")
        finally:
            await manager.cleanup()

        lines = said(app.session, "add_message")
        told = [line for line in lines if "while its conversation was unavailable" in line]
        assert not stored and state == "failed" and len(told) == 1
        assert lines[-1] == "No finished background jobs to resume."


class TestDroppedOnceItsDeleteWorks:
    """``/new`` whose delete raised left the conversation in the store, as
    when the workflow was down: it is dropped at the next point once deletes
    work again, and its job is told as unavailable, once."""

    async def test_the_next_turn_drops_it_and_its_job_is_told_unavailable(self, project):
        app, workflow, jobs, old = await with_a_finished_job(job_auto_resume=True)
        deletes = Deletes(workflow)
        deletes.failing = True
        await app.process_input("/new")
        new = app.session_id
        assert new != old and old in workflow.stored
        deletes.failing = False

        await app.process_input("Hello.")
        await app.process_input("Again.")

        assert not await workflow.session_exists(old)
        assert jobs.records[0].resume_state == "failed"
        assert deletes.asked == [old, old]
        assert workflow.calls[-3:] == [f"delete_session {old}", f"turn {new}", f"turn {new}"]
        assert said(app.session, "add_message").count(UNAVAILABLE) == 1

    async def test_one_whose_delete_fails_after_a_ctrl_c_is_dropped_all_the_same(
        self, project
    ):
        """Ctrl+C stops the wait, not the delete, which then raises: the
        conversation is still dropped at the next point."""
        app, workflow, jobs, old = await with_a_finished_job(job_auto_resume=True)
        gate, failed = asyncio.Event(), asyncio.Event()

        async def stuck(session_id, *, user_id=None) -> bool:
            workflow.deleting.set()
            await gate.wait()
            failed.set()
            raise RuntimeError("the store is busy")

        workflow.delete_session = stuck
        starting = asyncio.create_task(app.process_input("/new"))
        await asyncio.wait_for(workflow.deleting.wait(), 5)
        starting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(starting, 5)
        gate.set()
        await asyncio.wait_for(failed.wait(), 5)
        for _ in range(5):  # the drop's task ends
            await asyncio.sleep(0)
        del workflow.delete_session  # deletes work again
        assert app.session_id != old and old in workflow.stored

        await app.process_input("Hello.")

        assert not await workflow.session_exists(old)
        assert jobs.records[0].resume_state == "failed"
        assert said(app.session, "add_message").count(UNAVAILABLE) == 1


class TestDroppedWhenItsLookupFailed:
    """``/new`` cannot tell whether the conversation left is stored (its
    ``session_exists`` raised): it is dropped at the next point, as one
    left while the workflow was down."""

    async def test_the_next_turn_drops_it_and_its_job_is_told_unavailable(self, project):
        app, workflow, jobs, old = await with_a_finished_job(job_auto_resume=True)

        async def broken(session_id, *, user_id=None) -> bool:
            raise RuntimeError("the lookup failed")

        workflow.session_exists = broken
        await app.process_input("/new")
        new = app.session_id
        del workflow.session_exists  # lookups work again
        assert new != old and old in workflow.stored

        await app.process_input("Hello.")
        await app.process_input("Again.")

        assert not await workflow.session_exists(old)
        assert jobs.records[0].resume_state == "failed"
        assert workflow.calls[-3:] == [f"delete_session {old}", f"turn {new}", f"turn {new}"]
        assert said(app.session, "add_message").count(UNAVAILABLE) == 1

    async def test_a_workflow_without_sessions_has_none_to_drop(self, project):
        """No session store (``supports_sessions`` False, whose hooks raise
        ``NotImplementedError``): nothing is asked, at any point."""
        app, workflow, _, _ = await with_a_finished_job()
        workflow.supports_sessions = False
        asked: list[str] = []

        async def unsupported(session_id, *, user_id=None) -> bool:
            asked.append(session_id)
            raise NotImplementedError("this backend keeps no sessions")

        workflow.delete_session = unsupported

        await app.process_input("/new")
        await app.process_input("One.")
        await app.process_input("Two.")

        assert asked == [] and app._conversation.to_drop == []


class TestOnTheRealBackends:
    async def test_adk_drops_it_and_its_job_is_told_unavailable(self, project, monkeypatch):
        from types import SimpleNamespace

        from agentic_cli.tools.jobs import JobState

        serve(monkeypatch, Claude(solo=[says("Hello back.")]))
        manager = solo()  # on the memory store
        await manager.initialize_services()
        app = app_on(manager, manager._settings)
        try:
            old = app.session_id
            await app.process_input("Hello.")
            assert await manager.session_exists(old)
            record = SimpleNamespace(
                job_id="j1", name="build", tool="build", call_id="c1", call_name="build",
                session_id=old, user_id=app.settings.default_user,
                state=JobState.SUCCEEDED, exit_code=0, error=None,
            )

            new = await app.start_new_session()
            stored = await manager.session_exists(old)
            resumable = await manager.can_resume(record)
            told = await app._message_processor.process_resume(
                record, app._workflow_controller, app.session, app.settings,
                usage_tracker=app.usage_tracker,
            )
        finally:
            await manager.cleanup()

        assert not stored and not resumable and not told.delivered
        assert said(app.session, "add_message")[-2:] == [
            f"New conversation {new}. Previous: {old}, {DROPPED}.",
            "✓ Background job 'build' finished (succeeded) while its conversation was "
            "unavailable — fetch the result with /jobs j1.",
        ]

    async def test_deep_agents_drops_it(self, project, monkeypatch):
        pytest.importorskip("deepagents")
        from agentic_cli.workflow.deepagents import manager as deepagents_manager
        from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager
        from tests.conftest import MockContext
        from tests.workflow.deepagents.fakes import ScriptedModels, ai

        monkeypatch.setattr(deepagents_manager, "build_chat_model",
                            ScriptedModels(solo=[ai("A1."), ai("B1.")]))
        with MockContext(
            google_api_key="test-key", session_store="memory", orchestrator="deepagents"
        ) as context:
            manager = DeepAgentsWorkflowManager(
                agent_configs=[AgentConfig(name="solo", prompt="You help.")],
                settings=context.settings, model="gemini-2.5-flash",
            )
            await manager.initialize_services()
            app = app_on(manager, context.settings)
            try:
                old = app.session_id
                await app.process_input("One in A.")
                assert await manager.session_exists(old)

                await app.process_input("/new")
                await app.process_input("One in B.")
                stored = [row.session_id for row in await manager.list_sessions()]
            finally:
                await manager.cleanup()

        assert stored == [app.session_id] != [old]
