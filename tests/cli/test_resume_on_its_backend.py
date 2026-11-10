"""A conversation opens on its own backend: the app follows the conversation.

A conversation belongs to one backend for its whole life, and there is no
shared session format. ``/resume <id>``, the ``/resume`` picker,
``--resume <id>`` and ``--continue`` find the conversation on whichever
installed backend keeps it, and a conversation another backend keeps is
opened there: the switch every change of conversation makes
(``ConversationCore.switch``), at the conversation's place (its backend),
with the controller starting that backend first (``switch_backend``).

In order: the conversation is held where it is kept (one another process
has open is refused before anything starts), the memories of the
conversation left are proposed while its backend still runs, the
conversation's backend starts on the run's services (handed over: the
session grants, jobs, sandbox kernels, knowledge bases and embedding model
are the run's, not a backend's), and the switch is made there. A backend
that fails to start, or a Ctrl+C before it runs, changes nothing; a Ctrl+C
once it runs (the old one being released) opens the conversation there,
then the cancel goes on.

An id two backends keep (left by the live switch of backend an earlier
version made) opens on the running backend, and the app says the other has
one too; the picker's row names its backend, and that one opens.

Most tests run the app on fake managers of each backend (the controller
builds them) over each backend's stored conversations, which the other
backends' fake readers read too. The last class runs the real ADK and Deep
Agents managers on sqlite, offline.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import os
import threading

import pytest

pytest.importorskip("google.adk")
pytest.importorskip("deepagents")

from agentic_cli.cli import AppInfo  # noqa: E402
from agentic_cli.cli.app import BaseCLIApp  # noqa: E402
from agentic_cli.cli.conversation import Place  # noqa: E402
from agentic_cli.cli.conversation_lock import ConversationLock  # noqa: E402
from agentic_cli.cli.session_args import SessionStart  # noqa: E402
from agentic_cli.cli.session_resume import PickedConversation  # noqa: E402
from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.memory import MockEmbeddingService  # noqa: E402
from agentic_cli.tools import job_list  # noqa: E402
from agentic_cli.tools.jobs.manager import JobState  # noqa: E402
from agentic_cli.tools.knowledge_tools import kb_search  # noqa: E402
from agentic_cli.tools.sandbox import sandbox_execute  # noqa: E402
from agentic_cli.workflow import memory_services  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.deepagents import manager as deepagents_manager  # noqa: E402
from agentic_cli.workflow.permissions.rules import Effect, Rule, RuleSource  # noqa: E402
from agentic_cli.workflow.service_registry import (  # noqa: E402
    EMBEDDING_SERVICE,
    JOB_MANAGER,
    KNOWLEDGE_BASE,
    PERMISSION_ENGINE,
    SANDBOX_MANAGER,
)
from tests.cli.dialog_fakes import autospec_session, opening_boxes  # noqa: E402
from tests.cli.test_compact_and_new import app_on, quiet_app, said  # noqa: E402,F401
from tests.cli import test_conversation_switch  # noqa: E402
from tests.cli.test_resume_at_startup import Session  # noqa: E402
from tests.cli.test_resume_session import conversation, replayed  # noqa: E402
from tests.workflow.deepagents.fakes import ScriptedModels, ai  # noqa: E402
from tests.workflow.test_adk_compaction import (  # noqa: E402,F401
    PROMPTS,
    Claude,
    project,
    says,
    serve,
)

#: A fake manager of one backend (``fails`` makes its start fail).
Manager = test_conversation_switch.TestAPlaceChangeWhileTheWorkflowIsDown.Manager


class StoreReader:
    """A backend's reader over the conversations its fake manager keeps."""

    def __init__(self, backend: str, stored: dict) -> None:
        self.backend, self.stored, self.closed = backend, stored, False

    def _row(self, session_id):
        info, _ = self.stored[session_id]
        return dataclasses.replace(info, backend=self.backend)

    async def list(self, limit=None):
        rows = sorted((self._row(s) for s in self.stored), key=lambda r: r.updated, reverse=True)
        return rows if limit is None else rows[:limit]

    async def info(self, session_id):
        return self._row(session_id) if session_id in self.stored else None

    async def exists(self, session_id, *, user_id=None):
        return session_id in self.stored

    async def delete(self, session_id):
        return self.stored.pop(session_id, None) is not None

    async def aclose(self):
        self.closed = True


class Backends:
    """Each backend's stored conversations, the fake managers the controller
    builds over them, and the readers the app opens on them."""

    def __init__(self, monkeypatch) -> None:
        self.stored: dict[str, dict] = {"adk": {}, "deepagents": {}}
        self.built: list[Manager] = []
        self.fails: set[str] = set()
        self.unreadable: dict[str, Exception] = {}
        self.unreadable_once: dict[str, Exception] = {}
        self.readers: list[StoreReader] = []
        monkeypatch.setattr(
            "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings",
            self.create,
        )
        monkeypatch.setattr(
            "agentic_cli.cli.stored_conversations.open_session_reader", self.open
        )

    def store(self, backend: str, session_id: str, title: str, messages: int, *,
              hours: float = 2.0) -> None:
        self.stored[backend][session_id] = conversation(session_id, title, messages, hours=hours)

    def manager(self, backend: str) -> Manager:
        manager = Manager(backend, fails=backend in self.fails)
        manager.stored = self.stored[backend]
        manager.session_store = "sqlite"
        return manager

    def create(self, agent_configs, settings, app_name=None, model=None, orchestrator=None,
               session_store=None):
        backend = orchestrator or settings.orchestrator.value
        manager = self.manager(backend)
        self.built.append(manager)
        return manager

    async def open(self, backend, settings, *, store=None):
        if store == "memory":
            return None
        if backend in self.unreadable:
            raise self.unreadable[backend]
        if backend in self.unreadable_once:
            raise self.unreadable_once.pop(backend)
        reader = StoreReader(backend, self.stored[backend])
        self.readers.append(reader)
        return reader


@pytest.fixture
def backends(project, monkeypatch) -> Backends:
    """HOME is the test's: the conversation locks live in its workspace."""
    monkeypatch.setenv("HOME", str(project))
    return Backends(monkeypatch)


async def on_adk(backends: Backends, **settings) -> tuple:
    """An app running ADK, in its stored conversation ``a1``."""
    adk = backends.manager("adk")
    adk.is_initialized = True
    app = app_on(adk, BaseSettings(session_store="sqlite", **settings))
    assert await app.resume_session("a1")
    app.session.reset_mock()
    return app, adk


def held_by(app, backend: str, session_id: str) -> tuple:
    return dataclasses.replace(app._place(), backend=backend).key(session_id)


def free(app, key: tuple) -> bool:
    """Whether another process could open the conversation ``key`` names now."""
    other = ConversationLock(app.settings.sessions_dir / "locks")
    taken = other.take(key) is None
    other.release()
    return taken


def running(app) -> str:
    return app._workflow_controller.workflow.backend_type


@pytest.fixture
def two(backends):
    backends.store("adk", "a1", "On ADK", 2)
    backends.store("deepagents", "d1", "On Deep Agents", 4, hours=1)
    return backends


class TestResumeOnAnotherBackend:
    async def test_it_opens_there(self, two):
        app, adk = await on_adk(two)

        await app.process_input("/resume d1")

        assert (app.session_id, app.session_backend, running(app)) == (
            "d1", "deepagents", "deepagents"
        )
        (deepagents,) = two.built
        assert deepagents.calls == ["load_session d1"]
        adk.cleanup.assert_awaited_once()
        assert not free(app, held_by(app, "deepagents", "d1"))
        assert free(app, held_by(app, "adk", "a1"))
        assert replayed(app) == [
            ("user", "/resume d1"),
            ("system", "Resumed on deepagents: On Deep Agents (d1, 2 turns, last active 1h ago)."),
            ("user", "d1 message 0"), ("response", "d1 message 1"),
            ("user", "d1 message 2"), ("response", "d1 message 3"),
        ]
        assert two.readers and all(reader.closed for reader in two.readers)
        await app.process_input("Hello.")
        assert deepagents.turns == ["d1"]

    async def test_the_memories_of_the_one_left_are_proposed_while_its_backend_runs(
        self, two
    ):
        app, adk = await on_adk(two, auto_extract_session_facts=True)
        await app.process_input("Hello.")

        await app.process_input("/resume d1")

        assert adk.ended == [adk.session_ref("a1", app.settings.default_user)]
        (deepagents,) = two.built
        assert "on_session_end" not in deepagents.calls

    async def test_a_backend_that_fails_to_start_changes_nothing(self, two):
        two.fails.add("deepagents")
        app, adk = await on_adk(two)

        await app.process_input("/resume d1")

        assert (app.session_id, running(app)) == ("a1", "adk")
        assert app._workflow_controller.workflow is adk
        assert said(app.session, "add_error") == [
            "Could not open 'd1': deepagents did not start: "
            "deepagents cannot start (its extra is missing)"
        ]
        assert not free(app, held_by(app, "adk", "a1"))
        assert free(app, held_by(app, "deepagents", "d1"))

    async def test_one_open_in_another_process_there_is_refused_before_it_starts(self, two):
        app, _ = await on_adk(two)
        other = ConversationLock(app.settings.sessions_dir / "locks")
        assert other.take(held_by(app, "deepagents", "d1")) is None
        try:
            await app.process_input("/resume d1")
        finally:
            other.release()

        assert two.built == [] and (app.session_id, running(app)) == ("a1", "adk")
        assert said(app.session, "add_error") == [
            f"Could not open 'd1': it is open in another process ({os.getpid()})."
        ]

    async def test_ctrl_c_before_its_backend_runs_changes_nothing(self, two, monkeypatch):
        app, _ = await on_adk(two)
        starting = asyncio.Event()

        async def initialize_services():
            starting.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(Manager, "initialize_services", lambda self: initialize_services())
        resuming = asyncio.create_task(app.process_input("/resume d1"))
        await asyncio.wait_for(starting.wait(), 5)
        resuming.cancel()
        with pytest.raises(asyncio.CancelledError):
            await resuming

        assert (app.session_id, running(app)) == ("a1", "adk")
        assert not free(app, held_by(app, "adk", "a1"))
        assert free(app, held_by(app, "deepagents", "d1"))

    async def test_ctrl_c_once_its_backend_runs_opens_it(self, two):
        app, adk = await on_adk(two)
        releasing, released = asyncio.Event(), asyncio.Event()

        async def cleanup():
            releasing.set()
            await released.wait()

        adk.cleanup = cleanup
        resuming = asyncio.create_task(app.process_input("/resume d1"))
        await asyncio.wait_for(releasing.wait(), 5)
        resuming.cancel()
        with pytest.raises(asyncio.CancelledError):
            await resuming
        released.set()

        assert (app.session_id, app.session_backend, running(app)) == (
            "d1", "deepagents", "deepagents"
        )
        assert not free(app, held_by(app, "deepagents", "d1"))
        await app.process_input("Hello.")
        assert two.built[0].turns == ["d1"]
        await app._workflow_controller.close()

    async def test_ctrl_c_during_its_load_there_says_where_the_app_is(self, two, monkeypatch):
        """The switch is made before the load (the conversation left is not
        on the new backend): a Ctrl+C there leaves the app in it, and it says
        so, and draws the status line, before the cancel goes on."""
        app, _ = await on_adk(two)
        gate, built = asyncio.Event(), []
        build = two.manager

        def manager(backend):
            made = build(backend)
            made.load_gate = gate
            built.append(made)
            return made

        monkeypatch.setattr(two, "manager", manager)
        resuming = asyncio.create_task(app.process_input("/resume d1"))
        for _ in range(500):
            if built and built[0].loading.is_set():
                break
            await asyncio.sleep(0.01)
        assert built[0].loading.is_set()
        app.session.set_status.reset_mock()
        resuming.cancel()
        with pytest.raises(asyncio.CancelledError):
            await resuming

        assert (app.session_id, app.session_backend) == ("d1", "deepagents")
        assert said(app.session, "add_message")[-2:] == [
            "Resumed on deepagents: On Deep Agents (d1, 2 turns, last active 1h ago).",
            "Operation cancelled...",
        ]
        assert app.session.set_status.called
        gate.set()
        await app._workflow_controller.close()

    async def test_one_deleted_after_the_look_is_refused_before_its_backend_starts(
        self, two, monkeypatch
    ):
        """Another CLI's ``/sessions --delete`` between the look and the hold:
        looked at again once held, it is gone, and nothing starts."""
        from agentic_cli.cli import stored_conversations

        app, adk = await on_adk(two)
        looked = stored_conversations.StoredConversations.__aexit__

        async def deleted_after_the_look(self, *exc):
            two.stored["deepagents"].pop("d1", None)
            return await looked(self, *exc)

        monkeypatch.setattr(
            stored_conversations.StoredConversations, "__aexit__", deleted_after_the_look
        )
        await app.process_input("/resume d1")

        assert two.built == [] and (app.session_id, running(app)) == ("a1", "adk")
        assert said(app.session, "add_error") == ["No conversation 'd1' found."]
        assert free(app, held_by(app, "deepagents", "d1"))
        assert not free(app, held_by(app, "adk", "a1"))

    async def test_one_deleted_as_its_backend_starts_opens_a_new_one_there(
        self, two, monkeypatch
    ):
        """The last resort (a deleter that holds no lock, not another CLI):
        the backend runs, and the conversation left is not on it, so the app
        opens a new conversation there, never the id deleted."""
        app, _ = await on_adk(two)
        build = two.manager

        def manager(backend):
            made = build(backend)

            async def initialize_services():
                two.stored["deepagents"].pop("d1", None)
                made.is_initialized = True

            made.initialize_services = initialize_services
            return made

        monkeypatch.setattr(two, "manager", manager)
        await app.process_input("/resume d1")

        new = app.session_id
        assert new not in ("d1", "a1") and app.session_backend == "deepagents"
        assert said(app.session, "add_error") == [
            f"No conversation 'd1' found on deepagents: it was deleted as it opened. "
            f"New conversation {new} on deepagents."
        ]
        await app.process_input("Hello.")
        assert two.built[0].turns == [new]
        assert free(app, held_by(app, "deepagents", "d1"))

    async def test_not_found_anywhere_says_which_backends_were_not_read(self, two):
        two.unreadable["deepagents"] = RuntimeError("database is locked")
        app, _ = await on_adk(two)

        await app.process_input("/resume zzz")

        assert said(app.session, "add_error") == [
            "No conversation 'zzz' found. "
            "Could not read the conversations on deepagents: database is locked."
        ]


class TestAnIdTwoBackendsKeep:
    async def test_the_running_backends_opens_and_the_other_is_named(self, backends):
        backends.store("adk", "a1", "On ADK", 2)
        backends.store("adk", "x", "X on ADK", 2)
        backends.store("deepagents", "x", "X on Deep Agents", 2, hours=1)
        app, _ = await on_adk(backends)

        await app.process_input("/resume x")

        assert (app.session_id, running(app)) == ("x", "adk") and backends.built == []
        assert said(app.session, "add_message")[1].startswith("Resumed: X on ADK (x,")
        assert said(app.session, "add_message")[-1] == (
            "'x' is also a conversation on deepagents: /resume lists it."
        )

    async def test_the_one_picked_opens_on_its_backend(self, backends):
        backends.store("adk", "a1", "On ADK", 2)
        backends.store("adk", "x", "X on ADK", 2)
        backends.store("deepagents", "x", "X on Deep Agents", 2, hours=1)
        app, _ = await on_adk(backends)

        assert await app.resume_session(PickedConversation("x", "deepagents"))

        assert (app.session_id, app.session_backend) == ("x", "deepagents")
        assert said(app.session, "add_message")[0].startswith(
            "Resumed on deepagents: X on Deep Agents (x,"
        )

    async def test_the_picked_twin_of_the_one_it_is_in_starts_over(self, backends):
        """Not the open conversation moved: another one, with usage counters of
        its own."""
        backends.store("adk", "a1", "On ADK", 2)
        backends.store("adk", "x", "X on ADK", 2)
        backends.store("deepagents", "x", "X on Deep Agents", 2, hours=1)
        app, _ = await on_adk(backends)
        assert await app.resume_session("x")
        app.usage_tracker.record({"prompt_tokens": 1234, "completion_tokens": 5})
        app._conversation.written = True

        assert await app.resume_session(PickedConversation("x", "deepagents"))

        assert (app.session_id, app.session_backend) == ("x", "deepagents")
        assert (app.usage_tracker.prompt_tokens, app.usage_tracker.invocation_count) == (0, 0)
        assert app._conversation.written is False
        assert app._conversation.fresh is False  # stored there

    async def test_already_in_it_names_the_other(self, backends):
        backends.store("adk", "a1", "On ADK", 2)
        backends.store("deepagents", "a1", "A1 on Deep Agents", 2, hours=1)
        app, adk = await on_adk(backends)
        adk.calls.clear()

        assert await app.resume_session("a1") is False

        assert said(app.session, "add_message") == [
            "Already in this conversation.",
            "'a1' is also a conversation on deepagents: /resume lists it.",
        ]
        assert adk.calls == []


class TestAtStartup:
    """``--resume <id>`` and ``--continue`` find the conversation on every
    installed backend, and open it on its own."""

    @staticmethod
    def app(start: SessionStart) -> tuple[BaseCLIApp, Session]:
        session = Session(until=lambda s: any(t.startswith("Resumed") for t in s.said("system")))
        app = BaseCLIApp(
            app_info=AppInfo(name="Startup test", version="0"),
            agent_configs=[AgentConfig(name="assistant", prompt="Help.")],
            settings=BaseSettings(session_store="sqlite"),
        )
        app.session = session
        app.session_start = start
        return app, session

    async def test_resume_opens_it_on_its_backend(self, two):
        app, session = self.app(SessionStart("resume", "d1"))
        session.inputs = ["Hello."]

        await app.run()

        assert app.exit_status == 0
        assert session.said("system")[0].startswith("Resumed: On Deep Agents (d1,")
        assert app._workflow_controller.backend == "deepagents"
        deepagents = next(m for m in two.built if m.backend_type == "deepagents")
        assert deepagents.turns == ["d1"]

    async def test_continue_opens_the_newest_of_every_backend(self, two):
        app, session = self.app(SessionStart("continue"))
        session.inputs = ["Hello."]

        await app.run()

        assert session.said("system")[0].startswith("Resumed: On Deep Agents (d1,")
        deepagents = next(m for m in two.built if m.backend_type == "deepagents")
        assert deepagents.turns == ["d1"]

    async def test_continue_passes_over_one_open_elsewhere_on_another_backend(self, two):
        app, session = self.app(SessionStart("continue"))
        other = ConversationLock(app.settings.sessions_dir / "locks")
        place = dataclasses.replace(Place.of(app.settings), backend="deepagents")
        assert other.take(place.key("d1")) is None
        try:
            await app.run()
        finally:
            other.release()

        assert session.said("warning")[0] == (
            f"The newest conversation, 'd1', is open in another process ({os.getpid()}); "
            "continuing 'a1'."
        )
        assert session.said("system")[0].startswith("Resumed: On ADK (a1,")


class TestTheStartPicksTheBackendFirst:
    """``--resume <id>`` and ``--continue`` find the conversation's backend
    from every installed backend's store before the workflow is built, so
    the run builds that backend once: not the settings', then the other."""

    app = staticmethod(TestAtStartup.app)

    @staticmethod
    def built(backends: Backends) -> list[str]:
        return [manager.backend_type for manager in backends.built]

    async def test_resume_builds_its_backend_alone(self, two):
        app, session = self.app(SessionStart("resume", "d1"))

        await app.run()

        assert self.built(two) == ["deepagents"]
        assert session.said("system")[0].startswith("Resumed: On Deep Agents (d1,")

    async def test_continue_builds_the_newest_ones_backend_alone(self, two):
        app, session = self.app(SessionStart("continue"))

        await app.run()

        assert self.built(two) == ["deepagents"]
        assert session.said("system")[0].startswith("Resumed: On Deep Agents (d1,")

    async def test_the_runs_own_new_conversation_is_on_that_backend(self, two):
        app, _ = self.app(SessionStart("resume", "d1"))
        before = app.session_id
        places: list = []
        adopt = app._adopt_session_on_startup

        async def adopting():
            places.append(app._place())
            return await adopt()

        app._adopt_session_on_startup = adopting

        await app.run()

        assert places[0].backend == "deepagents" and before != "d1"

    async def test_one_the_settings_backend_keeps_too_is_opened_there(self, backends):
        backends.store("adk", "x", "X on ADK", 2)
        backends.store("deepagents", "x", "X on Deep Agents", 2, hours=1)
        app, session = self.app(SessionStart("resume", "x"))

        await app.run()

        assert self.built(backends) == ["adk"]
        assert session.said("system")[0].startswith("Resumed: X on ADK (x,")
        assert "'x' is also a conversation on deepagents: /resume lists it." in (
            session.said("system")
        )

    async def test_a_target_nobody_has_starts_as_before(self, two):
        app, session = self.app(SessionStart("resume", "zzz"))

        await app.run()

        assert app.exit_status == 1 and self.built(two) == ["adk"]
        assert session.said("error") == ["No conversation 'zzz' found."]

    async def test_its_backend_failing_to_start_keeps_the_prompt_and_says_how(self, two):
        """The next start builds that backend still: fixed in /settings,
        ``/resume`` opens the conversation there."""
        two.fails.add("deepagents")
        app, session = self.app(SessionStart("resume", "d1"))
        session.until = None
        free_after: list[bool] = []
        adopt = app._adopt_session_on_startup

        async def adopting():
            adopted = await adopt()
            free_after.append(free(app, held_by(app, "deepagents", "d1")))
            return adopted

        app._adopt_session_on_startup = adopting

        await app.run()

        assert self.built(two) == ["deepagents"]
        assert app._workflow_controller.backend == "deepagents"
        assert session.said("warning")[0] == (
            "Could not open 'd1': its backend, deepagents, did not start. "
            "Fix it with /settings, then /resume d1."
        )
        assert free_after == [True]

    async def test_a_lookup_that_fails_before_the_build_is_made_again_after_it(self, two):
        """The start finds the target as it always did, on every backend:
        one another backend keeps opens there once the prompt is up."""
        two.unreadable_once["deepagents"] = RuntimeError("database is locked")
        app, session = self.app(SessionStart("resume", "d1"))

        await app.run()

        assert self.built(two) == ["adk", "deepagents"]
        assert session.said("system")[0].startswith("Resumed on deepagents: On Deep Agents (d1,")

    async def test_continue_whose_backend_fails_names_the_conversation(self, two):
        two.fails.add("deepagents")
        app, session = self.app(SessionStart("continue"))
        session.until = None

        await app.run()

        assert self.built(two) == ["deepagents"]
        assert session.said("warning")[0] == (
            "Could not continue 'd1': its backend, deepagents, did not start. "
            "Fix it with /settings, then /resume d1."
        )


# --- The real backends ---------------------------------------------------------------------


#: What a hand-over keeps: the session grants (in the engine), the kernels,
#: the jobs, the knowledge bases and the embedding model.
KEPT = (PERMISSION_ENGINE, SANDBOX_MANAGER, JOB_MANAGER, KNOWLEDGE_BASE, EMBEDDING_SERVICE)


class TestOnTheRealBackends:
    """``/resume`` onto the other backend and back, on the real ADK and Deep
    Agents managers (sqlite; ADK on Claude through ``FakeAPI``, Deep Agents
    on scripted models): the run's services are the same objects throughout,
    a job started on one backend runs on, and a session grant still holds."""

    @pytest.fixture
    def real(self, project, monkeypatch):
        serve(monkeypatch, Claude(solo=[says("A1."), says("A2.")]))
        monkeypatch.setattr(deepagents_manager, "build_chat_model",
                            ScriptedModels(solo=[ai("D1."), ai("D2.")]))
        made: list = []

        def embedding(settings):
            made.append(MockEmbeddingService(embedding_dim=16))
            return made[-1]

        monkeypatch.setattr(memory_services, "build_embedding_service", embedding)
        return made

    @staticmethod
    async def started() -> BaseCLIApp:
        app_settings = BaseSettings(
            session_store="sqlite", default_model="claude-sonnet-4-6",
            stateful_executor_backend="local", knowledge_base_use_mock=False,
        )
        app = BaseCLIApp(
            app_info=AppInfo(name="Backends test", version="0"),
            agent_configs=[AgentConfig(
                name="solo", prompt=PROMPTS["solo"], include_state_tools=False,
                tools=[sandbox_execute, job_list, kb_search],
            )],
            settings=app_settings,
        )
        app.session = opening_boxes(autospec_session())
        app._conversation.place = Place.of(app_settings)  # as run() records it
        await app._workflow_controller.start_background_init()
        assert await app._adopt_session_on_startup()
        return app

    @staticmethod
    def services(app) -> dict:
        workflow = app._workflow_controller.workflow
        return {key: workflow.services[key] for key in KEPT}

    async def test_the_runs_services_go_with_it_onto_the_other_backend_and_back(self, real):
        app = await self.started()
        release = threading.Event()
        try:
            await app.process_input("One on ADK.")
            on_adk = app.session_id
            held = self.services(app)
            grant = Rule("http.read", "https://example.com/**", Effect.ALLOW, RuleSource.SESSION)
            held[PERMISSION_ENGINE]._session_rules.append(grant)
            job = held[JOB_MANAGER].submit(
                tool="t", backend="inprocess", spec={"target": functools.partial(release.wait, 10)},
            )

            await app.process_input("/new deepagents")
            await app.process_input("One on Deep Agents.")
            on_deepagents = app.session_id
            assert self.services(app) == held

            await app.process_input(f"/resume {on_adk}")
            assert (app.session_id, app.session_backend) == (on_adk, "adk")
            assert self.services(app) == held, "the services were rebuilt on ADK"

            await app.process_input(f"/resume {on_deepagents}")
            assert (app.session_id, app.session_backend) == (on_deepagents, "deepagents")
            assert self.services(app) == held, "the services were rebuilt on Deep Agents"

            assert all(self.services(app)[k] is held[k] for k in KEPT)
            assert grant in held[PERMISSION_ENGINE]._session_rules
            assert held[PERMISSION_ENGINE]._workflow is app._workflow_controller.workflow
            jobs = held[JOB_MANAGER]
            assert (await asyncio.to_thread(jobs.get, job.job_id)).state is JobState.RUNNING
            assert len(real) == 1, "the embedding model loaded again"
            assert said(app.session, "add_error") == []
            replay = [text for kind, text in replayed(app) if kind in ("user", "response")]
            assert replay[-2:] == ["One on Deep Agents.", "D1."]
        finally:
            release.set()
            await app._workflow_controller.close()

    async def test_after_a_failed_model_switch_a_new_backend_keeps_the_runs_services(
        self, real, monkeypatch
    ):
        """The model switch failed: its manager is kept, holding the run's
        services, and the workflow is down. ``/new deepagents`` then: the next
        start builds Deep Agents on those services, given over."""
        app = await self.started()
        controller = app._workflow_controller
        release = threading.Event()
        try:
            await app.process_input("One on ADK.")
            held = self.services(app)
            job = held[JOB_MANAGER].submit(
                tool="t", backend="inprocess", spec={"target": functools.partial(release.wait, 10)},
            )

            async def fails(self) -> None:
                raise RuntimeError("model switch failed")

            with monkeypatch.context() as patched:
                patched.setattr(GoogleADKWorkflowManager, "_do_initialize", fails)
                with pytest.raises(RuntimeError, match="model switch failed"):
                    await controller.reinitialize(model="claude-opus-4-6")
            assert not controller.is_ready

            await app.process_input("/new deepagents")
            assert await controller.ensure_initialized() is True
            await controller._join_releases()

            assert controller.backend == "deepagents"
            assert all(self.services(app)[k] is held[k] for k in KEPT), "rebuilt on Deep Agents"
            jobs = held[JOB_MANAGER]
            assert (await asyncio.to_thread(jobs.get, job.job_id)).state is JobState.RUNNING
            assert len(real) == 1, "the embedding model loaded again"
        finally:
            release.set()
            await controller.close()

    async def test_a_failed_start_of_the_other_backend_changes_nothing(self, real, monkeypatch):
        app = await self.started()
        release = threading.Event()
        try:
            await app.process_input("One on ADK.")
            on_adk = app.session_id
            await app.process_input("/new deepagents")
            await app.process_input("One on Deep Agents.")
            on_deepagents, workflow = app.session_id, app._workflow_controller.workflow
            held = self.services(app)
            job = held[JOB_MANAGER].submit(
                tool="t", backend="inprocess", spec={"target": functools.partial(release.wait, 10)},
            )

            async def fails(self) -> None:
                raise RuntimeError("no key for ADK")

            monkeypatch.setattr(GoogleADKWorkflowManager, "_do_initialize", fails)
            await app.process_input(f"/resume {on_adk}")

            assert said(app.session, "add_error") == [
                f"Could not open '{on_adk}': adk did not start: no key for ADK"
            ]
            assert (app.session_id, app.session_backend) == (on_deepagents, "deepagents")
            assert app._workflow_controller.workflow is workflow
            assert self.services(app) == held
            jobs = held[JOB_MANAGER]
            assert (await asyncio.to_thread(jobs.get, job.job_id)).state is JobState.RUNNING
            await app.process_input("Two on Deep Agents.")
            assert said(app.session, "add_error")[1:] == []
        finally:
            release.set()
            await app._workflow_controller.close()


def test_a_picked_conversation_is_its_id():
    picked = PickedConversation("abc", "deepagents")

    assert picked == "abc" and isinstance(picked, str) and picked.backend == "deepagents"
    assert type(str(picked)) is str
