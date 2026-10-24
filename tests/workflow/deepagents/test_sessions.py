"""Session identity and the session store on Deep Agents (``workflow/deepagents/sessions.py``).

A conversation's thread is keyed by the whole ``SessionRef``: ``thread_id``
is ``json.dumps([app_name, user_id, session_id])``, and the same three values
go into every checkpoint's metadata. ``CheckpointSessions`` answers the
session hooks from the checkpointer:

- a session exists when its root thread holds a message;
- listing keeps one app and one user, newest first, with each session's last
  checkpoint time and message count, and leaves out sub-agent checkpoints
  (namespace ``tools:<task-id>``) and threads that are not ours;
- deleting removes the whole thread, sub-agents included, and reports an
  unknown one;
- recent messages are the user's and the assistant's text.

Conversations are written by real agent trees on scripted models, into a real
``InMemorySaver`` and ``AsyncSqliteSaver``. The last classes run whole
managers: two apps on one file, awkward identities (Review Focus 4), and ADK
and Deep Agents sharing one sqlite file (Review Focus 3).
"""

from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("deepagents")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from agentic_cli.workflow.checkpointers import open_checkpointer  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.deepagents import manager as manager_module  # noqa: E402
from agentic_cli.workflow.deepagents.agent_builder import build_agent_tree  # noqa: E402
from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager  # noqa: E402
from agentic_cli.workflow.deepagents.sessions import (  # noqa: E402
    CheckpointSessions,
    ref_from_thread_id,
    run_config,
    thread_id,
)
from agentic_cli.workflow.sessions import SessionRef  # noqa: E402
from tests.conftest import MockContext  # noqa: E402
from tests.workflow.deepagents.fakes import ScriptedChat, ScriptedModels, ai, task  # noqa: E402

AWKWARD = SessionRef("app", "ü/ser:1", 'a "quoted"/id')
SOLO = AgentConfig(name="solo", prompt="p")
COORDINATOR = AgentConfig(
    name="coordinator", prompt="p", description="Leads.", sub_agents=["worker"]
)
WORKER = AgentConfig(name="worker", prompt="p", description="Works.")


def ref(user: str = "alice", session: str = "s1", app: str = "app") -> SessionRef:
    return SessionRef(app, user, session)


@pytest.fixture(params=["memory", "sqlite"])
async def saver(request, tmp_path):
    """A real checkpointer of each kind the backend opens."""
    saver, cm = await open_checkpointer(request.param, sqlite_path=tmp_path / "c.db")
    try:
        yield saver
    finally:
        if cm is not None:
            await cm.__aexit__(None, None, None)


def _tree(saver, configs, **scripts):
    return build_agent_tree(
        configs,
        chat_models={
            name: ScriptedChat(label=name, script=list(script))
            for name, script in scripts.items()
        },
        resolved={},
        tools={},
        prompts={config.name: f"You are {config.name}." for config in configs},
        settings=None,
        checkpointer=saver,
    )


async def say(saver, session: SessionRef, text: str = "Hello.", *answers) -> None:
    """One turn of a single agent in ``session``, answering ``answers``."""
    root = _tree(saver, [SOLO], solo=answers or [ai("Hi.")])
    await root.ainvoke({"messages": [HumanMessage(text)]}, run_config(session))


async def delegate(saver, session: SessionRef) -> None:
    """One turn in which the coordinator delegates to a worker."""
    root = _tree(
        saver,
        [COORDINATOR, WORKER],
        coordinator=[ai(calls=[task("worker", "Do it.")]), ai("Done, says the worker.")],
        worker=[ai("Worked.")],
    )
    await root.ainvoke({"messages": [HumanMessage("Delegate.")]}, run_config(session))


async def threads(saver) -> list[tuple[str, str]]:
    """Every (thread, namespace) the saver holds a checkpoint for."""
    found = set()
    async for item in saver.alist(None):
        configurable = item.config["configurable"]
        found.add((configurable["thread_id"], configurable.get("checkpoint_ns", "")))
    return sorted(found)


# --- identity ---------------------------------------------------------------------------


class TestThreadId:
    def test_it_is_the_whole_triple_as_json(self):
        assert thread_id(SessionRef("app", "u", "s")) == '["app", "u", "s"]'

    @pytest.mark.parametrize(
        "session",
        [AWKWARD, SessionRef("", "", ""), SessionRef("a/b", "c\\d", "e\nf"),
         SessionRef("日本", "😀", "[1]")],
    )
    def test_it_reads_back_as_the_same_triple(self, session):
        assert ref_from_thread_id(thread_id(session)) == session

    def test_no_two_triples_share_a_thread(self):
        """A joined key would give all three the same thread."""
        collide = [
            SessionRef("app", "a/b", "c"),
            SessionRef("app", "a", "b/c"),
            SessionRef("app/a", "b", "c"),
        ]
        assert len({thread_id(session) for session in collide}) == 3

    @pytest.mark.parametrize(
        "value",
        ["s1", "", "not json", "null", "{}", '{"a": 1}', "[1, 2, 3]", '["a", "b"]',
         '["a", "b", "c", "d"]', '["a", null, "c"]', '[["a"], "b", "c"]', '"a"'],
    )
    def test_anything_else_is_no_session(self, value):
        assert ref_from_thread_id(value) is None


class TestRunConfig:
    def test_it_keys_the_thread_and_records_the_triple(self):
        assert run_config(AWKWARD) == {
            "configurable": {"thread_id": thread_id(AWKWARD)},
            "metadata": {
                "app_name": "app", "user_id": "ü/ser:1", "session_id": 'a "quoted"/id',
            },
        }

    def test_more_keys_join_the_config(self):
        callbacks = [object()]
        config = run_config(AWKWARD, callbacks=callbacks)
        assert config["callbacks"] is callbacks
        assert config["configurable"] == {"thread_id": thread_id(AWKWARD)}

    async def test_every_checkpoint_carries_the_triple(self, saver):
        await delegate(saver, AWKWARD)

        metadata = [item.metadata async for item in saver.alist(None)]
        assert metadata
        for entry in metadata:
            assert (entry["app_name"], entry["user_id"], entry["session_id"]) == (
                "app", "ü/ser:1", 'a "quoted"/id'
            )


# --- the store --------------------------------------------------------------------------


class TestExists:
    async def test_a_session_exists_once_it_holds_a_message(self, saver):
        sessions = CheckpointSessions(saver)
        assert await sessions.exists(ref()) is False

        await say(saver, ref())

        assert await sessions.exists(ref()) is True

    async def test_the_same_id_of_another_user_or_app_does_not(self, saver):
        await say(saver, ref())
        sessions = CheckpointSessions(saver)

        assert await sessions.exists(ref(user="bob")) is False
        assert await sessions.exists(ref(app="other")) is False
        assert await sessions.exists(ref(session="s2")) is False


class TestList:
    async def test_this_app_and_user_only_newest_first(self, saver):
        await say(saver, ref(session="s1"))
        await say(saver, ref(session="s2"))
        await say(saver, ref(user="bob", session="s3"))
        await say(saver, ref(app="other", session="s4"))

        listed = await CheckpointSessions(saver).list("app", "alice")

        assert [entry["session_id"] for entry in listed] == ["s2", "s1"]
        assert all(type(entry["last_update"]) is float for entry in listed)
        assert listed[0]["last_update"] > listed[1]["last_update"]
        assert [entry["message_count"] for entry in listed] == [2, 2]

    async def test_a_later_turn_moves_a_session_first(self, saver):
        await say(saver, ref(session="s1"))
        await say(saver, ref(session="s2"))
        await say(saver, ref(session="s1"), "Again.")

        listed = await CheckpointSessions(saver).list("app", "alice")

        assert [entry["session_id"] for entry in listed] == ["s1", "s2"]

    async def test_sub_agent_checkpoints_are_not_sessions(self, saver):
        await delegate(saver, ref())
        assert any(ns.startswith("tools:") for _, ns in await threads(saver))

        listed = await CheckpointSessions(saver).list("app", "alice")

        # The root conversation: the question, the task call, its result, the answer.
        assert [(e["session_id"], e["message_count"]) for e in listed] == [("s1", 4)]

    async def test_a_sub_agents_later_checkpoint_does_not_stand_for_the_session(
        self, saver
    ):
        """A turn that failed inside the sub-agent leaves the thread's newest
        checkpoint in the sub-agent's namespace."""
        root = _tree(
            saver,
            [COORDINATOR, WORKER],
            coordinator=[ai(calls=[task("worker", "Do it.")])],
            worker=[RuntimeError("provider said 400")],
        )
        with pytest.raises(RuntimeError):
            await root.ainvoke({"messages": [HumanMessage("Delegate.")]}, run_config(ref()))

        listed = await CheckpointSessions(saver).list("app", "alice")

        # The question and the task call; the worker's brief is not the session's.
        assert [(e["session_id"], e["message_count"]) for e in listed] == [("s1", 2)]

    async def test_a_thread_that_is_not_ours_is_left_out(self, saver):
        """Metadata alone does not make a session: the thread id must name it."""
        root = _tree(saver, [SOLO], solo=[ai("Hi."), ai("Hi.")])
        claims_alice = {"app_name": "app", "user_id": "alice", "session_id": "s1"}
        await root.ainvoke(
            {"messages": [HumanMessage("Hello.")]},
            {"configurable": {"thread_id": "s1"}, "metadata": claims_alice},
        )
        await root.ainvoke(
            {"messages": [HumanMessage("Hello.")]},
            {"configurable": {"thread_id": thread_id(ref(user="bob"))},
             "metadata": claims_alice},
        )

        assert await CheckpointSessions(saver).list("app", "alice") == []

    async def test_nothing_stored_lists_nothing(self, saver):
        assert await CheckpointSessions(saver).list("app", "alice") == []


class TestDelete:
    async def test_it_removes_the_thread_with_its_sub_agents(self, saver):
        await delegate(saver, ref())
        sessions = CheckpointSessions(saver)

        assert await sessions.delete(ref()) is True

        assert await sessions.exists(ref()) is False
        assert [t for t, _ in await threads(saver) if t == thread_id(ref())] == []

    async def test_an_unknown_session_is_not_deleted(self, saver):
        await say(saver, ref())
        sessions = CheckpointSessions(saver)

        assert await sessions.delete(ref(session="nope")) is False
        assert await sessions.delete(ref(user="bob")) is False
        assert await sessions.exists(ref()) is True

    async def test_other_sessions_are_left_alone(self, saver):
        await say(saver, ref())
        await say(saver, ref(user="bob"))
        sessions = CheckpointSessions(saver)

        await sessions.delete(ref())

        assert await sessions.exists(ref(user="bob")) is True


class TestRecent:
    async def test_the_users_and_the_assistants_text(self, saver):
        thinking_then_text = AIMessage(content=[
            {"type": "reasoning", "reasoning": "Let me think."},
            {"type": "text", "text": "Part one."},
            {"type": "text", "text": "Part two."},
        ])
        await say(saver, ref(), "Tell me.", thinking_then_text)

        recent = await CheckpointSessions(saver).recent(ref(), 20)

        assert recent == [
            {"role": "user", "content": "Tell me."},
            {"role": "assistant", "content": "Part one. Part two."},
        ]

    async def test_tool_traffic_and_empty_answers_are_left_out(self, saver):
        await delegate(saver, ref())

        recent = await CheckpointSessions(saver).recent(ref(), 20)

        # The task call carries no text and its result is a tool message;
        # the worker's own messages are in its namespace, not the conversation.
        assert recent == [
            {"role": "user", "content": "Delegate."},
            {"role": "assistant", "content": "Done, says the worker."},
        ]

    async def test_the_limit_keeps_the_newest(self, saver):
        await say(saver, ref(), "One.", ai("Two."))
        await say(saver, ref(), "Three.", ai("Four."))

        recent = await CheckpointSessions(saver).recent(ref(), 3)

        assert [m["content"] for m in recent] == ["Two.", "Three.", "Four."]

    async def test_an_unknown_session_has_none(self, saver):
        await say(saver, ref())
        assert await CheckpointSessions(saver).recent(ref(user="bob"), 20) == []


# --- whole managers ---------------------------------------------------------------------


@pytest.fixture
def hermetic(tmp_path, monkeypatch):
    """A hermetic cwd and HOME."""
    (tmp_path / "home").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path


def _deepagents(settings, *, app_name: str | None = None) -> DeepAgentsWorkflowManager:
    return DeepAgentsWorkflowManager(
        agent_configs=[AgentConfig(name="solo", prompt="p")],
        settings=settings,
        app_name=app_name,
        model="gemini-2.5-flash",
    )


async def converse(manager, text: str, *, user: str, session: str) -> list:
    return [event async for event in manager.process(text, user, session_id=session)]


def _ids(listed: list[dict]) -> list[str]:
    return [entry["session_id"] for entry in listed]


class TestOneFileManyApps:
    async def test_two_apps_on_one_sqlite_file_keep_their_sessions(
        self, hermetic, monkeypatch
    ):
        models = ScriptedModels(solo=[ai("For one."), ai("For two.")])
        monkeypatch.setattr(manager_module, "build_chat_model", models)
        db = hermetic / "shared.db"
        with MockContext(google_api_key="test-key", sqlite_uri=f"sqlite:///{db}") as ctx:
            one = _deepagents(ctx.settings, app_name="one")
            two = _deepagents(ctx.settings, app_name="two")
            try:
                await converse(one, "Hello, one.", user="u", session="s")
                await converse(two, "Hello, two.", user="u", session="s")

                assert _ids(await one.list_sessions(user_id="u")) == ["s"]
                assert _ids(await two.list_sessions(user_id="u")) == ["s"]
                assert [m["content"] for m in await one.recent_messages("s", user_id="u")] == [
                    "Hello, one.", "For one."
                ]
                assert await one.delete_session("s", user_id="u") is True
                assert await two.session_exists("s", user_id="u") is True
            finally:
                await one.cleanup()
                await two.cleanup()


class TestAwkwardIdentities:
    """Review Focus 4: identities with JSON, path and Unicode characters."""

    async def test_session_identity_with_awkward_characters(self, hermetic, monkeypatch):
        models = ScriptedModels(solo=[ai("Hi, awkward."), ai("Hi, a."), ai("Hi, a/b.")])
        monkeypatch.setattr(manager_module, "build_chat_model", models)
        user, session = AWKWARD.user_id, AWKWARD.session_id
        with MockContext(google_api_key="test-key", session_store="sqlite") as ctx:
            manager = _deepagents(ctx.settings, app_name="app")
            try:
                await converse(manager, "Hello.", user=user, session=session)
                # Two more whose user/session joined with "/" would be one key.
                await converse(manager, "Hello.", user="a", session="b/c")
                await converse(manager, "Hello.", user="a/b", session="c")

                assert ref_from_thread_id(thread_id(manager.session_ref(session, user))) == (
                    manager.session_ref(session, user)
                )
                assert _ids(await manager.list_sessions(user_id=user)) == [session]
                assert _ids(await manager.list_sessions(user_id="a")) == ["b/c"]
                assert _ids(await manager.list_sessions(user_id="a/b")) == ["c"]
                assert _ids(await manager.list_sessions(user_id="ü")) == []
                assert await manager.list_sessions() == []  # the default user's

                assert await manager.session_exists(session, user_id=user) is True
                assert await manager.session_exists(session, user_id="a") is False
                assert await manager.delete_session(session, user_id="a/b") is False
                assert await manager.session_exists(session, user_id=user) is True

                assert await manager.delete_session(session, user_id=user) is True
                assert await manager.session_exists(session, user_id=user) is False
                assert _ids(await manager.list_sessions(user_id="a")) == ["b/c"]
                assert _ids(await manager.list_sessions(user_id="a/b")) == ["c"]
            finally:
                await manager.cleanup()


class TestSharedFileWithAdk:
    """Review Focus 3: one ``sqlite_uri`` for both backends, one file."""

    async def test_shared_sqlite_file_keeps_backends_apart(self, hermetic, monkeypatch):
        pytest.importorskip("google.adk")
        from google.adk.events import Event
        from google.genai import types

        from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager

        models = ScriptedModels(solo=[ai("The Deep Agents answer.")])
        monkeypatch.setattr(manager_module, "build_chat_model", models)
        db = hermetic / "shared.db"
        with MockContext(google_api_key="test-key", sqlite_uri=f"sqlite:///{db}") as ctx:
            adk = GoogleADKWorkflowManager(
                agent_configs=[AgentConfig(name="solo", prompt="p")],
                settings=ctx.settings,
                app_name="app",
                model="gemini-2.5-flash",
            )
            deep = _deepagents(ctx.settings, app_name="app")
            try:
                # The same user and session id on both backends.
                await adk.initialize_services()
                written = await adk._session_service.create_session(
                    app_name="app", user_id="u", session_id="s"
                )
                await adk._session_service.append_event(
                    session=written,
                    event=Event(author="user", content=types.Content(
                        role="user", parts=[types.Part.from_text(text="Hello, ADK.")]
                    )),
                )
                await converse(deep, "Hello, Deep Agents.", user="u", session="s")

                with sqlite3.connect(db) as connection:
                    tables = {row[0] for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )}
                assert {"sessions", "events", "checkpoints", "writes"} <= tables

                assert [r.session_id for r in await adk.list_sessions(user_id="u")] == ["s"]
                assert _ids(await deep.list_sessions(user_id="u")) == ["s"]
                assert [m["content"] for m in await adk.recent_messages("s", user_id="u")] == [
                    "Hello, ADK."
                ]
                assert [m["content"] for m in await deep.recent_messages("s", user_id="u")] == [
                    "Hello, Deep Agents.", "The Deep Agents answer."
                ]

                # Deleting one backend's session leaves the other's.
                assert await deep.delete_session("s", user_id="u") is True
                assert await adk.session_exists("s", user_id="u") is True
                assert await deep.session_exists("s", user_id="u") is False
            finally:
                await adk.cleanup()
                await deep.cleanup()
