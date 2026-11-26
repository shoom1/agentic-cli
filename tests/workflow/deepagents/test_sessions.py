"""Session identity and the session store on Deep Agents (``workflow/deepagents/sessions.py``).

A conversation's thread is keyed by the whole ``SessionRef``: ``thread_id``
is ``json.dumps([app_name, user_id, session_id])``, and the same three values
go into every checkpoint's metadata. ``CheckpointSessions`` answers the
session hooks from the checkpointer:

- a session exists when its root thread holds a message;
- listing keeps one app and one user, newest first, one ``SessionInfo`` row
  each: the first prompt as its title, the turns the user ran (a job's
  notification is none), its last checkpoint's time; it leaves out the
  sub-agent checkpoints an older version stored (namespace
  ``tools:<task-id>``) and threads that are not ours, and ``limit`` keeps the
  newest; one conversation's row is ``info``;
- deleting removes the whole thread, sub-agents' checkpoints included, every
  sub-agent instance's thread of the conversation (first), then its
  sub-agent records, then the thread and its row, and reports an unknown one;
- each sub-agent instance's transcript is a thread of its own
  (``instance_thread_id``), which never lists as a conversation; the
  instances a conversation keeps number a new manager's next ones;
- a conversation's sub-agent records are one item of the store beside the
  checkpointer, under ``("agentic_cli", "agents", <app>, <user>)``, never a
  row of the index;
- recent messages are the user's and the assistant's text.

Conversations are written by real agent trees on scripted models, into a real
``InMemorySaver`` and ``AsyncSqliteSaver``, without the index the manager
writes as each turn ends: these rows come from the scan a first listing makes
(``test_session_index.py`` covers the index). The last classes run whole
managers: two apps on one file, awkward identities (Review Focus 4), and ADK
and Deep Agents sharing one sqlite file (Review Focus 3).
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

pytest.importorskip("deepagents")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from agentic_cli.workflow.checkpointers import open_checkpointer, open_store  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.deepagents import manager as manager_module  # noqa: E402
from agentic_cli.workflow.deepagents.agent_builder import build_agent_tree  # noqa: E402
from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager  # noqa: E402
from agentic_cli.tools.jobs import JobState  # noqa: E402
from agentic_cli.workflow.deepagents.sessions import (  # noqa: E402
    CheckpointSessions,
    instance_run_config,
    instance_thread_id,
    job_notification,
    notice_message,
    ref_from_thread_id,
    run_config,
    thread_id,
)
from agentic_cli.workflow.sessions import SessionInfo, SessionRef  # noqa: E402
from tests.conftest import MockContext  # noqa: E402
from tests.loop_guard import no_blocking_calls  # noqa: E402
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
def kind(request) -> str:
    return request.param


@pytest.fixture
async def saver(kind, tmp_path):
    """A real checkpointer of each kind the backend opens."""
    saver, cm = await open_checkpointer(kind, sqlite_path=tmp_path / "c.db")
    try:
        yield saver
    finally:
        if cm is not None:
            await cm.__aexit__(None, None, None)


@pytest.fixture
async def store(kind, tmp_path):
    """The session index opened beside it, as the manager opens it."""
    store, cm = await open_store(kind, sqlite_path=tmp_path / "c.db")
    try:
        yield store
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


async def older_sub_agent_checkpoint(saver, session: SessionRef, *texts: str) -> None:
    """A checkpoint in a sub-agent's namespace holding ``texts``, as the
    versions that compiled sub-agents onto the root's saver stored them."""
    from langgraph.checkpoint.base import empty_checkpoint

    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"messages": [HumanMessage(text) for text in texts]}
    checkpoint["channel_versions"] = {"messages": "1"}
    config = {"configurable": {"thread_id": thread_id(session), "checkpoint_ns": "tools:t1"}}
    metadata = {"app_name": session.app_name, "user_id": session.user_id,
                "session_id": session.session_id, "source": "loop", "step": 1}
    await saver.aput(config, checkpoint, metadata, {"messages": "1"})


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


class TestAnInstancesThread:
    """Each sub-agent instance's transcript is a thread of its own, keyed by
    its conversation's triple and its id; its checkpoints carry both."""

    def test_it_is_the_triple_and_the_instance_as_json(self):
        assert instance_thread_id(SessionRef("app", "u", "s"), "researcher-1") == (
            '["app", "u", "s", "researcher-1"]'
        )

    def test_it_names_no_conversation(self):
        assert ref_from_thread_id(instance_thread_id(AWKWARD, "researcher-1")) is None

    def test_its_run_config_keys_the_thread_and_records_the_instance(self):
        callbacks = [object()]

        config = instance_run_config(AWKWARD, "researcher-1", callbacks=callbacks)

        assert config == {
            "configurable": {"thread_id": instance_thread_id(AWKWARD, "researcher-1")},
            "metadata": {
                "app_name": "app", "user_id": "ü/ser:1", "session_id": 'a "quoted"/id',
                "instance": "researcher-1", "thread_kind": "instance",
            },
            "callbacks": callbacks,
        }


async def instance_ran(saver, session: SessionRef, instance: str, *texts: str) -> None:
    """A run of the sub-agent instance ``instance`` of ``session`` on its own
    thread, as the manager runs one, saying ``texts`` in turn (one run each)."""
    graph = _tree(saver, [SOLO], solo=[ai(text) for text in texts] or [ai("Done.")])
    for text in texts or ("Done.",):
        await graph.ainvoke(
            {"messages": [HumanMessage(f"Then {text}")]}, instance_run_config(session, instance)
        )


class TestStoredInstances:
    async def test_the_highest_stored_number_of_each_type(self, saver, store):
        await say(saver, ref())
        for instance in ("researcher-1", "researcher-3", "writer-2", "notes"):
            await instance_ran(saver, ref(), instance)
        await instance_ran(saver, ref(session="s2"), "researcher-7")  # another conversation's
        await instance_ran(saver, ref(user="bob"), "writer-9")  # another user's
        sessions = CheckpointSessions(saver, store)
        # Records that do not read: the scan's filtering is what counts them.
        await sessions.save_agent_records(ref(), {"v": 99})

        assert await sessions.instances(ref()) == [
            "notes", "researcher-1", "researcher-3", "writer-2",
        ]
        assert await sessions.stored_instances(ref()) == {"researcher": 3, "writer": 2}
        # No instance's thread lists as a conversation.
        assert [e.session_id for e in await sessions.list("app", "alice")] == ["s1"]

    async def test_an_instance_run_twice_is_one(self, saver, store):
        await instance_ran(saver, ref(), "researcher-1", "One.", "Two.")

        assert await CheckpointSessions(saver, store).instances(ref()) == ["researcher-1"]

    async def test_none_is_none(self, saver, store):
        await say(saver, ref())

        assert await CheckpointSessions(saver, store).stored_instances(ref()) == {}

    async def test_readable_records_are_read_alone(self, saver, store, monkeypatch):
        """A first run's thread is made only once the records name it, so the
        records' ids and counters are every instance the conversation keeps:
        no thread is scanned."""
        await say(saver, ref())
        for instance in ("researcher-1", "researcher-2", "researcher-3"):
            await instance_ran(saver, ref(), instance)
        sessions = CheckpointSessions(saver, store)
        await sessions.save_agent_records(
            ref(), records_naming("researcher-1", researcher=2)
        )
        scans: list[dict] = []
        alist = saver.alist

        def listing(config, *, filter=None, **kwargs):
            if filter and "thread_kind" in filter:
                scans.append(filter)
            return alist(config, filter=filter, **kwargs)

        monkeypatch.setattr(saver, "alist", listing)

        with no_blocking_calls():
            counts = await sessions.stored_instances(ref())

        assert counts == {"researcher": 2}
        assert scans == []

    async def test_records_that_name_no_instance_count_none(self, saver, store, monkeypatch):
        await instance_ran(saver, ref(), "researcher-4")
        sessions = CheckpointSessions(saver, store)
        await sessions.save_agent_records(ref(), records_of("researcher-4#1"))
        scans: list[dict] = []
        alist = saver.alist

        def listing(config, *, filter=None, **kwargs):
            if filter and "thread_kind" in filter:
                scans.append(filter)
            return alist(config, filter=filter, **kwargs)

        monkeypatch.setattr(saver, "alist", listing)

        assert await sessions.stored_instances(ref()) == {}
        assert scans == []

    async def test_a_conversation_without_records_scans_nothing(self, saver, store, monkeypatch):
        """No records means no instance thread was ever made for it: a thread
        named like one of its instances is left."""
        await say(saver, ref())
        await instance_ran(saver, ref(), "researcher-4")
        sessions = CheckpointSessions(saver, store)
        scans: list[dict] = []
        alist = saver.alist

        def listing(config, *, filter=None, **kwargs):
            if filter and "thread_kind" in filter:
                scans.append(filter)
            return alist(config, filter=filter, **kwargs)

        monkeypatch.setattr(saver, "alist", listing)

        assert await sessions.stored_instances(ref()) == {}
        assert await sessions.delete(ref()) is True
        assert scans == []
        monkeypatch.undo()
        assert await sessions.instances(ref()) == ["researcher-4"]

    async def test_records_of_another_version_read_the_threads(self, saver, store):
        await instance_ran(saver, ref(), "researcher-4")
        sessions = CheckpointSessions(saver, store)
        await sessions.save_agent_records(ref(), {"v": 99})

        assert await sessions.stored_instances(ref()) == {"researcher": 4}


# --- the store --------------------------------------------------------------------------


def records_of(*runs: str) -> dict:
    """Stored records (``AgentRuns.export``'s shape) holding ``runs``' ids alone."""
    return {"v": 1, "saved": "2026-11-19T10:00:00", "numbered": {}, "instances": [],
            "runs": [{"run": run} for run in runs]}


class TestAgentRecords:
    """A conversation's sub-agent records are one item of the store beside the
    checkpointer, under ``("agentic_cli", "agents", <app>, <user>)``, keyed
    by the session id: never a row of the index, never another's."""

    async def test_written_read_and_written_over(self, saver, store):
        sessions = CheckpointSessions(saver, store)

        assert await sessions.agent_records(ref()) is None
        with no_blocking_calls():
            await sessions.save_agent_records(ref(), records_of("researcher-1#1"))
        await sessions.save_agent_records(ref(), records_of("researcher-1#1", "writer-1#1"))

        assert await sessions.agent_records(ref()) == records_of("researcher-1#1", "writer-1#1")
        for other in (ref(session="s2"), ref(user="bob"), ref(app="other")):
            assert await sessions.agent_records(other) is None
        item = await store.aget(("agentic_cli", "agents", '"app"', '"alice"'), "s1")
        assert item.value == records_of("researcher-1#1", "writer-1#1")

    async def test_names_with_periods_are_kept_apart(self, saver, store):
        sessions = CheckpointSessions(saver, store)

        await sessions.save_agent_records(ref(app="my.app", user="a.b"), records_of("x-1#1"))

        assert await sessions.agent_records(ref(app="my.app", user="a.b")) == records_of("x-1#1")
        assert await sessions.agent_records(ref(app="my", user="a.b")) is None

    async def test_they_are_no_conversation(self, saver, store):
        sessions = CheckpointSessions(saver, store)
        await say(saver, ref(session="s2"))
        await sessions.save_agent_records(ref(), records_of("researcher-1#1"))

        assert _ids(await sessions.list("app", "alice")) == ["s2"]
        assert await sessions.info(ref()) is None
        assert await sessions.exists(ref()) is False

    async def test_deleted(self, saver, store):
        sessions = CheckpointSessions(saver, store)
        await sessions.save_agent_records(ref(), records_of("researcher-1#1"))

        await sessions.delete_agent_records(ref())
        await sessions.delete_agent_records(ref())

        assert await sessions.agent_records(ref()) is None


class TestExists:
    async def test_a_session_exists_once_it_holds_a_message(self, saver, store):
        sessions = CheckpointSessions(saver, store)
        assert await sessions.exists(ref()) is False

        await say(saver, ref())

        assert await sessions.exists(ref()) is True

    async def test_the_same_id_of_another_user_or_app_does_not(self, saver, store):
        await say(saver, ref())
        sessions = CheckpointSessions(saver, store)

        assert await sessions.exists(ref(user="bob")) is False
        assert await sessions.exists(ref(app="other")) is False
        assert await sessions.exists(ref(session="s2")) is False


class TestList:
    async def test_this_app_and_user_only_newest_first(self, saver, store):
        await say(saver, ref(session="s1"))
        await say(saver, ref(session="s2"))
        await say(saver, ref(user="bob", session="s3"))
        await say(saver, ref(app="other", session="s4"))

        listed = await CheckpointSessions(saver, store).list("app", "alice")

        assert [entry.session_id for entry in listed] == ["s2", "s1"]
        assert all(entry.updated.tzinfo is not None for entry in listed)
        assert listed[0].updated > listed[1].updated
        assert [(entry.title, entry.turns) for entry in listed] == [("Hello.", 1)] * 2

    async def test_a_later_turn_moves_a_session_first(self, saver, store):
        await say(saver, ref(session="s1"))
        await say(saver, ref(session="s2"))
        await say(saver, ref(session="s1"), "Again.")

        listed = await CheckpointSessions(saver, store).list("app", "alice")

        assert [(entry.session_id, entry.turns) for entry in listed] == [("s1", 2), ("s2", 1)]

    async def test_a_row_is_the_first_prompt_and_the_turns(self, saver, store):
        await say(saver, ref(), "  Plan the\ntrip to   Lisbon ")
        await say(saver, ref(), "And the budget?")

        (row,) = await CheckpointSessions(saver, store).list("app", "alice")

        assert row == SessionInfo("s1", "Plan the trip to Lisbon", 2, row.updated)

    async def test_a_job_notification_is_no_turn(self, saver, store):
        await say(saver, ref(), "Build it.")
        record = SimpleNamespace(
            job_id="j1", tool="build", name="build", call_id="c1", call_name="build",
            session_id="s1", user_id="alice", state=JobState.SUCCEEDED, exit_code=0,
            error=None,
        )
        root = _tree(saver, [SOLO], solo=[ai("Built.")])
        await root.ainvoke(
            {"messages": [job_notification(record, "ok")]}, run_config(ref())
        )

        (row,) = await CheckpointSessions(saver, store).list("app", "alice")

        assert (row.title, row.turns) == ("Build it.", 1)

    async def test_a_notice_to_the_root_is_no_turn_and_no_title(self, saver, store):
        root = _tree(saver, [SOLO], solo=[ai("Noted.")])
        await root.ainvoke({"messages": [notice_message("The user worked with x-1.")]},
                           run_config(ref()))
        await say(saver, ref(), "Build it.")

        (row,) = await CheckpointSessions(saver, store).list("app", "alice")

        assert (row.title, row.turns) == ("Build it.", 1)

    async def test_the_limit_keeps_the_newest(self, saver, store):
        await say(saver, ref(session="s1"))
        await say(saver, ref(session="s2"))

        listed = await CheckpointSessions(saver, store).list("app", "alice", limit=1)

        assert [entry.session_id for entry in listed] == ["s2"]

    async def test_sub_agent_checkpoints_are_not_sessions(self, saver, store):
        await delegate(saver, ref())
        # Sub-agents store none now; a thread an older version wrote holds some.
        assert await threads(saver) == [(thread_id(ref()), "")]
        await older_sub_agent_checkpoint(saver, ref(), "Do it.")

        listed = await CheckpointSessions(saver, store).list("app", "alice")

        # The root conversation: one question; the worker's brief is no turn.
        assert [(e.session_id, e.title, e.turns) for e in listed] == [("s1", "Delegate.", 1)]

    async def test_a_sub_agents_later_checkpoint_does_not_stand_for_the_session(
        self, saver, store
    ):
        """A checkpoint in a sub-agent's namespace written after the root's
        newest (as a turn that failed inside the sub-agent left one, when
        sub-agents stored them) does not stand for the session."""
        root = _tree(
            saver,
            [COORDINATOR, WORKER],
            coordinator=[ai(calls=[task("worker", "Do it.")]), ai("The worker failed.")],
            worker=[RuntimeError("provider said 400")],
        )
        await root.ainvoke({"messages": [HumanMessage("Delegate.")]}, run_config(ref()))
        await older_sub_agent_checkpoint(saver, ref(), "Do it.", "And this.")

        listed = await CheckpointSessions(saver, store).list("app", "alice")

        # The question is the one turn; the worker's brief is not the session's.
        assert [(e.session_id, e.turns) for e in listed] == [("s1", 1)]

    async def test_a_thread_that_is_not_ours_is_left_out(self, saver, store):
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

        assert await CheckpointSessions(saver, store).list("app", "alice") == []

    async def test_nothing_stored_lists_nothing(self, saver, store):
        assert await CheckpointSessions(saver, store).list("app", "alice") == []

    async def test_a_thread_with_no_message_is_no_conversation(self, saver, store):
        """A turn cancelled before its first step leaves only LangGraph's
        input checkpoint: as ``info`` and ``exists`` say, no conversation."""
        from langgraph.checkpoint.base import empty_checkpoint

        config = {"configurable": {"thread_id": thread_id(ref()), "checkpoint_ns": ""}}
        metadata = {"app_name": "app", "user_id": "alice", "session_id": "s1",
                    "source": "input", "step": -1}
        await saver.aput(config, empty_checkpoint(), metadata, {})
        sessions = CheckpointSessions(saver, store)

        assert await sessions.list("app", "alice") == []
        assert await sessions.info(ref()) is None


class TestInfo:
    async def test_it_is_the_conversations_row(self, saver, store):
        await say(saver, ref(), "Hello there.")
        sessions = CheckpointSessions(saver, store)

        (listed,) = await sessions.list("app", "alice")

        assert await sessions.info(ref()) == listed
        assert listed.title == "Hello there."

    async def test_none_for_a_conversation_that_is_not_stored(self, saver, store):
        await say(saver, ref())
        sessions = CheckpointSessions(saver, store)

        assert await sessions.info(ref(session="s2")) is None
        assert await sessions.info(ref(user="bob")) is None


def records_naming(*instances: str, **numbered: int) -> dict:
    """Stored records that name ``instances`` (each with its first run) and number past ``numbered``."""
    return {
        "v": 1, "saved": "2026-11-19T10:00:00", "numbered": numbered,
        "instances": [{"instance": instance} for instance in instances],
        "runs": [{"run": f"{instance}#1"} for instance in instances],
    }


class TestDelete:
    async def test_it_removes_the_thread_with_its_sub_agents(self, saver, store):
        await delegate(saver, ref())
        await older_sub_agent_checkpoint(saver, ref(), "Do it.")
        sessions = CheckpointSessions(saver, store)

        assert await sessions.delete(ref()) is True

        assert await sessions.exists(ref()) is False
        assert [t for t, _ in await threads(saver) if t == thread_id(ref())] == []

    async def test_it_deletes_its_instances_threads_and_no_others(self, saver, store):
        await say(saver, ref())
        await instance_ran(saver, ref(), "researcher-1", "One.", "Two.")
        await instance_ran(saver, ref(session="s2"), "researcher-1")
        sessions = CheckpointSessions(saver, store)
        await sessions.save_agent_records(ref(), records_naming("researcher-1"))

        with no_blocking_calls():
            assert await sessions.delete(ref()) is True

        assert await sessions.instances(ref()) == []
        assert await sessions.instances(ref(session="s2")) == ["researcher-1"]
        assert [t for t, _ in await threads(saver)] == [
            instance_thread_id(ref(session="s2"), "researcher-1")
        ]

    async def test_its_records_go_after_its_instances_and_before_it(
        self, saver, store, monkeypatch
    ):
        await say(saver, ref())
        await instance_ran(saver, ref(), "researcher-1")
        sessions = CheckpointSessions(saver, store)
        await sessions.save_agent_records(ref(), records_naming("researcher-1"))
        await sessions.save_agent_records(ref(session="s2"), records_of("writer-1#1"))
        done: list[str] = []
        delete_thread, delete_item = type(saver).adelete_thread, type(store).adelete

        async def deleting_thread(self, thread: str) -> None:
            done.append("conversation" if thread == thread_id(ref()) else "instance")
            await delete_thread(self, thread)

        async def deleting_item(self, namespace, key) -> None:
            done.append(namespace[1])
            await delete_item(self, namespace, key)

        monkeypatch.setattr(type(saver), "adelete_thread", deleting_thread)
        monkeypatch.setattr(type(store), "adelete", deleting_item)

        assert await sessions.delete(ref()) is True
        assert done == ["instance", "agents", "conversation"]
        assert await sessions.agent_records(ref()) is None
        assert await sessions.agent_records(ref(session="s2")) == records_of("writer-1#1")

    async def test_a_conversation_already_gone_leaves_no_instance_behind(self, saver, store):
        await instance_ran(saver, ref(), "researcher-1")
        sessions = CheckpointSessions(saver, store)
        await sessions.save_agent_records(ref(), records_naming("researcher-1"))

        assert await sessions.delete(ref()) is False
        assert await threads(saver) == []
        assert await sessions.agent_records(ref()) is None

    async def test_a_delete_with_records_of_another_version_scans(self, saver, store):
        """Only the store can say what a conversation keeps when this manager
        wrote none of its records: its instances' threads are found by the scan."""
        await say(saver, ref())
        await instance_ran(saver, ref(), "researcher-1")
        await instance_ran(saver, ref(session="s2"), "researcher-1")
        sessions = CheckpointSessions(saver, store)
        await sessions.save_agent_records(ref(), {"v": 99})

        assert await sessions.delete(ref()) is True

        assert await sessions.instances(ref()) == []
        assert await sessions.instances(ref(session="s2")) == ["researcher-1"]

    async def test_an_unknown_session_is_not_deleted(self, saver, store):
        await say(saver, ref())
        sessions = CheckpointSessions(saver, store)

        assert await sessions.delete(ref(session="nope")) is False
        assert await sessions.delete(ref(user="bob")) is False
        assert await sessions.exists(ref()) is True

    async def test_other_sessions_are_left_alone(self, saver, store):
        await say(saver, ref())
        await say(saver, ref(user="bob"))
        sessions = CheckpointSessions(saver, store)

        await sessions.delete(ref())

        assert await sessions.exists(ref(user="bob")) is True


class TestRecent:
    async def test_the_users_and_the_assistants_text(self, saver, store):
        thinking_then_text = AIMessage(content=[
            {"type": "reasoning", "reasoning": "Let me think."},
            {"type": "text", "text": "Part one."},
            {"type": "text", "text": "Part two."},
        ])
        await say(saver, ref(), "Tell me.", thinking_then_text)

        recent = await CheckpointSessions(saver, store).recent(ref(), 20)

        assert recent == [
            {"role": "user", "content": "Tell me."},
            {"role": "assistant", "content": "Part one. Part two."},
        ]

    async def test_tool_traffic_and_empty_answers_are_left_out(self, saver, store):
        await delegate(saver, ref())

        recent = await CheckpointSessions(saver, store).recent(ref(), 20)

        # The task call carries no text and its result is a tool message;
        # the worker's own messages are in its namespace, not the conversation.
        assert recent == [
            {"role": "user", "content": "Delegate."},
            {"role": "assistant", "content": "Done, says the worker."},
        ]

    async def test_the_limit_keeps_the_newest(self, saver, store):
        await say(saver, ref(), "One.", ai("Two."))
        await say(saver, ref(), "Three.", ai("Four."))

        recent = await CheckpointSessions(saver, store).recent(ref(), 3)

        assert [m["content"] for m in recent] == ["Two.", "Three.", "Four."]

    async def test_an_unknown_session_has_none(self, saver, store):
        await say(saver, ref())
        assert await CheckpointSessions(saver, store).recent(ref(user="bob"), 20) == []

    async def test_a_notice_to_the_root_is_left_out_and_its_answer_kept(self, saver, store):
        await say(saver, ref(), "Tell me.", ai("Told."))
        root = _tree(saver, [SOLO], solo=[ai("Noted.")])
        await root.ainvoke({"messages": [notice_message("The user worked with x-1.")]},
                           run_config(ref()))

        recent = await CheckpointSessions(saver, store).recent(ref(), 20)

        assert [m["content"] for m in recent] == ["Tell me.", "Told.", "Noted."]


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


def _ids(listed: list[SessionInfo]) -> list[str]:
    return [entry.session_id for entry in listed]


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
                # Deep Agents' session index: tables ADK has none of.
                assert {"store", "store_migrations"} <= tables

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


class TestNumberedPastTheStored:
    """A new manager (a restart) numbers a conversation's instances past the
    ones its checkpointer keeps (``_stored_instances``), whose threads stay
    as they were."""

    async def test_a_restart_numbers_past_them(self, hermetic, monkeypatch):
        models = ScriptedModels(
            coordinator=[ai(calls=[task("worker", "Do it.", "c1")]), ai("Done."),
                         ai(calls=[task("worker", "Do it again.", "c2")]), ai("Done.")],
            worker=[ai("Worked."), ai("Worked again.")],
        )
        monkeypatch.setattr(manager_module, "build_chat_model", models)
        configs = [COORDINATOR, WORKER]
        with MockContext(google_api_key="test-key") as ctx:
            first = DeepAgentsWorkflowManager(
                agent_configs=configs, settings=ctx.settings, model="gemini-2.5-flash"
            )
            try:
                await converse(first, "Delegate.", user="u", session="s")
            finally:
                await first.cleanup()
            second = DeepAgentsWorkflowManager(
                agent_configs=configs, settings=ctx.settings, model="gemini-2.5-flash"
            )
            try:
                await second.initialize_services()
                with no_blocking_calls():
                    stored = await second._stored_instances(second.session_ref("s", "u"))
                await converse(second, "Again.", user="u", session="s")
                runs = [r.ref.run for r in second.agent_runs.records(second.session_ref("s", "u"))]
                kept = await second._sessions.instances(second.session_ref("s", "u"))
            finally:
                await second.cleanup()

        assert stored == {"worker": 1}
        # The first manager's run is in its records, loaded with the conversation.
        assert runs == ["worker-1#1", "worker-2#1"]
        assert kept == ["worker-1", "worker-2"]


class TestAKeptRow:
    """A conversation whose only work is a run the user started has a row
    ``keep`` wrote: a turn that stores no message leaves it, and a title it
    holds is its title."""

    async def test_a_turn_that_stored_no_message_leaves_the_kept_row(self, saver, store):
        sessions = CheckpointSessions(saver, store)
        await sessions.keep(ref(), "Objective title.")

        await sessions.record(ref(), [HumanMessage("Typed.")])  # the turn's start
        await sessions.record(ref())  # its end: the input was never stored

        assert await sessions.exists(ref())
        (row,) = await sessions.list("app", "alice")
        assert row.title == "Objective title."

    async def test_a_conversation_never_kept_is_titled_and_dropped_as_before(self, saver, store):
        sessions = CheckpointSessions(saver, store)

        await sessions.record(ref(), [HumanMessage("Typed.")])
        assert [(r.title, r.turns) for r in await sessions.list("app", "alice")] == [("Typed.", 1)]
        await sessions.record(ref())  # no message stored: its row goes

        assert await sessions.list("app", "alice") == []
