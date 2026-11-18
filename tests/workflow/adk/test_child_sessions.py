"""ADK's sub-agent transcripts live and die with their conversation.

Each sub-agent run's transcript is a child session in the conversation's own
session service (``adk/sessions.py``): app ``child_app(app)``, session
``<conversation>/<instance>``, the conversation's user.

- Deleting the conversation deletes its children (``delete_children``, which
  ``ServiceSessions.delete`` calls after the conversation's own delete,
  whatever that found), so the manager's ``delete_session`` and a
  ``SessionReader``'s leave none behind, on every store; their events go
  with them.
- A manager numbers a conversation's instances past the ones its store
  keeps (``_stored_instances``, asked once per conversation per turn), so
  neither a new manager (a restart) nor one whose turns another process's
  came between makes a second ``researcher-1``, and a stored transcript is
  never replaced: a run whose instance is stored fails.
- A model switch between turns rebuilds the runners and keeps the
  transcripts, the records and the numbering.
- No child session lists, or is found, as a conversation.
- A message steered to a run is stored where the run read it: after the
  tool result before it, before the answer it prompted, on every store.

Turns run the real manager on ``FakeAPI`` (Claude), on the memory store and
on sqlite.
"""

from __future__ import annotations

import contextlib
import sqlite3

import anthropic
import pytest

pytest.importorskip("google.adk")

from google.adk.events import Event  # noqa: E402
from google.adk.sessions import DatabaseSessionService, InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.tools.registry import register_tool  # noqa: E402
from agentic_cli.workflow.adk import sessions as adk_sessions  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.adk.sessions import child_app  # noqa: E402
from agentic_cli.workflow.permissions import EXEMPT  # noqa: E402
from agentic_cli.workflow.service_registry import WORKFLOW, get_service  # noqa: E402
from agentic_cli.workflow.sessions import get_active_run  # noqa: E402
from agentic_cli.workflow.session_readers import open_session_reader  # noqa: E402
from tests.loop_guard import no_blocking_calls  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI  # noqa: E402
from tests.workflow.test_adk_delegation import (  # noqa: E402,F401
    Team,
    calls,
    delegation,
    project,
    route,
    says,
    team,
)

APP = "app"


@pytest.fixture(params=["memory", "sqlite"])
async def service(request, tmp_path):
    """A session service of each kind ADK keeps conversations in."""
    if request.param == "memory":
        made = InMemorySessionService()
    else:
        made = DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{tmp_path / 'sessions.db'}")
    yield made
    if request.param == "sqlite":
        await made.close()


async def stored_ids(service, app_name: str, user_id: str = "u") -> list[str]:
    listed = await service.list_sessions(app_name=app_name, user_id=user_id)
    return sorted(session.id for session in listed.sessions)


def manager_on(store: str) -> GoogleADKWorkflowManager:
    return GoogleADKWorkflowManager(
        agent_configs=team(),
        settings=BaseSettings(
            session_store=store, default_model="claude-sonnet-4-6", default_user="u"
        ),
    )


async def say(manager, text: str, session_id: str = "s") -> list:
    return [event async for event in manager.process(text, "u", session_id=session_id)]


def two_delegations() -> Team:
    """The coordinator delegates once in each of two conversations."""
    return Team(
        coordinator=[delegation("toolu_c1"), says("Done."), delegation("toolu_c1"), says("Done.")],
        researcher=[says("X is 42."), says("Y is 7.")],
    )


def sqlite_rows(manager, query: str, *args) -> list[tuple]:
    """Rows of the sqlite file the manager's sessions are in."""
    with sqlite3.connect(manager._settings.sessions_dir / "sessions.db") as db:
        return db.execute(query, args).fetchall()


class TestDeleteChildren:
    async def test_it_deletes_the_conversations_children_alone(self, service):
        for app_name, user_id, session_id in [
            (child_app(APP), "u", "s/researcher-1"),
            (child_app(APP), "u", "s/researcher-2"),
            (child_app(APP), "u", "s/writer-1"),
            (child_app(APP), "u", "t/researcher-1"),     # another conversation's
            (child_app(APP), "u", "s/x/researcher-1"),   # the conversation "s/x"'s
            (child_app(APP), "v", "s/researcher-1"),     # another user's
            (APP, "u", "s"),                             # the conversation itself
        ]:
            await service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)

        deleted = await adk_sessions.delete_children(service, APP, "u", "s")

        assert deleted == 3
        assert await stored_ids(service, child_app(APP)) == ["s/x/researcher-1", "t/researcher-1"]
        assert await stored_ids(service, child_app(APP), "v") == ["s/researcher-1"]
        assert await stored_ids(service, APP) == ["s"]

    async def test_none_is_none(self, service):
        assert await adk_sessions.delete_children(service, APP, "u", "s") == 0

    async def test_their_events_go_with_them(self, tmp_path):
        database = tmp_path / "sessions.db"

        def events() -> int:
            with contextlib.closing(sqlite3.connect(database)) as db:
                return db.execute("select count(*) from events").fetchone()[0]

        service = DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{database}")
        try:
            child = await service.create_session(
                app_name=child_app(APP), user_id="u", session_id="s/researcher-1"
            )
            await service.append_event(child, Event(
                author="researcher",
                content=types.Content(role="model", parts=[types.Part(text="X is 42.")]),
            ))
            before = events()

            await adk_sessions.delete_children(service, APP, "u", "s")
        finally:
            await service.close()

        assert (before, events()) == (1, 0)


class TestDeletingTheConversation:
    @pytest.mark.parametrize("store", ["memory", "sqlite"])
    async def test_its_delete_deletes_its_runs_transcripts(self, store, project, monkeypatch):
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(two_delegations()))
        manager = manager_on(store)
        try:
            await say(manager, "Find X.", "s")
            await say(manager, "Find Y.", "t")
            before = await stored_ids(manager._session_service, child_app(manager.app_name))
            deleted = await manager.delete_session("s", user_id="u")
            after = await stored_ids(manager._session_service, child_app(manager.app_name))
        finally:
            await manager.cleanup()

        assert before == ["s/researcher-1", "t/researcher-1"]
        assert deleted is True
        assert after == ["t/researcher-1"]

    async def test_a_conversation_already_gone_leaves_none_behind(self, project, monkeypatch):
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(two_delegations()))
        manager = manager_on("memory")
        try:
            await say(manager, "Find X.")
            service = manager._session_service
            await service.delete_session(app_name=manager.app_name, user_id="u", session_id="s")
            deleted = await manager.delete_session("s", user_id="u")
            after = await stored_ids(service, child_app(manager.app_name))
        finally:
            await manager.cleanup()

        assert (deleted, after) == (False, [])  # the answer is still about the conversation

    async def test_a_readers_delete_deletes_them_too(self, project, monkeypatch):
        """``/sessions --delete`` of a conversation on a backend that is not running."""
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(two_delegations()))
        manager = manager_on("sqlite")
        try:
            await say(manager, "Find X.", "s")
            await say(manager, "Find Y.", "t")
        finally:
            await manager.cleanup()

        async with await open_session_reader("adk", manager._settings) as reader:
            deleted = await reader.delete("s")

        assert deleted is True
        assert sqlite_rows(manager, "select app_name, id from sessions order by id") == [
            (manager.app_name, "t"), (child_app(manager.app_name), "t/researcher-1"),
        ]
        assert {app for (app,) in sqlite_rows(manager, "select distinct session_id from events")} == {
            "t", "t/researcher-1",
        }

    async def test_a_delete_blocks_no_event_loop(self, project, monkeypatch):
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(two_delegations()))
        manager = manager_on("sqlite")
        try:
            await say(manager, "Find X.")
            with no_blocking_calls():
                deleted = await manager.delete_session("s", user_id="u")
            after = await stored_ids(manager._session_service, child_app(manager.app_name))
        finally:
            await manager.cleanup()

        assert (deleted, after) == (True, [])


def texts_of(session) -> list[str]:
    """The texts a session's events hold, in order."""
    return [
        part.text for event in session.events if event.content is not None
        for part in event.content.parts or () if part.text
    ]


async def child(manager, instance: str, conversation: str = "s"):
    return await manager._session_service.get_session(
        app_name=child_app(manager.app_name), user_id="u", session_id=f"{conversation}/{instance}"
    )


def instances(manager, conversation: str = "s") -> list[tuple[str, str]]:
    """The conversation's runs this manager recorded: (run, status), in start order."""
    return [
        (record.ref.run, record.status)
        for record in manager.agent_runs.records(manager.session_ref(conversation, "u"))
    ]


def both(*answers: dict) -> dict:
    """One answer that makes every call ``answers`` make."""
    return {**answers[0], "content": [block for answer in answers for block in answer["content"]]}


class TestStoredInstances:
    async def test_the_highest_stored_number_of_each_type(self, service):
        for session_id in [
            "s/researcher-1", "s/researcher-3", "s/writer-2",
            "t/researcher-7",            # another conversation's
            "s/x/researcher-9",          # the conversation "s/x"'s
            "s/notes", "s/writer-two",   # no instance of the framework's
        ]:
            await service.create_session(
                app_name=child_app(APP), user_id="u", session_id=session_id
            )

        counts = await adk_sessions.stored_instances(service, APP, "u", "s")

        assert counts == {"researcher": 3, "writer": 2}

    async def test_none_is_none(self, service):
        assert await adk_sessions.stored_instances(service, APP, "u", "s") == {}


class TestNumberedPastTheStored:
    """Instances are numbered per manager: each turn numbers past the
    instances the conversation's store keeps (``_stored_instances``), which
    another manager may have run (before a restart or a switch of backend
    or store, or in another process between two turns), so a stored
    transcript is never another run's."""

    async def test_a_restart_numbers_past_them_and_keeps_their_transcripts(
        self, project, monkeypatch
    ):
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(two_delegations()))
        first = manager_on("sqlite")
        try:
            await say(first, "Find X.")
        finally:
            await first.cleanup()

        second = manager_on("sqlite")
        try:
            await say(second, "Find Y.")
            runs = instances(second)
            earlier, later = await child(second, "researcher-1"), await child(second, "researcher-2")
        finally:
            await second.cleanup()

        assert runs == [("researcher-2#1", "completed")]
        assert texts_of(earlier)[1:] == ["X is 42."]
        assert texts_of(later)[1:] == ["Y is 7."]

    async def test_parallel_first_runs_after_a_restart_are_both_numbered_past_them(
        self, project, monkeypatch
    ):
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(Team(
            coordinator=[
                delegation("toolu_c1"), says("Done."),
                both(delegation("toolu_c1"), delegation("toolu_c2")), says("Both done."),
            ],
            researcher=[says("X is 42."), says("Y is 7."), says("Z is 3.")],
        )))
        first = manager_on("sqlite")
        try:
            await say(first, "Find X.")
        finally:
            await first.cleanup()

        second = manager_on("sqlite")
        asked: list = []
        looked = second._stored_instances

        async def counted(conversation):
            asked.append(conversation)
            return await looked(conversation)

        second._stored_instances = counted
        try:
            await say(second, "Find Y and Z.")
            runs = sorted(run for run, _ in instances(second))
            stored = await stored_ids(second._session_service, child_app(second.app_name))
        finally:
            await second.cleanup()

        assert runs == ["researcher-2#1", "researcher-3#1"]
        assert stored == ["s/researcher-1", "s/researcher-2", "s/researcher-3"]
        assert asked == [second.session_ref("s", "u")]

    async def test_turns_another_manager_ran_in_between_are_numbered_past(
        self, project, monkeypatch
    ):
        """Two processes take turns on one conversation (each holds it while
        it runs a turn: ``ConversationLock``): each turn's first run is
        numbered past what the other stored in between."""
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(Team(
            coordinator=[delegation("toolu_c1"), says("Done.")] * 3,
            researcher=[says("X is 42."), says("Y is 7."), says("Z is 3.")],
        )))
        one, two = manager_on("sqlite"), manager_on("sqlite")
        try:
            await say(one, "Find X.")
            await say(two, "Find Y.")
            await say(one, "Find Z.")
            runs = {"one": instances(one), "two": instances(two)}
            last = await child(one, "researcher-3")
        finally:
            await one.cleanup()
            await two.cleanup()

        assert runs == {
            "one": [("researcher-1#1", "completed"), ("researcher-3#1", "completed")],
            "two": [("researcher-2#1", "completed")],
        }
        assert texts_of(last)[1:] == ["Z is 3."]

    async def test_a_stored_transcript_is_never_replaced(self, project, monkeypatch):
        """A run whose instance the store already keeps (here: a store that
        could not tell what it keeps) fails, and the transcript stays."""
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(two_delegations()))
        first = manager_on("sqlite")
        try:
            await say(first, "Find X.")
        finally:
            await first.cleanup()

        second = manager_on("sqlite")

        async def keeps_none(conversation):
            return {}

        second._stored_instances = keeps_none
        try:
            await say(second, "Find Y.")
            (record,) = second.agent_runs.records(second.session_ref("s", "u"))
            kept = await child(second, "researcher-1")
        finally:
            await second.cleanup()

        assert (record.ref.run, record.status) == ("researcher-1#1", "failed")
        assert record.error == "Session with id s/researcher-1 already exists."
        assert texts_of(kept)[1:] == ["X is 42."]

    async def test_the_look_blocks_no_event_loop(self, project, monkeypatch):
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(two_delegations()))
        first = manager_on("sqlite")
        try:
            await say(first, "Find X.")
        finally:
            await first.cleanup()

        second = manager_on("sqlite")
        try:
            await second.initialize_services()
            with no_blocking_calls():
                counts = await second._stored_instances(second.session_ref("s", "u"))
        finally:
            await second.cleanup()

        assert counts == {"researcher": 1}


class TestAModelSwitchBetweenTurns:
    """A model switch between turns: ``/settings`` reinitializes the manager
    in place (``reinitialize(preserve_sessions=True)``)."""

    async def test_the_runners_are_rebuilt_and_the_transcripts_and_records_stay(
        self, project, monkeypatch
    ):
        answers = two_delegations()
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
        manager = manager_on("memory")
        try:
            await say(manager, "Find X.")
            before = manager._agent_runners["researcher"]
            await manager.reinitialize(model="claude-haiku-4-5", preserve_sessions=True)
            after = manager._agent_runners["researcher"]
            await say(manager, "Find Y.")
            runs = instances(manager)
            earlier, later = await child(manager, "researcher-1"), await child(manager, "researcher-2")
        finally:
            await manager.cleanup()

        assert after is not before
        assert [body["model"] for body in answers.bodies("researcher")] == [
            "claude-sonnet-4-6", "claude-haiku-4-5",
        ]
        assert runs == [("researcher-1#1", "completed"), ("researcher-2#1", "completed")]
        assert texts_of(earlier)[1:] == ["X is 42."]
        assert texts_of(later)[1:] == ["Y is 7."]

    async def test_a_new_managers_first_run_on_the_new_model_is_numbered_past_the_stored(
        self, project, monkeypatch
    ):
        answers = two_delegations()
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
        first = manager_on("sqlite")
        try:
            await say(first, "Find X.")
        finally:
            await first.cleanup()

        second = manager_on("sqlite")
        try:
            await second.initialize_services()
            await second.reinitialize(model="claude-haiku-4-5", preserve_sessions=True)
            await say(second, "Find Y.")
            runs = instances(second)
        finally:
            await second.cleanup()

        assert runs == [("researcher-2#1", "completed")]
        assert [body["model"] for body in answers.bodies("researcher")] == [
            "claude-sonnet-4-6", "claude-haiku-4-5",
        ]


class TestNeverListed:
    """A child session is under the conversation's child app, so none lists,
    or is found, as a conversation: not by a reader (``/sessions`` and the
    picker on a backend that is not running), nor by the CLI's look at every
    backend's conversations (``cli/stored_conversations.py``), the running
    manager's or the readers'."""

    async def test_no_reader_or_look_shows_one(self, project, monkeypatch):
        from agentic_cli.cli.stored_conversations import StoredConversations

        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(two_delegations()))
        manager = manager_on("sqlite")
        try:
            await say(manager, "Find X.")
            async with StoredConversations(
                manager._settings, "sqlite", workflow=manager, backend="adk"
            ) as look:
                running = await look.rows()
                found = await look.where("s/researcher-1")
        finally:
            await manager.cleanup()

        async with await open_session_reader("adk", manager._settings) as reader:
            listed = await reader.list()
            info = await reader.info("s/researcher-1")
            exists = await reader.exists("s/researcher-1")
        async with StoredConversations(manager._settings, "sqlite") as look:  # no manager runs
            read = await look.rows()

        assert [(row.session_id, row.backend) for row in running] == [("s", "adk")]
        assert found == []
        assert [row.session_id for row in listed] == ["s"]
        assert (info, exists) == (None, False)
        assert [(row.session_id, row.backend) for row in read] == [("s", "adk")]


@register_tool(capabilities=EXEMPT, description="Hold, and steer this run (a test tool).")
async def child_sessions_steered_hold() -> dict:
    """A message steered to the run that called it, as its parent would, while it works."""
    get_service(WORKFLOW)._delegations.control(get_active_run()).steer("Also find Z.")
    return {"success": True}


def what_happened(session) -> list[str]:
    """Each of a session's events, as what it holds: a text, a call, a result."""
    told: list[str] = []
    for event in session.events:
        for part in (event.content.parts if event.content is not None else ()) or ():
            if part.function_call:
                told.append(f"{event.author}: call {part.function_call.name}")
            elif part.function_response:
                told.append(f"{event.author}: result {part.function_response.name}")
            elif part.text:
                told.append(f"{event.author}: {part.text.splitlines()[0]}")
    return told


class TestASteeredMessage:
    @pytest.mark.parametrize("store", ["memory", "sqlite"])
    async def test_it_is_stored_between_the_result_and_the_answer_it_prompted(
        self, project, monkeypatch, store
    ):
        """ADK stamps a model call's answer before the call's callbacks run,
        so the message steered there was newer than the answer it prompted:
        a durable store, which orders a session's events by time, read it
        back after that answer."""
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(Team(
            coordinator=[delegation("toolu_c1"), says("Done.")],
            researcher=[calls("child_sessions_steered_hold", "toolu_r1"),
                        says("X is 42; Z is 42.")],
        )))
        manager = GoogleADKWorkflowManager(
            agent_configs=team(researcher_tools=(child_sessions_steered_hold,)),
            settings=BaseSettings(
                session_store=store, default_model="claude-sonnet-4-6", default_user="u"
            ),
        )
        try:
            await say(manager, "Find X.")
            stored = await child(manager, "researcher-1")
        finally:
            await manager.cleanup()

        assert what_happened(stored) == [
            "user: # Task: Find X",
            "researcher: call child_sessions_steered_hold",
            "researcher: result child_sessions_steered_hold",
            "user: Also find Z.",
            "researcher: X is 42; Z is 42.",
        ]
