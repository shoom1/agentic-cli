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
  keeps (``_stored_instances``, asked once per conversation), so a new
  manager (a restart) never makes a second ``researcher-1``, and a stored
  transcript is never replaced: a run whose instance is stored fails.
- A model switch between turns rebuilds the runners and keeps the
  transcripts, the records and the numbering.
- No child session lists, or is found, as a conversation.

Turns run the real manager on ``FakeAPI`` (Claude), on the memory store and
on sqlite.
"""

from __future__ import annotations

import sqlite3

import anthropic
import pytest

pytest.importorskip("google.adk")

from google.adk.events import Event  # noqa: E402
from google.adk.sessions import DatabaseSessionService, InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.workflow.adk import sessions as adk_sessions  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.adk.sessions import child_app  # noqa: E402
from agentic_cli.workflow.session_readers import open_session_reader  # noqa: E402
from tests.loop_guard import no_blocking_calls  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI  # noqa: E402
from tests.workflow.test_adk_delegation import (  # noqa: E402,F401
    Team,
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
        service = DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{database}")
        try:
            child = await service.create_session(
                app_name=child_app(APP), user_id="u", session_id="s/researcher-1"
            )
            await service.append_event(child, Event(
                author="researcher",
                content=types.Content(role="model", parts=[types.Part(text="X is 42.")]),
            ))

            await adk_sessions.delete_children(service, APP, "u", "s")
        finally:
            await service.close()

        with sqlite3.connect(database) as db:
            assert db.execute("select count(*) from events").fetchone() == (0,)


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
