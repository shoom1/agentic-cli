"""On ADK a conversation's row comes from two keys of its session state.

Each user turn saves them with its message (``Runner.run_async(state_delta=)``):
``agentic_cli.title``, its first prompt (``session_title``), and
``agentic_cli.turns``, the turns so far. ADK's listing returns every session's
state without its events, so ``list_sessions`` stays cheap. A conversation
from before the keys is read from its events, for the rows returned, and
nothing is written back: its place in the list stays. A job's resume adds no
turn, nor does a delegated request or a summary: nobody typed them.

Turns run on the real ADK manager, Claude on ``FakeAPI``; the conversations
from before the keys are written straight into a sqlite session store.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import anthropic
import pytest

pytest.importorskip("google.adk")

from google.adk.events import Event, EventActions  # noqa: E402
from google.adk.events.event_actions import EventCompaction  # noqa: E402
from google.genai import types  # noqa: E402

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.tools.jobs.resume import JOB_NOTICE_HEADER  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.adk.sessions import (  # noqa: E402
    TITLE_KEY,
    TURNS_KEY,
    typed_prompts,
)
from agentic_cli.workflow.sessions import SessionInfo  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI  # noqa: E402
from tests.workflow.test_adk_delegation import (  # noqa: E402,F401
    STARTED,
    TURNS,
    Team,
    _record,
    calls,
    delegating,
    delegation_probe_start_job,
    manager_for,
    project,
    route,
    says,
    team,
    texts,
    turn,
)


def chat(*replies: str) -> Team:
    """The coordinator answers each turn with the next reply."""
    return Team(coordinator=[says(reply) for reply in replies])


async def say(manager, text: str, session_id: str) -> None:
    async for _ in manager.process(text, "u", session_id=session_id):
        pass


def sqlite_manager() -> GoogleADKWorkflowManager:
    return GoogleADKWorkflowManager(
        agent_configs=team(),
        settings=BaseSettings(session_store="sqlite", default_model="claude-sonnet-4-6"),
    )


def text_event(author: str, text: str, *, thought: bool = False) -> Event:
    role = "user" if author == "user" else "model"
    return Event(author=author, content=types.Content(
        role=role, parts=[types.Part(text=text, thought=thought or None)]
    ))


async def seed(manager, session_id: str, events: list[Event]) -> None:
    """A conversation written as before the keys: events, no state."""
    service = manager._session_service
    session = await service.create_session(
        app_name=manager.app_name, user_id="u", session_id=session_id
    )
    for event in events:
        await service.append_event(session=session, event=event)


async def stored(manager, session_id: str):
    return await manager._session_service.get_session(
        app_name=manager.app_name, user_id="u", session_id=session_id
    )


class TestTypedPrompts:
    def test_only_what_the_user_typed(self):
        summary = Event(author="user", actions=EventActions(compaction=EventCompaction(
            start_timestamp=1.0, end_timestamp=2.0,
            compacted_content=types.Content(role="user", parts=[types.Part(text="Summary.")]),
        )))
        job_result = Event(author="user", content=types.Content(role="user", parts=[
            types.Part(function_response=types.FunctionResponse(
                id="c1", name="start_job", response={"result": "done"}
            ))
        ]))
        events = [
            text_event("user", "Plan the trip."),
            text_event("coordinator", "Sure."),
            text_event("coordinator", "Let me think.", thought=True),
            summary,
            job_result,
            text_event("user", f"{JOB_NOTICE_HEADER}\n{{\"job_id\": \"j1\"}}"),
            text_event("researcher", "X is 42."),
            text_event("user", "And the budget?"),
        ]

        assert typed_prompts(events) == ["Plan the trip.", "And the budget?"]


class TestRows:
    async def test_two_turns_give_the_first_prompt_and_two_turns(self, project, monkeypatch):
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(chat("Sure.", "About 900.")))
        manager = manager_for(team())
        try:
            await turn(manager, "  Plan the\ntrip to   Lisbon ")
            await turn(manager, "And the budget?")
            (row,) = await manager.list_sessions(user_id="u")
            state = (await stored(manager, "s")).state
        finally:
            await manager.cleanup()

        assert (row.session_id, row.title, row.turns) == ("s", "Plan the trip to Lisbon", 2)
        assert abs(datetime.now(timezone.utc) - row.updated) < timedelta(minutes=1)
        assert (state[TITLE_KEY], state[TURNS_KEY]) == ("Plan the trip to Lisbon", 2)

    async def test_newest_first_and_the_limit_keeps_the_newest(self, project, monkeypatch):
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(chat("A.", "B.", "C.")))
        manager = manager_for(team())
        try:
            await say(manager, "First.", "s1")
            await say(manager, "Second.", "s2")
            await say(manager, "First, again.", "s1")
            listed = await manager.list_sessions(user_id="u")
            newest = await manager.list_sessions(user_id="u", limit=1)
        finally:
            await manager.cleanup()

        assert [(r.session_id, r.title, r.turns) for r in listed] == [
            ("s1", "First.", 2), ("s2", "Second.", 1),
        ]
        assert newest == listed[:1]

    async def test_session_info_is_the_conversations_row(self, project, monkeypatch):
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(chat("Hi.")))
        manager = manager_for(team())
        try:
            await turn(manager, "Hello.")
            info = await manager.session_info("s", user_id="u")
            listed = await manager.list_sessions(user_id="u")
            missing = await manager.session_info("missing", user_id="u")
        finally:
            await manager.cleanup()

        assert listed == [info] and info.title == "Hello."
        assert missing is None

    async def test_a_jobs_resume_is_no_turn(self, project, monkeypatch):
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(Team(coordinator=[
            calls("delegation_probe_start_job", "toolu_j1", command="make"),
            says("Started job j1."),
            says("The build finished."),
        ])))
        manager = manager_for(team(tools=[delegation_probe_start_job]))
        try:
            await turn(manager, "Build it.")
            resumed = [e async for e in manager.resume_with_job_result(
                _record(STARTED[-1]), result="ok")]
            (row,) = await manager.list_sessions(user_id="u")
        finally:
            await manager.cleanup()

        assert "The build finished." in texts(resumed)  # the resume ran
        assert (row.title, row.turns) == ("Build it.", 1)

    async def test_a_blank_prompt_is_no_turn(self, project, monkeypatch):
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(chat("Yes?", "Hi.")))
        manager = manager_for(team())
        try:
            await turn(manager, "   ")
            await turn(manager, "Hello.")
            (row,) = await manager.list_sessions(user_id="u")
        finally:
            await manager.cleanup()

        assert (row.title, row.turns) == ("Hello.", 1)

    async def test_a_delegated_request_is_no_turn(self, project, monkeypatch):
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(delegating()))
        manager = manager_for(team("inherit"))
        try:
            for text in TURNS:
                await turn(manager, text)
            (row,) = await manager.list_sessions(user_id="u")
        finally:
            await manager.cleanup()

        assert (row.title, row.turns) == (TURNS[0], 3)


class TestTheListingReadsTheKeys:
    async def test_only_a_conversation_from_before_the_keys_is_read(self, project, monkeypatch):
        """The keys are what keeps a listing cheap: ADK's listing returns
        state without events, and only a conversation without keys is read."""
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(chat("A.", "B.")))
        manager = sqlite_manager()
        await manager.initialize_services()
        try:
            await say(manager, "First.", "s1")
            await seed(manager, "old", [
                text_event("user", "Before the keys."), text_event("coordinator", "Ok."),
            ])
            await say(manager, "Second.", "s2")
            service = manager._session_service
            reads: list[str] = []
            get_session = service.get_session

            async def counting(**kwargs):
                reads.append(kwargs["session_id"])
                return await get_session(**kwargs)

            monkeypatch.setattr(service, "get_session", counting)
            listed = await manager.list_sessions(user_id="u")
            newest = await manager.list_sessions(user_id="u", limit=1)
        finally:
            await manager.cleanup()

        assert [r.session_id for r in listed] == ["s2", "old", "s1"]
        assert [r.session_id for r in newest] == ["s2"]
        assert reads == ["old"]


class TestAConversationFromBeforeTheKeys:
    async def test_its_row_is_read_from_its_events_and_it_is_not_moved(
        self, project, monkeypatch
    ):
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(chat("New.")))
        manager = sqlite_manager()
        await manager.initialize_services()
        try:
            await seed(manager, "old", [
                text_event("user", "Plan the trip."), text_event("coordinator", "Sure."),
                text_event("user", "And the budget?"), text_event("coordinator", "900."),
            ])
            await say(manager, "Something else.", "new")
            before = (await stored(manager, "old")).last_update_time
            listed = await manager.list_sessions(user_id="u")
            info = await manager.session_info("old", user_id="u")
            after = await stored(manager, "old")
        finally:
            await manager.cleanup()

        assert [(r.session_id, r.title, r.turns) for r in listed] == [
            ("new", "Something else.", 1), ("old", "Plan the trip.", 2),
        ]
        assert info == listed[1]
        assert after.last_update_time == before
        assert TURNS_KEY not in after.state  # nothing written back

    async def test_its_next_turn_counts_the_turns_it_had(self, project, monkeypatch):
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(chat("Again.")))
        manager = sqlite_manager()
        await manager.initialize_services()
        try:
            await seed(manager, "old", [
                text_event("user", "Plan the trip."), text_event("coordinator", "Sure."),
            ])
            await say(manager, "And the budget?", "old")
            (row,) = await manager.list_sessions(user_id="u")
        finally:
            await manager.cleanup()

        assert row == SessionInfo("old", "Plan the trip.", 2, row.updated)

    async def test_a_session_without_events_is_not_listed(self, project):
        manager = sqlite_manager()
        await manager.initialize_services()
        try:
            await seed(manager, "empty", [])
            listed = await manager.list_sessions(user_id="u")
            info = await manager.session_info("empty", user_id="u")
        finally:
            await manager.cleanup()

        assert listed == [] and info is None
