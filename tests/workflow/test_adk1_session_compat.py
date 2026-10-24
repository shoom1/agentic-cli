"""Sessions that 0.6.x saved still open, whatever ADK is installed.

0.6.x ran on ADK 1.x, so its durable sessions are databases ADK 1.x wrote.
``tests/fixtures/adk1_sessions.sql`` is a text dump of one, made by ADK 1.36
(``tests/fixtures/make_adk1_sessions.py``): session state, a user turn, a
function call and its response with their IDs, a pending long-running call and
a model reply. These tests load it into a fresh file, point the ADK manager's
session store at it, read it back through the manager's own session hooks and
add a turn. They passed on ADK 1.36 before the move to ADK 2 and must keep
passing after it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("google.adk")

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "adk1_sessions.sql"
APP_NAME, USER_ID, SESSION_ID = "research_demo", "tester", "adk1-session"


def _load(db: Path) -> None:
    with sqlite3.connect(db) as conn:
        conn.executescript(FIXTURE.read_text())


def _manager(tmp_path: Path, db: Path) -> GoogleADKWorkflowManager:
    settings = BaseSettings(
        workspace_dir=tmp_path / "workspace",
        session_store="sqlite",
        sqlite_uri=f"sqlite:///{db}",
    )
    manager = GoogleADKWorkflowManager(
        agent_configs=[], settings=settings, app_name=APP_NAME
    )
    manager._session_service = manager._make_session_service()
    return manager


@pytest.fixture
async def open_manager(tmp_path):
    db = tmp_path / "sessions.db"
    _load(db)
    managers: list[GoogleADKWorkflowManager] = []

    def _open() -> GoogleADKWorkflowManager:
        managers.append(_manager(tmp_path, db))
        return managers[-1]

    yield _open
    for manager in managers:
        await manager._session_service.close()


async def test_the_session_hooks_find_the_saved_session(open_manager):
    manager = open_manager()

    assert await manager.session_exists(SESSION_ID, user_id=USER_ID) is True
    listed = await manager.list_sessions(user_id=USER_ID)
    # Saved before the title and turn keys: its row is read from its events.
    assert [(s.session_id, s.title, s.turns) for s in listed] == [
        (SESSION_ID, "Search arXiv for speculative decoding.", 1)
    ]
    assert await manager.recent_messages(SESSION_ID, user_id=USER_ID) == [
        {"role": "user", "content": "Search arXiv for speculative decoding."},
        {"role": "assistant", "content": "Found 2 papers; the analysis is running."},
    ]


async def test_state_call_ids_and_the_pending_call_survive(open_manager):
    manager = open_manager()

    session = await manager._session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )

    assert session.state["plan"] == "1. Search arXiv"
    assert session.state["user:lang"] == "en"
    calls = [
        (p.function_call.id, p.function_call.name)
        for e in session.events
        for p in e.content.parts
        if p.function_call
    ]
    assert calls == [("call-1", "search_arxiv"), ("call-2", "sandbox_execute")]
    responses = [
        (p.function_response.id, p.function_response.response)
        for e in session.events
        for p in e.content.parts
        if p.function_response
    ]
    assert responses == [("call-1", {"success": True, "count": 2})]
    pending = [e.long_running_tool_ids for e in session.events if e.long_running_tool_ids]
    assert pending == [{"call-2"}]


async def test_a_new_turn_is_saved_and_read_back(open_manager):
    from google.adk.events import Event
    from google.genai import types

    manager = open_manager()
    session = await manager._session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    await manager._session_service.append_event(
        session,
        Event(
            author="user",
            invocation_id="inv-2",
            content=types.Content(role="user", parts=[types.Part(text="And the second paper?")]),
        ),
    )

    # A second manager over the same file: the turn reached the database.
    reopened = open_manager()
    recent = await reopened.recent_messages(SESSION_ID, user_id=USER_ID)
    assert len(recent) == 3
    assert recent[-1] == {"role": "user", "content": "And the second paper?"}
