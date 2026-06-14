"""Tests that ADK session resume actually persists restored messages.

The bug: _inject_session_messages appended events to the copy returned by
create_session, leaving the stored session empty — so resume silently restored
nothing. The fix uses append_event with real Event objects.
"""

import pytest

pytest.importorskip("google.adk")

from google.adk.sessions import InMemorySessionService

from agentic_cli.config import BaseSettings
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager


def _manager(tmp_path) -> tuple[GoogleADKWorkflowManager, BaseSettings]:
    settings = BaseSettings(workspace_dir=tmp_path)
    mgr = GoogleADKWorkflowManager(
        agent_configs=[], settings=settings, model="gemini-2.0-flash"
    )
    mgr._session_service = InMemorySessionService()
    return mgr, settings


async def test_inject_persists_to_stored_session(tmp_path):
    mgr, settings = _manager(tmp_path)
    messages = [
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "name": "search", "args": {"q": "x"}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "name": "search", "content": '{"result": 42}'},
    ]

    await mgr._inject_session_messages("s1", messages, current_agent="root")

    stored = await mgr._session_service.get_session(
        app_name=mgr.app_name, user_id=settings.default_user, session_id="s1"
    )
    assert stored is not None
    # Was 0 before the fix (events appended to the discarded copy).
    assert len(stored.events) == 4


async def test_inject_extract_roundtrip(tmp_path):
    mgr, _ = _manager(tmp_path)
    messages = [
        {"role": "user", "content": "remember the alpha value"},
        {"role": "assistant", "content": "noted"},
    ]

    await mgr._inject_session_messages("s2", messages, current_agent="root")
    got, _agent = await mgr._extract_session_data("s2")

    assert [m["role"] for m in got] == ["user", "assistant"]
    assert "remember the alpha value" in got[0]["content"]
    assert got[1]["content"] == "noted"
