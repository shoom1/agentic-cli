"""ADK: a tool that brings outside content marks the conversation, in its
session state, so a resumed conversation keeps the mark (spec §9.1)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from google.adk.tools import FunctionTool

from agentic_cli.config import BaseSettings
from agentic_cli.tools.registry import ToolCategory, register_tool
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager
from agentic_cli.workflow.adk.permission_plugin import PermissionPlugin
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.permissions import EXEMPT
from agentic_cli.workflow.outside_content import (
    BEFORE_0_7_0,
    OUTSIDE_CONTENT_KEY,
    OutsideContentMarks,
)
from agentic_cli.workflow.sessions import SessionRef, reset_active_turn, set_active_turn
from tests.workflow.test_adk_compaction import Claude, calls, project, says, serve, turn  # noqa: F401  (project: a fixture)

REF = SessionRef(app_name="app", user_id="u", session_id="s")


@register_tool(category=ToolCategory.NETWORK, capabilities=EXEMPT,
               description="Fetch a probe page (an exempt test tool).", outside_content=True)
def marks_probe_fetch() -> dict:
    """Return a page, as a fetch would."""
    return {"success": True, "text": "A page."}


def _manager(project) -> GoogleADKWorkflowManager:
    return GoogleADKWorkflowManager(
        agent_configs=[AgentConfig(name="solo", prompt="You help.", tools=[marks_probe_fetch],
                                   include_state_tools=False)],
        settings=BaseSettings(session_store="sqlite", default_model="claude-sonnet-4-6",
                              workspace_dir=project / "w"),
    )


class _State(dict):
    """Stands in for ADK's delta-aware state: item writes are recorded."""


def _context(agent="root"):
    return SimpleNamespace(state=_State(), agent_name=agent, function_call_id="call-1")


@pytest.fixture
def active_turn():
    token = set_active_turn(REF)
    yield
    reset_active_turn(token)


async def test_an_allowed_web_fetch_marks_the_conversation_and_its_state(active_turn, monkeypatch):
    import agentic_cli.tools.webfetch_tool as webfetch

    marks = OutsideContentMarks()
    plugin = PermissionPlugin(marks=marks, root="root")
    monkeypatch.setattr(PermissionPlugin, "_gate", _allow)
    context = _context()

    assert await plugin.before_tool_callback(
        tool=FunctionTool(func=webfetch.web_fetch), tool_args={"url": "https://x.org"},
        tool_context=context,
    ) is None
    assert marks.get(REF) == "web_fetch"
    assert context.state[OUTSIDE_CONTENT_KEY] == "web_fetch"


async def test_a_denied_call_marks_nothing(active_turn, monkeypatch):
    marks = OutsideContentMarks()
    plugin = PermissionPlugin(marks=marks, root="root")

    async def deny(self, tool, defn, tool_args, tool_context):
        return {"success": False, "error": "Permission denied: no"}

    monkeypatch.setattr(PermissionPlugin, "_gate", deny)
    import agentic_cli.tools.webfetch_tool as webfetch

    await plugin.before_tool_callback(
        tool=FunctionTool(func=webfetch.web_fetch), tool_args={}, tool_context=_context()
    )
    assert marks.get(REF) is None


async def test_a_sub_agent_s_mark_is_written_at_the_root_s_next_model_call(active_turn):
    marks = OutsideContentMarks()
    marks.mark(REF, "kb_read")
    plugin = PermissionPlugin(marks=marks, root="root")
    sub, root = _context("researcher"), _context("root")
    await plugin.before_model_callback(callback_context=sub, llm_request=None)
    await plugin.before_model_callback(callback_context=root, llm_request=None)
    assert OUTSIDE_CONTENT_KEY not in sub.state
    assert root.state[OUTSIDE_CONTENT_KEY] == "kb_read"


async def _allow(self, tool, defn, tool_args, tool_context):
    return None


def test_marks_keep_the_first_tool():
    marks = OutsideContentMarks()
    assert marks.mark(REF, "web_fetch") is True
    assert marks.mark(REF, "kb_read") is False
    assert marks.get(REF) == "web_fetch"
    marks.seed(REF, "")
    assert marks.get(REF) == "web_fetch"


def test_langgraph_marks_every_conversation():
    marks = OutsideContentMarks(always="the LangGraph backend")
    assert marks.get(REF) == "the LangGraph backend"


async def test_a_resumed_adk_conversation_keeps_its_mark(project, monkeypatch):
    """Review Focus 4 (ADK): the mark is in the session's state, so a manager
    in a new process (here a new manager on the same sqlite file) sees it."""
    serve(monkeypatch, Claude(solo=[calls("marks_probe_fetch", "toolu_1"), says("Read it."), says("Hello.")]))
    first = _manager(project)
    try:
        await turn(first, "Fetch the page.")
    finally:
        await first.cleanup()
    second = _manager(project)
    ref = SessionRef(app_name=second.app_name, user_id="u", session_id="s")
    try:
        assert second._outside_marks.get(ref) is None
        await turn(second, "And now?")
        assert second._outside_marks.get(ref) == "marks_probe_fetch"
    finally:
        await second.cleanup()


async def test_a_conversation_from_before_0_7_0_counts_as_marked(project, monkeypatch):
    from google.adk.events import Event
    from google.genai import types

    serve(monkeypatch, Claude(solo=[says("Hello.")]))
    manager = _manager(project)
    try:
        await manager.initialize_services()
        service = manager._session_service
        session = await service.create_session(app_name=manager.app_name, user_id="u", session_id="old")
        await service.append_event(session, Event(
            author="user",
            content=types.Content(role="user", parts=[types.Part.from_text(text="A 0.6 turn.")]),
        ))
        await turn(manager, "Hi.", session_id="old")
        ref = SessionRef(app_name=manager.app_name, user_id="u", session_id="old")
        assert manager._outside_marks.get(ref) == BEFORE_0_7_0
    finally:
        await manager.cleanup()


async def test_a_new_conversation_records_that_it_is_unmarked(project, monkeypatch):
    serve(monkeypatch, Claude(solo=[says("Hello.")]))
    manager = _manager(project)
    try:
        await turn(manager, "Hi.", session_id="new")
        stored = await manager._session_service.get_session(
            app_name=manager.app_name, user_id="u", session_id="new"
        )
        assert stored.state[OUTSIDE_CONTENT_KEY] == ""
        ref = SessionRef(app_name=manager.app_name, user_id="u", session_id="new")
        assert manager._outside_marks.get(ref) is None
    finally:
        await manager.cleanup()

