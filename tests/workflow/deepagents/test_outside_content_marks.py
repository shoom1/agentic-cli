"""Deep Agents: the gate marks the conversation, and the mark is kept in the
checkpoint's state (first write wins), so a resumed conversation keeps it."""

from __future__ import annotations

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from agentic_cli.tools.registry import ToolCategory, register_tool
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.deepagents.middleware import PermissionMiddleware, with_state_update
from agentic_cli.workflow.deepagents.state import _first
from agentic_cli.workflow.deepagents.tools import to_langchain_tools
from agentic_cli.workflow.outside_content import OutsideContentMarks
from agentic_cli.workflow.permissions import EXEMPT, Capability
from agentic_cli.workflow.sessions import SessionRef, reset_active_turn, set_active_turn
from tests.conftest import MockContext
from tests.workflow.deepagents.fakes import ai, task
from tests.workflow.deepagents.test_manager import _manager, hermetic, scripted, turn  # noqa: F401  (hermetic: a fixture)
from tests.workflow.deepagents.test_permission_middleware import _engine, _gate

REF = SessionRef(app_name="app", user_id="u", session_id="s")


def test_first_write_wins():
    assert _first("", "web_fetch") == "web_fetch"
    assert _first("web_fetch", "kb_read") == "web_fetch"


def test_a_tool_message_becomes_a_command_with_the_mark():
    message = ToolMessage(content="page", tool_call_id="c1")
    command = with_state_update(message, {"outside_content": "web_fetch"})
    assert isinstance(command, Command)
    assert command.update == {"outside_content": "web_fetch", "messages": [message]}


def test_a_command_keeps_its_update():
    command = Command(update={"plan": "p", "messages": []})
    marked = with_state_update(command, {"outside_content": "web_fetch"})
    assert marked.update == {"plan": "p", "messages": [], "outside_content": "web_fetch"}


@register_tool(category=ToolCategory.NETWORK, capabilities=EXEMPT,
               description="Fetch a probe page (an exempt test tool).", outside_content=True)
def da_marks_probe_fetch() -> dict:
    """Return a page, as a fetch would."""
    return {"success": True, "text": "A page."}


@register_tool(category=ToolCategory.NETWORK, capabilities=[Capability("http.read", target="https://example.org/")],
               description="Fetch a gated probe page (a test tool).", outside_content=True)
def da_marks_probe_gated() -> dict:
    """Return a page, as a fetch would."""
    return {"success": True, "text": "A page."}


async def test_an_allowed_marking_tool_marks_and_returns_the_mark():
    marks = OutsideContentMarks()
    [tool] = to_langchain_tools([da_marks_probe_fetch])
    token = set_active_turn(REF)
    try:
        result, ran = await _gate(tool, engine=_engine(), gate=PermissionMiddleware("a", marks=marks))
    finally:
        reset_active_turn(token)
    assert ran == ["da_marks_probe_fetch"]
    assert isinstance(result, Command)
    assert result.update["outside_content"] == "da_marks_probe_fetch"
    assert marks.get(REF) == "da_marks_probe_fetch"


async def test_a_denied_call_marks_nothing():
    marks = OutsideContentMarks()
    [tool] = to_langchain_tools([da_marks_probe_gated])
    token = set_active_turn(REF)
    try:
        result, ran = await _gate(tool, engine=_engine(allowed=False),
                                  gate=PermissionMiddleware("a", marks=marks))
    finally:
        reset_active_turn(token)
    assert ran == [] and isinstance(result, ToolMessage)
    assert marks.get(REF) is None


PROBE = [AgentConfig(name="solo", prompt="Answer.", tools=[da_marks_probe_fetch])]


async def test_a_resumed_deep_agents_conversation_keeps_its_mark(hermetic, monkeypatch):
    """Review Focus 4 (Deep Agents): the mark is in the checkpoint's state."""
    scripted(monkeypatch, solo=[ai("", calls=[("da_marks_probe_fetch", {})]), ai("Read it."), ai("Hello.")])
    with MockContext(google_api_key="test-key", session_store="sqlite") as context:
        first = _manager(context.settings, PROBE)
        try:
            await turn(first, "Fetch the page.")
        finally:
            await first.cleanup()
        second = _manager(context.settings, PROBE)
        try:
            await turn(second, "And now?")
            ref = SessionRef(app_name=second.app_name, user_id="u", session_id="s")
            assert second._outside_marks.get(ref) == "da_marks_probe_fetch"
        finally:
            await second.cleanup()


async def test_a_sub_agent_s_mark_reaches_the_root_s_checkpoint(hermetic, monkeypatch):
    """A sub-agent's fetch marks the conversation, and the root's own checkpoint
    keeps the mark (the sub-agent has none of its own), so a new process sees it."""
    configs = [
        AgentConfig(name="coordinator", prompt="Lead.", sub_agents=["researcher"]),
        AgentConfig(name="researcher", prompt="Research.", description="Researches.",
                    tools=[da_marks_probe_fetch]),
    ]
    scripted(
        monkeypatch,
        coordinator=[ai("", calls=[task("researcher", "Fetch it.")]), ai("Done.")],
        researcher=[ai("", calls=[("da_marks_probe_fetch", {})]), ai("Read it.")],
    )
    with MockContext(google_api_key="test-key", session_store="sqlite") as context:
        manager = _manager(context.settings, configs)
        try:
            await turn(manager, "Research it.")
            ref = SessionRef(app_name=manager.app_name, user_id="u", session_id="s")
            assert manager._outside_marks.get(ref) == "da_marks_probe_fetch"
            assert await manager._sessions.outside_content(ref) == "da_marks_probe_fetch"
        finally:
            await manager.cleanup()
