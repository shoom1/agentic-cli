"""ADK: a call to a tool name the agent does not have gets ADK's own error.

The model made the name up, so it reads what Deep Agents answers too
(``tool_errors.unknown_tool_result``): "<name> failed: ValueError: Tool
'<name>' not found." and the agent's tools, by name. ADK resolves the name
to a bare ``BaseTool`` stand-in and runs the before-tool callbacks on it
first, so the permission gate lets that stand-in through (it runs nothing)
and ``ToolErrorPlugin`` answers ADK's lookup error. It is the model's
mistake, not the user's: the log line is at info, not a warning over the
prompt.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import anthropic
import pytest
from structlog.testing import capture_logs

pytest.importorskip("google.adk")

from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.agents.invocation_context import InvocationContext  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.adk.tools import BaseTool, FunctionTool  # noqa: E402
from google.genai import types  # noqa: E402

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.workflow.adk.interrupted_calls_plugin import InterruptedCallsPlugin  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.adk.permission_plugin import PermissionPlugin  # noqa: E402
from agentic_cli.workflow.adk.tool_error_plugin import ToolErrorPlugin  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from agentic_cli.workflow.interrupts import OpenCalls  # noqa: E402
from agentic_cli.workflow.sessions import get_active_tool_call  # noqa: E402
from agentic_cli.workflow.tool_errors import unknown_tool_result  # noqa: E402
from tests.workflow.adk.fakes import anthropic_message  # noqa: E402
from tests.workflow.test_adk_overflow import Scripted  # noqa: E402
from tests.workflow.test_manager_models import route  # noqa: E402

MODEL = "claude-sonnet-4-6"
#: The state tools every agent gets by default: what ADK lists as available.
STATE_TOOLS = ["save_plan", "get_plan", "save_tasks", "get_tasks"]


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def _calls(name: str) -> dict:
    message = anthropic_message(model=MODEL)
    message["content"] = [{"type": "tool_use", "id": "toolu_1", "name": name, "input": {}}]
    message["stop_reason"] = "tool_use"
    return message


def _tool_results(body: dict) -> list:
    return [
        block for message in body["messages"]
        if isinstance(message["content"], list)
        for block in message["content"] if block.get("type") == "tool_result"
    ]


def _stand_in(name: str = "nope") -> BaseTool:
    """ADK's stand-in for an unknown tool name (pinned below)."""
    return BaseTool(name=name, description="Tool not found")


def _not_found(name: str, available: list[str]) -> ValueError:
    """ADK's lookup error for an unknown name, as its first lines read (pinned below)."""
    return ValueError(
        f"Tool '{name}' not found.\nAvailable tools: {', '.join(available)}\n\n"
        "Possible causes:\n  1. LLM hallucinated the function name"
    )


async def test_the_model_reads_adks_error_and_no_warning_is_logged(project, monkeypatch):
    api = Scripted(_calls("nope"), anthropic_message({"type": "text", "text": "done"}, model=MODEL))
    route(monkeypatch, anthropic, "AsyncAnthropic", api)
    manager = GoogleADKWorkflowManager(
        agent_configs=[AgentConfig(name="solo", prompt="Help.")],
        settings=BaseSettings(session_store="memory", default_model=MODEL,
                              prompt_caching_enabled=False),
    )
    try:
        with capture_logs() as logs:
            events = [e async for e in manager.process("Hi", "u", session_id="s")]
    finally:
        await manager.cleanup()

    (result,) = _tool_results(json.loads(api.requests[1].content))
    assert json.loads(result["content"]) == unknown_tool_result("nope", STATE_TOOLS)
    assert [log["event"] for log in logs if log["log_level"] == "warning"] == []
    assert {"event": "tool_not_found", "tool": "nope", "agent": "solo"}.items() <= next(
        log for log in logs if log["event"] == "tool_not_found"
    ).items()
    (tool_result,) = [e for e in events if e.type is EventType.TOOL_RESULT]
    assert (tool_result.metadata["tool_name"], tool_result.metadata["success"]) == ("nope", False)


class TestThePlugins:
    async def test_the_permission_gate_lets_the_stand_in_through(self):
        """It runs nothing: ADK reports its lookup error next."""
        plugin = PermissionPlugin(OpenCalls())

        assert await plugin.before_tool_callback(
            tool=_stand_in(), tool_args={}, tool_context=None
        ) is None

    async def test_the_tool_error_plugin_answers_as_deep_agents_does(self):
        with capture_logs() as logs:
            result = await ToolErrorPlugin().on_tool_error_callback(
                tool=_stand_in(), tool_args={},
                tool_context=SimpleNamespace(agent_name="solo"),
                error=_not_found("nope", ["save_plan", "get_plan"]),
            )

        assert result == unknown_tool_result("nope", ["save_plan", "get_plan"])
        assert [(log["log_level"], log["event"]) for log in logs] == [("info", "tool_not_found")]

    async def test_a_tool_that_raises_a_value_error_is_still_a_tool_that_raised(self):
        error = ValueError("bad input")
        result = await ToolErrorPlugin().on_tool_error_callback(
            tool=FunctionTool(func=_answer), tool_args={}, tool_context=None, error=error,
        )
        assert result == {"success": False, "error": "_answer failed: ValueError: bad input"}

    async def test_the_stand_in_is_no_running_call(self):
        """ADK skips the after-tool callbacks for it, which would end the mark."""
        plugin = InterruptedCallsPlugin(OpenCalls())

        await plugin.before_tool_callback(
            tool=_stand_in(), tool_args={},
            tool_context=SimpleNamespace(function_call_id="c1"),
        )

        assert plugin._active == {}
        assert get_active_tool_call() is None


def _answer() -> dict:
    """Answer."""
    return {"success": True}


class TestTheADKItRestsOn:
    """ADK's stand-in for an unknown tool name and its lookup error (private ADK)."""

    async def test_an_unknown_name_is_a_bare_tool_and_a_lookup_error(self):
        from google.adk.flows.llm_flows.tools._caller import _prepare_single

        from agentic_cli.workflow.adk.tool_error_plugin import is_unknown_tool

        service = InMemorySessionService()
        session = await service.create_session(app_name="app", user_id="u", session_id="s")
        agent = LlmAgent(name="solo", model="gemini-2.5-flash")
        context = InvocationContext(
            session_service=service, invocation_id="i", agent=agent, session=session,
        )
        tools = {"save_plan": FunctionTool(func=_answer), "get_plan": FunctionTool(func=_answer)}

        prepared = await _prepare_single(
            context, types.FunctionCall(id="c1", name="nope", args={}), tools, agent
        )

        assert type(prepared.tool) is BaseTool
        assert (prepared.tool.name, prepared.tool.description) == ("nope", "Tool not found")
        assert is_unknown_tool(prepared.tool)
        assert isinstance(prepared.tool_lookup_error, ValueError)
        assert str(prepared.tool_lookup_error).splitlines()[:2] == [
            "Tool 'nope' not found.", "Available tools: save_plan, get_plan",
        ]

    def test_only_the_stand_in_is_one(self):
        from agentic_cli.workflow.adk.tool_error_plugin import is_unknown_tool

        assert is_unknown_tool(_stand_in())
        assert not is_unknown_tool(BaseTool(name="nope", description="Something else"))
        assert not is_unknown_tool(FunctionTool(func=_answer))
        assert not is_unknown_tool(SimpleNamespace(name="nope", description="Tool not found"))
