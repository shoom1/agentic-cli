"""Live: Claude takes the turn after a stopped one, on both backends.

Ctrl+C mid-tool, or during the model call, leaves every call with one result
(``workflow/interrupts.py``); a long-running call a crash left open on ADK is
answered on the next turn. Offline, a fake refuses what the real API refuses
(``test_interrupted_calls.py``); this checks the real API accepts the history
the seals make. Keys come from the live-test framework (``conftest``).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import uuid
from pathlib import Path

import pytest

pytest.importorskip("google.adk")
pytest.importorskip("deepagents")

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.tools.registry import register_tool  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from agentic_cli.workflow.permissions import EXEMPT  # noqa: E402

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="No Anthropic API key (set ANTHROPIC_API_KEY or ~/.research_demo/.env).",
    ),
]

HAIKU = os.getenv("LIVE_CONTEXT_HAIKU_MODEL", "anthropic:claude-haiku-4-5")
PROMPT = "You are a terse assistant. When asked to wait, call live_ic_wait at once."

#: Set when ``live_ic_wait`` runs.
STARTED: list[asyncio.Event] = []


@register_tool(capabilities=EXEMPT, description="Waits for a while (a live test tool).")
async def live_ic_wait() -> dict:
    """Wait, as a long tool does, until the turn is stopped."""
    STARTED[-1].set()
    await asyncio.Event().wait()
    return {"success": True}


@pytest.fixture
def app_name(tmp_path, monkeypatch):
    """A project of its own, and an app name nothing else uses (its folder removed after)."""
    monkeypatch.chdir(tmp_path)
    name = f"ic-live-{uuid.uuid4().hex[:8]}"
    STARTED.append(asyncio.Event())
    yield name
    shutil.rmtree(Path.home() / f".{name}", ignore_errors=True)


def manager_for(backend: str, app_name: str):
    settings = BaseSettings(app_name=app_name, default_model=HAIKU, session_store="memory")
    configs = [AgentConfig(name="solo", prompt=PROMPT, tools=[live_ic_wait],
                           include_state_tools=False)]
    if backend == "adk":
        return GoogleADKWorkflowManager(agent_configs=configs, settings=settings)
    return DeepAgentsWorkflowManager(agent_configs=configs, settings=settings)


async def turn(manager, text: str) -> list:
    return [event async for event in manager.process(text, "u", session_id="s")]


def answered(events: list) -> str:
    assert [e.content for e in events if e.type is EventType.ERROR] == []
    text = " ".join(e.content for e in events if e.type is EventType.TEXT).strip()
    assert text
    return text


@pytest.mark.parametrize("backend", ["adk", "deepagents"])
async def test_claude_takes_the_turn_after_one_stopped_mid_tool(app_name, backend):
    manager = manager_for(backend, app_name)
    try:
        running = asyncio.create_task(turn(manager, "Please wait."))
        await asyncio.wait_for(STARTED[-1].wait(), timeout=90)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        events = await turn(manager, "What happened to that call? One sentence.")
    finally:
        await manager.cleanup()

    answered(events)


@pytest.mark.parametrize("backend", ["adk", "deepagents"])
async def test_claude_takes_the_turn_after_one_stopped_before_any_reply(app_name, backend):
    manager = manager_for(backend, app_name)
    try:
        running = asyncio.create_task(turn(manager, "Write a long poem about rivers."))
        await asyncio.sleep(0.3)  # the request is on its way; no reply yet
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        events = await turn(manager, "Say hello in one word.")
    finally:
        await manager.cleanup()

    answered(events)


async def test_on_adk_claude_takes_the_turn_after_a_long_running_call_a_crash_left_open(
    app_name,
):
    """Without the answer ADK keeps the call in the request, and Claude refuses it."""
    from google.adk.events import Event
    from google.genai import types

    manager = manager_for("adk", app_name)
    try:
        answered(await turn(manager, "Say hi."))
        service = manager._session_service
        session = await service.get_session(app_name=manager.app_name, user_id="u",
                                            session_id="s")
        await service.append_event(session, Event(
            invocation_id="e-crashed", author="solo",
            content=types.Content(role="model", parts=[types.Part(
                function_call=types.FunctionCall(id="toolu_crashed", name="live_ic_wait",
                                                 args={}),
            )]),
            long_running_tool_ids={"toolu_crashed"},
        ))
        events = await turn(manager, "Say hello in one word.")
    finally:
        await manager.cleanup()

    answered(events)
