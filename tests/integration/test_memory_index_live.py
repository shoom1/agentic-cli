"""LIVE: a memory saved before a conversation reaches its main agent through
the index (spec §11), on ADK (Claude) and Deep Agents (Gemini).

The agent has the memory tools, so a right answer alone would prove nothing: it
could have come from a search. The turn must make no tool call at all, and the
answer must say the saved word (``tab``/``tabs`` as a word, not "established").

Run explicitly: conda run -n agenticcli python -m pytest tests/integration/test_memory_index_live.py -v -m llm
"""

from __future__ import annotations

import os
import re

import pytest

from agentic_cli.config import BaseSettings
from agentic_cli.memory import MemoryStore
from agentic_cli.tools import memory_tools
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.events import EventType
from tests.integration.helpers import find_tool_calls

pytestmark = [pytest.mark.llm]

CONFIG = [AgentConfig(name="solo", prompt="You are a concise assistant.",
                      tools=[memory_tools.read_memory, memory_tools.search_memory])]
QUESTION = "From what you know about me, which indentation do I prefer? Answer in one word."


def _answered_from_the_index(events) -> str:
    """The turn's answer, once it is known that no tool was called for it."""
    calls = [event.metadata.get("tool_name") for event in find_tool_calls(events)]
    assert calls == [], f"the agent searched instead of reading its index: {calls}"
    return " ".join(event.content for event in events if event.type is EventType.TEXT).lower()


async def _ask(manager) -> str:
    try:
        return _answered_from_the_index([e async for e in manager.process(QUESTION, "u", session_id="s")])
    finally:
        await manager.cleanup()


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="No Anthropic API key.")
async def test_adk_claude_answers_from_the_index(tmp_path, monkeypatch):
    from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager

    monkeypatch.chdir(tmp_path)
    settings = BaseSettings(session_store="memory", workspace_dir=tmp_path / "w",
                            default_model="claude-sonnet-4-6")
    MemoryStore(settings.workspace_dir / "memory").save(
        "Prefers tabs for indentation", "The user indents with tabs, never spaces.", "feedback", scope="user")
    assert re.search(r"\btabs?\b", await _ask(GoogleADKWorkflowManager(agent_configs=CONFIG, settings=settings)))


@pytest.mark.skipif(not os.environ.get("GOOGLE_API_KEY"), reason="No Google API key.")
async def test_deepagents_gemini_answers_from_the_index(tmp_path, monkeypatch):
    from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager

    monkeypatch.chdir(tmp_path)
    settings = BaseSettings(session_store="memory", workspace_dir=tmp_path / "w",
                            default_model="gemini-2.5-flash", orchestrator="deepagents")
    MemoryStore(settings.workspace_dir / "memory").save(
        "Prefers tabs for indentation", "The user indents with tabs, never spaces.", "feedback", scope="user")
    assert re.search(r"\btabs?\b", await _ask(DeepAgentsWorkflowManager(agent_configs=CONFIG, settings=settings)))
