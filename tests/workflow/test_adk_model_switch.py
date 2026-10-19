"""A model switch that keeps the session, on ADK: no model is sent another provider's thoughts.

An agent's own earlier events carry its model's thoughts, and ADK sends them
back with every request. After a ``/settings`` switch to another provider they
are another provider's, and the APIs that check them refuse the request: Claude
a thinking block it did not sign ("Invalid `signature` in `thinking` block"),
Gemini 3 a thought signature it did not issue ("Corrupted thought signature").
Both refused every later turn of the session. The ADK manager tags each model
response with its provider and leaves another provider's thoughts out of the
request; the session keeps them, and a model's own thoughts still go back to it.

Gemini and Claude answer through fake APIs, and the conftest guard gives every
manager an unlisted catalog, so nothing reaches the network.
"""

from __future__ import annotations

import base64
import json

import pytest

pytest.importorskip("google.adk")

import anthropic  # noqa: E402

from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI, anthropic_message  # noqa: E402
from tests.workflow.test_adk_context_cache import route_gemini  # noqa: E402
from tests.workflow.test_manager_models import KEYS, adk, keys, route, settings  # noqa: E402

CLAUDE = "anthropic:claude-sonnet-4-5"
GEMINI = "google:gemini-2.5-flash"
CLAUDE_SIGNATURE = "Q2xhdWRlIHNpZ25lZCB0aGlz"
GEMINI_SIGNATURE = base64.b64encode(b"Gemini signed this").decode()


@pytest.fixture(autouse=True)
def _hermetic(tmp_path, monkeypatch):
    """A hermetic cwd and HOME, with Google and Anthropic keys."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for variable in KEYS:
        monkeypatch.delenv(variable, raising=False)
    keys(monkeypatch, "google", "anthropic")


def gemini_thinking(request) -> dict:
    """Gemini's answer, after a signed thought."""
    return {
        "candidates": [{
            "content": {"role": "model", "parts": [
                {"text": "Gemini thought.", "thought": True, "thoughtSignature": GEMINI_SIGNATURE},
                {"text": "Gemini answer."},
            ]},
            "finishReason": "STOP",
        }],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2, "totalTokenCount": 7},
    }


def claude_thinking() -> dict:
    return anthropic_message(
        {"type": "thinking", "thinking": "Claude thought.", "signature": CLAUDE_SIGNATURE},
        {"type": "text", "text": "Claude answer."},
    )


def solo(model: str):
    return adk([AgentConfig(name="solo", prompt="Answer.")], settings(default_model=model))


async def turn(manager, text: str) -> list:
    return [event async for event in manager.process(text, user_id="u", session_id="s")]


async def _switch(before: str, after: str, *turns_after: str):
    manager = solo(before)
    try:
        await turn(manager, "Hello")
        await manager.reinitialize(model=after, preserve_sessions=True)
        for text in turns_after or ("Again",):
            await turn(manager, text)
        session = await manager._session_service.get_session(
            app_name=manager.app_name, user_id="u", session_id="s"
        )
    finally:
        await manager.cleanup()
    return session


async def test_claude_after_a_switch_from_gemini_gets_no_gemini_thought(monkeypatch):
    route_gemini(monkeypatch, FakeAPI(gemini_thinking))
    claude = FakeAPI(anthropic_message({"type": "text", "text": "Claude answer."}))
    route(monkeypatch, anthropic, "AsyncAnthropic", claude)

    await _switch(GEMINI, CLAUDE)

    sent = claude.bodies[0]
    assert "Gemini thought" not in json.dumps(sent)
    assistant = [m for m in sent["messages"] if m["role"] == "assistant"]
    assert [block["type"] for block in assistant[0]["content"]] == ["text"]
    assert assistant[0]["content"][0]["text"] == "Gemini answer."


async def test_gemini_after_a_switch_from_claude_gets_no_claude_thought(monkeypatch):
    route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(claude_thinking()))
    gemini = FakeAPI(gemini_thinking)
    route_gemini(monkeypatch, gemini)

    await _switch(CLAUDE, GEMINI)

    sent = json.dumps(gemini.bodies[0])
    assert "Claude thought" not in sent
    assert CLAUDE_SIGNATURE not in sent
    assert "Claude answer." in sent


async def test_a_model_still_gets_its_own_thoughts_back(monkeypatch):
    claude = FakeAPI(claude_thinking())
    route(monkeypatch, anthropic, "AsyncAnthropic", claude)

    await _switch(CLAUDE, CLAUDE)  # a reinitialize on the same provider

    (thinking,) = [
        block
        for message in claude.bodies[1]["messages"] if message["role"] == "assistant"
        for block in message["content"] if block["type"] == "thinking"
    ]
    assert (thinking["thinking"], thinking["signature"]) == ("Claude thought.", CLAUDE_SIGNATURE)


async def test_the_session_keeps_every_thought(monkeypatch):
    route_gemini(monkeypatch, FakeAPI(gemini_thinking))
    route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(claude_thinking()))

    session = await _switch(GEMINI, CLAUDE)

    thoughts = [
        part.text
        for event in session.events if event.content and event.author == "solo"
        for part in event.content.parts or () if part.thought
    ]
    assert thoughts == ["Gemini thought.", "Claude thought."]
