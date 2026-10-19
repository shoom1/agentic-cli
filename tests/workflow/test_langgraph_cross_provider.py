"""LangGraph across providers: one conversation, models of several providers.

A ``/settings`` model switch keeps the session, and the agents of a mixed tree
share its messages, so a model can be sent another provider's AI messages.
Their reasoning belongs to the model that produced it: Claude rejects a
thinking block it did not sign, and Gemini drops foreign reasoning with a
warning on every call. The agent node leaves another provider's reasoning out
of the request; the checkpoint keeps it, and a model's own reasoning still
goes back to it.
"""

from __future__ import annotations

import json
import logging

import pytest

pytest.importorskip("langgraph")

import anthropic  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from agentic_cli.models import ProviderKind  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI  # noqa: E402
from tests.workflow.langchain_fakes import (  # noqa: E402
    anthropic_reply,
    gemini_reply,
    route,
    route_google,
)
from tests.workflow.test_manager_models import (  # noqa: E402,F401 - fixtures
    _langgraph,
    home,
    keys,
    lg_turn,
    providers,
    settings,
)

SOLO = [AgentConfig(name="solo", prompt="Answer.")]


def gemini_thinking():
    return gemini_reply(
        {"text": "Gemini thought.", "thought": True},
        {"text": "Gemini answer.", "thoughtSignature": "R0VNSU5JX1NJRw=="},
    )


def claude_thinking():
    return anthropic_reply(
        {"type": "thinking", "thinking": "Claude thought.", "signature": "claude-sig"},
        {"type": "text", "text": "Claude answer."},
        model="claude-opus-4-7",
    )


def blocks(body: dict, role: str = "assistant") -> list:
    """The content blocks of every ``role`` message of a request body."""
    return [
        block
        for message in body["messages"]
        if message["role"] == role and isinstance(message["content"], list)
        for block in message["content"]
    ]


async def test_a_switch_from_gemini_to_claude_sends_claude_no_gemini_thoughts(
    providers, monkeypatch
):
    keys(monkeypatch, "google", "anthropic")
    route_google(monkeypatch, FakeAPI(gemini_thinking()))
    claude = FakeAPI(anthropic_reply({"type": "text", "text": "Claude answer."}))
    route(monkeypatch, anthropic, "AsyncClient", claude)
    manager = _langgraph(SOLO, settings(default_model="gemini-3-flash-preview"))
    try:
        await lg_turn(manager, "Hello")
        await manager.reinitialize(model="anthropic:claude-sonnet-4-5", preserve_sessions=True)
        events = await lg_turn(manager, "And now?")
    finally:
        await manager.cleanup()

    sent = claude.bodies[0]
    assert "Gemini thought" not in json.dumps(sent)
    assert [b["type"] for b in blocks(sent)] == ["text"]
    assert "Claude answer." in " ".join(str(e.content) for e in events)


async def test_a_switch_from_claude_to_gemini_warns_nothing(providers, monkeypatch, caplog):
    keys(monkeypatch, "google", "anthropic")
    route(monkeypatch, anthropic, "AsyncClient", FakeAPI(claude_thinking()))
    gemini = FakeAPI(gemini_reply({"text": "Gemini answer."}))
    route_google(monkeypatch, gemini)
    manager = _langgraph(SOLO, settings(default_model="claude-opus-4-7"))
    try:
        await lg_turn(manager, "one")
        await lg_turn(manager, "two")
        await manager.reinitialize(model="google:gemini-2.5-flash", preserve_sessions=True)
        caplog.clear()
        caplog.set_level(logging.WARNING)
        await lg_turn(manager, "three")
    finally:
        await manager.cleanup()

    assert [r.getMessage() for r in caplog.records if r.name.startswith("langchain")] == []
    assert "Claude thought" not in json.dumps(gemini.bodies[0])


async def test_claude_still_gets_its_own_thinking_back(providers, monkeypatch):
    keys(monkeypatch, "anthropic")
    claude = FakeAPI(claude_thinking())
    route(monkeypatch, anthropic, "AsyncClient", claude)
    manager = _langgraph(SOLO, settings(default_model="claude-opus-4-7"))
    try:
        await lg_turn(manager, "one")
        await lg_turn(manager, "two")
    finally:
        await manager.cleanup()

    (thinking,) = [b for b in blocks(claude.bodies[1]) if b["type"] == "thinking"]
    assert (thinking["thinking"], thinking["signature"]) == ("Claude thought.", "claude-sig")


def test_the_filter_keeps_the_conversation_and_changes_nothing_it_was_given():
    from agentic_cli.workflow.langchain_models import without_foreign_reasoning

    gemini = AIMessage(
        content=[{"type": "thinking", "thinking": "Gemini thought."},
                 {"type": "text", "text": "Gemini answer."}],
        response_metadata={"model_provider": "google_genai"},
    )
    endpoint = AIMessage(
        content="Endpoint answer.",
        additional_kwargs={"reasoning_content": "Endpoint thought."},
        response_metadata={"model_provider": "openai_compatible"},
    )
    claude = AIMessage(
        content=[{"type": "thinking", "thinking": "Claude thought.", "signature": "s"},
                 {"type": "text", "text": "Claude answer."}],
        response_metadata={"model_provider": "anthropic"},
    )
    history = [HumanMessage("hi"), gemini, endpoint, claude]

    for_claude = without_foreign_reasoning(history, ProviderKind.ANTHROPIC)
    for_openrouter = without_foreign_reasoning(history, ProviderKind.OPENROUTER)

    assert for_claude[1].content == [{"type": "text", "text": "Gemini answer."}]
    assert for_claude[3] is claude  # its own reasoning goes back as it was
    assert for_openrouter[2].additional_kwargs == {}
    assert for_openrouter[2].content == "Endpoint answer."
    # The checkpointed messages are never changed.
    assert gemini.content[0]["type"] == "thinking"
    assert endpoint.additional_kwargs == {"reasoning_content": "Endpoint thought."}
