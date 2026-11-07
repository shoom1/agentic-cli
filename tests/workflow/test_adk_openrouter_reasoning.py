"""On ADK, an OpenRouter model's reasoning details go back to it, and only to it.

OpenRouter sends a reasoning model's ``reasoning_details`` (Claude's signed
thinking, Gemini's thought signatures) beside its reasoning text, and the
model needs them back on the assistant message that called a tool.
``OpenAICompatibleLlm`` keeps them on the response's thought part, so the
session holds them, and sends them with each later request to the model that
wrote them: after the sqlite session store is reloaded by another manager too,
and never to another model, on OpenRouter or another provider.

OpenRouter and Claude answer through fake APIs, and the conftest guard gives
every manager an unlisted catalog, so nothing reaches the network.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("google.adk")

import anthropic  # noqa: E402
import openai  # noqa: E402

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from tests.models.fakes import openrouter_spec  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI, anthropic_message, chat_completion  # noqa: E402
from tests.workflow.test_manager_models import KEYS, keys, route  # noqa: E402

CLAUDE = "openrouter:anthropic/claude-sonnet-4.5"
GEMINI = "openrouter:google/gemini-3-pro-preview"
DIRECT_CLAUDE = "anthropic:claude-haiku-4-5"

#: Claude's signed thinking, as OpenRouter sends it.
SIGNED = [{
    "type": "reasoning.text", "text": "I should save a plan.", "signature": "sig-1",
    "format": "anthropic-claude-v1", "index": 0,
}]


@pytest.fixture(autouse=True)
def _hermetic(tmp_path, monkeypatch):
    """A hermetic cwd and HOME, with OpenRouter's and Anthropic's keys."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for variable in KEYS:
        monkeypatch.delenv(variable, raising=False)
    keys(monkeypatch, "openrouter", "anthropic")


def plans_after_thinking() -> dict[str, Any]:
    """OpenRouter's reply: Claude thinks, signed, then calls save_plan."""
    reply = chat_completion(model="anthropic/claude-sonnet-4.5")
    reply["choices"][0] = {
        "index": 0,
        "message": {
            "role": "assistant", "content": None,
            "reasoning": "I should save a plan.", "reasoning_details": SIGNED,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "save_plan", "arguments": json.dumps({"content": "- [ ] Answer"}),
            }}],
        },
        "finish_reason": "tool_calls",
    }
    return reply


class OpenRouter(FakeAPI):
    """OpenRouter: Claude thinks and calls save_plan, then answers each
    request after it with plain text."""

    def __init__(self) -> None:
        self.script = [plans_after_thinking()]
        super().__init__(
            lambda request: self.script.pop(0) if self.script else chat_completion("Done.")
        )


def manager(workspace, model: str = CLAUDE, *, store: str = "memory") -> GoogleADKWorkflowManager:
    return GoogleADKWorkflowManager(
        agent_configs=[AgentConfig(name="solo", prompt="Plan, then answer.")],
        settings=BaseSettings(
            workspace_dir=workspace, session_store=store, default_model=model,
            model_providers={"openrouter": openrouter_spec()}, prompt_caching_enabled=False,
        ),
    )


async def turn(manager, text: str) -> list:
    return [event async for event in manager.process(text, user_id="u", session_id="s")]


def details_sent(body: dict[str, Any]) -> list:
    """The reasoning details each assistant message of a request carries."""
    return [m.get("reasoning_details") for m in body["messages"] if m["role"] == "assistant"]


async def test_the_details_go_back_with_the_tool_result(monkeypatch, tmp_path):
    api = OpenRouter()
    route(monkeypatch, openai, "AsyncOpenAI", api)
    solo = manager(tmp_path)
    try:
        await turn(solo, "Plan it.")
    finally:
        await solo.cleanup()

    assert details_sent(api.bodies[1]) == [SIGNED]


async def test_a_reloaded_session_still_sends_them(monkeypatch, tmp_path):
    """The thought part that holds them is saved with the session: a new
    manager on the same sqlite file sends them with the next turn."""
    api = OpenRouter()
    route(monkeypatch, openai, "AsyncOpenAI", api)
    first = manager(tmp_path, store="sqlite")
    try:
        await turn(first, "Plan it.")
    finally:
        await first.cleanup()
    second = manager(tmp_path, store="sqlite")
    try:
        await turn(second, "And now?")
    finally:
        await second.cleanup()

    assert details_sent(api.bodies[2]) == [SIGNED, None]


async def _switch(workspace, after: str) -> None:
    """A turn on OpenRouter's Claude, then a /settings switch to ``after``
    that keeps the session, and a turn on it."""
    solo = manager(workspace)
    try:
        await turn(solo, "Plan it.")
        await solo.reinitialize(model=after, preserve_sessions=True)
        await turn(solo, "And now?")
    finally:
        await solo.cleanup()


async def test_another_openrouter_model_is_not_sent_them(monkeypatch, tmp_path):
    """The provider stays OpenRouter, so its thoughts stay in the request:
    the details in them go only to the model that wrote them."""
    api = OpenRouter()
    route(monkeypatch, openai, "AsyncOpenAI", api)

    await _switch(tmp_path, GEMINI)

    after = api.bodies[2]
    assert after["model"] == "google/gemini-3-pro-preview"
    assert details_sent(after) == [None, None]
    assert "sig-1" not in json.dumps(after)


async def test_the_same_model_after_a_reinitialize_still_gets_them(monkeypatch, tmp_path):
    api = OpenRouter()
    route(monkeypatch, openai, "AsyncOpenAI", api)

    await _switch(tmp_path, CLAUDE)

    assert details_sent(api.bodies[2]) == [SIGNED, None]


async def test_another_provider_is_not_sent_them(monkeypatch, tmp_path):
    route(monkeypatch, openai, "AsyncOpenAI", OpenRouter())
    claude = FakeAPI(anthropic_message({"type": "text", "text": "Claude answer."}))
    route(monkeypatch, anthropic, "AsyncAnthropic", claude)

    await _switch(tmp_path, DIRECT_CLAUDE)

    sent = json.dumps(claude.bodies[0])
    assert "sig-1" not in sent and "reasoning_details" not in sent
    assert "I should save a plan" not in sent
