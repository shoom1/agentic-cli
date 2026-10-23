"""What ADK's model adapters report as a call's prompt: all of it, cached tokens included.

The status bar's ``ctx`` is the main agent's last ``prompt_tokens``, and ADK's
own summary check reads the same number (the previous call's
``prompt_token_count``). Providers count a cache their own way:

- Claude's ``input_tokens`` leaves the cache out; ADK's adapter adds the
  cache reads and writes back (``_extract_prompt_token_count``).
- Gemini's ``promptTokenCount`` holds the cached content, which
  ``cachedContentTokenCount`` breaks out.
- OpenAI's ``input_tokens`` (Responses) and ``prompt_tokens`` (Chat
  Completions, which ``OpenAICompatibleLlm`` speaks for OpenRouter and
  endpoints) hold the cached tokens, which ``cached_tokens`` breaks out.

If an adapter stopped counting them, ``ctx`` (and on ADK the trigger) would
fall to the uncached remainder on every cache hit. These pin what ADK 2.12's
adapters pass on; what each API means by its counts is a live check. Deep
Agents' adapters are pinned in ``tests/integration/test_deepagents_turns.py``
(``TestUsageAndCache``).
"""

from __future__ import annotations

import anthropic
import openai
import pytest

pytest.importorskip("google.adk")

from agentic_cli.cli.usage_tracker import UsageTracker  # noqa: E402
from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from tests.workflow.adk.fakes import (  # noqa: E402
    FakeAPI,
    anthropic_message,
    chat_completion,
    openai_response,
)
from tests.workflow.test_adk_context_cache import route_gemini  # noqa: E402
from tests.workflow.test_manager_models import route  # noqa: E402

#: The user's server, an OpenAI-compatible endpoint.
ENDPOINTS = {"local": {"base_url": "http://localhost:8000/v1",
                       "models": {"small": {"context_window": 32_000}}}}


@pytest.fixture(autouse=True)
def hermetic(tmp_path, monkeypatch):
    """A hermetic cwd and HOME, and every provider's fake key."""
    (tmp_path / "home").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    for variable, key in (("ANTHROPIC_API_KEY", "a-test-key"), ("GOOGLE_API_KEY", "g-test-key"),
                          ("OPENAI_API_KEY", "o-test-key")):
        monkeypatch.setenv(variable, key)


async def usage_of_a_turn(model: str, **settings) -> tuple[dict, UsageTracker]:
    """The one LLM_USAGE of a turn on ``model``, and the status bar's tracker after it."""
    manager = GoogleADKWorkflowManager(
        agent_configs=[AgentConfig(name="solo", prompt="Answer.", include_state_tools=False)],
        settings=BaseSettings(session_store="memory", default_model=model,
                              prompt_caching_enabled=False, **settings),
    )
    try:
        events = [e async for e in manager.process("Hello.", "u", session_id="s")]
    finally:
        await manager.cleanup()
    (usage,) = [e.metadata for e in events if e.type is EventType.LLM_USAGE]
    tracker = UsageTracker()
    tracker.record(usage)
    return usage, tracker


async def test_claude_counts_the_cache_reads_and_writes_in_the_prompt(monkeypatch):
    reply = {**anthropic_message({"type": "text", "text": "Hi."}),
             "usage": {"input_tokens": 50, "output_tokens": 7,
                       "cache_read_input_tokens": 1_200, "cache_creation_input_tokens": 300}}
    route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(reply))

    usage, tracker = await usage_of_a_turn("anthropic:claude-sonnet-4-6")

    assert (usage["prompt_tokens"], usage["cached_tokens"]) == (1_550, 1_200)
    assert tracker.last_prompt_tokens == 1_550


async def test_gemini_passes_its_whole_prompt_count_on(monkeypatch):
    def answer(request) -> dict:
        return {
            "candidates": [{"content": {"role": "model", "parts": [{"text": "Hi."}]},
                            "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 900, "candidatesTokenCount": 5,
                              "totalTokenCount": 930, "cachedContentTokenCount": 800,
                              "thoughtsTokenCount": 25},
        }

    route_gemini(monkeypatch, FakeAPI(answer))

    usage, tracker = await usage_of_a_turn("google:gemini-2.5-flash")

    assert (usage["prompt_tokens"], usage["cached_tokens"], usage["thinking_tokens"]) == (
        900, 800, 25
    )
    assert tracker.last_prompt_tokens == 900


async def test_openai_responses_passes_its_whole_prompt_count_on(monkeypatch):
    reply = openai_response("Hi.")
    reply["usage"] = {"input_tokens": 1_550, "output_tokens": 7, "total_tokens": 1_557,
                      "input_tokens_details": {"cached_tokens": 1_200},
                      "output_tokens_details": {"reasoning_tokens": 0}}
    route(monkeypatch, openai, "AsyncOpenAI", FakeAPI(reply))

    usage, tracker = await usage_of_a_turn("openai:gpt-5")

    assert (usage["prompt_tokens"], usage["cached_tokens"]) == (1_550, 1_200)
    assert tracker.last_prompt_tokens == 1_550


async def test_chat_completions_pass_their_whole_prompt_count_on(monkeypatch):
    reply = chat_completion("Hi.")
    reply["usage"] = {"prompt_tokens": 1_550, "completion_tokens": 7, "total_tokens": 1_557,
                      "prompt_tokens_details": {"cached_tokens": 1_200}}
    route(monkeypatch, openai, "AsyncOpenAI", FakeAPI(reply))

    usage, tracker = await usage_of_a_turn("local:small", model_endpoints=ENDPOINTS)

    assert (usage["prompt_tokens"], usage["cached_tokens"]) == (1_550, 1_200)
    assert tracker.last_prompt_tokens == 1_550
