"""LIVE Claude-on-ADK tests — real Anthropic API calls (opt-in, costs money).

Marked ``@pytest.mark.llm`` (skipped by default in a plain run; the integration
``conftest`` loads real keys from ``~/.research_demo/.env`` or
``$AGENTIC_TEST_ENV_FILE``). Run explicitly with ``-m llm``:

    conda run -n agenticcli python -m pytest tests/integration/test_adk_claude_live.py -v -m llm
    # optional overrides:
    #   LIVE_CLAUDE_MODEL=claude-sonnet-4-6        default; supports budget thinking + effort
    #   LIVE_CLAUDE_47PLUS_MODEL=claude-opus-4-8   only if you have access; enables the 400 test

They exercise ``DirectAnthropicLlm.generate_content_async`` directly (no session
/ runner) so they target the new code with minimal scaffolding. Requests use a
tiny ``max_tokens`` to stay cheap. Assertions are structural (got a response /
got a 400), not exact text.

Deferred (can't be tested on this branch yet):
- Adaptive thinking via ADK — needs the google-adk >= 1.34 bump (1.33 raises on a
  negative thinking_budget before any request is sent).
- Effort routed through the manager's planner — the manager doesn't set effort yet.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("google.adk")
pytest.importorskip("anthropic")

from google.genai import types  # noqa: E402
from google.adk.models.llm_request import LlmRequest  # noqa: E402

from agentic_cli.workflow.adk.anthropic_llm import DirectAnthropicLlm  # noqa: E402


pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="No Anthropic API key (set ANTHROPIC_API_KEY or ~/.research_demo/.env).",
    ),
]

MODEL = os.getenv("LIVE_CLAUDE_MODEL", "claude-sonnet-4-6")
MODEL_47PLUS = os.getenv("LIVE_CLAUDE_47PLUS_MODEL")  # e.g. claude-opus-4-8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _request(text: str, *, thinking_budget: int | None = None, tools=None) -> LlmRequest:
    config = types.GenerateContentConfig(
        system_instruction="You are a helpful, concise assistant.",
    )
    if thinking_budget is not None:
        config.thinking_config = types.ThinkingConfig(thinking_budget=thinking_budget)
    if tools is not None:
        config.tools = tools
    return LlmRequest(
        model=MODEL,
        contents=[types.Content(role="user", parts=[types.Part(text=text)])],
        config=config,
    )


async def _collect(llm: DirectAnthropicLlm, request: LlmRequest) -> list:
    responses = []
    async for resp in llm.generate_content_async(request, stream=False):
        responses.append(resp)
    return responses


def _all_parts(responses) -> list:
    parts = []
    for r in responses:
        content = getattr(r, "content", None)
        parts.extend(getattr(content, "parts", None) or [])
    return parts


def _text(responses) -> str:
    return "".join(p.text for p in _all_parts(responses) if getattr(p, "text", None))


# ---------------------------------------------------------------------------
# Basic turn + tool call (no thinking)
# ---------------------------------------------------------------------------


class TestLiveBasic:
    async def test_basic_turn(self):
        llm = DirectAnthropicLlm(model=MODEL, max_tokens=64)
        responses = await _collect(llm, _request("Reply with exactly one word: pong"))
        assert _text(responses).strip(), "expected non-empty text response"

    async def test_tool_call(self):
        llm = DirectAnthropicLlm(model=MODEL, max_tokens=256)
        tool = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="get_weather",
                    description="Get the current weather for a city.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={"city": types.Schema(type=types.Type.STRING)},
                        required=["city"],
                    ),
                )
            ]
        )
        req = _request("What's the weather in Paris? Use the tool.", tools=[tool])
        responses = await _collect(llm, req)
        calls = [
            p.function_call
            for p in _all_parts(responses)
            if getattr(p, "function_call", None)
        ]
        assert calls, "expected a function_call part"
        assert calls[0].name == "get_weather"


# ---------------------------------------------------------------------------
# Budget thinking (works on <= 4.6; this is the deprecated-but-functional path)
# ---------------------------------------------------------------------------


class TestLiveBudgetThinking:
    async def test_budget_thinking_succeeds(self):
        # max_tokens must exceed the thinking budget (Anthropic requirement).
        llm = DirectAnthropicLlm(model=MODEL, max_tokens=4096)
        responses = await _collect(
            llm, _request("What is 17 * 24? Think it through.", thinking_budget=2048)
        )
        # Should not error; should produce some output. A thought part may or may
        # not be surfaced depending on the model/display defaults.
        assert _text(responses).strip() or any(
            getattr(p, "thought", None) for p in _all_parts(responses)
        )


# ---------------------------------------------------------------------------
# output_config.effort accepted on the wire (the key new-code assertion)
# ---------------------------------------------------------------------------


class TestLiveEffort:
    async def test_effort_low_accepted(self):
        llm = DirectAnthropicLlm(model=MODEL, max_tokens=256, effort="low")
        responses = await _collect(llm, _request("Name one primary color."))
        assert _text(responses).strip(), "effort=low request should return text"

    async def test_effort_high_accepted(self):
        llm = DirectAnthropicLlm(model=MODEL, max_tokens=512, effort="high")
        responses = await _collect(llm, _request("Name one primary color."))
        assert _text(responses).strip(), "effort=high request should return text"


# ---------------------------------------------------------------------------
# Confirm the limitation: budget thinking 400s on Opus 4.7+/Fable
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not MODEL_47PLUS,
    reason="Set LIVE_CLAUDE_47PLUS_MODEL (e.g. claude-opus-4-8) to run the 400-confirmation test.",
)
class TestLiveBudgetRejectedOn47Plus:
    async def test_budget_thinking_400s(self):
        import anthropic

        llm = DirectAnthropicLlm(model=MODEL_47PLUS, max_tokens=4096)
        req = LlmRequest(
            model=MODEL_47PLUS,
            contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
            config=types.GenerateContentConfig(
                system_instruction="You are a helpful, concise assistant.",
                thinking_config=types.ThinkingConfig(thinking_budget=2048),
            ),
        )
        with pytest.raises(anthropic.BadRequestError):
            async for _ in llm.generate_content_async(req, stream=False):
                pass
