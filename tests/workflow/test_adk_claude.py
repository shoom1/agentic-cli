"""Native Claude (direct-API ``AnthropicLlm``) support on the ADK backend.

Covers the thinking-budget mapping (Claude is budget-based, like Gemini 2.5),
``max_tokens`` coordination (Anthropic requires ``max_tokens > thinking budget``
and reads it from the model instance), and model-arg construction that avoids the
Vertex ``Claude`` class the registry would otherwise resolve a ``claude-*`` string
to.
"""

from __future__ import annotations

import pytest

pytest.importorskip("google.adk")

from google.adk.models.anthropic_llm import AnthropicLlm  # noqa: E402

from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.model_settings import (  # noqa: E402
    ModelSettings,
    ThinkingSettings,
)

CLAUDE = "claude-sonnet-4-5"


def _manager(mock_context, model: str) -> GoogleADKWorkflowManager:
    return GoogleADKWorkflowManager(
        agent_configs=[], settings=mock_context.settings, model=model
    )


def _cfg(model_settings=None, model=None) -> AgentConfig:
    return AgentConfig(name="a", prompt="p", model=model, model_settings=model_settings)


# ---------------------------------------------------------------------------
# Thinking → Anthropic budget mapping (via the planner)
# ---------------------------------------------------------------------------


class TestClaudePlannerThinking:
    @pytest.mark.parametrize(
        "effort,budget", [("low", 4096), ("medium", 10000), ("high", 32000)]
    )
    def test_effort_maps_to_budget(self, mock_context, effort, budget):
        mgr = _manager(mock_context, CLAUDE)
        planner = mgr._get_planner(
            _cfg(ModelSettings(thinking=ThinkingSettings(mode=effort)))
        )
        assert planner is not None
        assert planner.thinking_config.thinking_budget == budget
        assert planner.thinking_config.thinking_level is None  # never level-based

    def test_budget_mode_explicit(self, mock_context):
        mgr = _manager(mock_context, CLAUDE)
        planner = mgr._get_planner(
            _cfg(ModelSettings(thinking=ThinkingSettings(mode="budget", budget_tokens=20000)))
        )
        assert planner.thinking_config.thinking_budget == 20000

    def test_budget_mode_floored_at_anthropic_minimum(self, mock_context):
        mgr = _manager(mock_context, CLAUDE)
        planner = mgr._get_planner(
            _cfg(ModelSettings(thinking=ThinkingSettings(mode="budget", budget_tokens=200)))
        )
        assert planner.thinking_config.thinking_budget == 1024  # floored

    def test_global_effort_fallback(self, mock_context):
        mgr = _manager(mock_context, CLAUDE)
        mgr._settings.set_thinking_effort("high")
        planner = mgr._get_planner(_cfg())
        assert planner.thinking_config.thinking_budget == 32000

    def test_per_agent_none_disables(self, mock_context):
        mgr = _manager(mock_context, CLAUDE)
        mgr._settings.set_thinking_effort("high")
        planner = mgr._get_planner(
            _cfg(ModelSettings(thinking=ThinkingSettings(mode="none")))
        )
        assert planner is None


# ---------------------------------------------------------------------------
# Model-arg construction (AnthropicLlm instance, not Vertex Claude string)
# ---------------------------------------------------------------------------


class TestClaudeModelArg:
    def test_claude_returns_anthropic_llm_instance(self, mock_context):
        mgr = _manager(mock_context, CLAUDE)
        arg = mgr._build_model_arg(_cfg())
        assert isinstance(arg, AnthropicLlm)
        assert arg.model == CLAUDE

    def test_gemini_passes_through_as_string(self, mock_context):
        mgr = _manager(mock_context, "gemini-2.5-flash")
        assert mgr._build_model_arg(_cfg()) == "gemini-2.5-flash"

    def test_per_agent_model_override_to_claude(self, mock_context):
        mgr = _manager(mock_context, "gemini-2.5-flash")
        arg = mgr._build_model_arg(_cfg(model=CLAUDE))
        assert isinstance(arg, AnthropicLlm)
        assert arg.model == CLAUDE


# ---------------------------------------------------------------------------
# max_tokens coordination (must exceed thinking budget)
# ---------------------------------------------------------------------------


class TestClaudeMaxTokens:
    def test_default_without_thinking(self, mock_context):
        mgr = _manager(mock_context, CLAUDE)
        mgr._settings.set_thinking_effort("none")
        assert mgr._build_model_arg(_cfg()).max_tokens == 8192

    def test_exceeds_high_thinking_budget(self, mock_context):
        mgr = _manager(mock_context, CLAUDE)
        mgr._settings.set_thinking_effort("high")  # budget 32000
        arg = mgr._build_model_arg(_cfg())
        assert arg.max_tokens > 32000

    def test_honors_explicit_floor_without_thinking(self, mock_context):
        mgr = _manager(mock_context, CLAUDE)
        mgr._settings.set_thinking_effort("none")
        arg = mgr._build_model_arg(_cfg(ModelSettings(max_tokens=20000)))
        assert arg.max_tokens == 20000

    def test_grows_above_explicit_floor_when_budget_larger(self, mock_context):
        mgr = _manager(mock_context, CLAUDE)
        arg = mgr._build_model_arg(
            _cfg(ModelSettings(max_tokens=10000, thinking=ThinkingSettings(mode="high")))
        )
        assert arg.max_tokens > 32000  # budget+headroom beats the 10k floor


# ---------------------------------------------------------------------------
# generate_simple routes Claude through the Anthropic SDK (not the genai client)
# ---------------------------------------------------------------------------


class TestClaudeGenerateSimple:
    async def test_routes_to_anthropic_sdk(self, mock_context, monkeypatch):
        mgr = _manager(mock_context, CLAUDE)

        async def _noop():
            return None

        monkeypatch.setattr(mgr, "_ensure_initialized", _noop)

        captured: dict = {}

        class _Block:
            type = "text"
            text = "hello"

        class _Msg:
            content = [_Block()]

        class _Client:
            class messages:
                @staticmethod
                async def create(**kwargs):
                    captured.update(kwargs)
                    return _Msg()

        import anthropic

        monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda: _Client())

        out = await mgr.generate_simple("hi", max_tokens=123)

        assert out == "hello"
        assert captured["model"] == CLAUDE
        assert captured["max_tokens"] == 123
        assert captured["messages"] == [{"role": "user", "content": "hi"}]
