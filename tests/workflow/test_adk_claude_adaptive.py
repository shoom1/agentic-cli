"""Adaptive-thinking + effort policy for Claude on ADK.

Claude >= 4.6 rejects ``budget_tokens`` (400 on 4.7+/Fable), so the manager
switches those to adaptive thinking (negative budget) + ``output_config.effort``.
Claude <= 4.5 keeps the legacy numeric-budget path with no effort.

These assert the config the manager produces (planner budget, model effort/
max_tokens) — they don't make API calls, so they're version-independent (the
negative-budget -> adaptive mapping itself is ADK >= 1.34, exercised live).
"""

from __future__ import annotations

import pytest

pytest.importorskip("google.adk")

from agentic_cli.workflow.adk.manager import (  # noqa: E402
    GoogleADKWorkflowManager,
    _anthropic_uses_adaptive,
)
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.model_settings import (  # noqa: E402
    ModelSettings,
    ThinkingSettings,
)

ADAPTIVE = "claude-opus-4-8"   # >= 4.6
LEGACY = "claude-sonnet-4-5"   # <= 4.5


def _manager(mock_context, model: str) -> GoogleADKWorkflowManager:
    return GoogleADKWorkflowManager(
        agent_configs=[], settings=mock_context.settings, model=model
    )


def _cfg(model_settings=None, model=None) -> AgentConfig:
    return AgentConfig(name="a", prompt="p", model=model, model_settings=model_settings)


class TestAdaptiveClassification:
    @pytest.mark.parametrize(
        "model,adaptive",
        [
            ("claude-opus-4-8", True),
            ("claude-opus-4-7", True),
            ("claude-opus-4-6", True),
            ("claude-sonnet-4-6", True),
            ("claude-fable-5", True),
            ("claude-opus-4-5", False),
            ("claude-sonnet-4-5", False),
            ("claude-haiku-4-5", False),
            ("claude-sonnet-4", False),
            ("claude-3-5-sonnet", False),
            ("gemini-2.5-pro", False),
        ],
    )
    def test_uses_adaptive(self, model, adaptive):
        assert _anthropic_uses_adaptive(model) is adaptive


class TestAdaptivePlanner:
    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_adaptive_uses_negative_budget(self, mock_context, effort):
        mgr = _manager(mock_context, ADAPTIVE)
        planner = mgr._get_planner(
            _cfg(ModelSettings(thinking=ThinkingSettings(mode=effort)))
        )
        assert planner is not None
        assert planner.thinking_config.thinking_budget == -1  # adaptive sentinel
        assert planner.thinking_config.thinking_level is None

    def test_legacy_uses_positive_budget(self, mock_context):
        mgr = _manager(mock_context, LEGACY)
        planner = mgr._get_planner(
            _cfg(ModelSettings(thinking=ThinkingSettings(mode="high")))
        )
        assert planner.thinking_config.thinking_budget == 32000

    def test_none_disables_on_adaptive(self, mock_context):
        mgr = _manager(mock_context, ADAPTIVE)
        mgr._settings.set_thinking_effort("none")
        assert mgr._get_planner(_cfg()) is None


class TestAdaptiveEffort:
    @pytest.mark.parametrize(
        "mode,effort", [("low", "low"), ("medium", "medium"), ("high", "high")]
    )
    def test_effort_mapped_for_adaptive(self, mock_context, mode, effort):
        mgr = _manager(mock_context, ADAPTIVE)
        arg = mgr._build_model_arg(
            _cfg(ModelSettings(thinking=ThinkingSettings(mode=mode)))
        )
        assert arg.effort == effort

    def test_global_effort_fallback_for_adaptive(self, mock_context):
        mgr = _manager(mock_context, ADAPTIVE)
        mgr._settings.set_thinking_effort("medium")
        assert mgr._build_model_arg(_cfg()).effort == "medium"

    def test_no_effort_for_legacy(self, mock_context):
        mgr = _manager(mock_context, LEGACY)
        arg = mgr._build_model_arg(
            _cfg(ModelSettings(thinking=ThinkingSettings(mode="high")))
        )
        assert arg.effort is None

    def test_no_effort_for_budget_mode(self, mock_context):
        mgr = _manager(mock_context, ADAPTIVE)
        arg = mgr._build_model_arg(
            _cfg(ModelSettings(thinking=ThinkingSettings(mode="budget", budget_tokens=5000)))
        )
        assert arg.effort is None

    def test_no_effort_when_thinking_disabled(self, mock_context):
        mgr = _manager(mock_context, ADAPTIVE)
        mgr._settings.set_thinking_effort("none")
        assert mgr._build_model_arg(_cfg()).effort is None


class TestAdaptiveMaxTokens:
    def test_adaptive_does_not_inflate_max_tokens(self, mock_context):
        mgr = _manager(mock_context, ADAPTIVE)
        mgr._settings.set_thinking_effort("high")  # would be 32000 budget on legacy
        assert mgr._build_model_arg(_cfg()).max_tokens == 8192  # base default

    def test_adaptive_honors_explicit_max_tokens(self, mock_context):
        mgr = _manager(mock_context, ADAPTIVE)
        mgr._settings.set_thinking_effort("high")
        arg = mgr._build_model_arg(_cfg(ModelSettings(max_tokens=20000)))
        assert arg.max_tokens == 20000
