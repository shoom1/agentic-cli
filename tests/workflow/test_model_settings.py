"""Tests for per-agent ModelSettings (Phase 1) and its ADK translation.

Covers:
- the backend-neutral ``ModelSettings`` model,
- ADK ``_get_generate_content_config`` (neutral params -> GenerateContentConfig),
- ADK ``_get_planner`` per-agent thinking resolution (override + global fallback).
"""

from __future__ import annotations

import pytest

pytest.importorskip("google.adk")

from google.genai import types  # noqa: E402

from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.model_settings import (  # noqa: E402
    ModelSettings,
    ThinkingSettings,
)


def _manager(mock_context, model: str) -> GoogleADKWorkflowManager:
    """A manager with a pinned model (no API call / init needed)."""
    return GoogleADKWorkflowManager(
        agent_configs=[], settings=mock_context.settings, model=model
    )


def _cfg(model_settings=None, model=None) -> AgentConfig:
    return AgentConfig(
        name="a", prompt="p", model=model, model_settings=model_settings
    )


# ---------------------------------------------------------------------------
# ModelSettings model
# ---------------------------------------------------------------------------


class TestModelSettingsModel:
    def test_defaults_all_none(self):
        ms = ModelSettings()
        assert ms.common_params() == {}
        assert ms.thinking is None
        assert ms.extra == {}

    def test_common_params_only_non_none(self):
        ms = ModelSettings(temperature=0.5, max_tokens=100)
        assert ms.common_params() == {"temperature": 0.5, "max_tokens": 100}

    def test_thinking_defaults(self):
        ts = ThinkingSettings()
        assert ts.mode == "none"
        assert ts.budget_tokens is None


# ---------------------------------------------------------------------------
# ADK GenerateContentConfig translation
# ---------------------------------------------------------------------------


class TestGenerateContentConfig:
    def test_no_settings_only_http_options(self, mock_context):
        mgr = _manager(mock_context, "gemini-2.5-flash")
        cfg = mgr._get_generate_content_config(None)
        assert cfg.http_options is not None
        assert cfg.temperature is None

    def test_config_without_model_settings(self, mock_context):
        mgr = _manager(mock_context, "gemini-2.5-flash")
        cfg = mgr._get_generate_content_config(_cfg())
        assert cfg.http_options is not None
        assert cfg.temperature is None

    def test_neutral_params_mapped(self, mock_context):
        mgr = _manager(mock_context, "gemini-2.5-flash")
        ms = ModelSettings(
            temperature=0.3, top_p=0.9, top_k=40, max_tokens=1000,
            stop_sequences=["END"],
        )
        cfg = mgr._get_generate_content_config(_cfg(ms))
        assert cfg.temperature == 0.3
        assert cfg.top_p == 0.9
        assert cfg.top_k == 40
        assert cfg.max_output_tokens == 1000  # neutral max_tokens -> ADK name
        assert cfg.stop_sequences == ["END"]
        assert cfg.http_options is not None  # retry options preserved

    def test_extra_valid_key_passthrough_invalid_dropped(self, mock_context):
        mgr = _manager(mock_context, "gemini-2.5-flash")
        # Pick a real GenerateContentConfig field that isn't one of our neutral ones.
        neutral = {
            "temperature", "top_p", "top_k", "max_output_tokens",
            "stop_sequences", "http_options",
        }
        valid_key = next(
            k for k in types.GenerateContentConfig.model_fields if k not in neutral
        )
        ms = ModelSettings(
            temperature=0.5, extra={valid_key: 7, "definitely_not_a_field": 1}
        )
        out = mgr._generate_config_kwargs_from_settings(ms)
        assert out["temperature"] == 0.5
        assert out[valid_key] == 7
        assert "definitely_not_a_field" not in out


# ---------------------------------------------------------------------------
# ADK planner / thinking resolution
# ---------------------------------------------------------------------------


class TestPlannerThinking:
    def test_per_agent_overrides_global_disabled(self, mock_context):
        mgr = _manager(mock_context, "gemini-2.5-flash")
        mgr._settings.set_thinking_effort("none")
        planner = mgr._get_planner(
            _cfg(ModelSettings(thinking=ThinkingSettings(mode="high")))
        )
        assert planner is not None
        assert planner.thinking_config.thinking_budget == 24576  # 2.5 high

    def test_per_agent_none_disables_even_if_global_high(self, mock_context):
        mgr = _manager(mock_context, "gemini-2.5-flash")
        mgr._settings.set_thinking_effort("high")
        planner = mgr._get_planner(
            _cfg(ModelSettings(thinking=ThinkingSettings(mode="none")))
        )
        assert planner is None

    def test_global_fallback_when_no_model_settings(self, mock_context):
        mgr = _manager(mock_context, "gemini-2.5-flash")
        mgr._settings.set_thinking_effort("low")
        planner = mgr._get_planner(_cfg())
        assert planner is not None
        assert planner.thinking_config.thinking_budget == 4096  # 2.5 low

    def test_global_none_returns_no_planner(self, mock_context):
        mgr = _manager(mock_context, "gemini-2.5-flash")
        mgr._settings.set_thinking_effort("none")
        assert mgr._get_planner(_cfg()) is None

    def test_budget_mode_uses_explicit_budget(self, mock_context):
        mgr = _manager(mock_context, "gemini-2.5-flash")
        planner = mgr._get_planner(
            _cfg(ModelSettings(thinking=ThinkingSettings(mode="budget", budget_tokens=5000)))
        )
        assert planner.thinking_config.thinking_budget == 5000
        assert planner.thinking_config.thinking_level is None

    def test_gemini3_uses_thinking_level(self, mock_context):
        mgr = _manager(mock_context, "gemini-3-pro-preview")
        planner = mgr._get_planner(
            _cfg(ModelSettings(thinking=ThinkingSettings(mode="high")))
        )
        assert planner.thinking_config.thinking_level == types.ThinkingLevel.HIGH
        assert planner.thinking_config.thinking_budget is None

    def test_per_agent_model_override_selects_family(self, mock_context):
        # Manager default is 2.5 (budget path); agent overrides to gemini-3
        # flash, which must take the thinking_level path.
        mgr = _manager(mock_context, "gemini-2.5-flash")
        planner = mgr._get_planner(
            _cfg(
                ModelSettings(thinking=ThinkingSettings(mode="medium")),
                model="gemini-3-flash-preview",
            )
        )
        assert planner.thinking_config.thinking_level == types.ThinkingLevel.MEDIUM
        assert planner.thinking_config.thinking_budget is None

    def test_unsupported_model_returns_no_planner(self, mock_context):
        mgr = _manager(mock_context, "gpt-4o")
        planner = mgr._get_planner(
            _cfg(ModelSettings(thinking=ThinkingSettings(mode="high")))
        )
        assert planner is None
