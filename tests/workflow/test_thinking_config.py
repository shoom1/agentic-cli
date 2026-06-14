"""Regression tests for ADK thinking-config selection per model generation.

Gemini 2.5 models only accept a numeric ``thinking_budget`` and reject
``thinking_level`` with HTTP 400; Gemini 3 models use ``thinking_level``.
``GoogleADKWorkflowManager._get_planner`` must pick the right field — sending
``thinking_level`` to gemini-2.5-flash (the default model) broke every request.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("google.adk")

from google.genai import types  # noqa: E402

from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402


def _planner_for(model: str, effort: str, supports: bool = True):
    """Call _get_planner on a bare manager with a stubbed settings object."""
    mgr = GoogleADKWorkflowManager.__new__(GoogleADKWorkflowManager)
    # `model` is a lazily-resolved read-only property; pin it so the getter
    # returns our value instead of resolving from settings.
    mgr._model = model
    mgr._model_resolved = True
    mgr._settings = SimpleNamespace(
        thinking_effort=effort,
        supports_thinking_effort=lambda m=None: supports,
    )
    return mgr._get_planner()


class TestGemini25BudgetPath:
    @pytest.mark.parametrize(
        "effort,expected_budget",
        [("low", 4096), ("medium", 12288), ("high", 24576)],
    )
    def test_uses_thinking_budget_not_level(self, effort, expected_budget):
        planner = _planner_for("gemini-2.5-flash", effort)
        cfg = planner.thinking_config
        # The 400-causing field must be absent; the budget must be set.
        assert cfg.thinking_level is None
        assert cfg.thinking_budget == expected_budget
        assert cfg.include_thoughts is True

    def test_pro_also_uses_budget(self):
        cfg = _planner_for("gemini-2.5-pro", "high").thinking_config
        assert cfg.thinking_level is None
        assert cfg.thinking_budget == 24576


class TestGemini3LevelPath:
    def test_flash_medium_uses_level_medium(self):
        cfg = _planner_for("gemini-3-flash", "medium").thinking_config
        assert cfg.thinking_budget is None
        assert cfg.thinking_level == types.ThinkingLevel.MEDIUM

    def test_pro_medium_falls_back_to_high(self):
        cfg = _planner_for("gemini-3-pro", "medium").thinking_config
        assert cfg.thinking_budget is None
        assert cfg.thinking_level == types.ThinkingLevel.HIGH

    def test_low_and_high_map_directly(self):
        assert (
            _planner_for("gemini-3-pro", "low").thinking_config.thinking_level
            == types.ThinkingLevel.LOW
        )
        assert (
            _planner_for("gemini-3-flash", "high").thinking_config.thinking_level
            == types.ThinkingLevel.HIGH
        )


class TestNoPlanner:
    def test_none_effort_returns_no_planner(self):
        assert _planner_for("gemini-2.5-flash", "none") is None

    def test_unsupported_model_returns_no_planner(self):
        assert _planner_for("gemini-2.0-flash", "high", supports=False) is None
