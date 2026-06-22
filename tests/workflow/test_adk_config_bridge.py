"""Tests for reusing an existing native ADK config (Phase 6)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

pytest.importorskip("google.adk")

from agentic_cli.workflow.adk_config_bridge import (  # noqa: E402
    load_adk_agent_native,
    translate_adk_yaml,
)
from agentic_cli.workflow.factory import (  # noqa: E402
    create_workflow_manager_from_settings,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def _native_yaml(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "root_agent.yaml",
        """
        agent_class: LlmAgent
        name: root
        model: gemini-2.5-flash
        instruction: You are root.
        """,
    )


def _tree_yaml(tmp_path: Path) -> Path:
    _write(
        tmp_path / "child.yaml",
        """
        name: child
        model: gemini-2.5-flash
        instruction: Do work.
        tools:
          - name: web_search
        """,
    )
    return _write(
        tmp_path / "root.yaml",
        """
        name: coordinator
        model: gemini-2.5-pro
        description: Root
        instruction: Coordinate.
        generate_content_config:
          temperature: 0.3
          max_output_tokens: 2048
          thinking_config:
            thinking_budget: 8000
        tools:
          - name: read_file
          - name: kb_search
            args:
              - {name: x, value: 1}
        planner:
          thinking_config: {thinking_budget: 100}
        sub_agents:
          - config_path: child.yaml
        """,
    )


# ---------------------------------------------------------------------------
# Native passthrough
# ---------------------------------------------------------------------------


class TestNativeLoad:
    def test_loads_agent_tree(self, tmp_path):
        agent = load_adk_agent_native(_native_yaml(tmp_path))
        assert agent.name == "root"


# ---------------------------------------------------------------------------
# Translate
# ---------------------------------------------------------------------------


class TestTranslate:
    def test_translates_tree(self, tmp_path):
        configs = {c.name: c for c in translate_adk_yaml(_tree_yaml(tmp_path))}
        assert set(configs) == {"coordinator", "child"}

        coord = configs["coordinator"]
        assert coord.model == "gemini-2.5-pro"
        assert coord.description == "Root"
        assert coord.get_prompt() == "Coordinate."
        assert coord.sub_agents == ["child"]
        # tools keep names; args-bearing tool keeps its name (args dropped).
        assert coord.tools == ["read_file", "kb_search"]
        # generate_content_config -> model_settings
        assert coord.model_settings.temperature == 0.3
        assert coord.model_settings.max_tokens == 2048
        assert coord.model_settings.thinking.mode == "budget"
        assert coord.model_settings.thinking.budget_tokens == 8000

        assert configs["child"].tools == ["web_search"]

    def test_no_generate_config_means_no_model_settings(self, tmp_path):
        path = _write(
            tmp_path / "a.yaml",
            """
            name: solo
            model: gemini-2.5-flash
            instruction: hi
            """,
        )
        cfg = translate_adk_yaml(path)[0]
        assert cfg.model_settings is None


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


class TestFactoryIntegration:
    def test_native_mode_returns_adk_manager(self, tmp_path, mock_context):
        from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager

        mgr = create_workflow_manager_from_settings(
            [], mock_context.settings,
            adk_config_path=str(_native_yaml(tmp_path)),
            adk_config_mode="native",
        )
        assert isinstance(mgr, GoogleADKWorkflowManager)
        assert mgr._adk_config_path == str(_native_yaml(tmp_path))

    def test_translate_mode_uses_translated_configs(self, tmp_path, mock_context):
        mgr = create_workflow_manager_from_settings(
            [], mock_context.settings,
            adk_config_path=str(_tree_yaml(tmp_path)),
            adk_config_mode="translate",
        )
        assert {c.name for c in mgr.agent_configs} == {"coordinator", "child"}
        # Translated string tool refs were resolved at manager construction.
        coord = next(c for c in mgr.agent_configs if c.name == "coordinator")
        assert all(callable(t) for t in coord.tools)
