"""Tests for the unified YAML agent loader (Phase 3)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_cli.workflow.agent_loader import (
    create_workflow_manager_from_yaml,
    load_agents_from_yaml,
)
from agentic_cli.workflow.base_manager import BaseWorkflowManager


def _write(tmp_path: Path, text: str, name: str = "agents.yaml") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


class TestLoadAgentsFromYaml:
    def test_basic_load(self, tmp_path):
        path = _write(
            tmp_path,
            """
            agents:
              - name: coordinator
                model: gemini-2.5-pro
                description: Routes work
                instruction: You are the coordinator.
                tools: [kb_search, my_pkg.tools.custom]
                sub_agents: [researcher]
              - name: researcher
                prompt: Do research.
                tools: [web_search]
            """,
        )
        configs = load_agents_from_yaml(path)
        assert [c.name for c in configs] == ["coordinator", "researcher"]
        coord = configs[0]
        assert coord.model == "gemini-2.5-pro"
        assert coord.description == "Routes work"
        assert coord.get_prompt() == "You are the coordinator."
        # Tool refs stay strings (resolved later by the manager).
        assert coord.tools == ["kb_search", "my_pkg.tools.custom"]
        assert coord.sub_agents == ["researcher"]

    def test_model_settings_parsed(self, tmp_path):
        path = _write(
            tmp_path,
            """
            agents:
              - name: a
                instruction: hi
                model_settings:
                  temperature: 0.2
                  max_tokens: 1000
                  thinking:
                    mode: high
            """,
        )
        cfg = load_agents_from_yaml(path)[0]
        assert cfg.model_settings is not None
        assert cfg.model_settings.temperature == 0.2
        assert cfg.model_settings.max_tokens == 1000
        assert cfg.model_settings.thinking.mode == "high"

    def test_instruction_file_relative_to_yaml(self, tmp_path):
        (tmp_path / "prompts").mkdir()
        (tmp_path / "prompts" / "r.md").write_text("File prompt body", encoding="utf-8")
        path = _write(
            tmp_path,
            """
            agents:
              - name: a
                instruction_file: prompts/r.md
            """,
        )
        cfg = load_agents_from_yaml(path)[0]
        assert cfg.get_prompt() == "File prompt body"

    def test_bare_top_level_list(self, tmp_path):
        path = _write(
            tmp_path,
            """
            - name: a
              instruction: hi
            """,
        )
        configs = load_agents_from_yaml(path)
        assert len(configs) == 1 and configs[0].name == "a"

    def test_defaults(self, tmp_path):
        path = _write(tmp_path, "agents:\n  - name: a\n    instruction: hi\n")
        cfg = load_agents_from_yaml(path)[0]
        assert cfg.tools == []
        assert cfg.sub_agents == []
        assert cfg.description == ""
        assert cfg.model is None
        assert cfg.model_settings is None
        assert cfg.include_state_tools is True

    def test_missing_prompt_raises(self, tmp_path):
        path = _write(tmp_path, "agents:\n  - name: a\n")
        with pytest.raises(ValueError, match="must set 'prompt'"):
            load_agents_from_yaml(path)

    def test_unknown_key_rejected(self, tmp_path):
        path = _write(
            tmp_path,
            "agents:\n  - name: a\n    instruction: hi\n    bogus: 1\n",
        )
        with pytest.raises(ValidationError):
            load_agents_from_yaml(path)

    def test_empty_agents_rejected(self, tmp_path):
        path = _write(tmp_path, "agents: []\n")
        with pytest.raises(ValidationError):
            load_agents_from_yaml(path)

    def test_empty_file_raises(self, tmp_path):
        path = _write(tmp_path, "")
        with pytest.raises(ValueError, match="empty"):
            load_agents_from_yaml(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_agents_from_yaml(tmp_path / "nope.yaml")


class TestCreateManagerFromYaml:
    def test_builds_manager_and_resolves_tools(self, tmp_path, mock_context):
        path = _write(
            tmp_path,
            """
            agents:
              - name: a
                instruction: hi
                model: gemini-2.5-flash
                tools: [read_file]
            """,
        )
        mgr = create_workflow_manager_from_yaml(path, mock_context.settings)
        assert isinstance(mgr, BaseWorkflowManager)
        assert [c.name for c in mgr.agent_configs] == ["a"]
        # Tool string resolved to a callable at manager construction.
        assert callable(mgr.agent_configs[0].tools[0])
        assert getattr(mgr.agent_configs[0].tools[0], "__name__", None) == "read_file"
