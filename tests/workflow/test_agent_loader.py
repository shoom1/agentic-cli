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
from agentic_cli.workflow.agents.policy import AgentLimits, SubAgentPolicy
from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.config import AgentGraphError, validate_agent_graph


def _write(tmp_path: Path, text: str, name: str = "agents.yaml") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


class TestLoadAgentsFromYaml:
    def test_a_one_shot_sub_agent(self, tmp_path):
        path = _write(
            tmp_path,
            """
            agents:
              - name: coordinator
                instruction: Lead.
                sub_agents: [checker]
              - name: checker
                instruction: Check once.
                continuable: false
            """,
        )
        configs = {c.name: c for c in load_agents_from_yaml(path)}
        assert configs["checker"].continuable is False
        assert configs["coordinator"].continuable is True

    def test_a_continuable_that_is_not_a_bool_is_rejected(self, tmp_path):
        path = _write(
            tmp_path, "agents:\n  - name: a\n    instruction: hi\n    continuable: sometimes\n"
        )
        with pytest.raises(ValidationError) as exc:
            load_agents_from_yaml(path)
        message = str(exc.value)
        assert "continuable" in message and "valid boolean" in message

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
        assert cfg.inherit_context is False
        assert cfg.brief_required == ("objective",)
        assert cfg.limits is None
        assert cfg.continuable is True
        assert cfg.spawnable_by is None
        assert cfg.model_choices == ()

    def test_a_sub_agents_inherit_context(self, tmp_path):
        path = _write(
            tmp_path,
            """
            agents:
              - name: coordinator
                instruction: Lead.
                sub_agents: [reviewer, researcher]
              - name: reviewer
                instruction: Review what was said.
                inherit_context: true
              - name: researcher
                instruction: Find facts.
                inherit_context: false
            """,
        )
        configs = {c.name: c for c in load_agents_from_yaml(path)}
        assert configs["reviewer"].inherit_context is True
        assert configs["researcher"].inherit_context is False
        assert configs["coordinator"].inherit_context is False

    def test_an_inherit_context_that_is_not_a_bool_is_rejected(self, tmp_path):
        path = _write(
            tmp_path, "agents:\n  - name: a\n    instruction: hi\n    inherit_context: shared\n"
        )
        with pytest.raises(ValidationError) as exc:
            load_agents_from_yaml(path)
        message = str(exc.value)
        assert "inherit_context" in message and "valid boolean" in message

    @pytest.mark.parametrize("value", ["fresh", "inherit"])
    def test_the_old_context_key_is_refused(self, tmp_path, value):
        # Renamed before it shipped, with no alias: a YAML written against a
        # develop build fails naming the key, rather than starting fresh by
        # mistake.
        path = _write(
            tmp_path, f"agents:\n  - name: a\n    instruction: hi\n    context: {value}\n"
        )
        with pytest.raises(ValidationError) as exc:
            load_agents_from_yaml(path)
        message = str(exc.value)
        assert "agents.0.context" in message
        assert "Extra inputs are not permitted" in message

    def test_a_sub_agents_brief_and_limits(self, tmp_path):
        path = _write(
            tmp_path,
            """
            agents:
              - name: coordinator
                instruction: Lead.
                sub_agents: [researcher]
              - name: researcher
                instruction: Find facts.
                brief_required: [objective, deliverable]
                limits: {max_model_calls: 30, timeout: 600, max_tokens: 200000}
            """,
        )
        configs = {c.name: c for c in load_agents_from_yaml(path)}
        assert configs["researcher"].brief_required == ("objective", "deliverable")
        assert configs["researcher"].limits == AgentLimits(
            max_model_calls=30, timeout=600, max_tokens=200_000
        )
        assert configs["coordinator"].brief_required == ("objective",)
        assert configs["coordinator"].limits is None

    def test_limits_may_name_some_of_their_fields(self, tmp_path):
        path = _write(
            tmp_path, "agents:\n  - name: a\n    instruction: hi\n    limits: {timeout: 90.5}\n"
        )
        assert load_agents_from_yaml(path)[0].limits == AgentLimits(timeout=90.5)

    @pytest.mark.parametrize("limits, field, why", [
        ("{max_model_calls: 0}", "max_model_calls", "must be a positive int"),
        # Read as the YAML wrote them: true is no count, '5' no number.
        ("{max_model_calls: true}", "max_model_calls", "valid integer"),
        ("{max_model_calls: '5'}", "max_model_calls", "valid integer"),
        ("{timeout: -1}", "timeout", "must be a positive number"),
        ("{timeout: '30'}", "timeout", "valid number"),
        ("{max_tokens: 1.5}", "max_tokens", "valid integer"),
    ])
    def test_a_limit_that_is_no_limit_is_rejected(self, tmp_path, limits, field, why):
        path = _write(
            tmp_path, f"agents:\n  - name: a\n    instruction: hi\n    limits: {limits}\n"
        )
        with pytest.raises(ValidationError) as exc:
            load_agents_from_yaml(path)
        message = str(exc.value)
        assert "agents.0.limits" in message and field in message and why in message

    def test_an_unknown_limit_is_rejected(self, tmp_path):
        path = _write(
            tmp_path, "agents:\n  - name: a\n    instruction: hi\n    limits: {steps: 5}\n"
        )
        with pytest.raises(ValidationError) as exc:
            load_agents_from_yaml(path)
        message = str(exc.value)
        assert "agents.0.limits.steps" in message
        assert "Extra inputs are not permitted" in message

    def test_a_type_started_at_run_time(self, tmp_path):
        # Listed before the root: a free type is never the root.
        path = _write(
            tmp_path,
            """
            agents:
              - name: helper
                instruction: Help.
                spawnable_by: [agent, user]
              - name: coordinator
                instruction: Lead.
                sub_agents: [researcher]
              - name: researcher
                instruction: Find facts.
                spawnable_by: [skill]
            """,
        )
        configs = load_agents_from_yaml(path)
        found = {c.name: c.spawnable_by for c in configs}
        assert found == {
            "helper": frozenset({"agent", "user"}), "coordinator": None,
            "researcher": frozenset({"skill"}),
        }
        graph = validate_agent_graph(
            configs, policy=SubAgentPolicy(spawn_sources={"agent", "user", "skill"})
        )
        assert (graph.root_name, graph.free_types) == ("coordinator", ("helper",))

    def test_spawnable_by_is_a_list(self, tmp_path):
        path = _write(
            tmp_path, "agents:\n  - name: a\n    instruction: hi\n    spawnable_by: agent\n"
        )
        with pytest.raises(ValidationError) as exc:
            load_agents_from_yaml(path)
        message = str(exc.value)
        assert "agents.0.spawnable_by" in message and "valid list" in message

    def test_a_source_spawnable_by_does_not_name_is_the_graphs_to_refuse(self, tmp_path):
        path = _write(
            tmp_path,
            """
            agents:
              - name: coordinator
                instruction: Lead.
                sub_agents: [researcher]
              - name: researcher
                instruction: Find facts.
                spawnable_by: [agents]
            """,
        )
        configs = load_agents_from_yaml(path)
        with pytest.raises(AgentGraphError, match="'researcher' has spawnable_by frozenset"):
            validate_agent_graph(configs)

    def test_model_choices(self, tmp_path):
        path = _write(
            tmp_path,
            """
            agents:
              - name: coordinator
                instruction: Lead.
                sub_agents: [researcher]
              - name: researcher
                instruction: Find facts.
                model: strong
                model_choices: [fast, anthropic:claude-haiku-4-5]
            """,
        )
        configs = load_agents_from_yaml(path)
        assert [c.model_choices for c in configs] == [
            (), ("fast", "anthropic:claude-haiku-4-5"),
        ]
        assert validate_agent_graph(configs).root_name == "coordinator"

    def test_model_choices_is_a_list(self, tmp_path):
        path = _write(
            tmp_path,
            "agents:\n  - name: a\n    instruction: hi\n    model_choices: fast\n"
        )
        with pytest.raises(ValidationError) as exc:
            load_agents_from_yaml(path)
        message = str(exc.value)
        assert "agents.0.model_choices" in message and "valid list" in message

    def test_brief_required_is_a_list(self, tmp_path):
        path = _write(
            tmp_path,
            "agents:\n  - name: a\n    instruction: hi\n    brief_required: deliverable\n",
        )
        with pytest.raises(ValidationError) as exc:
            load_agents_from_yaml(path)
        message = str(exc.value)
        assert "agents.0.brief_required" in message and "valid list" in message

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

    def test_a_missing_instruction_file_is_a_value_error_naming_the_agent(self, tmp_path):
        path = _write(tmp_path, "agents:\n  - name: writer\n    instruction_file: nope.md\n")
        with pytest.raises(ValueError, match="Agent 'writer'.*instruction_file") as exc:
            load_agents_from_yaml(path)
        assert str(tmp_path / "nope.md") in str(exc.value)

    def test_an_instruction_file_that_is_a_directory_is_a_value_error(self, tmp_path):
        (tmp_path / "prompts").mkdir()
        path = _write(tmp_path, "agents:\n  - name: writer\n    instruction_file: prompts\n")
        with pytest.raises(ValueError, match="Agent 'writer'.*instruction_file"):
            load_agents_from_yaml(path)

    def test_an_instruction_file_that_is_not_utf8_is_a_value_error(self, tmp_path):
        (tmp_path / "p.md").write_bytes(b"\xff\xfe not utf-8")
        path = _write(tmp_path, "agents:\n  - name: writer\n    instruction_file: p.md\n")
        with pytest.raises(ValueError, match="Agent 'writer'.*instruction_file"):
            load_agents_from_yaml(path)

    def test_a_yaml_path_that_is_a_directory_is_a_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="cannot read"):
            load_agents_from_yaml(tmp_path)

    def test_yaml_that_does_not_parse_is_a_value_error_naming_the_file(self, tmp_path):
        path = _write(tmp_path, "agents:\n  - name: a\n    instruction: [unclosed\n")
        with pytest.raises(ValueError, match="not valid YAML") as exc:
            load_agents_from_yaml(path)
        assert str(path) in str(exc.value)


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
