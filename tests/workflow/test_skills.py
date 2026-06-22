"""Tests for skills support (Phase 5) — ADK SkillToolset wiring, scripts gated."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

pytest.importorskip("google.adk")

from agentic_cli.tools.skills import (  # noqa: E402
    SkillStore,
    make_skill_toolset,
    register_skill_tool_permissions,
)
from agentic_cli.workflow.config import AgentConfig  # noqa: E402


def _make_skill(parent: Path, name: str = "pdf-tools", desc: str = "Work with PDFs.") -> Path:
    """Create a minimal valid skill directory (dir name must match the name)."""
    skill_dir = parent / name
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        textwrap.dedent(
            f"""\
            ---
            name: {name}
            description: {desc}
            ---
            # {name}
            Follow these instructions.
            """
        ),
        encoding="utf-8",
    )
    (skill_dir / "scripts" / "run.py").write_text("print('hi')\n", encoding="utf-8")
    return skill_dir


# ---------------------------------------------------------------------------
# SkillStore
# ---------------------------------------------------------------------------


class TestSkillStore:
    def test_resolve_by_path(self, tmp_path):
        skill_dir = _make_skill(tmp_path)
        skills = SkillStore().resolve([str(skill_dir)])
        assert [s.name for s in skills] == ["pdf-tools"]

    def test_resolve_by_name_via_dirs(self, tmp_path):
        _make_skill(tmp_path)
        skills = SkillStore([str(tmp_path)]).resolve(["pdf-tools"])
        assert skills[0].name == "pdf-tools"

    def test_resolve_dedupes_by_name(self, tmp_path):
        skill_dir = _make_skill(tmp_path)
        skills = SkillStore([str(tmp_path)]).resolve([str(skill_dir), "pdf-tools"])
        assert len(skills) == 1

    def test_resolve_unknown_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            SkillStore([str(tmp_path)]).resolve(["nope"])


# ---------------------------------------------------------------------------
# make_skill_toolset (script gating)
# ---------------------------------------------------------------------------


class TestMakeSkillToolset:
    def test_scripts_disabled_by_default(self, tmp_path):
        skills = SkillStore().resolve([str(_make_skill(tmp_path))])
        ts = make_skill_toolset(skills)
        names = {t.name for t in ts._tools}
        assert "run_skill_script" not in names
        assert {"list_skills", "load_skill", "load_skill_resource"} <= names

    def test_scripts_enabled_includes_run_tool(self, tmp_path):
        skills = SkillStore().resolve([str(_make_skill(tmp_path))])
        ts = make_skill_toolset(skills, scripts_enabled=True)
        assert "run_skill_script" in {t.name for t in ts._tools}


# ---------------------------------------------------------------------------
# Permission registration
# ---------------------------------------------------------------------------


class TestSkillPermissions:
    def test_skill_tools_registered(self):
        from agentic_cli.tools.registry import get_registry
        from agentic_cli.workflow.permissions.capabilities import _CapabilityExempt

        register_skill_tool_permissions()
        reg = get_registry()
        for name in ("list_skills", "load_skill", "load_skill_resource"):
            assert isinstance(reg.get(name).capabilities, _CapabilityExempt)
        # run_skill_script is permissioned (non-exempt capability list).
        run = reg.get("run_skill_script")
        assert run is not None and not isinstance(
            run.capabilities, _CapabilityExempt
        )


# ---------------------------------------------------------------------------
# ADK manager + YAML loader integration
# ---------------------------------------------------------------------------


class TestSkillsIntegration:
    def test_manager_attaches_skill_toolset(self, tmp_path, mock_context):
        from google.adk.tools.skill_toolset import SkillToolset

        from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager

        skill_dir = _make_skill(tmp_path)
        cfg = AgentConfig(
            name="a", prompt="p", tools=[], include_state_tools=False,
            skills=[str(skill_dir)],
        )
        mgr = GoogleADKWorkflowManager(
            agent_configs=[cfg], settings=mock_context.settings, model="gemini-2.5-flash"
        )
        tools = mgr._assemble_agent_tools(cfg, mgr._get_service_tool_map())
        skill_toolsets = [t for t in tools if isinstance(t, SkillToolset)]
        assert len(skill_toolsets) == 1
        # Scripts disabled by default -> run_skill_script not exposed.
        assert "run_skill_script" not in {t.name for t in skill_toolsets[0]._tools}

    def test_yaml_loader_parses_skills(self, tmp_path):
        from agentic_cli.workflow.agent_loader import load_agents_from_yaml

        path = tmp_path / "agents.yaml"
        path.write_text(
            textwrap.dedent(
                """
                agents:
                  - name: a
                    instruction: hi
                    skills: [pdf-tools, ./skills/sql]
                """
            ),
            encoding="utf-8",
        )
        cfg = load_agents_from_yaml(path)[0]
        assert cfg.skills == ["pdf-tools", "./skills/sql"]
