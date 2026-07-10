"""Resolution + asset tests for the demo report_writer skill (no LLM)."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("google.adk")

from agentic_cli.tools.skills import SkillStore, make_skill_toolset

_SKILLS_DIR = Path(__file__).resolve().parents[2] / "examples" / "research_demo" / "skills"


def test_report_writer_skill_resolves():
    skills = SkillStore([str(_SKILLS_DIR)]).resolve(["report_writer"])
    assert len(skills) == 1
    assert skills[0].name == "report_writer"
    assert skills[0].description  # non-empty when-to-use


def test_report_writer_template_asset_present():
    tpl = _SKILLS_DIR / "report_writer" / "assets" / "report_template.tex"
    assert tpl.is_file()
    body = tpl.read_text()
    assert "\\documentclass" in body and "\\includegraphics" in body


def test_report_writer_toolset_excludes_scripts():
    skills = SkillStore([str(_SKILLS_DIR)]).resolve(["report_writer"])
    names = {t.name for t in make_skill_toolset(skills)._tools}
    assert "run_skill_script" not in names
    assert {"list_skills", "load_skill", "load_skill_resource"} <= names
