"""Config-level tests for the research_demo example agents.

Verifies that AGENT_CONFIGS is wired correctly without making any LLM calls.
"""


def test_data_analyst_wired():
    from research_demo.agents import AGENT_CONFIGS

    names = {a.name for a in AGENT_CONFIGS}
    assert "data_analyst" in names
    analyst = next(a for a in AGENT_CONFIGS if a.name == "data_analyst")
    tool_names = {t.__name__ for t in analyst.tools}
    assert "sandbox_execute" in tool_names
    coord = next(a for a in AGENT_CONFIGS if a.name == "research_coordinator")
    assert "data_analyst" in coord.sub_agents


def test_report_writer_wired():
    from research_demo.agents import AGENT_CONFIGS

    names = {a.name for a in AGENT_CONFIGS}
    assert "report_writer" in names
    rw = next(a for a in AGENT_CONFIGS if a.name == "report_writer")
    tool_names = {t.__name__ for t in rw.tools}
    assert "compile_document" in tool_names
    assert "sandbox_execute" not in tool_names       # no arbitrary code exec
    assert rw.skills == ["report-writer"]
    coord = next(a for a in AGENT_CONFIGS if a.name == "research_coordinator")
    assert "report_writer" in coord.sub_agents


def test_report_writer_skills_dir_configured():
    from research_demo.settings import ResearchDemoSettings

    s = ResearchDemoSettings()
    assert any(str(p).rstrip("/").endswith("skills") for p in s.skills_dirs)
