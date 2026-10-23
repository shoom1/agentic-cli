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


def test_the_arxiv_specialist_ingests_web_pages_through_web_fetch():
    from research_demo.agents import ARXIV_SPECIALIST_PROMPT

    assert "kb_ingest_url" not in ARXIV_SPECIALIST_PROMPT
    assert "saved_path" in ARXIV_SPECIALIST_PROMPT


def test_the_prompts_say_delegate_and_name_no_delegation_tool():
    """ADK names a delegation tool after the sub-agent, Deep Agents calls it
    ``task``, and neither offers ``transfer_to_agent`` any more: a prompt
    says to delegate to the sub-agent, and that it reports back."""
    from research_demo.agents import AGENT_CONFIGS

    for agent in AGENT_CONFIGS:
        assert "transfer_to_agent" not in agent.get_prompt(), agent.name
    coord = next(a for a in AGENT_CONFIGS if a.name == "research_coordinator")
    assert "reports back to you" in coord.get_prompt()


def test_a_specialist_answers_its_coordinator_not_the_user():
    """A sub-agent's answer goes to the coordinator, which replies to the user."""
    from research_demo.agents import AGENT_CONFIGS

    from pathlib import Path

    from research_demo import agents

    coord = next(a for a in AGENT_CONFIGS if a.name == "research_coordinator")
    for agent in AGENT_CONFIGS:
        if agent.name in coord.sub_agents:
            assert "to the user" not in agent.get_prompt(), agent.name
    # The report writer's skill is part of what it reads.
    skill = Path(agents.__file__).parent / "skills" / "report-writer" / "SKILL.md"
    assert "tell the user" not in skill.read_text()
