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

    def test_a_name_is_looked_up_only_under_skills_dirs(self, tmp_path, monkeypatch):
        """A checkout's folder named after a skill cannot stand in for it."""
        skills_dir = tmp_path / "skills"
        _make_skill(skills_dir, desc="The configured skill.")
        checkout = tmp_path / "checkout"
        _make_skill(checkout, desc="A folder in the checkout.")
        monkeypatch.chdir(checkout)
        skills = SkillStore([str(skills_dir)]).resolve(["pdf-tools"])
        assert [s.description for s in skills] == ["The configured skill."]

    def test_a_name_found_only_in_the_working_directory_is_not_found(
        self, tmp_path, monkeypatch
    ):
        _make_skill(tmp_path)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="not found"):
            SkillStore().resolve(["pdf-tools"])

    @pytest.mark.parametrize("ref", ["./pdf-tools", "skills/pdf-tools", ".."])
    def test_a_relative_path_is_refused(self, tmp_path, monkeypatch, ref):
        _make_skill(tmp_path)
        _make_skill(tmp_path / "skills")
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="absolute"):
            SkillStore([str(tmp_path / "skills")]).resolve([ref])

    def test_a_relative_skills_dir_is_refused(self, tmp_path, monkeypatch):
        """A relative entry would look names up in the working directory."""
        _make_skill(tmp_path / "skills")
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="absolute"):
            SkillStore(["skills"]).resolve(["pdf-tools"])

    def test_a_skills_dir_may_start_at_home(self, tmp_path, monkeypatch):
        _make_skill(tmp_path / "skills")
        monkeypatch.setenv("HOME", str(tmp_path))
        skills = SkillStore(["~/skills"]).resolve(["pdf-tools"])
        assert [s.name for s in skills] == ["pdf-tools"]


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

    def test_run_tool_appears_only_with_a_code_executor(self, tmp_path):
        """The tool is exposed by the thing that makes it work, not by a flag.

        A flag could enable it without an executor, and every call then failed
        with NO_CODE_EXECUTOR.
        """
        skills = SkillStore().resolve([str(_make_skill(tmp_path))])
        ts = make_skill_toolset(skills, code_executor=object())
        assert "run_skill_script" in {t.name for t in ts._tools}

    def test_manager_path_never_exposes_the_script_tool(self, tmp_path):
        """The supported manager path supplies no executor, so scripts stay off."""
        from types import SimpleNamespace

        import pytest

        pytest.importorskip("google.adk")
        from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager

        mgr = GoogleADKWorkflowManager.__new__(GoogleADKWorkflowManager)
        mgr._settings = SimpleNamespace(skills_dirs=[])
        toolset = mgr._build_skill_toolset([str(_make_skill(tmp_path))])
        assert "run_skill_script" not in {t.name for t in toolset._tools}

    def test_removed_setting_is_gone(self):
        from agentic_cli.config import BaseSettings

        assert "skill_scripts_enabled" not in BaseSettings.model_fields


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

    async def test_a_task_may_narrow_a_sub_agent_to_its_skill_tools_and_its_own(
        self, tmp_path
    ):
        """What ``task``'s ``tools`` may name: the agent's own tools by their names,
        its skill toolset's tools, never the state or delegation tools."""
        from agentic_cli.tools import read_file
        from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager
        from tests.conftest import MockContext

        configs = [
            AgentConfig(name="lead", prompt="p", sub_agents=["a"]),
            AgentConfig(name="a", prompt="p", tools=[read_file],
                        skills=[str(_make_skill(tmp_path))]),
        ]
        with MockContext(google_api_key="test-key", session_store="memory") as context:
            mgr = GoogleADKWorkflowManager(
                agent_configs=configs, settings=context.settings,
                model="gemini-2.5-flash"
            )
            await mgr.initialize_services()
        try:
            assert await mgr._agent_tool_names("a") == {
                "read_file", "list_skills", "load_skill", "load_skill_resource",
            }
            assert await mgr._agent_tool_names("nobody") == frozenset()
        finally:
            await mgr.cleanup()

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


# ---------------------------------------------------------------------------
# A skill that starts an agent: the framework's load_skill
# ---------------------------------------------------------------------------


def _starting_skill(parent: Path, name: str = "research", agent: str = "helper") -> Path:
    """A skill whose frontmatter names the agent type ``agent``."""
    skill_dir = parent / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Researches.\nagent: {agent}\n---\nLook into it.\n",
        encoding="utf-8",
    )
    return skill_dir


def _agents(sources: frozenset[str] | set[str] = frozenset({"skill"})):
    """The root ``lead``'s ``SkillAgents``, over delegations that may start ``helper``
    (from ``sources``)."""
    from agentic_cli.tools.skills import SkillAgents
    from agentic_cli.workflow.agents import AgentRuns, SubAgentPolicy
    from agentic_cli.workflow.agents.delegate import Delegations

    async def never(run):  # pragma: no cover - no run starts here
        raise AssertionError(run)

    delegations = Delegations(
        runs=AgentRuns(), policy=SubAgentPolicy(spawn_sources=sources), hook=never,
        types={
            "lead": AgentConfig(name="lead", prompt="p"),
            "helper": AgentConfig(name="helper", prompt="p", spawnable_by=frozenset({"skill"})),
        },
    )
    return SkillAgents("lead", lambda: delegations, root=True)


def _load_skill(toolset):
    (tool,) = [t for t in toolset._tools if t.name == "load_skill"]
    return tool


class TestTheFrameworksLoadSkill:
    def test_an_agent_with_a_skill_that_starts_one_gets_it_declared_as_adks(self, tmp_path):
        from google.adk.tools.skill_toolset import SkillToolset

        from agentic_cli.tools.registry import get_registry, identify_tool
        from agentic_cli.workflow.adk.skill_tool import SkillLoadTool

        skills = SkillStore().resolve([str(_make_skill(tmp_path)), str(_starting_skill(tmp_path))])

        toolset = make_skill_toolset(skills, agents=_agents())

        tool = _load_skill(toolset)
        assert type(tool) is SkillLoadTool
        assert identify_tool(tool) is get_registry().get("load_skill")
        adks = _load_skill(SkillToolset(skills=skills))
        assert tool._get_declaration() == adks._get_declaration()
        assert [t.name for t in toolset._tools] == [
            t.name for t in make_skill_toolset(skills)._tools
        ]

    def test_without_such_a_skill_or_without_its_agents_it_is_adks_own(self, tmp_path):
        from google.adk.tools.skill_toolset import LoadSkillTool

        plain = SkillStore().resolve([str(_make_skill(tmp_path))])
        starting = SkillStore().resolve([str(_starting_skill(tmp_path))])

        for toolset in (make_skill_toolset(plain, agents=_agents()), make_skill_toolset(starting)):
            assert type(_load_skill(toolset)) is LoadSkillTool

    @pytest.mark.parametrize("sources", [set(), {"agent"}], ids=["default", "agents-only"])
    def test_while_no_skill_may_start_one_the_toolset_is_adks_own(self, tmp_path, sources):
        """The key is ignored: ``load_skill`` is ADK's own, object for object."""
        from google.adk.tools.skill_toolset import LoadSkillTool

        skills = SkillStore().resolve([str(_starting_skill(tmp_path))])

        toolset = make_skill_toolset(skills, agents=_agents(sources))

        assert type(_load_skill(toolset)) is LoadSkillTool

    async def test_the_gate_lets_the_one_the_framework_built_through_and_no_other(
        self, tmp_path
    ):
        from agentic_cli.workflow.adk.permission_plugin import PermissionPlugin
        from agentic_cli.workflow.adk.skill_tool import SkillLoadTool

        skills = SkillStore().resolve([str(_starting_skill(tmp_path))])
        toolset = make_skill_toolset(skills, agents=_agents())
        unbound = SkillLoadTool(toolset, {}, _agents())

        async def gate(tool):
            return await PermissionPlugin().before_tool_callback(
                tool=tool, tool_args={"skill_name": "research"}, tool_context=None
            )

        assert await gate(_load_skill(toolset)) is None
        denied = await gate(unbound)
        assert denied["success"] is False and "not registered" in denied["error"]

    def test_what_it_rests_on_of_adks_load_skill(self, tmp_path):
        """ADK's toolset holds one exact ``LoadSkillTool``, made from the toolset,
        whose ``run_async`` takes ``args`` and ``tool_context`` by keyword."""
        import inspect

        from google.adk.tools.skill_toolset import LoadSkillTool, SkillToolset

        toolset = SkillToolset(skills=SkillStore().resolve([str(_make_skill(tmp_path))]))

        assert [type(t) for t in toolset._tools].count(LoadSkillTool) == 1
        assert LoadSkillTool(toolset).name == "load_skill"
        parameters = inspect.signature(LoadSkillTool.run_async).parameters
        assert {name: p.kind for name, p in parameters.items() if name != "self"} == {
            "args": inspect.Parameter.KEYWORD_ONLY,
            "tool_context": inspect.Parameter.KEYWORD_ONLY,
        }
