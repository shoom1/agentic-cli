"""Build an ADK ``SkillToolset`` for a set of skills.

Wraps ADK's native toolset. ``run_skill_script`` is exposed only when a code
executor is actually supplied — the tool cannot run a script without one, and
advertising it regardless produced a guaranteed ``NO_CODE_EXECUTOR`` failure.
The discovery/read tools (``list_skills``/``load_skill``/``load_skill_resource``)
and the L1 metadata prompt injection always work.

No supported manager path wires an executor today, so in practice scripts stay
off; the parameter exists for a caller that owns one.

A skill whose frontmatter names an agent type starts a run of it when loaded
(``agents.py``). For an agent with such a skill, given its ``SkillAgents``,
the toolset's ``load_skill`` is the framework's (``adk/skill_tool.py``); any
other toolset is ADK's own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentic_cli.tools.skills.agents import skill_agent

if TYPE_CHECKING:
    from agentic_cli.tools.skills.agents import SkillAgents


def make_skill_toolset(
    skills: list[Any],
    *,
    code_executor: Any | None = None,
    additional_tools: list[Any] | None = None,
    agents: SkillAgents | None = None,
) -> Any:
    """Create an ADK SkillToolset; script execution follows the executor.

    Args:
        skills: Loaded ADK ``Skill`` objects.
        code_executor: ADK code executor for script execution. When None
            (the default), ``run_skill_script`` is removed from the toolset
            rather than offered and then failing.
        additional_tools: Tools surfaced when a skill with ``adk_additional_tools``
            frontmatter is activated.
        agents: The loading agent's ``SkillAgents``. Given, and one of the
            skills naming an agent type, ``load_skill`` starts that agent
            (and the skills that name one the agent could not start are
            warned of). None: every skill loads as ADK loads it.

    Returns:
        A configured ``SkillToolset``.
    """
    from google.adk.tools.skill_toolset import LoadSkillTool, RunSkillScriptTool, SkillToolset

    toolset = SkillToolset(
        skills=skills,
        code_executor=code_executor,
        additional_tools=additional_tools or [],
    )
    if code_executor is None:
        toolset._tools = [
            t for t in toolset._tools if not isinstance(t, RunSkillScriptTool)
        ]
    starting = {s.name: s for s in skills if skill_agent(s) is not None}
    if agents is not None and starting:
        from agentic_cli.workflow.adk.skill_tool import SkillLoadTool

        agents.check(skills)
        toolset._tools = [
            SkillLoadTool(toolset, starting, agents) if type(t) is LoadSkillTool else t
            for t in toolset._tools
        ]
    _bind_skill_tool_identities(toolset)
    return toolset


def _bind_skill_tool_identities(toolset: Any) -> None:
    """Give each skill tool object the registry identity it implements.

    ADK's skill tools wrap no callable, so the permission plugin cannot verify
    them the way it verifies a function tool. Binding happens *here*, where the
    framework itself constructs them and their concrete types are known —
    rather than letting the plugin resolve them by ``tool.name``, which any
    application could pick to impersonate an EXEMPT tool.
    """
    from agentic_cli.tools.registry import bind_tool_identity, get_registry

    # The names/capabilities themselves are registered when the ``skills``
    # package is imported, which importing this module guarantees.
    registry = get_registry()
    for tool in getattr(toolset, "_tools", []) or []:
        definition = registry.get(getattr(tool, "name", ""))
        if definition is not None:
            bind_tool_identity(tool, definition)
