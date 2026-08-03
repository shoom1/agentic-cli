"""Build an ADK ``SkillToolset`` for a set of skills.

Wraps ADK's native toolset. When script execution is disabled (the default),
the ``run_skill_script`` tool is removed so it isn't advertised to the model;
the discovery/read tools (``list_skills``/``load_skill``/``load_skill_resource``)
and the L1 metadata prompt injection still work.
"""

from __future__ import annotations

from typing import Any


def make_skill_toolset(
    skills: list[Any],
    *,
    scripts_enabled: bool = False,
    code_executor: Any | None = None,
    additional_tools: list[Any] | None = None,
) -> Any:
    """Create an ADK SkillToolset, optionally excluding script execution.

    Args:
        skills: Loaded ADK ``Skill`` objects.
        scripts_enabled: If False (default), ``run_skill_script`` is removed.
        code_executor: ADK code executor for script execution (only meaningful
            when ``scripts_enabled`` is True).
        additional_tools: Tools surfaced when a skill with ``adk_additional_tools``
            frontmatter is activated.

    Returns:
        A configured ``SkillToolset``.
    """
    from google.adk.tools.skill_toolset import RunSkillScriptTool, SkillToolset

    toolset = SkillToolset(
        skills=skills,
        code_executor=code_executor,
        additional_tools=additional_tools or [],
    )
    if not scripts_enabled:
        toolset._tools = [
            t for t in toolset._tools if not isinstance(t, RunSkillScriptTool)
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
