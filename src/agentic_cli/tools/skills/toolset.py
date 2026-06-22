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
    return toolset
