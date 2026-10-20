"""Framework skills support: Agent Skills (SKILL.md folders) as agent tools.

``SkillStore`` loads skills with ADK's native loader, for either backend.
``make_skill_toolset`` builds ADK's ``SkillToolset`` from them, and
``make_skill_tools`` (``langchain_tools``) the same tools as LangChain tools,
for Deep Agents. It loads on first use, because it imports LangChain, which
only the ``deepagents`` extra installs.

Importing this package registers the skill tool names with the permission
registry, so both backends bind their skill tools to the same definitions
(the ADK PermissionPlugin and the Deep Agents gate both allow them).
"""

from typing import Any

from agentic_cli.tools.skills.permissions import register_skill_tool_permissions
from agentic_cli.tools.skills.store import SkillStore
from agentic_cli.tools.skills.toolset import make_skill_toolset

# Ensure skill tool names are permissioned as soon as skills are used.
register_skill_tool_permissions()

# make_skill_tools is left out: a star import would load LangChain.
__all__ = ["SkillStore", "make_skill_toolset", "register_skill_tool_permissions"]


def __getattr__(name: str) -> Any:
    """``make_skill_tools``, imported on first use (it needs the extra)."""
    if name == "make_skill_tools":
        from agentic_cli.tools.skills.langchain_tools import make_skill_tools

        return make_skill_tools
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
