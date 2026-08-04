"""Framework skills support, built on ADK's native Agent-Skills system.

Importing this package registers the ADK skill tool names with the permission
registry (so the ADK PermissionPlugin allows them).
"""

from agentic_cli.tools.skills.permissions import register_skill_tool_permissions
from agentic_cli.tools.skills.store import SkillStore
from agentic_cli.tools.skills.toolset import make_skill_toolset

# Ensure skill tool names are permissioned as soon as skills are used.
register_skill_tool_permissions()

__all__ = ["SkillStore", "make_skill_toolset", "register_skill_tool_permissions"]
