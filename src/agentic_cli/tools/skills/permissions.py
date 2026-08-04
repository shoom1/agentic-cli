"""Register ADK skill tool names with the framework permission registry.

ADK's ``SkillToolset`` tools aren't decorated with ``@register_tool``, so the
ADK ``PermissionPlugin`` would hard-deny them. We register their names here so
the discovery/read tools are EXEMPT and script execution is permissioned.

The registered function is a placeholder — the plugin only reads the declared
capabilities, never calls the function.
"""

from __future__ import annotations

_SKILL_READ_TOOLS = ("list_skills", "load_skill", "load_skill_resource")


def _noop(*args, **kwargs):  # pragma: no cover - placeholder, never invoked
    return None


def register_skill_tool_permissions() -> None:
    """Idempotently register skill tool names + capabilities in the registry."""
    from agentic_cli.tools.registry import ToolCategory, get_registry
    from agentic_cli.workflow.permissions import EXEMPT
    from agentic_cli.workflow.permissions.capabilities import Capability

    reg = get_registry()

    for name in _SKILL_READ_TOOLS:
        if name not in reg:
            reg.register(
                _noop,
                name=name,
                capabilities=EXEMPT,
                category=ToolCategory.KNOWLEDGE,
                description=f"ADK skill tool: {name}",
            )

    if "run_skill_script" not in reg:
        reg.register(
            _noop,
            name="run_skill_script",
            capabilities=[Capability("skill.script.exec", target_arg="file_path")],
            category=ToolCategory.EXECUTION,
            description="ADK skill tool: run_skill_script",
        )
