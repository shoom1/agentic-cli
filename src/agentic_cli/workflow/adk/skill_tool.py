"""ADK's ``load_skill`` for an agent with a skill that starts an agent.

A skill whose frontmatter names an agent type starts a run of it when loaded
(``tools/skills/agents.py``). ADK's own ``LoadSkillTool`` hands a skill's
instructions back, so an agent with such a skill gets this subclass in its
place (``make_skill_toolset(agents=)``): a skill that names a type goes to
``SkillAgents.start`` and answers as ``task`` does (the run's envelope as JSON
text, the outside-content mark read from and written back to the caller's
state, an inheriting type given the conversation); any other skill loads as
ADK loads it, so an agent whose skills name no type keeps ADK's own tool,
object for object.

The declaration is ADK's: the subclass changes what a call does, not what the
model is offered. ``make_skill_toolset`` binds the tool to the ``load_skill``
declaration as it binds ADK's skill tools, so the permission gate knows it by
identity, and denies a ``SkillLoadTool`` built anywhere else. What it relies on
of ADK (``LoadSkillTool(toolset)``, ``run_async(args=, tool_context=)``) is
contract-tested. It is a module of its own because it subclasses ADK's class,
which the skills package imports lazily: ``make_skill_toolset`` imports this
module only when it builds the tool.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from google.adk.tools.skill_toolset import LoadSkillTool, SkillToolset

from agentic_cli.workflow.adk.task_tool import SHARED_KEYS, inherited_events

if TYPE_CHECKING:
    from google.adk.tools.tool_context import ToolContext

    from agentic_cli.tools.skills.agents import SkillAgents


class SkillLoadTool(LoadSkillTool):
    """``load_skill``, starting the agent a skill names.

    Args:
        toolset: The skill toolset the tool belongs to.
        starting: The toolset's skills that name an agent type, by name.
        agents: The loading agent's ``SkillAgents``.
    """

    def __init__(
        self, toolset: SkillToolset, starting: Mapping[str, Any], agents: SkillAgents
    ) -> None:
        super().__init__(toolset)
        self._starting = dict(starting)
        self._agents = agents

    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        name = args.get("skill_name")
        skill = self._starting.get(name) if isinstance(name, str) else None
        if skill is None:
            return await super().run_async(args=args, tool_context=tool_context)
        state = tool_context.state
        session = tool_context.session
        delegated = await self._agents.start(
            skill,
            call_id=tool_context.function_call_id or "",
            state={key: state[key] for key in SHARED_KEYS if key in state},
            history=lambda: inherited_events(session.events),
        )
        for key, value in delegated.state.items():
            state[key] = value
        return json.dumps(delegated.result, ensure_ascii=False)
