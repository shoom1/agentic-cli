"""A skill that starts an agent.

A skill's ``SKILL.md`` may name one of the app's agent types at the top of its
frontmatter (``agent: researcher``). When an agent loads such a skill, a run of
that type starts, instead of the skill's instructions being handed back: its
task message is the skill's body under its name, its parent the loading
agent's run, ``started_by`` ``skill:<name>``, and the load's result the run's
envelope with the skill's name added.

The key takes effect only where the app lets a skill start an agent
(``"skill"`` in ``SubAgentPolicy.spawn_sources``); otherwise it is ignored, the
skill loads as instructions, as any skill does, and the build warns of it
(``skill_agent_ignored``: other tools, Claude Code's among them, give skills
an ``agent`` key of their own).

A load starts a run only where any run-time start is allowed: the type is one
of the app's, the type's ``spawnable_by`` and the policy's ``spawn_sources``
both hold ``"skill"``, and the loading run may start a type at its depth
(``max_spawn_depth``). Otherwise the load is refused and starts nothing, and a
skill whose agent its agent's first run could not start is warned of as the
agent is built: the app still starts, since a skill file can come from a
user's folder.

``SkillAgents`` is one loading agent's: ADK's ``adk/skill_tool.py::SkillLoadTool``
and Deep Agents' ``task_tool.start_skill_run`` call it. Imports no backend.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from agentic_cli.logging import Loggers
from agentic_cli.workflow.agents.brief import Brief
from agentic_cli.workflow.agents.delegate import Delegated, StartRequest
from agentic_cli.workflow.agents.spawn import SKILL_SOURCE, may_spawn, spawnable
from agentic_cli.workflow.sessions import get_active_run

if TYPE_CHECKING:
    from agentic_cli.workflow.agents.delegate import Delegations

logger = Loggers.tools()

#: The frontmatter key (top level) that names the agent type a skill starts.
SKILL_AGENT_KEY = "agent"

#: What a refused load says; ``reason`` is one of the four below.
SKILL_AGENT_ERROR = "The skill '{skill}' starts the agent '{agent}', which {reason}."
UNKNOWN_TYPE = "is not one of this app's agent types"
NOT_FOR_SKILLS = "this app does not let a skill start"
NOT_HERE = "cannot be started here: sub-agents cannot start sub-agents"
NO_TASK = "needs the skill's instructions as its task, and this skill has none"
#: The build's warning for a key that is ignored, its text.
IGNORED = (
    "The skill '{skill}' names the agent '{agent}', but this app lets no skill "
    "start an agent: it loads as instructions."
)


def skill_started_by(name: str) -> str:
    """``started_by`` of a run the skill ``name`` starts."""
    return f"skill:{name}"


def skill_agent(skill: Any) -> str | None:
    """The agent type ``skill`` names, or None.

    A value that is not text is taken as written, so it names no type and its
    load is refused, never silently loaded as instructions.
    """
    value = getattr(skill.frontmatter, SKILL_AGENT_KEY, None)
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


class SkillAgents:
    """What one agent's skills may start, and the start.

    Args:
        agent: The loading agent's type name.
        delegations: The manager's ``Delegations``, read when asked (it is
            replaced with the workflow).
        root: Whether the agent is the root (its first run is at depth 0,
            any other agent's at 1).
        warned: The (agent, skill) pairs already warned of, shared by the
            builds of one agent (a type and its model choices): each is
            logged once. None: this object's own.
    """

    def __init__(
        self,
        agent: str,
        delegations: Callable[[], Delegations],
        *,
        root: bool,
        warned: set[tuple[str, str]] | None = None,
    ) -> None:
        self._agent = agent
        self._delegations = delegations
        self._root = root
        self._warned = warned if warned is not None else set()

    def starts(self, skill: Any) -> bool:
        """Whether loading ``skill`` is a start: it names an agent type and the app
        lets a skill start one (``"skill"`` in ``spawn_sources``)."""
        return (
            skill_agent(skill) is not None
            and SKILL_SOURCE in self._delegations().policy.spawn_sources
        )

    def _reason(self, skill: Any, name: str, depth: int) -> str | None:
        """Why this agent's run at ``depth`` may not start the type ``name`` for ``skill``."""
        delegations = self._delegations()
        types = delegations.types
        config = types.get(name)
        if config is None:
            return UNKNOWN_TYPE
        policy = delegations.policy
        if name not in spawnable(list(types.values()), policy, SKILL_SOURCE):
            return NOT_FOR_SKILLS
        caller = types.get(self._agent)
        if (
            not may_spawn(depth, policy)
            or (caller is not None and caller.inherit_context)
            or (depth > 0 and config.inherit_context)
        ):
            return NOT_HERE
        if not skill.instructions.strip():
            return NO_TASK
        return None

    def refusal(self, skill: Any, depth: int) -> str | None:
        """The error a start of ``skill`` from a run at ``depth`` gets, or None to start it.

        A skill that names no agent is no start (None). One whose key the app
        ignores (``starts`` is false) is refused too: a load treats it as
        instructions before asking, and a start that comes anyway starts nothing.
        """
        name = skill_agent(skill)
        if name is None:
            return None
        reason = self._reason(skill, name, depth) if self.starts(skill) else NOT_FOR_SKILLS
        if reason is None:
            return None
        return SKILL_AGENT_ERROR.format(skill=skill.name, agent=name, reason=reason)

    def check(self, skills: Sequence[Any]) -> None:
        """Warn, once per skill, of each that names an agent this agent's first run
        could not start, or that the app lets no skill start (the key is ignored)."""
        for skill in skills:
            name = skill_agent(skill)
            if name is None or (self._agent, skill.name) in self._warned:
                continue
            if not self.starts(skill):
                self._warned.add((self._agent, skill.name))
                logger.warning(
                    "skill_agent_ignored",
                    agent=self._agent, skill=skill.name, agent_type=name,
                    message=IGNORED.format(skill=skill.name, agent=name),
                )
                continue
            reason = self._reason(skill, name, 0 if self._root else 1)
            if reason is None:
                continue
            self._warned.add((self._agent, skill.name))
            logger.warning(
                "skill_agent_not_startable",
                agent=self._agent, skill=skill.name, agent_type=name, reason=reason,
            )

    async def start(
        self,
        skill: Any,
        *,
        call_id: str | None,
        state: Mapping[str, Any],
        history: Callable[[], Any] | None,
    ) -> Delegated:
        """Load ``skill``, which ``starts``: start its agent's run, or refuse and start none."""
        caller = get_active_run()
        refused = self.refusal(skill, caller.depth if caller is not None else 0)
        if refused is not None:
            return Delegated(result={"success": False, "error": refused}, state={}, run=None)
        delegated = await self._delegations().start(StartRequest(
            agent_type=skill_agent(skill) or "",
            brief=Brief(title=skill.name, objective=skill.instructions),
            started_by=skill_started_by(skill.name),
            caller=caller, call_id=call_id, state=state, history=history, counted=True,
        ))
        if delegated.run is None:
            return delegated
        return Delegated(
            result={**delegated.result, "skill": skill.name},
            state=delegated.state, run=delegated.run,
        )
