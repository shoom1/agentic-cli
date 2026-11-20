"""The parent's tools that continue its sub-agents, as both backends call them.

``message_agent``, ``stop_agent`` and ``check_agents`` (``FOLLOW_UP_TOOLS``,
declared in ``tools/_core/delegation.py``) are offered beside ``task`` to an
agent with sub-agents when ``SubAgentPolicy.follow_ups`` is on. Each backend
builds its own (ADK's ``adk/task_tool.py::FollowUpTool``, Deep Agents'
``deepagents/task_tool.py::build_follow_up_tools``) from the same schema and
description (``agents/brief.py``), and hands each call to
``call_follow_up_tool``, which reads the model's arguments and calls the
manager's ``Delegations`` (``message``, ``stop``, ``check``) as the run in
progress (``get_active_run``: the calling agent's run; None for the root),
in the active turn's conversation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from agentic_cli.tools._core.delegation import (
    CHECK_AGENTS_TOOL,
    MESSAGE_AGENT_TOOL,
    STOP_AGENT_TOOL,
)
from agentic_cli.workflow.agents.delegate import Delegated, FollowUpRequest, started_by
from agentic_cli.workflow.sessions import get_active_run

if TYPE_CHECKING:
    from agentic_cli.workflow.agents.delegate import Delegations


async def call_follow_up_tool(
    delegations: Delegations,
    name: str,
    arguments: Mapping[str, Any],
    *,
    call_id: str | None,
    state: Mapping[str, Any],
) -> Delegated:
    """A call of ``name``, one of the parent's tools, with the model's ``arguments``.

    ``agent`` names the sub-agent (stripped; anything but text is none, which
    no sub-agent is); ``message_agent``'s ``message`` goes as written, and
    ``interrupt``, like ``stop_agent``'s ``close``, holds only when it is
    ``true``. ``call_id`` is the calling tool call, ``state`` the caller's
    shared state (the outside-content mark), which a follow-up run starts
    from.

    Returns:
        ``message``'s answer (the envelope of the run it started, the steer
        result, or an error result, with the shared state a run changed),
        or ``stop``'s or ``check``'s result with no state and no run.

    Raises:
        ValueError: ``name`` is not one of the parent's tools.
    """
    caller = get_active_run()
    agent = arguments.get("agent")
    agent = agent.strip() if isinstance(agent, str) and agent.strip() else None
    if name == MESSAGE_AGENT_TOOL:
        return await delegations.message(
            FollowUpRequest(
                instance=agent or "",
                message=arguments.get("message"),  # type: ignore[arg-type]  # message() checks it
                started_by=started_by(caller),
                caller=caller,
                call_id=call_id,
                state=dict(state),
            ),
            interrupt=arguments.get("interrupt") is True,
        )
    if name == STOP_AGENT_TOOL:
        result = delegations.stop(None, caller, agent or "", close=arguments.get("close") is True)
    elif name == CHECK_AGENTS_TOOL:
        result = delegations.check(None, caller, agent)
    else:
        raise ValueError(f"{name!r} is not one of the parent's tools.")
    return Delegated(result=result, state={}, run=None)
