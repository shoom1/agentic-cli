"""A sub-agent run's identity on ADK: the branch its work runs on.

ADK runs a sub-agent its parent delegated to (``_SingleTurnAgentTool``) on
the branch ``<agent>@<call id>``, its parent's call that started it, under
the parent's own branch, dot-joined: a leaf two levels down runs on
``middle@toolu_c1.leaf@toolu_m1``. Every event of the run carries the branch
(``Event.branch``), and every callback of it sees it
(``callback_context.branch``). The root, and an agent the root transfers to,
run on no such branch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_cli.workflow.agents.runs import AgentRuns
    from agentic_cli.workflow.sessions import RunRef, SessionRef


def delegations(branch: str | None) -> list[tuple[str, str]]:
    """The delegations ``branch`` runs in, outermost first: ``(agent, call id)``.

    A segment without ``@`` (a workflow node's) is no delegation.
    """
    found: list[tuple[str, str]] = []
    for segment in (branch or "").split("."):
        agent, at, call_id = segment.rpartition("@")
        if at and agent and call_id:
            found.append((agent, call_id))
    return found


def branch_run(
    runs: "AgentRuns", conversation: "SessionRef | None", branch: str | None
) -> "RunRef | None":
    """The run ``branch`` is, each delegation under the one before it; None for none."""
    run: RunRef | None = None
    for agent, call_id in delegations(branch):
        run = runs.run_for(conversation, run, call_id, agent)
    return run


def parent_branch_run(
    runs: "AgentRuns", conversation: "SessionRef | None", branch: str | None
) -> "RunRef | None":
    """The run that made the delegation ``branch`` is: its parent's; None for the root's."""
    run: RunRef | None = None
    for agent, call_id in delegations(branch)[:-1]:
        run = runs.run_for(conversation, run, call_id, agent)
    return run
