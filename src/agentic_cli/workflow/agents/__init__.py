"""Sub-agents, as the framework keeps them: their runs.

A sub-agent run is one piece of work a sub-agent does for its parent: one
delegation (its parent's tool call). ``runs`` names each run within its
conversation and records it from its ``AGENT_RUN`` events.
"""

from agentic_cli.workflow.agents.runs import (
    RESULT_MAX,
    RUN_CANCELLED,
    RUN_COMPLETED,
    RUN_END_STATUSES,
    RUN_FAILED,
    RUN_INTERRUPTED,
    RUN_REFUSED,
    RUN_STARTED,
    SUBAGENT_ERROR_PREFIX,
    TITLE_MAX,
    AgentRuns,
    RunRecord,
    result_head,
    run_title,
)

__all__ = [
    "AgentRuns",
    "RESULT_MAX",
    "RUN_CANCELLED",
    "RUN_COMPLETED",
    "RUN_END_STATUSES",
    "RUN_FAILED",
    "RUN_INTERRUPTED",
    "RUN_REFUSED",
    "RUN_STARTED",
    "RunRecord",
    "SUBAGENT_ERROR_PREFIX",
    "TITLE_MAX",
    "result_head",
    "run_title",
]
