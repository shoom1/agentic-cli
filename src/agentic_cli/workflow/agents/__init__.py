"""Sub-agents, as the framework keeps them: their runs, and how they run.

A sub-agent run is one piece of work a sub-agent does for its parent: one
delegation (its parent's tool call) or one follow-up. An instance outlives
its runs: it is idle between them and takes the next. ``runs`` names each run
and instance within its conversation and records them from its
``AGENT_RUN`` and ``AGENT_INSTANCE`` events; ``policy`` holds
an app's sub-agent policy and a run's limits; ``slots`` how many runs work
at once.
"""

from agentic_cli.workflow.agents.policy import DEFAULT_LIMITS, AgentLimits, SubAgentPolicy
from agentic_cli.workflow.agents.runs import (
    INSTANCE_CLOSED,
    INSTANCE_IDLE,
    INSTANCE_RUNNING,
    INSTANCE_STATUSES,
    INSTANCE_STOPPED,
    RESULT_MAX,
    RUN_CANCELLED,
    RUN_COMPLETED,
    RUN_END_STATUSES,
    RUN_FAILED,
    RUN_INPUT_REQUIRED,
    RUN_INTERRUPTED,
    RUN_QUEUED,
    RUN_REFUSED,
    RUN_RUNNING,
    RUN_STARTED,
    RUN_WORKING_STATUSES,
    SUBAGENT_ERROR_PREFIX,
    TITLE_MAX,
    AgentRuns,
    InstanceRecord,
    RunRecord,
    result_head,
    run_label,
    run_title,
)

__all__ = [
    "AgentLimits",
    "AgentRuns",
    "DEFAULT_LIMITS",
    "INSTANCE_CLOSED",
    "INSTANCE_IDLE",
    "INSTANCE_RUNNING",
    "INSTANCE_STATUSES",
    "INSTANCE_STOPPED",
    "InstanceRecord",
    "RESULT_MAX",
    "RUN_CANCELLED",
    "RUN_COMPLETED",
    "RUN_END_STATUSES",
    "RUN_FAILED",
    "RUN_INPUT_REQUIRED",
    "RUN_INTERRUPTED",
    "RUN_QUEUED",
    "RUN_REFUSED",
    "RUN_RUNNING",
    "RUN_STARTED",
    "RUN_WORKING_STATUSES",
    "RunRecord",
    "SUBAGENT_ERROR_PREFIX",
    "SubAgentPolicy",
    "TITLE_MAX",
    "result_head",
    "run_label",
    "run_title",
]
