"""Long-running job substrate (Tier A §3.2).

`JobManager` + pluggable execution backends behind one interface, plus the
``job_*`` tools and a reference long-running tool (``run_shell_job``). See
``docs/plans/2026-06-14-job-control-design.md``.
"""

from agentic_cli.tools.jobs.backends import (
    InProcessBackend,
    JobBackend,
    JobState,
    SubprocessBackend,
    default_backends,
)
from agentic_cli.tools.jobs.manager import JobManager, JobRecord
from agentic_cli.tools.jobs.tools import (
    job_cancel,
    job_list,
    job_logs,
    job_result,
    job_status,
    run_shell_job,
)

__all__ = [
    "JobManager",
    "JobRecord",
    "JobState",
    "JobBackend",
    "SubprocessBackend",
    "InProcessBackend",
    "default_backends",
    "run_shell_job",
    "job_status",
    "job_result",
    "job_logs",
    "job_cancel",
    "job_list",
]
