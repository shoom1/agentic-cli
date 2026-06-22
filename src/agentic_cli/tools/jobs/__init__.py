"""Long-running job substrate (Tier A §3.2).

`JobManager` + pluggable execution backends behind one interface, plus the
generic observe-only ``job_*`` tools. Typed long-running tools (the ones that
actually start work) are application-provided — see ``run_shell_job`` in
``examples/jobs_demo.py``. See ``docs/plans/2026-06-14-job-control-design.md``.
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
)

__all__ = [
    "JobManager",
    "JobRecord",
    "JobState",
    "JobBackend",
    "SubprocessBackend",
    "InProcessBackend",
    "default_backends",
    "job_status",
    "job_result",
    "job_logs",
    "job_cancel",
    "job_list",
]
