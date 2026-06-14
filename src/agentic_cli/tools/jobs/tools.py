"""Job tools.

Two kinds, per the design:

- **Typed long-running tools** start work and return a ``job_id`` immediately;
  the LLM only ever sees these. ``run_shell_job`` is the milestone-1 reference
  (subprocess-backed). They declare ``long_running=True`` and a
  ``longrunning.<toolname>`` capability (default-ASK → user verification).
- **Observe-only generic tools** read/manage existing jobs but never start one
  (capability ``jobs.manage``). To keep the agent's tool surface small,
  ``job_status`` is the *recommended* companion to a long-running tool — it
  returns state, a stdout tail, and the result once finished, so most agents
  need only it. ``job_result``/``job_logs``/``job_cancel``/``job_list`` remain
  available as opt-in extras (and power the ``/jobs`` command), but are not in
  the default bundle.

``JobManager`` itself is never exposed to the LLM — tools reach it via the
service registry.
"""

from __future__ import annotations

from agentic_cli.tools.registry import ToolCategory, register_tool
from agentic_cli.workflow.permissions import Capability
from agentic_cli.workflow.service_registry import JOB_MANAGER, get_service


def _manager():
    """Return the JobManager, or an error dict if it isn't available."""
    jm = get_service(JOB_MANAGER)
    if jm is None:
        return {"success": False, "error": "job manager not available"}
    return jm


# ---------------------------------------------------------------------------
# Reference long-running tool (subprocess-backed)
# ---------------------------------------------------------------------------


@register_tool(
    category=ToolCategory.EXECUTION,
    capabilities=[Capability("longrunning.run_shell_job")],
    long_running=True,
    description="Run a shell command as a detached background job; returns a job_id immediately.",
)
def run_shell_job(command: str, name: str = "", cwd: str = "") -> dict:
    """Start ``command`` as a background job and return its ``job_id``.

    Use the ``job_*`` tools to check status, read logs, fetch the result, or
    cancel. The job keeps running across turns and survives a CLI restart.

    Args:
        command: Shell command to run.
        name: Optional human-friendly name.
        cwd: Optional working directory.

    Returns:
        Dict with ``job_id`` and initial ``state``.
    """
    jm = _manager()
    if isinstance(jm, dict):
        return jm
    rec = jm.submit(
        tool="run_shell_job",
        backend="subprocess",
        spec={"command": command, "cwd": cwd or None},
        name=name or None,
    )
    return {"success": True, "job_id": rec.job_id, "state": rec.state.value}


# ---------------------------------------------------------------------------
# Observe-only management tools
# ---------------------------------------------------------------------------


@register_tool(
    category=ToolCategory.EXECUTION,
    capabilities=[Capability("jobs.manage")],
    description="Check a background job: state, exit code, a stdout tail, and the result once finished.",
)
def job_status(job_id: str) -> dict:
    """One-stop check for a background job.

    Returns state, elapsed time, exit code, and a short stdout tail; once the
    job is finished it also includes ``result``. This is the only job tool most
    agents need alongside the long-running tool that started the job.
    """
    from agentic_cli.tools.jobs.backends import TERMINAL_STATES

    jm = _manager()
    if isinstance(jm, dict):
        return jm
    rec = jm.get(job_id)
    if rec is None:
        return {"success": False, "error": f"no such job: {job_id}"}
    out = rec.summary()
    out["success"] = True
    out["stdout_tail"] = jm.tail(job_id, 10, "stdout")
    if rec.state in TERMINAL_STATES:
        out["result"] = jm.result(job_id)
    return out


@register_tool(
    category=ToolCategory.EXECUTION,
    capabilities=[Capability("jobs.manage")],
    description="Get the result of a finished background job.",
)
def job_result(job_id: str) -> dict:
    """Return the job's result, or an error if it isn't finished yet."""
    jm = _manager()
    if isinstance(jm, dict):
        return jm
    rec = jm.get(job_id)
    if rec is None:
        return {"success": False, "error": f"no such job: {job_id}"}
    from agentic_cli.tools.jobs.backends import TERMINAL_STATES

    if rec.state not in TERMINAL_STATES:
        return {"success": False, "error": f"job {job_id} not finished (state={rec.state.value})"}
    return {"success": True, "state": rec.state.value, "result": jm.result(job_id)}


@register_tool(
    category=ToolCategory.EXECUTION,
    capabilities=[Capability("jobs.manage")],
    description="Read recent log lines (stdout/stderr) of a background job.",
)
def job_logs(job_id: str, last_n: int = 50, stream: str = "stdout") -> dict:
    """Return the last ``last_n`` lines of the job's ``stdout`` or ``stderr``."""
    jm = _manager()
    if isinstance(jm, dict):
        return jm
    if jm.get(job_id) is None:
        return {"success": False, "error": f"no such job: {job_id}"}
    return {"success": True, "lines": jm.tail(job_id, last_n, stream)}


@register_tool(
    category=ToolCategory.EXECUTION,
    capabilities=[Capability("jobs.manage")],
    description="Cancel a running background job.",
)
def job_cancel(job_id: str) -> dict:
    """Best-effort cancel a running job."""
    jm = _manager()
    if isinstance(jm, dict):
        return jm
    rec = jm.cancel(job_id)
    if rec is None:
        return {"success": False, "error": f"no such job: {job_id}"}
    return {"success": True, "state": rec.state.value}


@register_tool(
    category=ToolCategory.EXECUTION,
    capabilities=[Capability("jobs.manage")],
    description="List background jobs, optionally filtered by state or tag.",
)
def job_list(state: str = "", tag: str = "") -> dict:
    """List jobs (most recent first), optionally filtered by state/tag."""
    jm = _manager()
    if isinstance(jm, dict):
        return jm
    from agentic_cli.tools.jobs.backends import JobState

    state_filter = JobState(state) if state else None
    recs = jm.list(state=state_filter, tag=tag or None)
    return {"success": True, "jobs": [r.summary() for r in recs], "count": len(recs)}
