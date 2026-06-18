"""Live end-to-end validation of ADK push/resume (phase-2 milestone 2).

Drives the *real* resume mechanic against a live Gemini model in a single
process: turn 1 calls a long-running tool (which starts a job and leaves the
call pending); we then hand the finished result back via
``resume_with_job_result`` and confirm the model actually reacts to it.

This is the test that answers the open "pending-response shape" question — it is
parametrized on whether the tool's initial return is a ``{"status":"pending"}``
dict or ``None`` — and proves ADK accepts a second ``FunctionResponse`` for an
already-pending long-running call.

Marked ``@pytest.mark.llm`` (skipped by default; costs a few real calls). ADK
only, so it requires GOOGLE_API_KEY:

    conda run -n agenticcli python -m pytest tests/integration/test_live_job_resume.py -v -m llm
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

from agentic_cli.config import BaseSettings, set_settings
from agentic_cli.tools.registry import ToolCategory, register_tool
from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.events import EventType
from agentic_cli.workflow.factory import create_workflow_manager_from_settings
from agentic_cli.workflow.permissions import Capability
from agentic_cli.workflow.service_registry import JOB_MANAGER, get_service

from tests.integration.helpers import find_events, find_tool_calls

_has_google = bool(os.environ.get("GOOGLE_API_KEY"))

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(not _has_google, reason="ADK resume needs GOOGLE_API_KEY"),
]

# Toggled per-parametrization to test both initial-return shapes.
_SHAPE = {"return_none": False}

_TOKEN = "RESUME_TOKEN_8731"


@register_tool(
    category=ToolCategory.EXECUTION,
    capabilities=[Capability("longrunning.live_resume_probe")],
    long_running=True,
    description="Start a background shell job; returns a job_id immediately.",
)
def live_resume_probe(command: str = "", tool_context=None):
    """Submit a subprocess job flagged for resume, capturing the call id."""
    jm = get_service(JOB_MANAGER)
    if jm is None:
        return {"success": False, "error": "job manager not available"}
    jm.submit(
        tool="live_resume_probe",
        backend="subprocess",
        spec={"command": command or f"echo {_TOKEN}"},
        resume_on_complete=True,
        call_id=getattr(tool_context, "function_call_id", None),
        call_name="live_resume_probe",
    )
    # The shape under test: an explicit pending dict vs. None.
    return None if _SHAPE["return_none"] else {"status": "pending"}


_AGENT = AgentConfig(
    name="job_runner",
    prompt=(
        "You run background jobs. When asked, call live_resume_probe with a shell "
        f"command that prints {_TOKEN} (e.g. 'echo {_TOKEN}'), then tell the user "
        "you started it. Never ask for confirmation. You have no other tools."
    ),
    tools=[live_resume_probe],
    description="Starts a long-running background job.",
)


@pytest.fixture
def live_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolated settings + a job tool mapped to the JobManager service."""
    app_name = "agentic_cli_live_resume_test"
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    monkeypatch.chdir(workspace)

    # Make the probe tool require the JobManager (detection is by tool name).
    monkeypatch.setattr(
        BaseWorkflowManager,
        "_TOOL_SERVICE_MAP",
        {**BaseWorkflowManager._TOOL_SERVICE_MAP, "live_resume_probe": "job_manager"},
    )

    settings = BaseSettings(
        app_name=app_name, workspace_dir=workspace, permissions_enabled=False
    )
    set_settings(settings)
    try:
        yield settings
    finally:
        shutil.rmtree(Path.home() / f".{app_name}", ignore_errors=True)


def _wait_terminal(jm, job_id: str, timeout: float = 10.0):
    from agentic_cli.tools.jobs import JobState

    terminal = {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED, JobState.UNKNOWN}
    end = time.time() + timeout
    while time.time() < end:
        rec = jm.get(job_id)
        if rec is not None and rec.state in terminal:
            return rec
        time.sleep(0.1)
    return jm.get(job_id)


@pytest.mark.parametrize("return_none", [False, True], ids=["pending_dict", "none"])
async def test_long_running_tool_then_resume(live_settings, return_none: bool):
    _SHAPE["return_none"] = return_none

    manager = create_workflow_manager_from_settings(
        agent_configs=[_AGENT], settings=live_settings
    )
    assert hasattr(manager, "resume_with_job_result"), "expected the ADK manager"

    async def _auto_input(request) -> str:  # noqa: ANN001
        return "yes, proceed"

    manager.set_input_callback(_auto_input)
    try:
        # --- Turn 1: model calls the long-running tool, job starts pending. ---
        session_id = "live-resume-1"
        turn1 = []
        async for ev in manager.process(
            message="Start a background job now.", user_id="tester", session_id=session_id
        ):
            turn1.append(ev)

        assert find_tool_calls(turn1, "live_resume_probe"), (
            "model did not call the long-running tool; calls: "
            f"{[c.metadata.get('tool_name') for c in find_tool_calls(turn1)]}"
        )

        jm = manager.job_manager
        assert jm is not None, "JobManager was not created"
        awaiting = [r for r in jm.list() if r.resume_on_complete]
        assert len(awaiting) == 1, f"expected one resume-flagged job, got {awaiting}"
        record = awaiting[0]
        # The crux of association: the tool captured the ADK function_call id.
        assert record.call_id, "call_id was not captured from tool_context"
        assert record.session_id == session_id and record.user_id == "tester"

        _wait_terminal(jm, record.job_id)
        record = jm.get(record.job_id)

        # --- Turn 2: deliver the result back to the pending call and resume. ---
        turn2 = []
        async for ev in manager.resume_with_job_result(record):
            turn2.append(ev)

        # If ADK rejected a 2nd FunctionResponse for the pending call, run_async
        # would have raised above. Reaching here with a model reply == success.
        text_events = find_events(turn2, EventType.TEXT)
        assert text_events, (
            f"resume produced no model text (shape={'none' if return_none else 'pending_dict'}); "
            f"events: {[(e.type, e.content[:40]) for e in turn2]}"
        )
    finally:
        manager.clear_input_callback()
        await manager.cleanup()
