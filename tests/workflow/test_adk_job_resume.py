"""ADK idiomatic push/resume execution (phase-2 milestone 2).

Covers wrapping ``long_running`` tools as ``LongRunningFunctionTool`` and
``resume_with_job_result`` building the right ``FunctionResponse`` and streaming
the follow-up turn — exercised against a fake runner (no live model).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("google.adk")

from google.adk.tools import LongRunningFunctionTool  # noqa: E402

from agentic_cli.tools.jobs import JobState  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.events import EventType, WorkflowEvent  # noqa: E402


# --- _wrap_long_running --------------------------------------------------------


def test_wrap_long_running_wraps_only_flagged_tools():
    from agentic_cli.tools.registry import ToolCategory, get_registry, register_tool
    from agentic_cli.workflow.permissions import EXEMPT

    @register_tool(
        category=ToolCategory.EXECUTION, capabilities=EXEMPT,
        long_running=True, description="probe long-running",
    )
    def _resume_probe_lr(x: str = "") -> dict:
        return {"success": True}

    @register_tool(
        category=ToolCategory.OTHER, capabilities=EXEMPT, description="probe normal",
    )
    def _resume_probe_normal(x: str = "") -> dict:
        return {"success": True}

    mgr = GoogleADKWorkflowManager.__new__(GoogleADKWorkflowManager)
    wrapped = mgr._wrap_long_running([_resume_probe_lr, _resume_probe_normal])

    assert isinstance(wrapped[0], LongRunningFunctionTool)
    assert wrapped[1] is _resume_probe_normal
    # Idempotent: an already-wrapped tool is not double-wrapped.
    again = mgr._wrap_long_running(wrapped)
    assert isinstance(again[0], LongRunningFunctionTool)
    assert again[1] is _resume_probe_normal
    assert get_registry().get("_resume_probe_lr").long_running is True


# --- resume_with_job_result ----------------------------------------------------


class _FakeRunner:
    def __init__(self) -> None:
        self.calls: list = []

    async def run_async(self, *, session_id, user_id, new_message, run_config):
        self.calls.append(
            SimpleNamespace(session_id=session_id, user_id=user_id, new_message=new_message)
        )
        for n in ("ev-1", "ev-2"):
            yield n


def _resume_manager(runner: _FakeRunner, *, session_exists: bool = True) -> GoogleADKWorkflowManager:
    mgr = GoogleADKWorkflowManager.__new__(GoogleADKWorkflowManager)
    mgr._settings = SimpleNamespace(app_name="test", context_window_enabled=False)
    mgr._app_name = "test"
    mgr._services = {}
    mgr._active_session_id = None
    mgr._active_user_id = None
    mgr._model = "gemini-2.5-flash"
    mgr._model_resolved = True
    mgr._on_event = None
    mgr._runner = runner
    mgr._llm_logging_plugin = None
    mgr._task_progress_plugin = None

    async def _process_event(adk_event, session_id):
        yield WorkflowEvent(type=EventType.TEXT, content=str(adk_event))

    mgr._event_processor = SimpleNamespace(model=None, process_event=_process_event)

    async def _ensure_initialized():
        return None

    mgr._ensure_initialized = _ensure_initialized

    class _SessionService:
        async def get_session(self, *, app_name, user_id, session_id):
            return object() if session_exists else None

    mgr._session_service = _SessionService()
    return mgr


def _record(**over):
    base = dict(
        job_id="j1", tool="run_shell_job", call_id="fc-1", call_name="run_shell_job",
        session_id="s1", user_id="u1", state=JobState.SUCCEEDED, exit_code=0, error=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


async def test_resume_sends_function_response_and_streams():
    runner = _FakeRunner()
    mgr = _resume_manager(runner)
    events = [ev async for ev in mgr.resume_with_job_result(_record(), result="build output")]

    # One WorkflowEvent per adk event from the fake runner.
    assert [e.content for e in events] == ["ev-1", "ev-2"]
    assert len(runner.calls) == 1

    part = runner.calls[0].new_message.parts[0]
    fr = part.function_response
    assert fr.id == "fc-1"
    assert fr.name == "run_shell_job"
    assert fr.response["state"] == "succeeded"
    assert fr.response["exit_code"] == 0
    assert fr.response["result_summary"] == "build output"
    assert "job_result('j1')" in fr.response["hint"]
    # Active turn was cleared after the resume.
    assert mgr.active_session_id is None


async def test_resume_fetches_result_from_job_manager_when_omitted():
    runner = _FakeRunner()
    mgr = _resume_manager(runner)
    mgr._services["job_manager"] = SimpleNamespace(result=lambda jid: f"fetched:{jid}")

    [ev async for ev in mgr.resume_with_job_result(_record())]
    fr = runner.calls[0].new_message.parts[0].function_response
    assert fr.response["result_summary"] == "fetched:j1"


async def test_resume_missing_call_id_yields_nothing():
    runner = _FakeRunner()
    mgr = _resume_manager(runner)
    events = [ev async for ev in mgr.resume_with_job_result(_record(call_id=None), result="x")]
    assert events == []
    assert runner.calls == []


async def test_resume_missing_session_yields_nothing():
    runner = _FakeRunner()
    mgr = _resume_manager(runner, session_exists=False)
    events = [ev async for ev in mgr.resume_with_job_result(_record(), result="x")]
    assert events == []
    assert runner.calls == []
