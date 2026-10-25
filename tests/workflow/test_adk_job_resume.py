"""ADK idiomatic push/resume execution (phase-2 milestone 2).

Covers wrapping ``long_running`` tools as ``LongRunningFunctionTool`` and
``resume_with_job_result`` building the right ``FunctionResponse`` and streaming
the follow-up turn — exercised against a fake runner (no live model).
"""

from __future__ import annotations

import asyncio
import json

from types import SimpleNamespace

import pytest

pytest.importorskip("google.adk")

from google.adk.tools import LongRunningFunctionTool  # noqa: E402

from agentic_cli.tools.jobs import JobState  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.interrupts import OpenCalls  # noqa: E402
from agentic_cli.tools.jobs.resume import JOB_NOTICE_ASK, JOB_NOTICE_HEADER  # noqa: E402
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

    async def run_async(self, *, session_id, user_id, new_message, state_delta=None,
                        abort_signal=None):
        self.calls.append(
            SimpleNamespace(session_id=session_id, user_id=user_id, new_message=new_message)
        )
        for n in ("ev-1", "ev-2"):
            yield n


def _resume_manager(
    runner: _FakeRunner, *, session_exists: bool = True, native: bool = False
) -> GoogleADKWorkflowManager:
    mgr = GoogleADKWorkflowManager.__new__(GoogleADKWorkflowManager)
    mgr._settings = SimpleNamespace(app_name="test", default_user="default_user")
    mgr._open_calls = OpenCalls()
    mgr._adk_config_path = "root_agent.yaml" if native else None
    mgr._app_name = "test"
    mgr._services = {}
    # Turns serialize on the manager's turn lock (see the concurrency contract).
    mgr._turn_lock = asyncio.Lock()
    mgr._lifecycle_lock = asyncio.Lock()
    mgr._on_event = None
    # Turn admission re-checks that the backend is live while holding the turn
    # lock, so the double has to present a complete one.
    mgr._initialized = True
    mgr._runner = runner
    mgr._root_agent = SimpleNamespace(name="root")
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


def _notice_payload(runner: _FakeRunner) -> dict:
    """The payload of the job notice the fake runner was given."""
    text = runner.calls[0].new_message.parts[0].text
    header, ask, payload = text.split("\n", 2)
    assert (header, ask) == (JOB_NOTICE_HEADER, JOB_NOTICE_ASK)
    return json.loads(payload)


async def test_resume_sends_the_job_notice_and_streams():
    """One user-role notice, as on Deep Agents: a result for the pending call
    itself made ADK leave out every turn since it (its rearranger)."""
    runner = _FakeRunner()
    mgr = _resume_manager(runner)
    events = [ev async for ev in mgr.resume_with_job_result(_record(), result="build output")]

    assert [e.content for e in events] == ["ev-1", "ev-2"]
    assert len(runner.calls) == 1
    assert runner.calls[0].new_message.role == "user"
    payload = _notice_payload(runner)
    assert payload["state"] == "succeeded"
    assert payload["exit_code"] == 0
    assert payload["result_summary"] == "build output"


async def test_a_native_adk_app_still_answers_the_pending_call():
    """A native ADK YAML app's agents are ADK's own: its job answers its call."""
    runner = _FakeRunner()
    mgr = _resume_manager(runner, native=True)
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
    assert _notice_payload(runner)["result_summary"] == "fetched:j1"


# What ADK sent before the payload moved to ``tools/jobs/resume.py``, byte for
# byte (json.dumps tells 0 from false and keeps the key order).
_HINT = "Full result via job_result('{0}'); logs via job_logs('{0}')."
_PINNED_PAYLOADS = [
    pytest.param(
        {}, "build output",
        '{"job_id": "j1", "tool": "run_shell_job", "state": "succeeded", "exit_code": 0, '
        '"result_summary": "build output", "hint": "' + _HINT.format("j1") + '"}',
        id="succeeded",
    ),
    pytest.param(
        {"state": JobState.FAILED, "exit_code": 2, "error": "exit 2: no such file"}, None,
        '{"job_id": "j1", "tool": "run_shell_job", "state": "failed", "exit_code": 2, '
        '"error": "exit 2: no such file", "hint": "' + _HINT.format("j1") + '"}',
        id="failed",
    ),
    pytest.param(
        {"state": JobState.CANCELLED, "exit_code": None}, None,
        '{"job_id": "j1", "tool": "run_shell_job", "state": "cancelled", '
        '"hint": "' + _HINT.format("j1") + '"}',
        id="cancelled",
    ),
    pytest.param(
        {}, "x" * 1500,
        '{"job_id": "j1", "tool": "run_shell_job", "state": "succeeded", "exit_code": 0, '
        '"result_summary": "' + "x" * 1000 + '\\u2026 (+500 chars)", '
        '"hint": "' + _HINT.format("j1") + '"}',
        id="long",
    ),
    pytest.param(
        {}, {"rows": 3, "ok": True},
        '{"job_id": "j1", "tool": "run_shell_job", "state": "succeeded", "exit_code": 0, '
        '"result_summary": "{\'rows\': 3, \'ok\': True}", "hint": "' + _HINT.format("j1") + '"}',
        id="not-text",
    ),
    pytest.param(
        {}, "café ✓",
        '{"job_id": "j1", "tool": "run_shell_job", "state": "succeeded", "exit_code": 0, '
        '"result_summary": "caf\\u00e9 \\u2713", "hint": "' + _HINT.format("j1") + '"}',
        id="unicode",
    ),
    pytest.param(
        {"error": ""}, "",
        '{"job_id": "j1", "tool": "run_shell_job", "state": "succeeded", "exit_code": 0, '
        '"result_summary": "", "hint": "' + _HINT.format("j1") + '"}',
        id="empty",
    ),
]


@pytest.mark.parametrize(("over", "result", "sent"), _PINNED_PAYLOADS)
async def test_resume_payload_is_unchanged(over, result, sent):
    runner = _FakeRunner()
    mgr = _resume_manager(runner)

    [ev async for ev in mgr.resume_with_job_result(_record(**over), result=result)]

    assert _notice_payload(runner) == json.loads(sent)


@pytest.mark.parametrize(("over", "result", "sent"), _PINNED_PAYLOADS)
def test_the_shared_payload_is_the_one_adk_sends(over, result, sent):
    """Deep Agents' notification carries the same payload (``tools/jobs/resume.py``)."""
    from agentic_cli.tools.jobs.resume import job_result_payload

    assert json.dumps(job_result_payload(_record(**over), result)) == sent


async def test_a_job_without_its_call_id_is_still_told():
    """The notice names the job, not the call: the conversation is enough."""
    runner = _FakeRunner()
    mgr = _resume_manager(runner)
    [ev async for ev in mgr.resume_with_job_result(_record(call_id=None), result="x")]
    assert _notice_payload(runner)["job_id"] == "j1"


async def test_on_a_native_app_a_missing_call_id_yields_nothing():
    runner = _FakeRunner()
    mgr = _resume_manager(runner, native=True)
    events = [ev async for ev in mgr.resume_with_job_result(_record(call_id=None), result="x")]
    assert events == []
    assert runner.calls == []


async def test_resume_missing_session_yields_nothing():
    runner = _FakeRunner()
    mgr = _resume_manager(runner, session_exists=False)
    events = [ev async for ev in mgr.resume_with_job_result(_record(), result="x")]
    assert events == []
    assert runner.calls == []


# --- can_resume ----------------------------------------------------------------


class _SessionSvc:
    def __init__(self, present: bool) -> None:
        self._present = present

    async def get_session(self, *, app_name, user_id, session_id):
        return object() if self._present else None


def _can_resume_manager(present: bool, *, native: bool = False) -> GoogleADKWorkflowManager:
    mgr = GoogleADKWorkflowManager.__new__(GoogleADKWorkflowManager)
    mgr._app_name = "test"
    mgr._session_service = _SessionSvc(present)
    mgr._adk_config_path = "root_agent.yaml" if native else None
    return mgr


async def test_can_resume_true_when_session_present():
    assert await _can_resume_manager(True).can_resume(_record()) is True


async def test_can_resume_false_when_session_missing():
    # Sessions are durable by default and survive a restart; this is the case
    # where the conversation is genuinely gone — deleted, or an explicitly
    # ephemeral session_store="memory" run whose process restarted.
    assert await _can_resume_manager(False).can_resume(_record()) is False


async def test_can_resume_false_when_ids_missing():
    mgr = _can_resume_manager(True)
    assert await mgr.can_resume(_record(session_id=None)) is False
    assert await mgr.can_resume(_record(user_id=None)) is False


async def test_a_job_without_its_call_id_can_resume_but_not_on_a_native_app():
    assert await _can_resume_manager(True).can_resume(_record(call_id=None)) is True
    native = _can_resume_manager(True, native=True)
    assert await native.can_resume(_record(call_id=None)) is False
