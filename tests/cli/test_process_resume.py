"""MessageProcessor.process_resume: resume vs. graceful notice (milestone 4 / A).

A finished job resumes only when the backend supports it AND the originating
conversation is still available (workflow.can_resume). Otherwise a one-line
notice is posted and no turn runs — the "finished while its conversation was
unavailable" case (e.g. after a CLI restart, the in-memory ADK session is gone).
"""

from __future__ import annotations

from types import SimpleNamespace

from agentic_cli.cli.message_processor import MessageProcessor
from agentic_cli.workflow.events import EventType, WorkflowEvent

from tests.event_replay import RecordingSession


class _ResumableWorkflow:
    backend_type = "fake"

    def __init__(self, can: bool = True) -> None:
        self._can = can
        self.resumed: list[str] = []

    async def can_resume(self, record) -> bool:
        return self._can

    def set_input_callback(self, cb) -> None:  # used by _run_turn
        pass

    def clear_input_callback(self) -> None:
        pass

    async def resume_with_job_result(self, record, result=None):
        self.resumed.append(record.job_id)
        yield WorkflowEvent(type=EventType.TEXT, content="resumed reply")


class _NoResumeWorkflow:
    """A backend without resume support (e.g. LangGraph today)."""

    backend_type = "langgraph"

    async def can_resume(self, record) -> bool:
        return True  # irrelevant: no resume_with_job_result attribute


class _Ctrl:
    def __init__(self, workflow) -> None:
        self.workflow = workflow

    async def ensure_initialized(self, ui=None) -> bool:
        return True

    def update_status_bar(self, ui) -> None:
        pass


def _record(job_id="j1", name="build", state="succeeded"):
    return SimpleNamespace(job_id=job_id, name=name, state=SimpleNamespace(value=state))


_SETTINGS = SimpleNamespace(default_user="u", verbose_thinking=False)


async def test_resumes_when_resumable():
    wf = _ResumableWorkflow(can=True)
    ui = RecordingSession()
    await MessageProcessor().process_resume(
        record=_record(), workflow_controller=_Ctrl(wf), ui=ui, settings=_SETTINGS
    )
    assert wf.resumed == ["j1"]
    assert "resumed reply" in ui.responses()
    assert any("resuming" in c[2] for c in ui.of("message"))


async def test_notifies_when_conversation_unavailable():
    wf = _ResumableWorkflow(can=False)
    ui = RecordingSession()
    await MessageProcessor().process_resume(
        record=_record(), workflow_controller=_Ctrl(wf), ui=ui, settings=_SETTINGS
    )
    assert wf.resumed == []  # never resumed
    assert not ui.responses()  # no model turn
    assert any("/jobs j1" in c[2] for c in ui.of("message"))


async def test_notifies_when_backend_has_no_resume_support():
    ui = RecordingSession()
    await MessageProcessor().process_resume(
        record=_record(), workflow_controller=_Ctrl(_NoResumeWorkflow()), ui=ui,
        settings=_SETTINGS,
    )
    assert not ui.responses()
    assert any("/jobs j1" in c[2] for c in ui.of("message"))
