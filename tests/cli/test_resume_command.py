"""``/resume [<id>]``: reopen a stored conversation, from a picker or by its id.

``/resume`` opens the picker (``app.pick_session()``) and resumes what it
returns (``app.resume_session(id)``), nothing on Esc; ``/resume <id>``
resumes that one. Before the workflow is ready, or on a backend without
sessions, it answers as ``/sessions`` does. (``/jobs resume`` now runs the
resumes of finished background jobs: ``tests/cli/test_jobs_command.py``.)
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from agentic_cli.cli.builtin_commands import ResumeSessionCommand
from agentic_cli.cli.commands import CommandRegistry
from tests.event_replay import RecordingSession


class _Workflow:
    supports_sessions = True
    backend_type = "adk"

    async def delete_session(self, session_id, *, user_id=None) -> bool:
        return True


class _SessionlessWorkflow:
    supports_sessions = False
    backend_type = "custom"


class _App:
    def __init__(self, workflow, picked: str | None = "abc", session_id: str = "cur") -> None:
        self._wf = workflow
        self.session = RecordingSession()
        self.session_id = session_id
        self.pick_session = AsyncMock(return_value=picked)
        self.resume_session = AsyncMock(return_value=True)

    @property
    def workflow(self):
        if self._wf is None:
            raise RuntimeError("workflow not initialized")
        return self._wf


async def test_without_an_id_it_resumes_the_conversation_picked():
    app = _App(_Workflow(), picked="abc")

    await ResumeSessionCommand().execute("", app)

    app.pick_session.assert_awaited_once_with()
    app.resume_session.assert_awaited_once_with("abc")


async def test_escape_in_the_picker_resumes_nothing():
    app = _App(_Workflow(), picked=None)

    await ResumeSessionCommand().execute("", app)

    app.resume_session.assert_not_awaited()


async def test_with_an_id_it_resumes_that_one_without_a_picker():
    app = _App(_Workflow())

    await ResumeSessionCommand().execute(" 3f2a9c1d0b7e ", app)

    app.pick_session.assert_not_awaited()
    app.resume_session.assert_awaited_once_with("3f2a9c1d0b7e")


async def test_before_the_workflow_is_ready_it_says_so():
    app = _App(None)

    await ResumeSessionCommand().execute("", app)

    assert app.session.warnings() == ["Sessions not available yet — workflow is initializing."]
    app.pick_session.assert_not_awaited()


async def test_a_backend_without_sessions_says_so():
    app = _App(_SessionlessWorkflow())

    await ResumeSessionCommand().execute("abc", app)

    (warning,) = app.session.warnings()
    assert "does not persist sessions" in warning
    app.resume_session.assert_not_awaited()


def test_resume_is_the_session_command():
    registry = CommandRegistry()
    registry.register(ResumeSessionCommand())

    command = registry.get("resume")
    assert isinstance(command, ResumeSessionCommand)
    assert command.usage == "/resume [<id>]"
