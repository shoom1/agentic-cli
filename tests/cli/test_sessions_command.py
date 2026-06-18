"""/sessions command backed by the native session store (durable sessions)."""

from __future__ import annotations

import pytest

from agentic_cli.cli.builtin_commands import SessionsCommand

from tests.event_replay import RecordingSession


class _Workflow:
    def __init__(self, sessions: list[dict]) -> None:
        self._sessions = sessions
        self.deleted: list[str] = []

    async def list_sessions(self) -> list[dict]:
        return self._sessions

    async def delete_session(self, session_id: str) -> bool:
        if any(s["session_id"] == session_id for s in self._sessions):
            self.deleted.append(session_id)
            return True
        return False


class _App:
    def __init__(self, workflow, session_id: str = "cur") -> None:
        self._wf = workflow
        self.session = RecordingSession()
        self.session_id = session_id

    @property
    def workflow(self):
        if self._wf is None:
            raise RuntimeError("workflow not initialized")
        return self._wf


async def test_lists_sessions():
    app = _App(_Workflow([
        {"session_id": "a", "last_update": None, "message_count": 3},
        {"session_id": "cur", "last_update": 1_700_000_000.0, "message_count": 1},
    ]))
    await SessionsCommand().execute("", app)
    assert app.session.of("rich")  # rendered a table


async def test_empty_message():
    app = _App(_Workflow([]))
    await SessionsCommand().execute("", app)
    assert any("No saved sessions" in c[2] for c in app.session.of("message"))


async def test_delete_existing():
    wf = _Workflow([{"session_id": "a", "last_update": None, "message_count": 1}])
    app = _App(wf)
    await SessionsCommand().execute("--delete=a", app)
    assert wf.deleted == ["a"]
    assert app.session.of("success")


async def test_delete_missing_errors():
    app = _App(_Workflow([]))
    await SessionsCommand().execute("--delete=zzz", app)
    assert app.session.errors()


async def test_not_ready_warns():
    app = _App(None)
    await SessionsCommand().execute("", app)
    assert app.session.warnings()
