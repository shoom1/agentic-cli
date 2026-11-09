"""The conversation the turns write to is the one held and loaded.

A conversation lives in one CLI process at a time (``cli/conversation_lock.py``),
so the lock must follow the conversation the app is in, however it changes:
``/new``, ``/resume``, the conversation a run starts in (``--continue``,
``--resume``, a new one), an orchestrator switch in ``/settings``, and the
``session_id`` setter. Each goes through one protocol
(``ConversationCore.switch``): the target's lock is taken (one another
process holds is refused), the memories of the conversation left are
proposed, the workflow loads the target, then the switch is made (the lock
kept, the one before let go, the task list and the usage counters reset)
or abandoned, and the status line is drawn. A Ctrl+C at any step leaves the
app in exactly one conversation, holding exactly its lock.

The place a conversation is held at (its backend and store) is recorded
when the lock is taken: a ``/settings`` change of ``session_store`` takes
effect at the next start, and the lock keeps the store the workflow runs.

The app is a real ``BaseCLIApp`` whose session is a fake and whose
controller holds a fake workflow (``tests/cli/test_resume_session.py``).
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("google.adk")

from agentic_cli.cli.conversation_lock import ConversationLock  # noqa: E402
from agentic_cli.config import BaseSettings  # noqa: E402
from tests.cli.test_compact_and_new import (  # noqa: E402,F401
    app_on,
    project,
    quiet_app,
    said,
)
from tests.cli.test_resume_session import Workflow, conversation  # noqa: E402


@pytest.fixture
def home(project, monkeypatch):
    """HOME is the test's: the app's conversation locks live in its workspace."""
    monkeypatch.setenv("HOME", str(project))
    return project


def open_elsewhere(app, key: tuple) -> ConversationLock:
    """Another process that has the conversation ``key`` names open."""
    other = ConversationLock(app.settings.sessions_dir / "locks")
    assert other.take(key) is None
    return other


def free(app, key: tuple) -> bool:
    """Whether another process could open the conversation ``key`` names now."""
    other = ConversationLock(app.settings.sessions_dir / "locks")
    taken = other.take(key) is None
    other.release()
    return taken


class TestASessionStoreChange:
    """``session_store`` takes effect at the next start: until then the
    workflow keeps its store, and so does the lock."""

    async def test_the_lock_keeps_the_store_the_workflow_runs(self, home):
        workflow = Workflow(conversation("abc", "Plan the trip", 2))
        app = app_on(workflow, BaseSettings(session_store="sqlite"))
        assert app._hold(app.session_id) is None
        sqlite_key = app._conversation_key("abc")

        assert await app.apply_settings({"session_store": "memory"})

        assert app._conversation_key("abc") == sqlite_key
        assert not free(app, app._conversation_key(app.session_id))

    async def test_one_open_elsewhere_is_still_refused(self, home):
        workflow = Workflow(conversation("abc", "Plan the trip", 2))
        app = app_on(workflow, BaseSettings(session_store="sqlite"))
        other = open_elsewhere(app, app._conversation_key("abc"))
        before = app.session_id

        assert await app.apply_settings({"session_store": "memory"})
        resumed = await app.resume_session("abc")
        other.release()

        assert resumed is False and app.session_id == before
        assert said(app.session, "add_error") == [
            f"Could not open 'abc': it is open in another process ({os.getpid()})."
        ]
