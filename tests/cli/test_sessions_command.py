"""/sessions command backed by the native session store (durable sessions).

The table shows each stored conversation's row (``SessionInfo``): its id (the
current one marked ``* ``), its title (the first prompt, never read as Rich
markup; "(no prompt yet)" without one), its turns, and when it last changed.
``--delete=<id>`` deletes one, but never the conversation the user is in.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any

from rich.console import Console

from agentic_cli.cli.builtin_commands import SessionsCommand
from agentic_cli.workflow.sessions import SessionInfo

from tests.event_replay import RecordingSession

WHEN = datetime(2026, 10, 24, 9, 30, tzinfo=timezone.utc)


class _Screen(RecordingSession):
    """Records a table as the terminal would show it."""

    def add_rich(self, renderable: Any) -> None:
        console = Console(file=io.StringIO(), record=True, width=160)
        console.print(renderable)
        self.calls.append(("rich", console.export_text()))


class _Workflow:
    supports_sessions = True
    backend_type = "adk"

    def __init__(self, sessions: list[SessionInfo]) -> None:
        self._sessions = sessions
        self.deleted: list[str] = []

    async def list_sessions(
        self, *, user_id: str | None = None, limit: int | None = None
    ) -> list[SessionInfo]:
        return self._sessions

    async def delete_session(
        self, session_id: str, *, user_id: str | None = None
    ) -> bool:
        if any(s.session_id == session_id for s in self._sessions):
            self.deleted.append(session_id)
            return True
        return False


class _SessionlessWorkflow:
    """A backend with no durable session store."""

    supports_sessions = False
    backend_type = "custom"


class _App:
    def __init__(self, workflow, session_id: str = "cur") -> None:
        self._wf = workflow
        self.session = _Screen()
        self.session_id = session_id

    @property
    def workflow(self):
        if self._wf is None:
            raise RuntimeError("workflow not initialized")
        return self._wf


def _table(app: _App) -> str:
    (rendered,) = app.session.rich()
    return rendered


async def test_lists_each_conversations_title_turns_and_last_update():
    app = _App(_Workflow([
        SessionInfo("a", "Plan the trip to Lisbon", 3, WHEN),
        SessionInfo("cur", "And now?", 1, WHEN),
    ]))

    await SessionsCommand().execute("", app)

    table = _table(app)
    for heading in ("Session ID", "Title", "Turns", "Last Update"):
        assert heading in table
    assert "Messages" not in table
    row_a = next(line for line in table.splitlines() if " a " in f" {line} ")
    assert "Plan the trip to Lisbon" in row_a and " 3 " in f"{row_a} "
    assert WHEN.astimezone().strftime("%Y-%m-%d %H:%M") in row_a
    assert "* cur" in table


async def test_a_title_is_shown_as_typed_never_as_markup():
    app = _App(_Workflow([SessionInfo("a", "Make [bold]this[/bold] bold", 1, WHEN)]))

    await SessionsCommand().execute("", app)

    assert "Make [bold]this[/bold] bold" in _table(app)


async def test_an_id_is_shown_as_given_never_as_markup():
    app = _App(_Workflow([SessionInfo("[/x]", "Hi", 1, WHEN), SessionInfo("[red]y", "Hi", 1, WHEN)]),
               session_id="[red]y")

    await SessionsCommand().execute("", app)

    table = _table(app)
    assert "[/x]" in table and "* [red]y" in table


async def test_without_a_prompt_the_title_says_so():
    app = _App(_Workflow([SessionInfo("a", "", 0, None)]))

    await SessionsCommand().execute("", app)

    assert "(no prompt yet)" in _table(app)


async def test_a_backend_that_cannot_tell_leaves_title_and_turns_blank():
    """LangGraph knows the id alone: it cannot say there was no prompt."""
    app = _App(_Workflow([SessionInfo("a", "", None, None)]))

    await SessionsCommand().execute("", app)

    table = _table(app)
    assert "(no prompt yet)" not in table and "None" not in table


async def test_emoji_codes_are_shown_as_typed():
    """Rich turns ":x:" into an emoji in a plain cell; an id must stay copyable."""
    app = _App(_Workflow([SessionInfo("ab:x:cd", "Parse a:b:c and :smile: then", 1, WHEN)]))

    await SessionsCommand().execute("", app)

    table = _table(app)
    assert "Parse a:b:c and :smile: then" in table and "ab:x:cd" in table


async def test_empty_message():
    app = _App(_Workflow([]))
    await SessionsCommand().execute("", app)
    assert any("No saved sessions" in c[2] for c in app.session.of("message"))


async def test_delete_existing():
    wf = _Workflow([SessionInfo("a", "Hi", 1, WHEN)])
    app = _App(wf)
    await SessionsCommand().execute("--delete=a", app)
    assert wf.deleted == ["a"]
    assert app.session.of("success")


async def test_delete_missing_errors():
    app = _App(_Workflow([]))
    await SessionsCommand().execute("--delete=zzz", app)
    assert app.session.errors()


async def test_the_conversation_the_user_is_in_is_not_deleted():
    wf = _Workflow([SessionInfo("cur", "Hi", 1, WHEN)])
    app = _App(wf, session_id="cur")

    await SessionsCommand().execute("--delete=cur", app)

    assert wf.deleted == []
    assert app.session.errors() == [
        "You are in this conversation; /new or /resume another first."
    ]


async def test_not_ready_warns():
    app = _App(None)
    await SessionsCommand().execute("", app)
    assert app.session.warnings()


async def test_backend_without_sessions_says_so():
    """An empty list would read as 'no saved sessions' — be explicit instead."""
    app = _App(_SessionlessWorkflow())
    await SessionsCommand().execute("", app)
    warnings = app.session.warnings()
    assert warnings and any("does not" in str(w) for w in warnings)
    assert not app.session.of("rich")
