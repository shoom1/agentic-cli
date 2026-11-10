"""The conversations of every installed backend: listed, picked and deleted.

A conversation belongs to one backend for its whole life, and each backend
keeps its own store. The app reads them all: the running workflow its own
backend's, each other installed backend's through its ``SessionReader``
(``workflow/session_readers.py``), opened for one look and closed after it
(``cli/stored_conversations.py``). ``BaseCLIApp.list_conversations()`` merges
them newest first, each row naming its backend (``SessionInfo.backend``), so
an id two backends keep (left by the live switch of backend an earlier
version made) is two rows; ``session_backend`` names the backend of the
conversation the app is in.

- The ``/resume`` picker lists every backend's conversations but the one the
  app is in, naming each row's backend once a row is on another backend than
  the running one, and returns the id picked with its backend.
- ``/sessions`` shows a Backend column and marks the conversation the app is
  in (its id on its backend); ``--delete=<id>`` deletes it on the backend
  that keeps it, held there against another process; an id two backends keep
  is deleted on the one ``--backend=<name>`` names, and refused, naming both,
  without it.
- A backend whose store cannot be read leaves its rows out; one whose extra
  is not installed too, without a word.

The app is a real ``BaseCLIApp`` on a fake running workflow (ADK, the
settings' orchestrator), on the sqlite store; the other backends' readers
are fakes holding the rows given.
"""

from __future__ import annotations

import dataclasses
import io
import os
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("google.adk")

from rich.console import Console  # noqa: E402

from agentic_cli.cli.conversation_lock import ConversationLock  # noqa: E402
from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.workflow.sessions import SessionInfo  # noqa: E402
from tests.cli.test_compact_and_new import app_on, project, quiet_app, said  # noqa: E402,F401
from tests.cli.test_resume_session import Workflow, conversation  # noqa: E402


def row(session_id: str, title: str, *, hours: float, turns: int = 1) -> SessionInfo:
    return SessionInfo(
        session_id, title, turns, datetime.now(timezone.utc) - timedelta(hours=hours)
    )


class FakeReader:
    """A backend's reader, holding ``rows``; it records what it was asked."""

    def __init__(self, backend: str, rows: list[SessionInfo]) -> None:
        self.backend = backend
        self.rows = {r.session_id: dataclasses.replace(r, backend=backend) for r in rows}
        self.deleted: list[str] = []
        self.closed = False

    async def list(self, limit=None):
        assert not self.closed
        listed = sorted(self.rows.values(), key=lambda r: r.updated, reverse=True)
        return listed if limit is None else listed[:limit]

    async def info(self, session_id):
        assert not self.closed
        return self.rows.get(session_id)

    async def exists(self, session_id, *, user_id=None):
        return session_id in self.rows

    async def delete(self, session_id):
        self.deleted.append(session_id)
        return self.rows.pop(session_id, None) is not None

    async def aclose(self):
        self.closed = True


class Readers:
    """The readers the app opens: each backend's rows, and every reader opened."""

    def __init__(self) -> None:
        self.stored: dict[str, list[SessionInfo]] = {}
        self.opened: list[FakeReader] = []
        self.failing: dict[str, Exception] = {}

    async def open(self, backend, settings, *, store=None):
        if store == "memory":
            return None
        if backend in self.failing:
            raise self.failing[backend]
        reader = FakeReader(backend, self.stored.get(backend, []))
        self.opened.append(reader)
        return reader

    def of(self, backend: str) -> list[FakeReader]:
        return [reader for reader in self.opened if reader.backend == backend]


@pytest.fixture
def readers(project, monkeypatch) -> Readers:
    """HOME is the test's (the conversation locks live in its workspace)."""
    monkeypatch.setenv("HOME", str(project))
    fakes = Readers()
    monkeypatch.setattr("agentic_cli.cli.stored_conversations.open_session_reader", fakes.open)
    return fakes


def on_adk(*stored) -> tuple:
    """An app running the ADK backend (a fake workflow holding ``stored``)."""
    workflow = Workflow(*stored)
    app = app_on(workflow, BaseSettings(session_store="sqlite"))
    return app, workflow


def table_of(app) -> str:
    (renderable,) = [c.args[0] for c in app.session.add_rich.call_args_list]
    console = Console(file=io.StringIO(), record=True, width=200)
    console.print(renderable)
    return console.export_text()


def line_of(table: str, *words: str) -> str:
    return next(line for line in table.splitlines() if all(w in line for w in words))


def open_elsewhere(app, key: tuple) -> ConversationLock:
    """Another process that has the conversation ``key`` names open."""
    other = ConversationLock(app.settings.sessions_dir / "locks")
    assert other.take(key) is None
    return other


class TestListConversations:
    async def test_every_backend_newest_first_each_naming_its_backend(self, readers):
        readers.stored["deepagents"] = [row("d1", "On Deep Agents", hours=1),
                                        row("d2", "Older there", hours=3)]
        app, _ = on_adk(conversation("a1", "On ADK", 2, hours=2))

        listed = await app.list_conversations()

        assert [(r.session_id, r.backend) for r in listed] == [
            ("d1", "deepagents"), ("a1", "adk"), ("d2", "deepagents"),
        ]
        assert app.session_backend == "adk"
        assert [r.backend for r in readers.opened] == ["deepagents"]
        assert all(r.closed for r in readers.opened)

    async def test_the_limit_keeps_the_newest_of_all(self, readers):
        readers.stored["deepagents"] = [row("d1", "There", hours=1)]
        app, _ = on_adk(conversation("a1", "Here", 2, hours=2))

        assert [r.session_id for r in await app.list_conversations(limit=1)] == ["d1"]

    async def test_a_backend_that_cannot_be_read_is_left_out(self, readers):
        readers.failing["deepagents"] = RuntimeError("database disk image is malformed")
        app, _ = on_adk(conversation("a1", "Here", 2))

        assert [r.session_id for r in await app.list_conversations()] == ["a1"]

    async def test_one_whose_extra_is_missing_is_left_out(self, readers):
        readers.failing["deepagents"] = ImportError("needs the 'deepagents' extra")
        app, _ = on_adk(conversation("a1", "Here", 2))

        assert [r.session_id for r in await app.list_conversations()] == ["a1"]
        assert said(app.session, "add_warning") == [] and said(app.session, "add_error") == []

    async def test_the_memory_store_lists_the_running_workflows_alone(self, readers):
        workflow = Workflow(conversation("a1", "Here", 2))
        app = app_on(workflow, BaseSettings(session_store="memory"))

        assert [r.session_id for r in await app.list_conversations()] == ["a1"]
        assert readers.opened == []


class TestTheLook:
    async def test_a_look_stopped_while_it_opens_closes_the_readers_it_opened(self, readers):
        """No workflow runs (before the first start): every backend's reader
        opens, and a Ctrl+C at the second closes the first."""
        import asyncio

        from agentic_cli.cli.stored_conversations import StoredConversations

        readers.failing["deepagents"] = asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            async with StoredConversations(BaseSettings(session_store="sqlite"), "sqlite"):
                pass

        assert [r.backend for r in readers.opened] == ["adk"]
        assert readers.opened[0].closed


class TestThePicker:
    async def test_it_lists_every_backend_and_returns_the_pick_with_its_backend(self, readers):
        readers.stored["deepagents"] = [row("d1", "On Deep Agents", hours=1)]
        app, workflow = on_adk(conversation("a1", "On ADK", 2, hours=2))
        workflow.stored[app.session_id] = conversation(app.session_id, "This one", 2, hours=0)

        def pick_the_first(dialog):
            (field,) = dialog.fields
            pick_the_first.options = field.options
            return field.options[0]

        app.session.show_dialog.side_effect = pick_the_first
        picked = await app.pick_session()

        assert pick_the_first.options == [
            "1h ago   · 1 turn    · deepagents · On Deep Agents · d1",
            "2h ago   · 1 turn    · adk        · On ADK · a1",
        ]
        assert picked == "d1" and picked.backend == "deepagents"
        assert all(r.closed for r in readers.opened)

    async def test_an_id_two_backends_keep_is_two_rows_but_the_one_the_app_is_in(
        self, readers
    ):
        app, workflow = on_adk(conversation("x", "On ADK", 2, hours=2))
        readers.stored["deepagents"] = [row("x", "On Deep Agents", hours=1)]
        app.session.show_dialog.return_value = None

        await app.pick_session()
        (field,) = app.session.show_dialog.call_args.args[0].fields
        assert [o.split(" · ")[2:] for o in field.options] == [
            ["deepagents", "On Deep Agents", "x"], ["adk       ", "On ADK", "x"],
        ]

        app._conversation.session_id = "x"  # in ADK's x now
        app.session.show_dialog.reset_mock()
        await app.pick_session()
        (field,) = app.session.show_dialog.call_args.args[0].fields
        assert [o.split(" · ")[2:] for o in field.options] == [
            ["deepagents", "On Deep Agents", "x"],
        ]

    async def test_with_the_running_backends_alone_no_backend_is_named(self, readers):
        app, _ = on_adk(conversation("a1", "On ADK", 2, hours=2))
        app.session.show_dialog.return_value = None

        await app.pick_session()

        (field,) = app.session.show_dialog.call_args.args[0].fields
        assert field.options == ["2h ago   · 1 turn    · On ADK · a1"]


class TestTheSessionsCommand:
    async def test_it_lists_every_backend_with_a_backend_column(self, readers):
        readers.stored["deepagents"] = [row("d1", "On Deep Agents", hours=1)]
        app, workflow = on_adk(conversation("a1", "On ADK", 2, hours=2))
        workflow.stored[app.session_id] = conversation(app.session_id, "This one", 2, hours=0)

        await app.process_input("/sessions")

        table = table_of(app)
        assert "Backend" in table
        assert "deepagents" in line_of(table, "d1", "On Deep Agents")
        assert "adk" in line_of(table, "a1", "On ADK")
        assert f"* {app.session_id}" in line_of(table, "This one")

    async def test_the_mark_is_on_the_conversation_the_app_is_in_alone(self, readers):
        app, workflow = on_adk(conversation("x", "On ADK", 2, hours=2))
        readers.stored["deepagents"] = [row("x", "On Deep Agents", hours=1)]
        app._conversation.session_id = "x"

        await app.process_input("/sessions")

        table = table_of(app)
        assert "* x" in line_of(table, "On ADK")
        assert "* x" not in line_of(table, "On Deep Agents")

    async def test_delete_deletes_on_the_backend_that_keeps_it(self, readers):
        readers.stored["deepagents"] = [row("d1", "On Deep Agents", hours=1)]
        app, workflow = on_adk(conversation("a1", "On ADK", 2))

        await app.process_input("/sessions --delete=d1")

        assert [r.deleted for r in readers.of("deepagents") if r.deleted] == [["d1"]]
        assert not any(c.startswith("delete_session") for c in workflow.calls)
        assert said(app.session, "add_success") == ["Session 'd1' deleted."]
        assert all(r.closed for r in readers.opened)

    async def test_one_open_in_another_process_there_is_not_deleted(self, readers):
        readers.stored["deepagents"] = [row("d1", "On Deep Agents", hours=1)]
        app, _ = on_adk()
        key = dataclasses.replace(app._place(), backend="deepagents").key("d1")
        other = open_elsewhere(app, key)
        try:
            await app.process_input("/sessions --delete=d1")
        finally:
            other.release()

        assert [r.deleted for r in readers.opened if r.deleted] == []
        assert said(app.session, "add_error") == [
            f"Could not delete 'd1': it is open in another process ({os.getpid()})."
        ]

    async def test_an_id_two_backends_keep_is_refused_naming_both(self, readers):
        app, workflow = on_adk(conversation("x", "On ADK", 2, hours=2))
        readers.stored["deepagents"] = [row("x", "On Deep Agents", hours=1)]

        await app.process_input("/sessions --delete=x")

        assert "x" in workflow.stored and [r.deleted for r in readers.opened if r.deleted] == []
        assert said(app.session, "add_error") == [
            "'x' is kept by adk and deepagents: "
            "/sessions --delete=x --backend=<adk or deepagents> deletes one."
        ]

    async def test_the_backend_named_is_the_one_deleted(self, readers):
        app, workflow = on_adk(conversation("x", "On ADK", 2, hours=2))
        readers.stored["deepagents"] = [row("x", "On Deep Agents", hours=1)]

        await app.process_input("/sessions --delete=x --backend=deepagents")

        assert "x" in workflow.stored
        assert [r.deleted for r in readers.of("deepagents") if r.deleted] == [["x"]]
        assert said(app.session, "add_success") == ["Session 'x' deleted."]

    async def test_the_one_the_app_is_in_stays_but_its_twin_can_go(self, readers):
        app, workflow = on_adk(conversation("x", "On ADK", 2, hours=2))
        readers.stored["deepagents"] = [row("x", "On Deep Agents", hours=1)]
        app._conversation.session_id = "x"

        await app.process_input("/sessions --delete=x")
        assert said(app.session, "add_error")[0].startswith(
            "You are in this conversation; /new or /resume another first."
        )
        assert "x" in workflow.stored

        await app.process_input("/sessions --delete=x --backend=deepagents")
        assert [r.deleted for r in readers.of("deepagents") if r.deleted] == [["x"]]
        assert "x" in workflow.stored

    async def test_the_one_the_app_is_in_names_how_to_delete_its_twin(self, readers):
        app, _ = on_adk(conversation("x", "On ADK", 2, hours=2))
        readers.stored["deepagents"] = [row("x", "On Deep Agents", hours=1)]
        app._conversation.session_id = "x"

        await app.process_input("/sessions --delete=x")

        assert said(app.session, "add_error") == [
            "You are in this conversation; /new or /resume another first. deepagents "
            "keeps one with this id too: /sessions --delete=x --backend=deepagents "
            "deletes that one."
        ]

    async def test_an_unknown_backend_is_said_to_be_unknown(self, readers):
        app, _ = on_adk(conversation("a1", "On ADK", 2))

        await app.process_input("/sessions --delete=a1 --backend=nope")

        assert said(app.session, "add_error") == [
            "Unknown backend 'nope': adk, deepagents or langgraph."
        ]

    async def test_one_no_backend_keeps_is_not_found(self, readers):
        app, _ = on_adk(conversation("a1", "On ADK", 2))

        await app.process_input("/sessions --delete=zzz")

        assert said(app.session, "add_error") == ["Session 'zzz' not found."]
