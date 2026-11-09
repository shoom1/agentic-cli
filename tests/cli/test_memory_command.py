"""/memory: review what memory holds (spec §12)."""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest
from thinking_prompt import ChoiceField

from agentic_cli.cli.memory_command import NOT_READY, MemoryCommand, proposals_line
from agentic_cli.memory import Candidate, MemoryStore
from tests.cli.dialog_fakes import autospec_session
from tests.memory.memory_files import T0, Clock


class _App:
    def __init__(self, store, ready: bool = True, required: frozenset[str] | None = None) -> None:
        self.session = autospec_session()
        self._store, self._ready, self._required = store, ready, required

    @property
    def workflow(self):
        if not self._ready:
            raise RuntimeError("Workflow not initialized yet")
        workflow = SimpleNamespace(memory_store=self._store)
        if self._required is not None:
            workflow.required_managers = self._required
        return workflow


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(tmp_path, clock) -> MemoryStore:
    return MemoryStore(tmp_path / "user", tmp_path / "proj" / "memory", project_name="app", now=clock)


def shown(app) -> list:
    """The dialogs shown, in order."""
    return [call.args[0] for call in app.session.show_dialog.await_args_list]


def picking(*answers):
    """``show_dialog``'s answers in turn; a function picks from the dialog's options."""
    queue = list(answers)

    def answer(dialog):
        value = queue.pop(0)
        if callable(value):
            (field,) = dialog.fields
            return value(field.options)
        return value

    return answer


async def test_before_the_workflow_is_ready_it_says_so(store):
    app = _App(store, ready=False)
    await MemoryCommand().execute("", app)
    app.session.add_warning.assert_called_once_with(NOT_READY)


async def test_an_app_without_memory_says_so():
    app = _App(None)
    await MemoryCommand().execute("", app)
    app.session.add_message.assert_called_once_with("system", "This app has no memory.")


async def test_a_memory_folder_that_cannot_open_says_the_log_has_why():
    app = _App(None, required=frozenset({"memory_store"}))
    await MemoryCommand().execute("", app)
    [(role, text)] = [call.args for call in app.session.add_message.call_args_list]
    assert role == "system"
    assert "unavailable" in text and "memory_store_unavailable" in text
    assert "no memory" not in text


async def test_an_app_whose_tools_need_no_memory_says_it_has_none():
    app = _App(None, required=frozenset({"knowledge_base"}))
    await MemoryCommand().execute("", app)
    app.session.add_message.assert_called_once_with("system", "This app has no memory.")


async def test_no_memories_yet(store):
    app = _App(store)
    await MemoryCommand().execute("", app)
    app.session.add_message.assert_called_once_with("system", "No memories saved yet.")


async def test_the_list_newest_first_with_counts(store, clock):
    store.save("Prefers tabs", "Tabs.", "feedback", scope="user")
    clock.now = T0 + timedelta(minutes=1)
    newer = store.save("KB on SQLite", "kb.sqlite", "project")
    store.propose(Candidate("new", None, "Uses metric", "km.", "user", "user"))
    app = _App(store)
    app.session.show_dialog.return_value = None
    await MemoryCommand().execute("", app)
    [dialog] = shown(app)
    assert dialog.title == "Memory: 1 project, 1 user · 1 memory proposal: /memory inbox"
    (field,) = dialog.fields
    assert isinstance(field, ChoiceField)
    assert field.options[0] == f"2026-11-08 · project · project · agent · KB on SQLite · {newer.id}"
    assert len(field.options) == 2


async def test_choosing_a_memory_shows_it_and_delete_deletes(store):
    memory = store.save("Prefers tabs", "Tabs, width 4.", "feedback", scope="user")
    app = _App(store)
    app.session.show_dialog.side_effect = picking(lambda options: options[0], "delete")
    await MemoryCommand().execute("", app)
    view = shown(app)[1]
    assert view.title == f"Memory {memory.id}"
    assert "Tabs, width 4." in str(view.body) and str(memory.path) in str(view.body)
    assert [b.text for b in view.buttons] == ["Delete", "Close"]
    assert store.read(memory.id) is None
    app.session.add_success.assert_called_once_with(
        f"Deleted memory {memory.id} (restorable for 30 days: /memory history).")


async def test_memory_by_id_and_its_history_button(store, clock):
    memory = store.save("Prefers tabs", "Tabs.", "feedback", scope="user")
    clock.now = T0 + timedelta(minutes=1)
    store.update(memory.id, content="Tabs, width 4.")
    app = _App(store)
    app.session.show_dialog.return_value = None
    await MemoryCommand().execute(memory.id, app)
    [view] = shown(app)
    assert [b.text for b in view.buttons] == ["Delete", "History", "Close"]


async def test_an_unknown_id(store):
    app = _App(store)
    await MemoryCommand().execute("deadbeef", app)
    app.session.add_message.assert_called_once_with("system", "No memory with id deadbeef.")


async def test_problems_are_a_row_of_their_own(store, tmp_path):
    (tmp_path / "user").mkdir(parents=True, exist_ok=True)
    (tmp_path / "user" / "broken-a1.md").write_text("nonsense\n")
    app = _App(store)
    app.session.show_dialog.side_effect = picking(lambda options: options[-1], None)
    await MemoryCommand().execute("", app)
    listing, problems = shown(app)
    assert listing.fields[0].options == ["Problems (1)"]
    assert "broken-a1.md" in str(problems.body) and "bad_frontmatter" in str(problems.body)


async def test_stale_lists_memories_unchanged_for_90_days(store, clock):
    old = store.save("Old fact", "x", "user", scope="user")
    clock.now = T0 + timedelta(days=100)
    store.save("New fact", "y", "user", scope="user")
    app = _App(store)
    app.session.show_dialog.return_value = None
    await MemoryCommand(now=clock).execute("stale", app)
    [dialog] = shown(app)
    assert [option.split(" · ")[-1] for option in dialog.fields[0].options] == [old.id]


async def test_stale_with_none(store, clock):
    store.save("Fresh", "x", "user", scope="user")
    app = _App(store)
    await MemoryCommand(now=clock).execute("stale", app)
    app.session.add_message.assert_called_once_with("system", "No memory has gone 90 days without an update.")


def test_the_proposals_line():
    assert proposals_line(0) is None
    assert proposals_line(1) == "1 memory proposal: /memory inbox"
    assert proposals_line(3) == "3 memory proposals: /memory inbox"


def test_the_builtin_commands_include_memory():
    from agentic_cli.cli.app import BaseCLIApp
    from agentic_cli.cli.commands import CommandRegistry

    app = SimpleNamespace(command_registry=CommandRegistry())
    BaseCLIApp._register_builtin_commands(app)
    assert isinstance(app.command_registry.get("memory"), MemoryCommand)
