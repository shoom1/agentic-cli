"""/memory: review what memory holds (spec §12)."""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest
from thinking_prompt import ChoiceField

from agentic_cli.cli.memory_command import NOT_READY, MemoryCommand, proposals_line
from agentic_cli.memory import Candidate, MemoryStore
from tests.cli.dialog_fakes import autospec_session
from tests.memory.memory_files import T0, Clock, memory_text, write_memory


class _App:
    def __init__(
        self, store, ready: bool = True, required: frozenset[str] | None = None,
        placement: str = "box",
    ) -> None:
        self.session = autospec_session()
        self.settings = SimpleNamespace(dialog_placement=placement)
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


async def test_its_dialogs_are_drawn_where_the_user_chose_since_the_start(store):
    """A ``/settings`` change of dialog_placement: the session keeps the
    placement it started with, so each dialog passes the live one."""
    store.save("Prefers tabs", "Tabs.", "feedback", scope="user")
    app = _App(store, placement="inline")
    app.session.show_dialog.side_effect = picking(lambda options: options[0], None)

    await MemoryCommand().execute("", app)

    assert [dialog.placement for dialog in shown(app)] == ["inline", "inline"]


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


async def test_history_restores_a_version(store, clock):
    memory = store.save("Prefers tabs", "Tabs.", "feedback", scope="user")
    clock.now = T0 + timedelta(minutes=1)
    store.update(memory.id, content="Tabs, width 4.")
    clock.now = T0 + timedelta(minutes=2)
    app = _App(store)
    app.session.show_dialog.side_effect = picking(lambda options: options[0], "restore")
    await MemoryCommand().execute("history", app)
    listing, version = shown(app)
    assert listing.fields[0].options == [f"2026-11-08 14:04 · updated · Prefers tabs · {memory.id}"]
    assert "Saved 2026-11-08 14:04 UTC:" in str(version.body) and "Now:" in str(version.body)
    assert [b.text for b in version.buttons] == ["Restore", "Close"]
    assert store.read(memory.id).content == "Tabs."
    app.session.add_success.assert_called_once_with(f"Restored memory {memory.id}: Prefers tabs.")


async def test_history_of_a_deleted_memory(store):
    memory = store.save("Prefers tabs", "Tabs.", "feedback", scope="user")
    store.delete(memory.id)
    app = _App(store)
    app.session.show_dialog.side_effect = picking(lambda options: options[0], None)
    await MemoryCommand().execute("history", app)
    listing, version = shown(app)
    assert " · deleted · " in listing.fields[0].options[0]
    assert str(version.body).endswith("Now:\ndeleted")


async def test_history_of_a_memory_whose_file_a_person_broke_does_not_call_it_deleted(store, clock):
    memory = store.save("Prefers tabs", "Tabs.", "feedback", scope="user")
    clock.now = T0 + timedelta(minutes=1)
    store.update(memory.id, content="Tabs, width 4.")
    path = store.read(memory.id).path
    path.write_text(path.read_text().replace("type: feedback", "type: preference"))
    app = _App(store)
    app.session.show_dialog.side_effect = picking(lambda options: options[0], None)
    await MemoryCommand().execute("history", app)
    listing, version = shown(app)
    assert " · updated · " in listing.fields[0].options[0]
    now = str(version.body).split("Now:\n")[1]
    assert str(path) in now and "deleted" not in now


async def test_a_restore_onto_a_file_in_the_way_names_the_file(store, clock):
    memory = store.save("Prefers tabs", "Tabs.", "feedback", scope="user")
    store.delete(memory.id)
    memory.path.write_text("nonsense, my notes\n")
    app = _App(store)
    app.session.show_dialog.side_effect = picking(lambda options: options[0], "restore")
    await MemoryCommand().execute("history", app)
    [error] = [call.args[0] for call in app.session.add_error.call_args_list]
    assert error.startswith(f"Could not restore memory {memory.id}:")
    assert str(memory.path) in error and "move or remove it" in error
    app.session.add_success.assert_not_called()
    assert memory.path.read_text() == "nonsense, my notes\n"


async def test_no_history(store):
    app = _App(store)
    await MemoryCommand().execute("history", app)
    app.session.add_message.assert_called_once_with("system", "No changes in the last 30 days.")


async def test_the_inbox_accepts_discards_and_skips(store, clock):
    target = store.save("Prefers spaces", "Spaces.", "feedback", scope="user")
    store.propose(Candidate("new", None, "Uses metric", "km.", "user", "user"),
                  conversation="c-1", outside_content="web_fetch")
    clock.now = T0 + timedelta(minutes=1)
    store.propose(Candidate("update", target.id, "Prefers tabs", "Tabs.", "feedback", "user"))
    clock.now = T0 + timedelta(minutes=2)
    third = store.propose(Candidate("new", None, "Likes otters", "Otters.", "user", "user"))
    app = _App(store)
    app.session.show_dialog.side_effect = picking("accept", "discard", "skip")
    await MemoryCommand().execute("inbox", app)
    one, two, three = shown(app)
    assert one.title == "Memory proposal 1 of 3"
    assert "From a conversation that read outside content (web_fetch)." in str(one.body)
    assert "Now:\nPrefers spaces" in str(two.body) and "Proposed:\nPrefers tabs" in str(two.body)
    assert [b.text for b in one.buttons] == ["Accept", "Discard", "Skip"]
    assert {m.description for m in store.memories()} == {"Prefers spaces", "Uses metric"}
    assert [p.id for p in store.proposals()] == [third.id]
    app.session.add_message.assert_called_with("system", "Accepted 1, discarded 1, 1 waiting.")


async def test_escape_stops_the_inbox(store, clock):
    store.propose(Candidate("new", None, "Uses metric", "km.", "user", "user"))
    clock.now = T0 + timedelta(minutes=1)
    store.propose(Candidate("new", None, "Likes otters", "Otters.", "user", "user"))
    app = _App(store)
    app.session.show_dialog.return_value = None
    await MemoryCommand().execute("inbox", app)
    assert len(shown(app)) == 1
    app.session.add_message.assert_called_with("system", "Accepted 0, discarded 0, 2 waiting.")


async def test_an_accept_the_store_refuses_says_why(store):
    proposal = store.propose(Candidate("new", None, "Prefers tabs", "Tabs.", "feedback", "user"))
    store.save("Prefers tabs", "By hand.", "feedback")
    app = _App(store)
    app.session.show_dialog.side_effect = picking("accept")
    await MemoryCommand().execute("inbox", app)
    [error] = [call.args[0] for call in app.session.add_error.call_args_list]
    assert error.startswith("Could not accept 'Prefers tabs':")
    assert [p.id for p in store.proposals()] == [proposal.id]


async def test_an_accept_onto_a_file_in_the_way_names_the_file(store, tmp_path):
    alpha = store.save("alpha", "a", "user", scope="user")
    mine = tmp_path / "user" / f"beta-{alpha.id}.md"
    mine.write_text("not frontmatter, my notes\n")
    proposal = store.propose(Candidate("update", alpha.id, "beta", "b", "user", "user"))
    app = _App(store)
    app.session.show_dialog.side_effect = picking("accept")
    await MemoryCommand().execute("inbox", app)
    [error] = [call.args[0] for call in app.session.add_error.call_args_list]
    assert error.startswith("Could not accept 'beta':")
    assert str(mine) in error and "move or remove it" in error
    assert mine.read_text() == "not frontmatter, my notes\n"
    assert [p.id for p in store.proposals()] == [proposal.id]


async def test_an_empty_inbox(store):
    app = _App(store)
    await MemoryCommand().execute("inbox", app)
    app.session.add_message.assert_called_once_with("system", "No memory proposals waiting.")


async def test_a_discard_the_store_cannot_do_says_discard(store, monkeypatch):
    from agentic_cli.memory import MemoryBusy

    proposal = store.propose(Candidate("new", None, "Uses metric", "km.", "user", "user"))

    def busy(proposal_id):
        raise MemoryBusy("another process holds the lock")

    monkeypatch.setattr(store, "discard", busy)
    app = _App(store)
    app.session.show_dialog.side_effect = picking("discard")
    await MemoryCommand().execute("inbox", app)
    [error] = [call.args[0] for call in app.session.add_error.call_args_list]
    assert error == "Could not discard 'Uses metric': another process holds the lock"
    assert [p.id for p in store.proposals()] == [proposal.id]


def focused_button(dialog) -> str:
    """The button a real dialog box puts the cursor on: Enter presses it.

    Built and focused as ``ThinkingPromptSession.show_dialog`` does for the
    box placement: a ``ButtonConfig(focused=True)`` button, else the widget's
    first focusable element, its first button."""
    from prompt_toolkit.layout import Layout
    from thinking_prompt.dialog_box import build_box

    box = build_box(dialog)
    layout = Layout(box.widget)
    layout.focus(box.initial_focus or box.widget)
    return next(button.text for button in box.buttons if button.window is layout.current_window)


# -- Enter must never press a destructive button ------------------------------------------------


@pytest.mark.parametrize("with_history", [False, True])
def test_enter_in_the_memory_view_closes_it(store, with_history):
    from agentic_cli.cli.memory_command import memory_view

    memory = store.save("Prefers tabs", "Tabs.", "feedback", scope="user")
    assert focused_button(memory_view(memory, with_history)) == "Close"


async def test_enter_in_the_version_view_does_not_restore(store, clock):
    memory = store.save("Prefers tabs", "Tabs.", "feedback", scope="user")
    clock.now = T0 + timedelta(minutes=1)
    store.update(memory.id, content="Tabs, width 4.")
    app = _App(store)
    app.session.show_dialog.side_effect = picking(lambda options: options[0], None)
    await MemoryCommand().execute("history", app)
    _, version = shown(app)
    assert focused_button(version) == "Close"


async def test_enter_in_the_inbox_skips_even_a_proposal_from_outside_content(store):
    store.propose(Candidate("new", None, "Uses metric", "km.", "user", "user"), outside_content="web_fetch")
    store.propose(Candidate("new", None, "Likes otters", "Otters.", "user", "user"))
    app = _App(store)
    app.session.show_dialog.return_value = None
    await MemoryCommand().execute("inbox", app)
    [first] = shown(app)
    assert focused_button(first) == "Skip"


# -- What was shown is what a button acts on -----------------------------------------------------


async def test_delete_of_a_memory_removed_elsewhere_says_so(store):
    memory = store.save("Prefers tabs", "Tabs.", "feedback", scope="user")
    app = _App(store)

    def removed_then_delete(dialog):
        store.delete(memory.id)  # another process, after the view opened
        return "delete"

    app.session.show_dialog.side_effect = removed_then_delete
    await MemoryCommand().execute(memory.id, app)
    app.session.add_message.assert_called_once_with("system", f"Memory {memory.id} was already deleted.")
    app.session.add_success.assert_not_called()


async def test_delete_does_not_remove_a_version_that_was_never_shown(store, tmp_path, clock):
    memory = store.save("Prefers tabs", "Tabs.", "feedback", scope="user")
    other = MemoryStore(tmp_path / "user", tmp_path / "proj" / "memory", project_name="app", now=clock)
    app = _App(store)
    answers = iter(["delete", None])

    def updated_then_delete(dialog):
        if len(shown(app)) == 1:  # another process updates it after the first view opened
            clock.now = T0 + timedelta(minutes=1)
            other.update(memory.id, content="Tabs, width 4, after a long talk.")
        return next(answers)

    app.session.show_dialog.side_effect = updated_then_delete
    await MemoryCommand().execute(memory.id, app)
    assert store.read(memory.id).content == "Tabs, width 4, after a long talk."
    app.session.add_success.assert_not_called()
    app.session.add_message.assert_called_once_with(
        "system", f"Memory {memory.id} changed since it was shown, so it was not deleted. It is shown again."
    )
    first, again = shown(app)
    assert "Tabs, width 4, after a long talk." not in str(first.body)
    assert "Tabs, width 4, after a long talk." in str(again.body)


async def test_delete_of_a_memory_whose_file_a_person_broke_since_says_so(store):
    memory = store.save("Prefers tabs", "Tabs.", "feedback", scope="user")
    app = _App(store)

    def broken_then_delete(dialog):
        memory.path.write_text(memory.path.read_text().replace("type: feedback", "type: preference"))
        return "delete"

    app.session.show_dialog.side_effect = broken_then_delete
    await MemoryCommand().execute(memory.id, app)
    [(_, text)] = [call.args for call in app.session.add_message.call_args_list]
    assert str(memory.path) in text and "Problems" in text and "already deleted" not in text
    assert memory.path.exists()


async def test_discard_of_a_proposal_handled_elsewhere_says_so(store):
    proposal = store.propose(Candidate("new", None, "Uses metric", "km.", "user", "user"))
    app = _App(store)

    def discarded_then_discard(dialog):
        store.discard(proposal.id)  # another session, after the dialog opened
        return "discard"

    app.session.show_dialog.side_effect = discarded_then_discard
    await MemoryCommand().execute("inbox", app)
    assert [call.args[1] for call in app.session.add_message.call_args_list] == [
        "That proposal was already handled.", "Accepted 0, discarded 0, 0 waiting.",
    ]


# -- An update proposal shows the memory it changes, whatever state its file is in ---------------


async def test_an_update_proposal_shows_the_type_it_changes(store):
    target = store.save("Prefers spaces", "Spaces.", "feedback", scope="user")
    store.propose(Candidate("update", target.id, "Prefers tabs", "Tabs.", "user", "user"))
    app = _App(store)
    app.session.show_dialog.return_value = None
    await MemoryCommand().execute("inbox", app)
    [dialog] = shown(app)
    body = str(dialog.body)
    assert "Now:\nPrefers spaces\nType: feedback" in body
    assert "Proposed:\nPrefers tabs\nType: user" in body


async def test_an_update_proposal_whose_memory_is_deleted_suggests_discarding(store):
    target = store.save("Prefers spaces", "Spaces.", "feedback", scope="user")
    store.propose(Candidate("update", target.id, "Prefers tabs", "Tabs.", "feedback", "user"))
    store.delete(target.id)
    app = _App(store)
    app.session.show_dialog.return_value = None
    await MemoryCommand().execute("inbox", app)
    [dialog] = shown(app)
    assert "Now:\n(deleted: discard this proposal)" in str(dialog.body)


async def test_an_update_proposal_whose_memory_file_is_broken_names_the_file_and_does_not_suggest_discarding(store):
    target = store.save("Prefers spaces", "Spaces.", "feedback", scope="user")
    store.propose(Candidate("update", target.id, "Prefers tabs", "Tabs.", "feedback", "user"))
    target.path.write_text(target.path.read_text().replace("type: feedback", "type: preference"))
    app = _App(store)
    app.session.show_dialog.side_effect = picking("accept")
    await MemoryCommand().execute("inbox", app)
    [dialog] = shown(app)
    body = str(dialog.body)
    assert f"its file cannot be used (see Problems in /memory): {target.path}" in body
    assert "deleted" not in body and "discard" not in body
    [error] = [call.args[0] for call in app.session.add_error.call_args_list]
    assert str(target.path) in error and "fix or remove it" in error
    assert len(store.proposals()) == 1


# -- Ids, and files the store cannot use ---------------------------------------------------------


async def test_memory_by_id_ignores_the_case_it_is_typed_in(tmp_path, clock):
    write_memory(tmp_path / "user", "mine-4F2A9C1E.md", memory_text("4F2A9C1E"))
    store = MemoryStore(tmp_path / "user", now=clock)
    for typed in ("4f2a9c1e", "4F2A9C1E"):
        app = _App(store)
        app.session.show_dialog.return_value = None
        await MemoryCommand().execute(typed, app)
        [view] = shown(app)
        assert view.title == "Memory 4F2A9C1E"


async def test_an_id_a_broken_file_holds_points_to_the_problems_row(store):
    memory = store.save("Prefers tabs", "Tabs.", "feedback", scope="user")
    memory.path.write_text(memory.path.read_text().replace("type: feedback", "type: preference"))
    app = _App(store)
    await MemoryCommand().execute(memory.id.upper(), app)
    [(role, text)] = [call.args for call in app.session.add_message.call_args_list]
    assert role == "system" and "No memory with id" not in text
    assert str(memory.path) in text and "bad_field" in text and "Problems" in text


async def test_a_broken_proposal_file_is_in_the_problems_row(store, tmp_path):
    inbox = tmp_path / "user" / ".inbox"
    inbox.mkdir(parents=True)
    (inbox / "nonsense.md").write_text("no frontmatter\n")
    app = _App(store)
    app.session.show_dialog.side_effect = picking(lambda options: options[-1], None)
    await MemoryCommand().execute("", app)
    listing, problems = shown(app)
    assert listing.fields[0].options == ["Problems (1)"]
    assert "nonsense.md" in str(problems.body) and "bad_proposal" in str(problems.body)
