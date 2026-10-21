"""``/settings`` is a 0.4 ``Dialog`` with fields and Save and Cancel buttons.

0.4 has no settings dialog: a dialog's ``values`` hold every field, and the
command works out what changed against each field's starting value, so
``apply_settings(changes)`` still receives only the changed keys. Save
returns True; Cancel and Escape return None. A cancel and a Save with nothing
changed both say "Settings unchanged." and apply nothing. Ctrl+C is not a
cancel: it interrupts the command, as thinking-prompt 0.4 does by default,
and agentic-cli adds nothing to that. The dialog is drawn where
``dialog_placement`` says now: the session's own placement is the one it
started with.

The dialogs here are real ``Dialog`` objects (``edited_dialog``) on an
autospecced session, so a constructor argument or a session method 0.4 lacks
fails the test.
"""

from __future__ import annotations

import asyncio
import functools
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest
from thinking_prompt import ButtonConfig, Dialog

from agentic_cli.cli.app import BaseCLIApp
from agentic_cli.cli.settings_command import SettingsCommand
from agentic_cli.config import BaseSettings
from tests.cli.dialog_fakes import autospec_session, edited_dialog

# A model picker (a dropdown of the catalog's references), a Literal, an int
# and a float in text fields, a bool and an Enum.
KEYS = [
    "model",
    "thinking_effort",
    "retry_max_attempts",
    "retry_initial_delay",
    "verbose_thinking",
    "orchestrator",
]


@pytest.fixture(autouse=True)
def hermetic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for variable in (
        "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY",
        "AGENTIC_DIALOG_PLACEMENT",
    ):
        monkeypatch.delenv(variable, raising=False)


class App(BaseCLIApp):
    def get_ui_setting_keys(self) -> list[str]:
        return KEYS


@pytest.fixture
def app() -> App:
    """The dialog's side of an app: real fields from real settings, and an
    ``apply_settings`` that records what it is given."""
    app = App.__new__(App)
    # Two providers: more models than a select shows, so the picker is a dropdown.
    app._settings = BaseSettings(google_api_key="k-g", anthropic_api_key="k-a")
    app._workflow_controller = SimpleNamespace(is_ready=False)
    app.session = autospec_session()
    app.apply_settings = AsyncMock(return_value=True)
    app.save_settings = AsyncMock()
    return app


async def _settings(app: App, monkeypatch, edits: dict, *, saved: bool = True) -> Dialog:
    """Run ``/settings``: the user makes ``edits``, then presses Save (or
    Cancel). Returns the dialog the command showed."""
    dialog_cls = edited_dialog(edits)
    monkeypatch.setattr("agentic_cli.cli.settings_command.Dialog", dialog_cls)
    app.session.show_dialog.return_value = True if saved else None

    await SettingsCommand().execute("", app)

    (dialog,) = dialog_cls.built
    app.session.show_dialog.assert_awaited_once_with(dialog)
    return dialog


def test_every_kind_of_field_is_in_the_dialog(app):
    """What the tests below rely on: each kind of value is a field."""
    fields = {field.key: type(field).__name__ for field in app._build_ui_items()}

    assert fields == {
        "model": "DropdownField",
        "thinking_effort": "SelectField",
        "retry_max_attempts": "TextField",
        "retry_initial_delay": "TextField",
        "verbose_thinking": "CheckboxField",
        "orchestrator": "SelectField",
    }


async def test_the_dialog_has_the_fields_and_save_and_cancel(app, monkeypatch):
    dialog = await _settings(app, monkeypatch, {})

    assert dialog.title == "Settings"
    assert list(dialog.fields) == app._build_ui_items()
    assert list(dialog.buttons) == [ButtonConfig("Save", result=True), ButtonConfig("Cancel")]
    assert dialog.width == 80
    assert dialog.placement == "box"


async def test_the_dialog_is_drawn_where_the_user_chose(app, monkeypatch):
    app._settings.update_setting("dialog_placement", "inline")

    dialog = await _settings(app, monkeypatch, {})

    assert dialog.placement == "inline"


async def test_a_placement_switch_applies_to_the_next_dialog(app, monkeypatch):
    """A Save that switches the placement: the very next /settings uses it."""
    app.get_ui_setting_keys = lambda: ["dialog_placement"]
    app.apply_settings = functools.partial(BaseCLIApp.apply_settings, app)
    saved = SimpleNamespace(project_path=Path("p"), user_path=None, user_scoped_keys=())
    app.save_settings = AsyncMock(return_value=saved)

    first = await _settings(app, monkeypatch, {"dialog_placement": "inline"})
    app.session.show_dialog.reset_mock()
    second = await _settings(app, monkeypatch, {})

    assert (first.placement, second.placement) == ("box", "inline")


async def test_save_applies_only_the_changed_keys(app, monkeypatch):
    await _settings(
        app,
        monkeypatch,
        {
            "thinking_effort": "high",
            "retry_max_attempts": "5",
            "verbose_thinking": True,  # what it already was
            "model": "google:gemini-2.5-pro",
        },
    )

    app.apply_settings.assert_awaited_once_with(
        {"model": "google:gemini-2.5-pro", "thinking_effort": "high", "retry_max_attempts": "5"}
    )


async def test_cancel_and_escape_leave_the_settings_unchanged(app, monkeypatch):
    await _settings(app, monkeypatch, {"thinking_effort": "high"}, saved=False)

    app.apply_settings.assert_not_awaited()
    assert app.session.add_message.call_args_list == [call("system", "Settings unchanged.")]


async def test_ctrl_c_interrupts_the_dialog_and_says_nothing(app, monkeypatch):
    """Esc and Cancel cancel the dialog; Ctrl+C interrupts the command. 0.4
    delivers it by cancelling the command's task, which closes the dialog,
    and agentic-cli adds nothing: no line, nothing applied or saved."""
    monkeypatch.setattr(
        "agentic_cli.cli.settings_command.Dialog", edited_dialog({"thinking_effort": "high"})
    )
    app.session.show_dialog.side_effect = asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await SettingsCommand().execute("", app)

    assert [name for name, *_ in app.session.method_calls] == ["show_dialog"]
    app.apply_settings.assert_not_awaited()
    app.save_settings.assert_not_awaited()


async def test_save_without_a_change_leaves_the_settings_unchanged(app, monkeypatch):
    """Untouched fields hold their starting values: an int or a float shown as
    text, a bool, an Enum and a picker's reference never read as a change."""
    await _settings(app, monkeypatch, {})

    app.apply_settings.assert_not_awaited()
    assert app.session.add_message.call_args_list == [call("system", "Settings unchanged.")]


async def test_save_with_every_field_set_to_its_value_changes_nothing(app, monkeypatch):
    values = {field.key: field.default for field in app._build_ui_items()}

    await _settings(app, monkeypatch, values)

    app.apply_settings.assert_not_awaited()
    assert app.session.add_message.call_args_list == [call("system", "Settings unchanged.")]


async def test_without_fields_there_is_no_dialog(app, monkeypatch):
    app._build_ui_items = lambda: []
    dialog_cls = edited_dialog({})
    monkeypatch.setattr("agentic_cli.cli.settings_command.Dialog", dialog_cls)

    await SettingsCommand().execute("", app)

    assert dialog_cls.built == []
    app.session.show_dialog.assert_not_awaited()
    app.session.add_error.assert_called_once_with(
        "No settings available. Please configure API keys first."
    )
