"""Guard: /settings exposes project-scoped settings, and the user's preferences.

User-scoped keys (not in PROJECT_SETTABLE_KEYS) must never render in the
dialog, regardless of what a domain app returns from get_ui_setting_keys() —
otherwise a /settings edit would persist to the user ~/.{app}/settings.json
and apply across all projects. The one exception is USER_PREFERENCE_KEYS
(ui_theme, dialog_placement): how the CLI looks is meant to apply in every
project, so an app may list them, and a Save writes them to the user file.
The default dialog does not show them.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import call

import pytest
import structlog
from thinking_prompt import SelectField

from agentic_cli.cli.app import BaseCLIApp
from agentic_cli.cli.settings_command import SettingsCommand
from tests.cli.dialog_fakes import autospec_session, edited_dialog

EXCLUDED_EVENT = "user_scoped_setting_excluded_from_ui"


def _make_app(settings, keys: list[str]) -> BaseCLIApp:
    class _App(BaseCLIApp):
        def get_ui_setting_keys(self) -> list[str]:
            return keys

    app = _App.__new__(_App)
    app._settings = settings
    app._workflow_controller = SimpleNamespace(is_ready=False)
    return app


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A temp cwd and HOME, so no real config file loads; returns HOME."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    for variable in ("AGENTIC_UI_THEME", "AGENTIC_DIALOG_PLACEMENT"):
        monkeypatch.delenv(variable, raising=False)
    return home


@pytest.fixture
def isolated_settings(home):
    """Hermetic BaseSettings: temp cwd/HOME so no real config files load."""
    from agentic_cli.config import BaseSettings

    # A key so the model picker has options and "model" can render
    return BaseSettings(google_api_key="test-key")


def _item_keys(items) -> list[str]:
    return [item.key for item in items]


class TestSettingsUiScopeGuard:
    def test_user_scoped_key_is_excluded_with_warning(self, isolated_settings):
        """A non-allowlisted field never renders; the exclusion is logged."""
        app = _make_app(isolated_settings, ["thinking_effort", "raw_llm_logging"])

        with structlog.testing.capture_logs() as logs:
            items = app._build_ui_items()

        assert _item_keys(items) == ["thinking_effort"]
        assert any(
            e.get("event") == EXCLUDED_EVENT and e.get("key") == "raw_llm_logging"
            for e in logs
        )

    def test_allowlisted_keys_render_without_warning(self, isolated_settings):
        app = _make_app(isolated_settings, ["thinking_effort", "verbose_thinking"])

        with structlog.testing.capture_logs() as logs:
            items = app._build_ui_items()

        assert set(_item_keys(items)) == {"thinking_effort", "verbose_thinking"}
        assert not any(e.get("event") == EXCLUDED_EVENT for e in logs)

    def test_model_synthetic_key_passes_guard(self, isolated_settings):
        """"model" is not a field but writes default_model (allowlisted)."""
        app = _make_app(isolated_settings, ["model"])

        with structlog.testing.capture_logs() as logs:
            items = app._build_ui_items()

        assert _item_keys(items) == ["model"]
        assert not any(e.get("event") == EXCLUDED_EVENT for e in logs)

    def test_dangling_nonexistent_key_is_excluded_with_warning(
        self, isolated_settings
    ):
        """A key for a removed field (e.g. airesearcher's log_activity) now
        warns instead of being silently skipped."""
        app = _make_app(isolated_settings, ["log_activity"])

        with structlog.testing.capture_logs() as logs:
            items = app._build_ui_items()

        assert items == []
        assert any(
            e.get("event") == EXCLUDED_EVENT and e.get("key") == "log_activity"
            for e in logs
        )


class TestOrchestratorPicker:
    """An app that shows ``orchestrator`` gets a picker of every backend."""

    def test_it_offers_every_orchestrator(self, isolated_settings):
        from thinking_prompt import SelectField

        app = _make_app(isolated_settings, ["orchestrator"])

        (item,) = app._build_ui_items()

        assert isinstance(item, SelectField)
        assert item.options == ["adk", "langgraph", "deepagents"]
        assert item.default == "adk"

    def test_it_shows_the_current_one(self, isolated_settings):
        isolated_settings.orchestrator = "deepagents"
        app = _make_app(isolated_settings, ["orchestrator"])

        (item,) = app._build_ui_items()

        assert item.default == "deepagents"


class TestUserPreferences:
    """ui_theme and dialog_placement: in the dialog when the app lists them,
    saved to the user file."""

    def test_the_default_dialog_does_not_show_them(self, isolated_settings):
        app = BaseCLIApp.__new__(BaseCLIApp)
        app._settings = isolated_settings
        app._workflow_controller = SimpleNamespace(is_ready=False)

        keys = _item_keys(app._build_ui_items())

        assert keys == ["model", "model_role:fast", "model_role:strong", "thinking_effort"]

    def test_an_app_that_lists_them_shows_them_as_selects(self, isolated_settings):
        isolated_settings.update_setting("ui_theme", "light")
        app = _make_app(
            isolated_settings,
            ["dialog_placement", "ui_theme", "verbose_thinking", "thinking_effort"],
        )

        with structlog.testing.capture_logs() as logs:
            items = app._build_ui_items()

        # After the model and thinking settings, whatever order the app lists.
        assert _item_keys(items) == [
            "thinking_effort", "verbose_thinking", "ui_theme", "dialog_placement",
        ]
        theme, placement = items[2:]
        assert isinstance(theme, SelectField) and isinstance(placement, SelectField)
        assert (theme.label, theme.options, theme.default) == (
            "UI Theme", ["dark", "light", "mono", "terminal", "auto"], "light",
        )
        assert (placement.label, placement.options, placement.default) == (
            "Dialog Placement", ["box", "inline"], "box",
        )
        assert not any(e.get("event") == EXCLUDED_EVENT for e in logs)

    def test_any_other_user_scoped_key_is_still_refused(self, isolated_settings):
        app = _make_app(isolated_settings, ["ui_theme", "raw_llm_logging", "model_endpoints"])

        with structlog.testing.capture_logs() as logs:
            items = app._build_ui_items()

        assert _item_keys(items) == ["ui_theme"]
        excluded = {e.get("key") for e in logs if e.get("event") == EXCLUDED_EVENT}
        assert excluded == {"raw_llm_logging", "model_endpoints"}


class TestSavingAPreference:
    """A /settings Save of a user preference writes the user file, and says so."""

    async def _save(self, monkeypatch, edits: dict):
        """Run /settings on an app listing both preferences; the user makes
        ``edits`` and presses Save. Returns the app."""
        from agentic_cli.config import BaseSettings

        app = _make_app(BaseSettings(), ["ui_theme", "dialog_placement"])
        app.session = autospec_session()
        app.session.show_dialog.return_value = True  # Save
        monkeypatch.setattr("agentic_cli.cli.settings_command.Dialog", edited_dialog(edits))

        await SettingsCommand().execute("", app)
        return app

    async def test_it_goes_to_the_user_file_not_the_project_s(
        self, home, tmp_path, monkeypatch
    ):
        from agentic_cli.config import BaseSettings

        app = await self._save(monkeypatch, {"ui_theme": "light"})

        user = home / ".agentic_cli" / "settings.json"
        assert json.loads(user.read_text()) == {"ui_theme": "light"}
        assert not (tmp_path / ".agentic_cli" / "settings.json").exists()
        app.session.add_success.assert_called_with(f"user-scoped (ui_theme) saved to {user}")
        app.session.add_error.assert_not_called()
        assert BaseSettings().ui_theme == "light"  # the next start

    async def test_the_report_names_it_as_the_dialog_does(self, home, monkeypatch):
        app = await self._save(monkeypatch, {"ui_theme": "light"})

        assert app.session.add_success.call_args_list[0] == call("UI Theme: light")

    async def test_switching_back_to_dark_removes_it_so_the_next_start_is_dark(
        self, home, monkeypatch
    ):
        from agentic_cli.config import BaseSettings

        user = home / ".agentic_cli" / "settings.json"
        user.parent.mkdir()
        user.write_text(json.dumps({"ui_theme": "light", "other": 1}))

        app = await self._save(monkeypatch, {"ui_theme": "dark"})

        assert json.loads(user.read_text()) == {"other": 1}
        app.session.add_success.assert_called_with(f"user-scoped (ui_theme) saved to {user}")
        assert BaseSettings().ui_theme == "dark"
