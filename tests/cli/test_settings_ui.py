"""``/settings`` lives in ``cli/settings_ui.py``: ``SettingsUI`` builds the
dialog's fields (the model pickers among them) and applies and saves a Save.

``BaseCLIApp`` keeps the methods an app overrides (``_build_ui_items``,
``apply_settings``, ``save_settings``), and each delegates to ``SettingsUI``,
so ``/settings`` still goes through the app: an app's override of one of
them, or of a hook ``SettingsUI`` calls (``get_ui_setting_keys``,
``get_theme``), still applies.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from thinking_prompt import ThinkingPromptStyles

from agentic_cli.cli.app import BaseCLIApp
from agentic_cli.cli.settings_command import SettingsCommand
from agentic_cli.cli.workflow_controller import WorkflowState
from agentic_cli.config import BaseSettings
from tests.cli.dialog_fakes import autospec_session, edited_dialog


@pytest.fixture(autouse=True)
def hermetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "home").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for variable in ("AGENTIC_UI_THEME", "AGENTIC_THINKING_EFFORT"):
        monkeypatch.delenv(variable, raising=False)


def _bare(cls: type[BaseCLIApp] = BaseCLIApp) -> BaseCLIApp:
    """An app with settings, a controller that has not started, and a session."""
    app = cls.__new__(cls)
    app._settings = BaseSettings(session_store="memory")
    app._workflow_controller = SimpleNamespace(
        is_ready=False, state=WorkflowState.UNINITIALIZED, agent_configs=[]
    )
    app.session = autospec_session()
    return app


class TestTheAppDelegates:
    """Each of the app's ``/settings`` methods is ``SettingsUI``'s, on the app."""

    def test_the_fields(self):
        from agentic_cli.cli.settings_ui import SettingsUI

        app = _bare()
        with patch.object(SettingsUI, "fields", autospec=True, return_value=["built"]) as fields:
            assert app._build_ui_items() == ["built"]

        (ui,), _ = fields.call_args
        assert ui.app is app

    async def test_a_save_s_effect(self):
        from agentic_cli.cli.settings_ui import SettingsUI

        app = _bare()
        with patch.object(SettingsUI, "apply", autospec=True, return_value=False) as apply:
            assert await app.apply_settings({"thinking_effort": "high"}) is False

        (ui, changes), _ = apply.call_args
        assert ui.app is app and changes == {"thinking_effort": "high"}

    async def test_the_save(self):
        from agentic_cli.cli.settings_ui import SettingsUI

        app = _bare()
        with patch.object(SettingsUI, "save", autospec=True, return_value="saved") as save:
            assert await app.save_settings(keys={"thinking_effort"}) == "saved"

        (ui, keys), _ = save.call_args
        assert ui.app is app and keys == {"thinking_effort"}


class TestAnAppsOverridesStillApply:
    """``/settings`` on an app that overrides each method and hook it goes
    through: every override is called, and each ``super()`` does its part."""

    class App(BaseCLIApp):
        calls: list[tuple]

        def get_ui_setting_keys(self) -> list[str]:
            self.calls.append(("get_ui_setting_keys",))
            return ["thinking_effort", "ui_theme"]

        def _build_ui_items(self):
            items = super()._build_ui_items()
            self.calls.append(("_build_ui_items", [item.key for item in items]))
            return items

        async def apply_settings(self, changes):
            self.calls.append(("apply_settings", dict(changes)))
            return await super().apply_settings(changes)

        async def save_settings(self, keys):
            self.calls.append(("save_settings", set(keys)))
            return SimpleNamespace(project_path=Path("p"), user_path=None, user_scoped_keys=())

        def get_theme(self, name: str) -> str | ThinkingPromptStyles:
            self.calls.append(("get_theme", name))
            return super().get_theme(name)

    async def test_each_override_is_called(self, monkeypatch):
        app = _bare(self.App)
        app.calls = []
        app.session.show_dialog.return_value = True  # Save
        edits = {"thinking_effort": "high", "ui_theme": "light"}
        monkeypatch.setattr("agentic_cli.cli.settings_command.Dialog", edited_dialog(edits))

        await SettingsCommand().execute("", app)

        assert app.calls == [
            ("get_ui_setting_keys",),
            ("_build_ui_items", ["thinking_effort", "ui_theme"]),
            ("apply_settings", edits),
            ("get_theme", "light"),
            ("save_settings", {"thinking_effort", "ui_theme"}),
        ]
        assert app.settings.thinking_effort == "high"
        app.session.set_theme.assert_called_once_with("light", repaint=True)

    async def test_an_override_of_save_settings_is_what_saves(self, monkeypatch):
        """The base's own save is not reached: the app's override replaced it."""
        app = _bare(self.App)
        app.calls = []
        app.session.show_dialog.return_value = True
        monkeypatch.setattr(
            "agentic_cli.cli.settings_command.Dialog", edited_dialog({"thinking_effort": "low"})
        )
        persist = AsyncMock()
        monkeypatch.setattr("agentic_cli.settings_persistence.SettingsPersistence.save", persist)

        await SettingsCommand().execute("", app)

        persist.assert_not_called()
        assert ("save_settings", {"thinking_effort"}) in app.calls
