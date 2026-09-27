"""Saving settings persists what the user changed, and never a credential.

``/settings`` used to dump the whole live settings object: a value that came
from the environment (``os_sandbox_enabled=False`` for one run) or was derived
at startup was written into the user config and kept applying on every later
run, and an application's own API-key fields (not in ``SECRET_FIELDS``) were
written in plaintext.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import Field, SecretStr

from agentic_cli.config import BaseSettings
from agentic_cli.settings_persistence import SettingsPersistence

APP = "agentic_cli"


@pytest.fixture
def fs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    return {
        "project": tmp_path / f".{APP}" / "settings.json",
        "user": home / f".{APP}" / "settings.json",
    }


def _read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


class TestSaveOnlyGivenKeys:
    def test_unchanged_non_default_values_are_not_persisted(self, fs):
        # As if these came from the environment for this run only.
        settings = BaseSettings(os_sandbox_enabled=False, raw_llm_logging=True)
        settings.update_setting("thinking_effort", "high")

        SettingsPersistence(APP).save(settings, keys={"thinking_effort"})

        assert _read(fs["project"]) == {"thinking_effort": "high"}
        assert _read(fs["user"]) == {}

    def test_a_changed_user_scoped_key_goes_to_the_user_file(self, fs):
        settings = BaseSettings()
        settings.update_setting("raw_llm_logging", True)
        SettingsPersistence(APP).save(settings, keys={"raw_llm_logging"})
        assert _read(fs["user"]) == {"raw_llm_logging": True}
        assert _read(fs["project"]) == {}

    def test_other_project_entries_are_kept_and_stale_ones_dropped(self, fs):
        fs["project"].parent.mkdir(parents=True)
        fs["project"].write_text(json.dumps({"log_level": "debug", "raw_llm_logging": True}))
        settings = BaseSettings()
        settings.update_setting("thinking_effort", "low")
        SettingsPersistence(APP).save(settings, keys={"thinking_effort"})
        assert _read(fs["project"]) == {"log_level": "debug", "thinking_effort": "low"}

    def test_a_key_changed_back_to_default_leaves_the_user_file(self, fs):
        fs["user"].parent.mkdir(parents=True)
        fs["user"].write_text(json.dumps({"raw_llm_logging": True, "other": 1}))
        settings = BaseSettings(raw_llm_logging=False)  # the user switched it back
        SettingsPersistence(APP).save(settings, keys={"raw_llm_logging"})
        assert _read(fs["user"]) == {"other": 1}

    def test_no_keys_writes_nothing(self, fs):
        SettingsPersistence(APP).save(BaseSettings(raw_llm_logging=True), keys=set())
        assert not fs["project"].exists()
        assert not fs["user"].exists()


class AppSettings(BaseSettings):
    openai_api_key: str | None = Field(default=None)
    service_token: SecretStr | None = Field(default=None)
    report_style: str = Field(default="plain")


class TestCredentialFieldsAreNeverWritten:
    def test_full_save_skips_domain_credentials(self, fs):
        settings = AppSettings(
            openai_api_key="value-1", service_token="value-2", report_style="fancy",
        )
        SettingsPersistence(APP).save(settings)
        written = {**_read(fs["project"]), **_read(fs["user"])}
        assert written.get("report_style") == "fancy"
        assert "openai_api_key" not in written
        assert "service_token" not in written

    def test_keyed_save_skips_domain_credentials(self, fs):
        settings = AppSettings(openai_api_key="value-1")
        SettingsPersistence(APP).save(settings, keys={"openai_api_key"})
        assert "openai_api_key" not in {**_read(fs["project"]), **_read(fs["user"])}


class TestSettingsCommandSavesTheChanges:
    """The command persists the fields the dialog actually changed."""

    @pytest.fixture(autouse=True)
    def _no_real_dialog(self, monkeypatch):
        monkeypatch.setattr(
            "agentic_cli.cli.settings_command.SettingsDialog", lambda **kw: None
        )

    def _app(self, settings, dialog_result):
        from agentic_cli.cli.app import BaseCLIApp

        app = SimpleNamespace()
        app.settings = settings
        app._settings = settings
        app._build_ui_items = lambda: ["item"]
        app.session = SimpleNamespace(
            show_dialog=AsyncMock(return_value=dialog_result),
            add_message=lambda *a: None, add_error=lambda *a: None,
            add_success=lambda *a: None, add_warning=lambda *a: None,
        )
        app._workflow_controller = SimpleNamespace(is_ready=False)
        app.apply_settings = lambda changes: BaseCLIApp.apply_settings(app, changes)
        app.save_settings = AsyncMock(return_value=SimpleNamespace(
            project_path=Path("p"), user_path=None, user_scoped_keys=()))
        return app

    @pytest.mark.asyncio
    async def test_saves_only_the_changed_fields(self, fs):
        from agentic_cli.cli.settings_command import SettingsCommand

        settings = BaseSettings(os_sandbox_enabled=False)
        app = self._app(settings, {"thinking_effort": "high"})
        await SettingsCommand().execute("", app)
        app.save_settings.assert_awaited_once_with(keys={"thinking_effort"})

    @pytest.mark.asyncio
    async def test_nothing_changed_means_nothing_saved(self, fs):
        from agentic_cli.cli.settings_command import SettingsCommand

        settings = BaseSettings()
        app = self._app(settings, {"thinking_effort": settings.thinking_effort})
        await SettingsCommand().execute("", app)
        app.save_settings.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_override_without_keys_is_still_called(self, fs):
        """An app that overrode save_settings() before keys= existed keeps
        working (it gets its own full save)."""
        from agentic_cli.cli.settings_command import SettingsCommand

        settings = BaseSettings()
        app = self._app(settings, {"thinking_effort": "high"})
        calls = []

        async def legacy_save_settings():
            calls.append("called")
            return SimpleNamespace(project_path=Path("p"), user_path=None, user_scoped_keys=())

        app.save_settings = legacy_save_settings
        await SettingsCommand().execute("", app)
        assert calls == ["called"]
