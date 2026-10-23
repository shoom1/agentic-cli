"""Saving settings persists what the user changed, and never a credential.

``/settings`` used to dump the whole live settings object: a value that came
from the environment (``os_sandbox_enabled=False`` for one run) or was derived
at startup was written into the user config and kept applying on every later
run, and an application's own API-key fields (not in ``SECRET_FIELDS``) were
written in plaintext.

Telling a credential by its name matched ``token`` anywhere in it, so a token
count such as ``context_max_tokens`` was never saved: changed in ``/settings``,
it was back to its default on the next start.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import Field, SecretStr

from agentic_cli.config import BaseSettings
from agentic_cli.settings_persistence import (
    SECRET_FIELDS,
    SettingsPersistence,
    is_secret_field,
)

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
    mistral_api_key: str | None = Field(default=None)
    service_token: SecretStr | None = Field(default=None)
    github_token: str | None = Field(default=None)
    report_style: str = Field(default="plain")
    summary_max_tokens: int = Field(default=1000)


class TestCredentialFieldsAreNeverWritten:
    def test_full_save_skips_domain_credentials(self, fs):
        settings = AppSettings(
            mistral_api_key="value-1", service_token="value-2", github_token="value-3",
            report_style="fancy",
        )
        SettingsPersistence(APP).save(settings)
        written = {**_read(fs["project"]), **_read(fs["user"])}
        assert written.get("report_style") == "fancy"
        assert "mistral_api_key" not in written
        assert "service_token" not in written
        assert "github_token" not in written

    def test_keyed_save_skips_domain_credentials(self, fs):
        settings = AppSettings(mistral_api_key="value-1")
        SettingsPersistence(APP).save(settings, keys={"mistral_api_key"})
        assert "mistral_api_key" not in {**_read(fs["project"]), **_read(fs["user"])}


class TestTokenCountsAreSaved:
    def test_a_keyed_save_writes_the_context_ceiling(self, fs):
        settings = BaseSettings()
        settings.update_setting("context_max_tokens", 50000)
        SettingsPersistence(APP).save(settings, keys={"context_max_tokens"})
        assert _read(fs["project"]) == {"context_max_tokens": 50000}

    def test_a_full_save_writes_token_counts(self, fs):
        settings = AppSettings(context_max_tokens=50000, summary_max_tokens=2000)
        SettingsPersistence(APP).save(settings)
        written = {**_read(fs["project"]), **_read(fs["user"])}
        assert written["context_max_tokens"] == 50000
        assert written["summary_max_tokens"] == 2000

    def test_the_framework_s_only_secret_fields_are_its_credentials(self):
        fields = BaseSettings.model_fields
        assert {n for n, f in fields.items() if is_secret_field(n, f)} == SECRET_FIELDS

    @pytest.mark.parametrize("name", [
        "github_token", "hf_token", "token", "access_token_file", "mistral_api_key",
        "apikey", "client_secret", "db_password", "credentials_path",
    ])
    def test_a_credential_name_is_secret(self, name):
        assert is_secret_field(name)

    @pytest.mark.parametrize("name", [
        "context_max_tokens", "max_tokens", "summary_max_tokens", "tokens_per_minute",
        "tokenizer", "embedding_tokenizer_name",
    ])
    def test_a_token_count_or_tokenizer_is_not(self, name):
        assert not is_secret_field(name)


class TestSettingsCommandSavesTheChanges:
    """The command persists the fields the dialog actually changed."""

    def _app(self, settings, monkeypatch, edits):
        """An app whose ``/settings`` dialog shows ``thinking_effort`` and is
        saved with ``edits``."""
        from agentic_cli.cli.app import BaseCLIApp
        from agentic_cli.cli.settings_introspection import field_to_ui_item
        from tests.cli.dialog_fakes import autospec_session, edited_dialog

        monkeypatch.setattr("agentic_cli.cli.settings_command.Dialog", edited_dialog(edits))
        app = SimpleNamespace()
        app.settings = settings
        app._settings = settings
        app._build_ui_items = lambda: [
            field_to_ui_item(
                "thinking_effort",
                BaseSettings.model_fields["thinking_effort"],
                settings.thinking_effort,
            )
        ]
        app.session = autospec_session()
        app.session.show_dialog.return_value = True  # Save
        app._workflow_controller = SimpleNamespace(is_ready=False)
        app.apply_settings = lambda changes: BaseCLIApp.apply_settings(app, changes)
        app.save_settings = AsyncMock(return_value=SimpleNamespace(
            project_path=Path("p"), user_path=None, user_scoped_keys=()))
        return app

    @pytest.mark.asyncio
    async def test_saves_only_the_changed_fields(self, fs, monkeypatch):
        from agentic_cli.cli.settings_command import SettingsCommand

        settings = BaseSettings(os_sandbox_enabled=False)
        app = self._app(settings, monkeypatch, {"thinking_effort": "high"})
        await SettingsCommand().execute("", app)
        app.save_settings.assert_awaited_once_with(keys={"thinking_effort"})

    @pytest.mark.asyncio
    async def test_nothing_changed_means_nothing_saved(self, fs, monkeypatch):
        from agentic_cli.cli.settings_command import SettingsCommand

        settings = BaseSettings()
        app = self._app(settings, monkeypatch, {"thinking_effort": settings.thinking_effort})
        await SettingsCommand().execute("", app)
        app.save_settings.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_override_without_keys_is_still_called(self, fs, monkeypatch):
        """An app that overrode save_settings() before keys= existed keeps
        working (it gets its own full save)."""
        from agentic_cli.cli.settings_command import SettingsCommand

        settings = BaseSettings()
        app = self._app(settings, monkeypatch, {"thinking_effort": "high"})
        calls = []

        async def legacy_save_settings():
            calls.append("called")
            return SimpleNamespace(project_path=Path("p"), user_path=None, user_scoped_keys=())

        app.save_settings = legacy_save_settings
        await SettingsCommand().execute("", app)
        assert calls == ["called"]
