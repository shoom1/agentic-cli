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
import threading
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
from agentic_cli.workflow.settings import PermissionsConfig

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


class TestTheSaveTakesTheKeysToWrite:
    """One writer: a save names the keys it writes. The keyless save wrote
    every allowlisted field's live value, which froze what came from the
    environment, the app's defaults and the start-up into files."""

    def test_keys_are_required(self, fs):
        with pytest.raises(TypeError, match="keys"):
            SettingsPersistence(APP).save(BaseSettings(raw_llm_logging=True))
        assert not fs["project"].exists()
        assert not fs["user"].exists()

    def test_a_path_is_not_taken(self, fs, tmp_path):
        with pytest.raises(TypeError):
            SettingsPersistence(APP).save(
                BaseSettings(), path=tmp_path / "dump.json", keys={"thinking_effort"}
            )

    def test_the_app_name_is_required(self):
        with pytest.raises(TypeError, match="app_name"):
            SettingsPersistence()

    @pytest.mark.asyncio
    async def test_the_app_s_save_requires_keys(self, fs):
        from agentic_cli.cli.app import BaseCLIApp

        app = SimpleNamespace(_settings=BaseSettings(raw_llm_logging=True))
        with pytest.raises(TypeError, match="keys"):
            await BaseCLIApp.save_settings(app)
        assert not fs["project"].exists()
        assert not fs["user"].exists()

    @pytest.mark.asyncio
    async def test_the_app_s_save_writes_off_the_event_loop(self, fs, monkeypatch):
        """The files are written on a worker thread, as any async code must
        keep its blocking work off the loop."""
        from agentic_cli.cli.app import BaseCLIApp

        threads: list[threading.Thread] = []
        save = SettingsPersistence.save

        def recording(self, settings, **kwargs):
            threads.append(threading.current_thread())
            return save(self, settings, **kwargs)

        monkeypatch.setattr(SettingsPersistence, "save", recording)
        settings = BaseSettings()
        settings.update_setting("thinking_effort", "low")
        app = SimpleNamespace(_settings=settings)

        saved = await BaseCLIApp.save_settings(app, keys={"thinking_effort"})

        assert threads and threads[0] is not threading.current_thread()
        assert saved.project_path == fs["project"]
        assert _read(fs["project"]) == {"thinking_effort": "low"}


class AppSettings(BaseSettings):
    mistral_api_key: str | None = Field(default=None)
    service_token: SecretStr | None = Field(default=None)
    github_token: str | None = Field(default=None)
    report_style: str = Field(default="plain")
    summary_max_tokens: int = Field(default=1000)


class TestCredentialFieldsAreNeverWritten:
    def test_a_save_naming_every_field_skips_domain_credentials(self, fs):
        settings = AppSettings(
            mistral_api_key="value-1", service_token="value-2", github_token="value-3",
            report_style="fancy",
        )
        SettingsPersistence(APP).save(settings, keys=set(AppSettings.model_fields))
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

    def test_a_keyed_save_writes_token_counts(self, fs):
        settings = AppSettings(context_max_tokens=50000, summary_max_tokens=2000)
        SettingsPersistence(APP).save(settings, keys={"context_max_tokens", "summary_max_tokens"})
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
    async def test_an_override_without_keys_fails_at_the_save_with_its_reason(
        self, fs, monkeypatch
    ):
        """The change stays applied; the dialog says it was not saved, and why."""
        from agentic_cli.cli.settings_command import SettingsCommand

        settings = BaseSettings()
        app = self._app(settings, monkeypatch, {"thinking_effort": "high"})
        calls = []

        async def save_settings_without_keys():
            calls.append("called")

        app.save_settings = save_settings_without_keys
        await SettingsCommand().execute("", app)

        assert calls == []
        assert settings.thinking_effort == "high"
        (warning,) = [c.args[0] for c in app.session.add_warning.call_args_list]
        assert warning.startswith("Settings applied but not saved: ")
        assert "keys" in warning
        assert not any(
            "saved" in call.args[0].lower() for call in app.session.add_success.call_args_list
        )


class TestTheProjectPermissionRules:
    """The project file's ``permissions`` section is no setting: the permission
    engine reads its deny rules from the file itself, so no settings save may
    drop them (a /settings change used to turn a denied read into an allowed one)."""

    DENY = {"deny": [{"capability": "filesystem.read", "target": "notes/**"}]}

    def _seed(self, fs, data: dict) -> None:
        fs["project"].parent.mkdir(parents=True, exist_ok=True)
        fs["project"].write_text(json.dumps(data))

    @staticmethod
    def _project_denials() -> list[tuple[str, str]]:
        """The deny rules the engine reads from the project file."""
        from unittest.mock import MagicMock

        from agentic_cli.workflow.permissions import (
            Effect,
            PermissionContext,
            PermissionEngine,
            RuleSource,
        )

        ctx = PermissionContext(workdir=Path.cwd(), home=Path.home(), app_name=APP)
        engine = PermissionEngine(settings=BaseSettings(), workflow=MagicMock(), ctx=ctx)
        return [
            (r.capability, r.target) for r in engine.rules
            if r.source is RuleSource.PROJECT and r.effect is Effect.DENY
        ]

    def test_saving_a_changed_key_keeps_the_project_permission_rules(self, fs):
        self._seed(fs, {"log_level": "debug", "permissions": self.DENY})
        denied = self._project_denials()
        assert denied, "the seeded deny rule is read"
        settings = BaseSettings()
        settings.update_setting("thinking_effort", "low")

        SettingsPersistence(APP).save(settings, keys={"thinking_effort"})

        assert _read(fs["project"])["permissions"] == self.DENY
        assert self._project_denials() == denied

    def test_a_save_naming_every_field_keeps_the_project_permission_rules(self, fs):
        self._seed(fs, {"log_level": "debug", "permissions": self.DENY})
        denied = self._project_denials()

        SettingsPersistence(APP).save(BaseSettings(), keys=set(BaseSettings.model_fields))

        assert _read(fs["project"])["permissions"] == self.DENY
        assert self._project_denials() == denied

    def test_a_save_leaves_a_malformed_project_file_as_it_is(self, fs):
        """Rewriting a file that cannot be read would drop any rules it holds."""
        settings = BaseSettings()  # loaded before the file broke
        fs["project"].parent.mkdir(parents=True)
        fs["project"].write_text('{"permissions": {"deny": [')

        with pytest.raises(ValueError):
            SettingsPersistence(APP).save(settings, keys={"thinking_effort"})

        assert fs["project"].read_text() == '{"permissions": {"deny": ['

    def test_a_save_takes_no_project_rule_into_the_user_file(self, fs):
        """The project file is untrusted, so its allow rules never apply; a
        save must not carry them into the trusted user file."""
        allow = {"allow": [{"capability": "filesystem.write", "target": "/**"}], **self.DENY}
        self._seed(fs, {"permissions": allow})
        settings = BaseSettings()
        assert settings.permissions == BaseSettings.model_fields["permissions"].get_default(
            call_default_factory=True
        )

        SettingsPersistence(APP).save(settings, keys=set(BaseSettings.model_fields))

        assert "permissions" not in _read(fs["user"])
        assert _read(fs["project"])["permissions"] == allow


class TestTheUnionOfRulesIsNeverSaved:
    """``settings.permissions`` is the union of the app's default, the user
    file, the environment and the constructor. A save writes none of it
    (``HAND_EDITED_KEYS``): the user file keeps its own section as it is,
    no other source's rules are copied into a file, and the project file
    keeps its deny section (``PROJECT_PRESERVED_KEYS``)."""

    USER = {"allow": [{"capability": "http.read", "target": "https://docs.example.org/**"}]}
    PROJECT = {"deny": [{"capability": "filesystem.read", "target": "notes/**"}]}

    @pytest.fixture
    def loaded(self, fs, monkeypatch):
        """Settings whose rules came from every trusted source, and a project
        file with its own deny section."""
        for path, body in ((fs["user"], {"log_level": "info", "permissions": self.USER}),
                           (fs["project"], {"permissions": self.PROJECT})):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(body))
        monkeypatch.setenv("AGENTIC_PERMISSIONS", json.dumps(
            {"deny": [{"capability": "http.read", "target": "https://env.example.net/**"}]}
        ))

        class AppSettings(BaseSettings):
            permissions: PermissionsConfig = Field(
                default_factory=lambda: PermissionsConfig.model_validate(
                    {"deny": [{"capability": "shell.exec", "target": "rm -rf *"}]}
                )
            )

        settings = AppSettings(
            permissions={"deny": [{"capability": "http.read", "target": "https://ctor.test/**"}]}
        )
        assert len(settings.permissions.deny) == 3, "the union was loaded"
        return settings

    def _assert_each_file_keeps_its_own_section(self, fs) -> None:
        assert _read(fs["user"])["permissions"] == self.USER
        assert _read(fs["project"])["permissions"] == self.PROJECT

    def test_a_save_naming_every_field(self, fs, loaded):
        SettingsPersistence(APP).save(loaded, keys=set(type(loaded).model_fields))

        self._assert_each_file_keeps_its_own_section(fs)

    def test_a_keyed_save_of_another_setting(self, fs, loaded):
        loaded.update_setting("thinking_effort", "low")

        SettingsPersistence(APP).save(loaded, keys={"thinking_effort", "log_level"})

        self._assert_each_file_keeps_its_own_section(fs)

    def test_a_keyed_save_naming_the_rules(self, fs, loaded):
        SettingsPersistence(APP).save(loaded, keys={"permissions"})

        self._assert_each_file_keeps_its_own_section(fs)
