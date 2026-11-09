"""Tests for settings persistence — secret field exclusion and atomic writes."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agentic_cli.settings_persistence import SECRET_FIELDS, SettingsPersistence


@pytest.fixture(autouse=True)
def isolated_fs(tmp_path, monkeypatch):
    """Isolate cwd + HOME so a save hits temp files only."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    for var in (
        "AGENTIC_RAW_LLM_LOGGING",
        "AGENTIC_STATEFUL_EXECUTOR_BACKEND",
        "AGENTIC_DEFAULT_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def _written(root: Path, app_name: str) -> dict:
    """What both settings files of ``app_name`` hold, merged."""
    data: dict = {}
    for path in (
        root / "home" / f".{app_name}" / "settings.json",
        root / f".{app_name}" / "settings.json",
    ):
        if path.exists():
            data.update(json.loads(path.read_text()))
    return data


class TestSecretFields:
    """Tests for SECRET_FIELDS completeness (C1)."""

    def test_secret_fields_excludes_all_api_keys(self):
        """Every credential-bearing field in BaseSettings is in SECRET_FIELDS."""
        expected = {
            "google_api_key",
            "anthropic_api_key",
            "openai_api_key",
            "tavily_api_key",
            "brave_api_key",
            "postgres_uri",  # embeds user:password@host
        }
        assert SECRET_FIELDS == expected

    def test_save_excludes_secrets(self, tmp_path):
        """Secret values are never written to a settings file, even named."""
        from agentic_cli.config import BaseSettings

        settings = BaseSettings(
            google_api_key="secret-google",
            anthropic_api_key="secret-anthropic",
            tavily_api_key="secret-tavily",
            brave_api_key="secret-brave",
            postgres_uri="postgresql://user:pass@host/db",
            search_backend="tavily",  # non-secret
        )

        persistence = SettingsPersistence(app_name="test")
        persistence.save(settings, keys=set(BaseSettings.model_fields))

        data = _written(tmp_path, "test")
        assert data["search_backend"] == "tavily"
        for secret in SECRET_FIELDS:
            assert secret not in data, f"{secret} leaked into saved JSON"

    def test_save_includes_non_secret_fields(self, tmp_path):
        """Non-secret settings are saved correctly."""
        from agentic_cli.config import BaseSettings

        settings = BaseSettings(search_backend="brave")

        persistence = SettingsPersistence(app_name="test")
        persistence.save(settings, keys={"search_backend"})

        assert _written(tmp_path, "test")["search_backend"] == "brave"

    def test_save_excludes_identity_fields(self, tmp_path):
        """Identity fields (app_name, workspace_dir) are excluded."""
        from agentic_cli.config import BaseSettings
        from agentic_cli.settings_persistence import IDENTITY_FIELDS

        settings = BaseSettings(app_name="my_app", workspace_dir=tmp_path)

        persistence = SettingsPersistence(app_name="my_app")
        persistence.save(settings, keys=set(BaseSettings.model_fields))

        data = _written(tmp_path, "my_app")
        assert "log_level" in data  # the other fields were saved
        for field in IDENTITY_FIELDS:
            assert field not in data, f"{field} leaked into saved JSON"

    def test_save_includes_default_values(self, tmp_path):
        """A project-settable key the user names is saved, even at its default."""
        from agentic_cli.config import BaseSettings

        settings = BaseSettings()

        persistence = SettingsPersistence(app_name="test")
        persistence.save(settings, keys={"verbose_thinking", "thinking_effort"})

        data = _written(tmp_path, "test")
        assert data == {
            "verbose_thinking": settings.verbose_thinking,
            "thinking_effort": settings.thinking_effort,
        }


class TestTrustSplitSave:
    """P0-1 follow-up: save() must agree with the allowlist-filtered loader.

    A save splits by trust: allowlisted keys → project settings.json,
    non-allowlisted (user-scoped) keys → user ~/.{app}/settings.json, so a
    /settings change to a security-relevant key survives a restart instead of
    being silently dropped by _AllowlistFilterSource.
    """

    def _persistence(self):
        return SettingsPersistence(app_name="agentic_cli")

    def test_keyed_save_keeps_project_file_allowlisted(self, isolated_fs):
        """Project file receives only allowlisted keys (and heals stale ones)."""
        from agentic_cli.config import BaseSettings
        from agentic_cli.settings_persistence import PROJECT_SETTABLE_KEYS

        # Pre-seed a stale kitchen-sink project file (pre-P0-1 artifact)
        proj_dir = isolated_fs / ".agentic_cli"
        proj_dir.mkdir()
        (proj_dir / "settings.json").write_text(
            json.dumps({"raw_llm_logging": True, "default_model": "gemini-2.5-pro"})
        )

        settings = BaseSettings(
            raw_llm_logging=True, default_model="claude-sonnet-4-6"
        )
        self._persistence().save(settings, keys={"default_model", "raw_llm_logging"})

        data = json.loads((proj_dir / "settings.json").read_text())
        assert set(data) <= PROJECT_SETTABLE_KEYS
        assert data["default_model"] == "claude-sonnet-4-6"
        assert "raw_llm_logging" not in data

    def test_keyed_save_writes_user_scoped_keys_to_user_config(self, isolated_fs):
        """Non-allowlisted keys that differ from defaults land in user config."""
        from agentic_cli.config import BaseSettings

        settings = BaseSettings(
            raw_llm_logging=True, stateful_executor_backend="local"
        )
        self._persistence().save(
            settings, keys={"raw_llm_logging", "stateful_executor_backend"}
        )

        user_file = isolated_fs / "home" / ".agentic_cli" / "settings.json"
        assert user_file.exists()
        data = json.loads(user_file.read_text())
        assert data["raw_llm_logging"] is True
        assert data["stateful_executor_backend"] == "local"

    def test_keyed_save_at_defaults_does_not_create_user_config(self, isolated_fs):
        """A user-scoped key at its default, absent from the file, writes nothing."""
        from agentic_cli.config import BaseSettings

        self._persistence().save(BaseSettings(), keys={"raw_llm_logging"})

        assert not (isolated_fs / "home" / ".agentic_cli" / "settings.json").exists()

    def test_user_config_merge_preserves_unmanaged_keys(self, isolated_fs):
        """Hand-stored secrets and unknown keys in user config survive a save."""
        from agentic_cli.config import BaseSettings

        user_dir = isolated_fs / "home" / ".agentic_cli"
        user_dir.mkdir(parents=True)
        (user_dir / "settings.json").write_text(
            json.dumps({"anthropic_api_key": "sk-stored", "domain_custom_key": 7})
        )

        settings = BaseSettings(raw_llm_logging=True)
        self._persistence().save(settings, keys={"raw_llm_logging"})

        data = json.loads((user_dir / "settings.json").read_text())
        assert data["anthropic_api_key"] == "sk-stored"
        assert data["domain_custom_key"] == 7
        assert data["raw_llm_logging"] is True

    def test_revert_to_default_removes_user_config_key(self, isolated_fs):
        """Reverting a user-scoped key to its default removes it from user config."""
        from agentic_cli.config import BaseSettings

        user_dir = isolated_fs / "home" / ".agentic_cli"
        user_dir.mkdir(parents=True)
        (user_dir / "settings.json").write_text(
            json.dumps({"raw_llm_logging": True, "domain_custom_key": 7})
        )

        settings = BaseSettings()  # loads raw_llm_logging=True from user config
        settings.update_setting("raw_llm_logging", False)  # user reverts
        self._persistence().save(settings, keys={"raw_llm_logging"})

        data = json.loads((user_dir / "settings.json").read_text())
        assert "raw_llm_logging" not in data
        assert data["domain_custom_key"] == 7  # unmanaged key untouched

    def test_round_trip_user_scoped_change_persists(self, isolated_fs):
        """THE regression: a /settings change to a non-allowlisted key must
        survive save + fresh load, without tripping the untrusted-key filter."""
        import structlog

        from agentic_cli.config import BaseSettings

        settings = BaseSettings()
        settings.update_setting("stateful_executor_backend", "local")
        self._persistence().save(settings, keys={"stateful_executor_backend"})

        with structlog.testing.capture_logs() as logs:
            reloaded = BaseSettings()

        assert reloaded.stateful_executor_backend == "local"
        assert not any(
            e.get("event") == "untrusted_project_setting_ignored" for e in logs
        )

    def test_save_result_reports_split(self, isolated_fs):
        """A save reports both paths and the user-scoped keys."""
        from agentic_cli.config import BaseSettings

        result = self._persistence().save(
            BaseSettings(raw_llm_logging=True, thinking_effort="low"),
            keys={"raw_llm_logging", "thinking_effort"},
        )

        assert result.project_path == isolated_fs / ".agentic_cli" / "settings.json"
        assert (
            result.user_path
            == isolated_fs / "home" / ".agentic_cli" / "settings.json"
        )
        assert result.project_keys == ("thinking_effort",)
        assert result.user_scoped_keys == ("raw_llm_logging",)


class TestAtomicWrite:
    """Tests for atomic settings write (C2)."""

    def test_save_uses_atomic_write(self, isolated_fs):
        """save() delegates to atomic_write_text."""
        from agentic_cli.config import BaseSettings

        settings = BaseSettings(search_backend="tavily")
        persistence = SettingsPersistence(app_name="test")

        with patch("agentic_cli.file_utils.atomic_write_text") as mock_aw:
            persistence.save(settings, keys={"search_backend"})
            mock_aw.assert_called_once()
            call_path, call_content = mock_aw.call_args[0]
            assert call_path == isolated_fs / ".test" / "settings.json"
            # Content should be valid JSON
            parsed = json.loads(call_content)
            assert parsed["search_backend"] == "tavily"

    def test_save_creates_parent_dirs(self, isolated_fs):
        """save() creates parent directories before writing."""
        from agentic_cli.config import BaseSettings

        settings = BaseSettings(search_backend="brave")
        persistence = SettingsPersistence(app_name="test")
        assert not (isolated_fs / ".test").exists()

        persistence.save(settings, keys={"search_backend"})

        data = json.loads((isolated_fs / ".test" / "settings.json").read_text())
        assert data["search_backend"] == "brave"
