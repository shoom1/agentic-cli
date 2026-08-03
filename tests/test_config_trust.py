"""P0-1 config trust-model tests.

A project ``./.{app}/settings.json`` (and a cwd-relative ``.env``) may set only
non-security allowlisted keys; sensitive keys are dropped with a warning. Real
environment variables and user ``~/.{app}/settings.json`` stay trusted.
"""

import json
from pathlib import Path


def _write_project_settings(root: Path, app: str, data: dict) -> None:
    d = root / f".{app}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "settings.json").write_text(json.dumps(data))


class TestProjectSettingsAllowlist:
    def test_allowlisted_key_is_applied(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("AGENTIC_DEFAULT_MODEL", raising=False)
        _write_project_settings(tmp_path, "agentic_cli", {"default_model": "claude-sonnet-4-6"})
        from agentic_cli.config import BaseSettings
        assert BaseSettings().default_model == "claude-sonnet-4-6"

    def test_sensitive_keys_are_dropped(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        for var in ("AGENTIC_STATEFUL_EXECUTOR_BACKEND", "AGENTIC_RAW_LLM_LOGGING"):
            monkeypatch.delenv(var, raising=False)
        _write_project_settings(tmp_path, "agentic_cli", {
            "stateful_executor_backend": "local",
            "raw_llm_logging": True,
            "workspace_dir": "/tmp/evil",
            "sandbox_data_mounts": ["/etc:etc"],
            "default_model": "kept",
        })
        from agentic_cli.config import BaseSettings
        s = BaseSettings()
        # sensitive → dropped (defaults preserved)
        assert s.stateful_executor_backend == "none"
        assert s.raw_llm_logging is False
        assert s.sandbox_data_mounts == []
        assert str(s.workspace_dir) != "/tmp/evil"
        # benign → applied
        assert s.default_model == "kept"

    def test_user_config_sensitive_key_is_trusted(self, tmp_path, monkeypatch):
        proj = tmp_path / "proj"; proj.mkdir()
        monkeypatch.chdir(proj)  # cwd has no project settings.json
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("AGENTIC_RAW_LLM_LOGGING", raising=False)
        ud = tmp_path / "home" / ".agentic_cli"
        ud.mkdir(parents=True)
        (ud / "settings.json").write_text(json.dumps({"raw_llm_logging": True}))
        from agentic_cli.config import BaseSettings
        assert BaseSettings().raw_llm_logging is True

    def test_real_env_var_sensitive_key_is_trusted(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("AGENTIC_STATEFUL_EXECUTOR_BACKEND", "local")
        from agentic_cli.config import BaseSettings
        assert BaseSettings().stateful_executor_backend == "local"

    def test_dropped_key_logs_warning(self, tmp_path, monkeypatch):
        import structlog
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        _write_project_settings(tmp_path, "agentic_cli", {"raw_llm_logging": True})
        from agentic_cli.config import BaseSettings
        with structlog.testing.capture_logs() as logs:
            BaseSettings()
        assert any(
            e.get("event") == "untrusted_project_setting_ignored"
            and e.get("key") == "raw_llm_logging"
            for e in logs
        )


class TestCwdDotenvFiltering:
    def _subclass_with_env_file(self, env_file):
        from agentic_cli.config import BaseSettings
        from pydantic_settings import SettingsConfigDict

        class _DomainSettings(BaseSettings):
            model_config = SettingsConfigDict(
                env_prefix="AGENTIC_",
                env_file=env_file,
                env_file_encoding="utf-8",
                env_nested_delimiter="__",
                extra="ignore",
            )

        return _DomainSettings

    def test_cwd_relative_env_drops_sensitive_key(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        for v in ("AGENTIC_RAW_LLM_LOGGING", "AGENTIC_DEFAULT_MODEL"):
            monkeypatch.delenv(v, raising=False)
        (tmp_path / ".env").write_text(
            "AGENTIC_RAW_LLM_LOGGING=true\nAGENTIC_DEFAULT_MODEL=envmodel\n"
        )
        s = self._subclass_with_env_file(".env")()
        assert s.raw_llm_logging is False       # sensitive dropped
        assert s.default_model == "envmodel"     # benign kept

    def test_absolute_env_file_is_trusted(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("AGENTIC_RAW_LLM_LOGGING", raising=False)
        abs_env = tmp_path / "user.env"
        abs_env.write_text("AGENTIC_RAW_LLM_LOGGING=true\n")
        s = self._subclass_with_env_file(str(abs_env))()
        assert s.raw_llm_logging is True         # absolute env_file trusted

    def test_list_env_file_is_trusted(self, tmp_path, monkeypatch):
        # A list/tuple env_file must stay trusted (unfiltered) — only a single
        # cwd-relative env_file is filtered.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("AGENTIC_RAW_LLM_LOGGING", raising=False)
        abs_env = tmp_path / "user.env"
        abs_env.write_text("AGENTIC_RAW_LLM_LOGGING=true\n")
        s = self._subclass_with_env_file([str(abs_env)])()
        assert s.raw_llm_logging is True   # list env_file trusted

    def test_tilde_env_file_is_trusted(self, tmp_path, monkeypatch):
        # A "~"-prefixed env_file is a user-level path (pydantic expands ~ when
        # reading it), so it must stay TRUSTED (unfiltered).
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.delenv("AGENTIC_RAW_LLM_LOGGING", raising=False)
        (fake_home / "user.env").write_text("AGENTIC_RAW_LLM_LOGGING=true\n")
        s = self._subclass_with_env_file("~/user.env")()
        assert s.raw_llm_logging is True   # tilde (user-level) env_file trusted

    def test_list_env_file_with_cwd_relative_entry_is_filtered(self, tmp_path, monkeypatch):
        # If ANY entry in a list env_file is cwd-relative, the whole dotenv
        # source is filtered (fail-safe over-filter) — a repo could otherwise
        # ship ./.env alongside a trusted absolute file and stay unfiltered.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("AGENTIC_RAW_LLM_LOGGING", raising=False)
        abs_env = tmp_path / "abs.env"
        abs_env.write_text("")  # trusted absolute entry (empty)
        (tmp_path / ".env").write_text("AGENTIC_RAW_LLM_LOGGING=true\n")  # cwd-relative
        s = self._subclass_with_env_file([str(abs_env), ".env"])()
        assert s.raw_llm_logging is False   # cwd-relative entry → whole source filtered


class TestCredentialInputSurface:
    """Credential fields accept their field name *and* the provider env name.

    The bare ``validation_alias`` bound only the env var, so a programmatic
    ``BaseSettings(google_api_key=...)`` was silently dropped by
    ``extra="ignore"``. Widening the alias must not widen the P0-1 trust
    boundary: an untrusted project file still cannot inject a key.
    """

    def _clean_env(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        for var in ("GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "TAVILY_API_KEY", "BRAVE_API_KEY"):
            monkeypatch.delenv(var, raising=False)

    def test_constructor_field_name_is_retained(self, tmp_path, monkeypatch):
        self._clean_env(monkeypatch, tmp_path)
        from agentic_cli.config import BaseSettings

        settings = BaseSettings(
            google_api_key="ctor-google", anthropic_api_key="ctor-anthropic"
        )
        assert settings.google_api_key == "ctor-google"
        assert settings.anthropic_api_key == "ctor-anthropic"
        assert settings.has_any_api_key is True

    def test_environment_variable_still_binds(self, tmp_path, monkeypatch):
        self._clean_env(monkeypatch, tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        from agentic_cli.config import BaseSettings

        assert BaseSettings().anthropic_api_key == "env-key"

    def test_constructor_beats_environment(self, tmp_path, monkeypatch):
        self._clean_env(monkeypatch, tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        from agentic_cli.config import BaseSettings

        assert BaseSettings(anthropic_api_key="ctor-key").anthropic_api_key == "ctor-key"

    def test_secrets_stay_out_of_repr(self, tmp_path, monkeypatch):
        self._clean_env(monkeypatch, tmp_path)
        from agentic_cli.config import BaseSettings

        settings = BaseSettings(google_api_key="super-secret")
        assert "super-secret" not in repr(settings)
        assert "super-secret" not in str(settings)

    def test_misspelled_credential_kwarg_raises(self, tmp_path, monkeypatch):
        self._clean_env(monkeypatch, tmp_path)
        import pytest

        from agentic_cli.config import BaseSettings

        with pytest.raises(ValueError, match="Unknown credential setting"):
            BaseSettings(anthropic_apikey="typo")

    def test_pydantic_settings_own_kwargs_are_not_mistaken_for_credentials(
        self, tmp_path, monkeypatch
    ):
        """``_secrets_dir`` matches the credential shape but is a library kwarg."""
        self._clean_env(monkeypatch, tmp_path)
        from agentic_cli.config import BaseSettings

        secrets_dir = tmp_path / "secrets"
        secrets_dir.mkdir()
        BaseSettings(_secrets_dir=str(secrets_dir))  # must not raise

    def test_unknown_non_credential_kwarg_still_ignored(self, tmp_path, monkeypatch):
        """Only credential-shaped keys are strict; config files stay permissive."""
        self._clean_env(monkeypatch, tmp_path)
        from agentic_cli.config import BaseSettings

        BaseSettings(some_future_option=True)  # must not raise

    def test_project_settings_json_still_cannot_inject_a_key(self, tmp_path, monkeypatch):
        self._clean_env(monkeypatch, tmp_path)
        _write_project_settings(
            tmp_path,
            "agentic_cli",
            {"google_api_key": "from-untrusted-repo", "GOOGLE_API_KEY": "also-untrusted"},
        )
        from agentic_cli.config import BaseSettings

        assert BaseSettings().google_api_key is None

    def test_cwd_dotenv_still_cannot_inject_a_key(self, tmp_path, monkeypatch):
        self._clean_env(monkeypatch, tmp_path)
        (tmp_path / ".env").write_text("GOOGLE_API_KEY=from-untrusted-repo\n")
        from agentic_cli.config import BaseSettings

        class _DomainSettings(BaseSettings):
            model_config = {**BaseSettings.model_config, "env_file": ".env"}

        assert _DomainSettings().google_api_key is None
