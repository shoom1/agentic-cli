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
