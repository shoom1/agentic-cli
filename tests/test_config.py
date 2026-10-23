"""Tests for configuration module."""

import os
from pathlib import Path
from typing import get_args
from unittest.mock import patch

import pytest

from agentic_cli.config import (
    BaseSettings,
    SettingsContext,
    SettingsValidationError,
    get_context_settings,
    get_settings,
    reload_settings,
    set_context_settings,
    set_settings,
    validate_settings,
)
from agentic_cli.workflow.config import AgentConfig


class TestBaseSettings:
    """Tests for BaseSettings class."""

    def test_default_values(self, temp_workspace: Path):
        """Test default settings values."""
        with patch.dict(os.environ, {}, clear=True):
            settings = BaseSettings(workspace_dir=temp_workspace)

        assert settings.default_model is None
        assert settings.thinking_effort == "medium"
        assert settings.log_level == "warning"
        assert settings.log_format == "console"
        assert settings.app_name == "agentic_cli"
        assert settings.embedding_model == "all-MiniLM-L6-v2"
        assert settings.python_executor_timeout == 30

    def test_workspace_path_expansion(self):
        """Test that ~ is expanded in workspace_dir."""
        with patch.dict(os.environ, {}, clear=True):
            settings = BaseSettings(workspace_dir="~/test_workspace")

        assert not str(settings.workspace_dir).startswith("~")
        assert settings.workspace_dir == Path.home() / "test_workspace"

    def test_derived_paths(self, temp_workspace: Path):
        """Test derived directory paths."""
        with patch.dict(os.environ, {}, clear=True):
            settings = BaseSettings(workspace_dir=temp_workspace)

        assert settings.sessions_dir == temp_workspace / "sessions"
        assert settings.knowledge_base_dir == temp_workspace / "knowledge_base"

    @pytest.mark.parametrize("name", ["knowledge_base_documents_dir", "knowledge_base_embeddings_dir"])
    def test_knowledge_base_layout_paths_are_gone(self, temp_workspace: Path, name: str):
        """The knowledge base lays out its own folder; settings name only the folder."""
        with patch.dict(os.environ, {}, clear=True):
            settings = BaseSettings(workspace_dir=temp_workspace)

        with pytest.raises(AttributeError):
            getattr(settings, name)


class TestModelConfiguration:
    """``set_model`` (the ``/settings`` model picker's setter) checks the reference only.

    Whether the provider has a key, and whether the model exists, is decided
    when the workflow resolves its models (and rolls the change back if it
    cannot start), not by the setter.
    """

    def test_set_model_keeps_the_reference_as_given(self, settings_google_only: BaseSettings):
        settings_google_only.set_model("gemini-2.5-flash")
        assert settings_google_only.default_model == "gemini-2.5-flash"

        settings_google_only.set_model("google:gemini-2.5-pro")
        assert settings_google_only.default_model == "google:gemini-2.5-pro"

    def test_set_model_does_not_check_the_key(self, settings_google_only: BaseSettings):
        settings_google_only.set_model("anthropic:claude-sonnet-4-5")
        assert settings_google_only.default_model == "anthropic:claude-sonnet-4-5"

    @pytest.mark.parametrize("spec", ["strong", "invalid-model", "gemini", "anthropic:", ""])
    def test_set_model_rejects_a_role_or_a_malformed_reference(
        self, settings_google_only: BaseSettings, spec: str
    ):
        with pytest.raises(ValueError):
            settings_google_only.set_model(spec)
        assert settings_google_only.default_model is None

    def test_default_model_cannot_be_a_role_in_a_settings_source(self, temp_workspace: Path):
        with patch.dict(os.environ, {"AGENTIC_DEFAULT_MODEL": "fast"}, clear=True):
            with pytest.raises(ValueError, match="default_model"):
                BaseSettings(workspace_dir=temp_workspace)


class TestThinkingEffort:
    """Tests for thinking effort configuration."""

    def test_set_thinking_effort_valid(self, temp_workspace: Path):
        """Every value the field takes is accepted by the setter."""
        with patch.dict(os.environ, {}, clear=True):
            settings = BaseSettings(workspace_dir=temp_workspace)

        levels = get_args(BaseSettings.model_fields["thinking_effort"].annotation)
        assert levels == ("none", "low", "medium", "high")
        for level in levels:
            settings.set_thinking_effort(level)
            assert settings.thinking_effort == level

    def test_set_thinking_effort_invalid(self, temp_workspace: Path):
        """Test setting invalid thinking effort level."""
        with patch.dict(os.environ, {}, clear=True):
            settings = BaseSettings(workspace_dir=temp_workspace)

        with pytest.raises(ValueError, match="Invalid thinking effort"):
            settings.set_thinking_effort("invalid")

    def test_a_native_level_is_per_agent_only(self, temp_workspace: Path):
        with patch.dict(os.environ, {}, clear=True):
            settings = BaseSettings(workspace_dir=temp_workspace)

        with pytest.raises(ValueError, match="Invalid thinking effort"):
            settings.set_thinking_effort("xhigh")


class TestGlobalSettings:
    """Tests for global settings management."""

    def test_get_settings_creates_default(self):
        """Test get_settings creates default instance."""
        reload_settings()  # Clear any existing

        with patch.dict(os.environ, {}, clear=True):
            settings = get_settings()

        assert isinstance(settings, BaseSettings)

    def test_set_settings(self, temp_workspace: Path):
        """Test set_settings replaces global instance."""
        with patch.dict(os.environ, {}, clear=True):
            custom_settings = BaseSettings(
                workspace_dir=temp_workspace,
                app_name="custom_app",
            )

        set_settings(custom_settings)
        retrieved = get_settings()

        assert retrieved.app_name == "custom_app"
        assert retrieved.workspace_dir == temp_workspace

    def test_reload_settings(self, temp_workspace: Path):
        """Test reload_settings clears and recreates."""
        with patch.dict(os.environ, {}, clear=True):
            custom = BaseSettings(workspace_dir=temp_workspace, app_name="custom")
        set_settings(custom)

        with patch.dict(os.environ, {}, clear=True):
            reloaded = reload_settings()

        assert reloaded.app_name == "agentic_cli"  # Default value


class TestWorkspaceOperations:
    """Tests for workspace operations."""

    def test_export_api_keys_to_env(self, temp_workspace: Path):
        """Test API key export to environment."""
        # Use a unique env var name for testing to avoid conflicts
        test_key = "TEST_EXPORT_KEY_12345"
        original_value = os.environ.pop("GOOGLE_API_KEY", None)

        try:
            # Set the key in environment first (simulating .env file load)
            os.environ["GOOGLE_API_KEY"] = test_key

            # Create settings (it will pick up the env var)
            settings = BaseSettings(workspace_dir=temp_workspace)

            # Verify settings has the key
            assert settings.google_api_key == test_key

            # Clear env var to test export
            del os.environ["GOOGLE_API_KEY"]
            assert os.environ.get("GOOGLE_API_KEY") is None

            # Export should restore it
            settings.export_api_keys_to_env()
            assert os.environ.get("GOOGLE_API_KEY") == test_key
        finally:
            # Restore original value
            if original_value is not None:
                os.environ["GOOGLE_API_KEY"] = original_value
            else:
                os.environ.pop("GOOGLE_API_KEY", None)

    def test_export_overwrites_stale_env_key(self, temp_workspace: Path, monkeypatch):
        """Export re-asserts this instance's key over a mutated env var.

        The key fields bind only via their env alias, so settings and env
        diverge when the process env changed after this instance loaded —
        exactly what an earlier manager's export does in a two-settings
        process. The previous set-if-absent guard then silently kept the
        first manager's credentials for every later one.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "current-key")
        settings = BaseSettings(workspace_dir=temp_workspace)
        assert settings.anthropic_api_key == "current-key"

        # Another settings instance exported its key in the meantime
        monkeypatch.setenv("ANTHROPIC_API_KEY", "stale-key")

        settings.export_api_keys_to_env()

        assert os.environ["ANTHROPIC_API_KEY"] == "current-key"

    def test_export_leaves_env_when_setting_unset(
        self, temp_workspace: Path, monkeypatch, tmp_path: Path
    ):
        """A key unset in settings never clobbers an externally-set env var."""
        monkeypatch.setenv("HOME", str(tmp_path))  # no user config leakage
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        settings = BaseSettings(workspace_dir=temp_workspace)
        assert settings.google_api_key is None

        monkeypatch.setenv("GOOGLE_API_KEY", "external-key")

        settings.export_api_keys_to_env()

        assert os.environ["GOOGLE_API_KEY"] == "external-key"


class TestSettingsContext:
    """Tests for context-based settings management."""

    def test_settings_context_basic(self, temp_workspace: Path):
        """Test SettingsContext sets and clears context."""
        with patch.dict(os.environ, {}, clear=True):
            global_settings = BaseSettings(workspace_dir=temp_workspace)
            context_settings = BaseSettings(
                workspace_dir=temp_workspace,
                app_name="context_app",
            )

        set_settings(global_settings)

        # Before context, get_settings returns global
        assert get_settings().app_name == "agentic_cli"

        # Inside context, get_settings returns context settings
        with SettingsContext(context_settings):
            assert get_settings().app_name == "context_app"

        # After context, get_settings returns global again
        assert get_settings().app_name == "agentic_cli"

    def test_settings_context_nested(self, temp_workspace: Path):
        """Test nested SettingsContext works correctly."""
        with patch.dict(os.environ, {}, clear=True):
            outer_settings = BaseSettings(
                workspace_dir=temp_workspace,
                app_name="outer",
            )
            inner_settings = BaseSettings(
                workspace_dir=temp_workspace,
                app_name="inner",
            )

        with SettingsContext(outer_settings):
            assert get_settings().app_name == "outer"

            with SettingsContext(inner_settings):
                assert get_settings().app_name == "inner"

            # Back to outer after inner context exits
            assert get_settings().app_name == "outer"

    def test_set_context_settings_direct(self, temp_workspace: Path):
        """Test set_context_settings function."""
        with patch.dict(os.environ, {}, clear=True):
            context_settings = BaseSettings(
                workspace_dir=temp_workspace,
                app_name="direct_context",
            )

        # Initially no context
        assert get_context_settings() is None

        # Set context directly
        set_context_settings(context_settings)
        assert get_context_settings() == context_settings
        assert get_settings().app_name == "direct_context"

        # Clear context
        set_context_settings(None)
        assert get_context_settings() is None

    def test_context_takes_precedence_over_global(self, temp_workspace: Path):
        """Test that context settings take precedence over global."""
        with patch.dict(os.environ, {}, clear=True):
            global_settings = BaseSettings(
                workspace_dir=temp_workspace,
                app_name="global",
            )
            context_settings = BaseSettings(
                workspace_dir=temp_workspace,
                app_name="context",
            )

        set_settings(global_settings)
        set_context_settings(context_settings)

        # Context should win
        assert get_settings().app_name == "context"

        # Clear context, global should be returned
        set_context_settings(None)
        assert get_settings().app_name == "global"


class TestSettingsValidation:
    """``validate_settings``: the static check, before any network call (R11).

    Every reference parses, every endpoint referenced exists, every provider a
    model will run on has a key, and some provider has one. Whether a model
    exists, and its thinking modes, are checked when the workflow resolves its
    agents against the model catalog.
    """

    @staticmethod
    def _settings(temp_workspace: Path, env: dict | None = None, **values) -> BaseSettings:
        with patch.dict(os.environ, env or {}, clear=True):
            return BaseSettings(workspace_dir=temp_workspace, **values)

    def test_no_provider_is_an_error_naming_every_variable(self, temp_workspace: Path):
        settings = self._settings(temp_workspace)

        with pytest.raises(SettingsValidationError) as caught:
            validate_settings(settings)

        message = str(caught.value)
        for variable in ("GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                         "OPENROUTER_API_KEY", "model_endpoints"):
            assert variable in message

    @pytest.mark.parametrize("variable", ["GOOGLE_API_KEY", "ANTHROPIC_API_KEY",
                                          "OPENAI_API_KEY", "OPENROUTER_API_KEY"])
    def test_any_one_provider_key_is_enough(self, temp_workspace: Path, variable: str):
        validate_settings(self._settings(temp_workspace, {variable: "test-key"}))

    def test_an_endpoint_is_a_provider(self, temp_workspace: Path):
        settings = self._settings(
            temp_workspace,
            model_endpoints={"local-vllm": {"base_url": "http://localhost:8000/v1"}},
            default_model="local-vllm:Qwen/Qwen3-32B",
        )

        validate_settings(settings)

    LAB = {"lab": {"base_url": "http://gpu-box:8000/v1", "api_key_env": "LAB_KEY"}}

    def test_an_endpoint_s_unset_key_variable_is_an_error_naming_it(
        self, temp_workspace: Path, monkeypatch
    ):
        """Its requests would carry a placeholder key and fail with a bare 401."""
        monkeypatch.delenv("LAB_KEY", raising=False)
        settings = self._settings(temp_workspace, model_endpoints=self.LAB)
        agents = [AgentConfig(name="writer", prompt="p", model="lab:Qwen/Qwen3-32B")]

        with pytest.raises(SettingsValidationError) as caught:
            validate_settings(settings, agents)

        message = str(caught.value)
        assert "Agent 'writer'" in message and "'lab'" in message
        assert "LAB_KEY is not set" in message

    def test_an_endpoint_whose_key_variable_is_set_runs(self, temp_workspace: Path, monkeypatch):
        monkeypatch.setenv("LAB_KEY", "lab-test-key")
        settings = self._settings(temp_workspace, model_endpoints=self.LAB)
        agents = [AgentConfig(name="writer", prompt="p", model="lab:Qwen/Qwen3-32B")]

        validate_settings(settings, agents)

    def test_an_unused_endpoint_s_unset_key_variable_is_no_error(
        self, temp_workspace: Path, monkeypatch
    ):
        monkeypatch.delenv("LAB_KEY", raising=False)
        settings = self._settings(
            temp_workspace, {"GOOGLE_API_KEY": "test-key"}, model_endpoints=self.LAB
        )

        validate_settings(settings)

    def test_a_valid_model(self, temp_workspace: Path):
        settings = self._settings(
            temp_workspace, {"GOOGLE_API_KEY": "test-key"}, default_model="gemini-2.5-pro"
        )

        assert validate_settings(settings) is None

    def test_an_unknown_model_id_is_left_to_the_catalog(self, temp_workspace: Path):
        """Existence needs a listing; the static check accepts a well-formed id."""
        settings = self._settings(
            temp_workspace, {"GOOGLE_API_KEY": "test-key"}, default_model="gemini-9-ultra"
        )

        validate_settings(settings)

    def test_the_default_model_needs_its_provider_s_key(self, temp_workspace: Path):
        settings = self._settings(
            temp_workspace, {"GOOGLE_API_KEY": "test-key"}, default_model="claude-sonnet-4"
        )

        with pytest.raises(SettingsValidationError, match="default_model.*set ANTHROPIC_API_KEY"):
            validate_settings(settings)

    def test_an_agent_model_needs_its_provider_s_key(self, temp_workspace: Path):
        """A per-agent override fails as hard as a bad default, only later."""
        settings = self._settings(temp_workspace, {"GOOGLE_API_KEY": "test-key"})
        agents = [AgentConfig(name="writer", prompt="p", model="claude-sonnet-4-5")]

        with pytest.raises(SettingsValidationError, match="Agent 'writer'.*ANTHROPIC_API_KEY"):
            validate_settings(settings, agents)

        validate_settings(settings, [AgentConfig(name="w", prompt="p", model="gemini-2.5-pro")])

    def test_a_role_an_agent_uses_needs_its_provider_s_key(self, temp_workspace: Path):
        settings = self._settings(
            temp_workspace, {"GOOGLE_API_KEY": "test-key"},
            model_roles={"strong": "anthropic:claude-opus-4-7"},
        )
        agents = [AgentConfig(name="critic", prompt="p", model="strong")]

        with pytest.raises(SettingsValidationError, match="Agent 'critic'.*ANTHROPIC_API_KEY"):
            validate_settings(settings, agents)
        # A mapping no agent uses costs nothing.
        validate_settings(settings, [AgentConfig(name="w", prompt="p")])

    def test_a_malformed_agent_model_is_an_error(self, temp_workspace: Path):
        settings = self._settings(temp_workspace, {"GOOGLE_API_KEY": "test-key"})
        agents = [AgentConfig(name="writer", prompt="p", model="gemini")]

        with pytest.raises(SettingsValidationError, match="Agent 'writer'.*'gemini'"):
            validate_settings(settings, agents)

    def test_an_undefined_endpoint_is_an_error_naming_the_setting(self, temp_workspace: Path):
        settings = self._settings(
            temp_workspace, {"GOOGLE_API_KEY": "test-key"},
            model_roles={"fast": "local-vllm:qwen"}, pinned_models=["other-box:llama"],
        )
        agents = [AgentConfig(name="writer", prompt="p", model="third-box:mistral")]

        with pytest.raises(SettingsValidationError) as caught:
            validate_settings(settings, agents)

        message = str(caught.value)
        assert "model_roles['fast']" in message and "'local-vllm'" in message
        assert "pinned_models[0]" in message and "'other-box'" in message
        assert "Agent 'writer'" in message and "'third-box'" in message

    def test_every_problem_is_reported_at_once(self, temp_workspace: Path):
        settings = self._settings(temp_workspace, default_model="claude-sonnet-4")
        agents = [AgentConfig(name="writer", prompt="p", model="gpt-5.5")]

        with pytest.raises(SettingsValidationError) as caught:
            validate_settings(settings, agents)

        lines = str(caught.value).splitlines()
        assert len(lines) == 3  # no provider, default_model, the writer

    def test_returns_none_and_keeps_its_public_signature(self, temp_workspace: Path):
        import inspect

        settings = self._settings(temp_workspace, {"GOOGLE_API_KEY": "test-key"})

        assert validate_settings(settings, []) is None
        assert list(inspect.signature(validate_settings).parameters) == [
            "settings", "agent_configs",
        ]


class TestSettingsJsonLoading:
    """Tests for JSON config loading with constructor app_name."""

    def test_init_kwargs_app_name_used_for_json_path(self, tmp_path):
        """Settings from JSON are loaded when app_name is passed via constructor."""
        import json

        # Create a project-level JSON config for a custom app
        config_dir = tmp_path / ".my_app"
        config_dir.mkdir()
        config_file = config_dir / "settings.json"
        config_file.write_text(json.dumps({
            "thinking_effort": "high",
            "log_level": "debug",
        }))

        # Create settings with constructor app_name, from the directory with the config
        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            with patch.dict(os.environ, {}, clear=True):
                settings = BaseSettings(
                    app_name="my_app",
                    workspace_dir=tmp_path / "workspace",
                )
        finally:
            os.chdir(original_cwd)

        # Values from JSON should be loaded
        assert settings.thinking_effort == "high"
        assert settings.log_level == "debug"

    def test_a_settings_file_naming_removed_settings_still_loads(self, tmp_path):
        """webfetch_model and sandbox_packages were removed; a file or env var
        that still sets them is ignored, not an error."""
        import json

        home = tmp_path / "home"
        (home / ".my_app").mkdir(parents=True)
        (home / ".my_app" / "settings.json").write_text(json.dumps({
            "webfetch_model": "gemini-2.5-flash",
            "sandbox_packages": ["numpy"],
            "log_level": "debug",
        }))

        env = {"HOME": str(home), "AGENTIC_WEBFETCH_MODEL": "gemini-2.5-flash"}
        with patch.dict(os.environ, env, clear=True):
            settings = BaseSettings(app_name="my_app", workspace_dir=tmp_path / "workspace")

        assert settings.log_level == "debug"
        assert not hasattr(settings, "webfetch_model")
        assert not hasattr(settings, "sandbox_packages")


def test_package_exports_new_modules():
    """Test that new modules are exported from package."""
    # Memory
    from agentic_cli.memory import MemoryItem, MemoryStore

    # Tools
    from agentic_cli.tools import shell_executor, read_file, write_file, diff_compare

    # All imports should succeed
    assert MemoryStore is not None
    assert MemoryItem is not None
    assert shell_executor is not None


def test_a_06_project_file_prints_nothing_at_startup(tmp_path):
    """0.6.0 and 0.6.1's /settings saved the removed context_window_* counts.

    Apps build their settings before ``BaseCLIApp`` configures logging, when
    structlog still prints every level, a debug log included, above the UI.
    So a key that names no setting is dropped without a log. A fresh
    interpreter: other tests configure structlog for the whole process.
    """
    import json
    import subprocess
    import sys

    project = tmp_path / "project"
    (project / ".agentic_cli").mkdir(parents=True)
    (project / ".agentic_cli" / "settings.json").write_text(json.dumps({
        "context_window_trigger_tokens": 100_000, "context_window_target_tokens": 80_000,
    }))
    (tmp_path / "home").mkdir()
    env = {**os.environ, "HOME": str(tmp_path / "home")}
    done = subprocess.run(
        [sys.executable, "-c", "from agentic_cli.config import BaseSettings; BaseSettings()"],
        cwd=project, env=env, capture_output=True, text=True, check=True,
    )
    assert done.stdout == ""
    assert done.stderr == ""
