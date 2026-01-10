"""Tests for configuration module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from agentic_cli.config import (
    ALL_MODELS,
    ANTHROPIC_MODELS,
    GOOGLE_MODELS,
    THINKING_EFFORT_LEVELS,
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
        assert settings.artifacts_dir == temp_workspace / "workspace"
        assert settings.knowledge_base_dir == temp_workspace / "knowledge_base"
        assert settings.templates_dir == temp_workspace / "templates"
        assert settings.reports_dir == temp_workspace / "reports"


class TestAPIKeyManagement:
    """Tests for API key detection and management."""

    def test_no_api_keys(self, settings_no_keys: BaseSettings):
        """Test behavior when no API keys are set."""
        assert not settings_no_keys.has_google_key
        assert not settings_no_keys.has_anthropic_key
        assert not settings_no_keys.has_any_api_key

        with pytest.raises(RuntimeError, match="No API keys found"):
            settings_no_keys.get_model()

    def test_google_key_only(self, settings_google_only: BaseSettings):
        """Test behavior with only Google API key."""
        assert settings_google_only.has_google_key
        assert not settings_google_only.has_anthropic_key
        assert settings_google_only.has_any_api_key

        model = settings_google_only.get_model()
        assert model == settings_google_only.default_model_google

    def test_anthropic_key_only(self, settings_anthropic_only: BaseSettings):
        """Test behavior with only Anthropic API key."""
        assert not settings_anthropic_only.has_google_key
        assert settings_anthropic_only.has_anthropic_key
        assert settings_anthropic_only.has_any_api_key

        model = settings_anthropic_only.get_model()
        assert model == settings_anthropic_only.default_model_anthropic

    def test_both_keys(self, settings_both_keys: BaseSettings):
        """Test behavior with both API keys - Google should be preferred."""
        assert settings_both_keys.has_google_key
        assert settings_both_keys.has_anthropic_key
        assert settings_both_keys.has_any_api_key

        # Google is preferred when both are available
        model = settings_both_keys.get_model()
        assert model == settings_both_keys.default_model_google

    def test_get_available_models(self, settings_both_keys: BaseSettings):
        """Test getting available models based on API keys."""
        models = settings_both_keys.get_available_models()

        for google_model in GOOGLE_MODELS:
            assert google_model in models
        for anthropic_model in ANTHROPIC_MODELS:
            assert anthropic_model in models

    def test_api_key_status(self, settings_both_keys: BaseSettings):
        """Test API key status report."""
        status = settings_both_keys.get_api_key_status()

        assert status == {"google": True, "anthropic": True}


class TestModelConfiguration:
    """Tests for model configuration."""

    def test_set_model_valid(self, settings_google_only: BaseSettings):
        """Test setting a valid model."""
        settings_google_only.set_model("gemini-2.5-pro")
        assert settings_google_only.default_model == "gemini-2.5-pro"
        assert settings_google_only.get_model() == "gemini-2.5-pro"

    def test_set_model_invalid(self, settings_google_only: BaseSettings):
        """Test setting an invalid model raises error."""
        with pytest.raises(ValueError, match="not available"):
            settings_google_only.set_model("invalid-model")

    def test_set_model_unavailable_provider(self, settings_google_only: BaseSettings):
        """Test setting a model from unavailable provider."""
        with pytest.raises(ValueError, match="not available"):
            settings_google_only.set_model("claude-sonnet-4")

    def test_is_google_model(self, settings_google_only: BaseSettings):
        """Test Google model detection."""
        assert settings_google_only.is_google_model("gemini-2.5-pro")
        assert settings_google_only.is_google_model("gemini-3-flash-preview")
        assert not settings_google_only.is_google_model("claude-sonnet-4")

    def test_is_anthropic_model(self, settings_anthropic_only: BaseSettings):
        """Test Anthropic model detection."""
        assert settings_anthropic_only.is_anthropic_model("claude-sonnet-4")
        assert settings_anthropic_only.is_anthropic_model("claude-opus-4")
        assert not settings_anthropic_only.is_anthropic_model("gemini-2.5-pro")


class TestThinkingEffort:
    """Tests for thinking effort configuration."""

    def test_set_thinking_effort_valid(self, temp_workspace: Path):
        """Test setting valid thinking effort levels."""
        with patch.dict(os.environ, {}, clear=True):
            settings = BaseSettings(workspace_dir=temp_workspace)

        for level in THINKING_EFFORT_LEVELS:
            settings.set_thinking_effort(level)
            assert settings.thinking_effort == level

    def test_set_thinking_effort_invalid(self, temp_workspace: Path):
        """Test setting invalid thinking effort level."""
        with patch.dict(os.environ, {}, clear=True):
            settings = BaseSettings(workspace_dir=temp_workspace)

        with pytest.raises(ValueError, match="Invalid thinking effort"):
            settings.set_thinking_effort("invalid")

    def test_supports_thinking_effort(self, settings_both_keys: BaseSettings):
        """Test thinking effort support detection."""
        # Anthropic models support thinking
        assert settings_both_keys.supports_thinking_effort("claude-sonnet-4")
        assert settings_both_keys.supports_thinking_effort("claude-opus-4")

        # Gemini 2.5+ supports thinking
        assert settings_both_keys.supports_thinking_effort("gemini-2.5-pro")
        assert settings_both_keys.supports_thinking_effort("gemini-3-flash-preview")


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

    def test_ensure_workspace_exists(self, temp_workspace: Path):
        """Test workspace directory creation."""
        workspace = temp_workspace / "new_workspace"

        with patch.dict(os.environ, {}, clear=True):
            settings = BaseSettings(workspace_dir=workspace)

        assert not workspace.exists()
        settings.ensure_workspace_exists()
        assert workspace.exists()

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


class TestModelConstants:
    """Tests for model constant lists."""

    def test_all_models_includes_all_providers(self):
        """Test ALL_MODELS includes both providers."""
        for model in GOOGLE_MODELS:
            assert model in ALL_MODELS
        for model in ANTHROPIC_MODELS:
            assert model in ALL_MODELS

    def test_thinking_effort_levels(self):
        """Test thinking effort level constants."""
        assert "none" in THINKING_EFFORT_LEVELS
        assert "low" in THINKING_EFFORT_LEVELS
        assert "medium" in THINKING_EFFORT_LEVELS
        assert "high" in THINKING_EFFORT_LEVELS


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
    """Tests for settings validation."""

    def test_validate_no_api_keys(self, temp_workspace: Path):
        """Test validation fails with no API keys."""
        with patch.dict(os.environ, {}, clear=True):
            settings = BaseSettings(workspace_dir=temp_workspace)

        with pytest.raises(SettingsValidationError, match="No API keys"):
            validate_settings(settings)

    def test_validate_with_api_key(self, temp_workspace: Path):
        """Test validation passes with API key."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True):
            settings = BaseSettings(workspace_dir=temp_workspace)

        # Should not raise
        validate_settings(settings)

    def test_validate_invalid_model(self, temp_workspace: Path):
        """Test validation fails with invalid model."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True):
            settings = BaseSettings(
                workspace_dir=temp_workspace,
                default_model="invalid-model",
            )

        with pytest.raises(SettingsValidationError, match="not available"):
            validate_settings(settings)

    def test_validate_valid_model(self, temp_workspace: Path):
        """Test validation passes with valid model."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True):
            settings = BaseSettings(
                workspace_dir=temp_workspace,
                default_model="gemini-2.5-pro",
            )

        # Should not raise
        validate_settings(settings)
