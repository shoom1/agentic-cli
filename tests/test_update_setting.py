"""Tests for BaseSettings.update_setting()."""

import pytest


class TestUpdateSetting:
    def test_update_regular_setting(self, mock_context):
        settings = mock_context.settings
        original = settings.verbose_thinking
        settings.update_setting("verbose_thinking", not original)
        assert settings.verbose_thinking is not original

    def test_update_model_delegates_to_setter(self, mock_context):
        settings = mock_context.settings
        with pytest.raises((ValueError, Exception)):
            settings.update_setting("model", "nonexistent-model-xyz")

    def test_update_thinking_effort(self, mock_context):
        settings = mock_context.settings
        settings.update_setting("thinking_effort", "high")
        assert settings.thinking_effort == "high"

    def test_rejects_invalid_type(self, mock_context):
        """An out-of-type value must be rejected, not written to the instance."""
        settings = mock_context.settings
        original = settings.context_window_trigger_tokens
        with pytest.raises(ValueError):
            settings.update_setting("context_window_trigger_tokens", "not-an-int")
        # The garbage value must not have landed on the instance.
        assert settings.context_window_trigger_tokens == original

    def test_rejects_unknown_key(self, mock_context):
        """Unknown setting keys must raise rather than silently attach."""
        settings = mock_context.settings
        with pytest.raises(ValueError):
            settings.update_setting("totally_unknown_key_xyz", 1)
        assert not hasattr(settings, "totally_unknown_key_xyz")

    def test_coerces_valid_value(self, mock_context):
        """A valid (coercible) value is accepted and type-coerced."""
        settings = mock_context.settings
        settings.update_setting("verbose_thinking", "true")  # str -> bool
        assert settings.verbose_thinking is True
