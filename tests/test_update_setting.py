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
