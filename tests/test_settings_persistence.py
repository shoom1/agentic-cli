"""Tests for settings persistence — secret field exclusion and atomic writes."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agentic_cli.settings_persistence import SECRET_FIELDS, SettingsPersistence


class TestSecretFields:
    """Tests for SECRET_FIELDS completeness (C1)."""

    def test_secret_fields_excludes_all_api_keys(self):
        """All API key fields in BaseSettings are in SECRET_FIELDS."""
        expected = {
            "google_api_key",
            "anthropic_api_key",
            "serper_api_key",
            "tavily_api_key",
            "brave_api_key",
        }
        assert SECRET_FIELDS == expected

    def test_save_excludes_secrets(self, tmp_path):
        """API key values are never written to the JSON file."""
        from agentic_cli.config import BaseSettings

        settings = BaseSettings(
            google_api_key="secret-google",
            anthropic_api_key="secret-anthropic",
            serper_api_key="secret-serper",
            tavily_api_key="secret-tavily",
            brave_api_key="secret-brave",
            search_backend="tavily",  # non-secret, non-default
        )

        persistence = SettingsPersistence(app_name="test")
        out = tmp_path / "settings.json"
        persistence.save(settings, exclude_defaults=False, path=out)

        data = json.loads(out.read_text())
        for secret in SECRET_FIELDS:
            assert secret not in data, f"{secret} leaked into saved JSON"

    def test_save_includes_non_secret_fields(self, tmp_path):
        """Non-secret settings are saved correctly."""
        from agentic_cli.config import BaseSettings

        settings = BaseSettings(search_backend="brave")

        persistence = SettingsPersistence(app_name="test")
        out = tmp_path / "settings.json"
        persistence.save(settings, exclude_defaults=True, path=out)

        data = json.loads(out.read_text())
        assert data["search_backend"] == "brave"


class TestAtomicWrite:
    """Tests for atomic settings write (C2)."""

    def test_save_uses_atomic_write(self, tmp_path):
        """save() delegates to atomic_write_text."""
        from agentic_cli.config import BaseSettings

        settings = BaseSettings(search_backend="tavily")
        persistence = SettingsPersistence(app_name="test")
        out = tmp_path / "settings.json"

        with patch("agentic_cli.persistence._utils.atomic_write_text") as mock_aw:
            persistence.save(settings, exclude_defaults=True, path=out)
            mock_aw.assert_called_once()
            call_path, call_content = mock_aw.call_args[0]
            assert call_path == out
            # Content should be valid JSON
            parsed = json.loads(call_content)
            assert parsed["search_backend"] == "tavily"

    def test_save_creates_parent_dirs(self, tmp_path):
        """save() creates parent directories before writing."""
        from agentic_cli.config import BaseSettings

        settings = BaseSettings(search_backend="brave")
        persistence = SettingsPersistence(app_name="test")
        nested = tmp_path / "a" / "b" / "settings.json"

        persistence.save(settings, exclude_defaults=True, path=nested)

        assert nested.parent.exists()
        data = json.loads(nested.read_text())
        assert data["search_backend"] == "brave"
