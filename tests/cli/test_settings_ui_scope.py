"""Guard: the /settings dialog exposes only project-scoped (allowlisted) settings.

User-scoped keys (not in PROJECT_SETTABLE_KEYS) must never render in the
dialog, regardless of what a domain app returns from get_ui_setting_keys() —
otherwise a /settings edit would persist to the user ~/.{app}/settings.json
and apply across all projects.
"""

from __future__ import annotations

import pytest
import structlog

from agentic_cli.cli.app import BaseCLIApp

EXCLUDED_EVENT = "user_scoped_setting_excluded_from_ui"


def _make_app(settings, keys: list[str]) -> BaseCLIApp:
    class _App(BaseCLIApp):
        def get_ui_setting_keys(self) -> list[str]:
            return keys

    app = _App.__new__(_App)
    app._settings = settings
    return app


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    """Hermetic BaseSettings: temp cwd/HOME so no real config files load."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    from agentic_cli.config import BaseSettings

    # A key so get_available_models() is non-empty and "model" can render
    return BaseSettings(google_api_key="test-key")


def _item_keys(items) -> list[str]:
    return [item.key for item in items]


class TestSettingsUiScopeGuard:
    def test_user_scoped_key_is_excluded_with_warning(self, isolated_settings):
        """A non-allowlisted field never renders; the exclusion is logged."""
        app = _make_app(isolated_settings, ["thinking_effort", "raw_llm_logging"])

        with structlog.testing.capture_logs() as logs:
            items = app._build_ui_items()

        assert _item_keys(items) == ["thinking_effort"]
        assert any(
            e.get("event") == EXCLUDED_EVENT and e.get("key") == "raw_llm_logging"
            for e in logs
        )

    def test_allowlisted_keys_render_without_warning(self, isolated_settings):
        app = _make_app(isolated_settings, ["thinking_effort", "verbose_thinking"])

        with structlog.testing.capture_logs() as logs:
            items = app._build_ui_items()

        assert set(_item_keys(items)) == {"thinking_effort", "verbose_thinking"}
        assert not any(e.get("event") == EXCLUDED_EVENT for e in logs)

    def test_model_synthetic_key_passes_guard(self, isolated_settings):
        """"model" is not a field but writes default_model (allowlisted)."""
        app = _make_app(isolated_settings, ["model"])

        with structlog.testing.capture_logs() as logs:
            items = app._build_ui_items()

        assert _item_keys(items) == ["model"]
        assert not any(e.get("event") == EXCLUDED_EVENT for e in logs)

    def test_dangling_nonexistent_key_is_excluded_with_warning(
        self, isolated_settings
    ):
        """A key for a removed field (e.g. airesearcher's log_activity) now
        warns instead of being silently skipped."""
        app = _make_app(isolated_settings, ["log_activity"])

        with structlog.testing.capture_logs() as logs:
            items = app._build_ui_items()

        assert items == []
        assert any(
            e.get("event") == EXCLUDED_EVENT and e.get("key") == "log_activity"
            for e in logs
        )
