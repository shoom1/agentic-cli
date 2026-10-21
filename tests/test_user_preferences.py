"""``ui_theme`` and ``dialog_placement``: the user's preferences, never a project's.

How the CLI looks is the user's choice in every project, so both settings are
user-scoped, like every key outside ``PROJECT_SETTABLE_KEYS``: the user file
``~/.{app}/settings.json`` and the environment set them, and a project file or
a cwd ``.env``, which a cloned repository can ship, is dropped with the P0-1
warning. ``USER_PREFERENCE_KEYS`` names them, so that an app may show them in
``/settings`` (``tests/cli/test_settings_ui_scope.py``). A save writes them to
the user file, and one set back to its default leaves it, so the next start
follows the default again.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import structlog
from pydantic import ValidationError

from agentic_cli.config import BaseSettings
from agentic_cli.settings_persistence import (
    PROJECT_SETTABLE_KEYS,
    USER_PREFERENCE_KEYS,
    SettingsPersistence,
)

APP = "agentic_cli"
CHOSEN = {"ui_theme": "light", "dialog_placement": "inline"}


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temp home and project, and neither setting in the environment."""
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path / "project")
    for variable in ("AGENTIC_UI_THEME", "AGENTIC_DIALOG_PLACEMENT", "AGENTIC_THINKING_EFFORT"):
        monkeypatch.delenv(variable, raising=False)
    return tmp_path


def _project_file(root: Path) -> Path:
    return root / "project" / f".{APP}" / "settings.json"


def _user_file(root: Path) -> Path:
    return root / "home" / f".{APP}" / "settings.json"


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _chosen(settings: BaseSettings) -> dict[str, str]:
    return {"ui_theme": settings.ui_theme, "dialog_placement": settings.dialog_placement}


def _dropped(logs: list[dict], source: str) -> set[str]:
    """The keys the P0-1 filter dropped from ``source``, with its warning."""
    return {
        entry["key"]
        for entry in logs
        if entry.get("event") == "untrusted_project_setting_ignored"
        and entry.get("source") == source
        and entry.get("log_level") == "warning"
    }


# --- the settings ----------------------------------------------------------------------


def test_the_defaults_are_today_s_look():
    assert _chosen(BaseSettings()) == {"ui_theme": "dark", "dialog_placement": "box"}


def test_they_are_user_preferences_and_never_project_settings():
    assert USER_PREFERENCE_KEYS == {"ui_theme", "dialog_placement"}
    assert USER_PREFERENCE_KEYS.isdisjoint(PROJECT_SETTABLE_KEYS)
    assert USER_PREFERENCE_KEYS <= set(BaseSettings.model_fields)


@pytest.mark.parametrize(
    ("key", "value"), [("ui_theme", "solarized"), ("dialog_placement", "floating")]
)
def test_a_bad_value_in_the_user_file_fails_at_load(isolated, key, value):
    _write(_user_file(isolated), {key: value})

    with pytest.raises(ValidationError, match=key):
        BaseSettings()


# --- where they come from -------------------------------------------------------------


def test_the_user_file_sets_them(isolated):
    _write(_user_file(isolated), CHOSEN)

    assert _chosen(BaseSettings()) == CHOSEN


def test_the_environment_sets_them(monkeypatch):
    monkeypatch.setenv("AGENTIC_UI_THEME", "mono")
    monkeypatch.setenv("AGENTIC_DIALOG_PLACEMENT", "inline")

    assert _chosen(BaseSettings()) == {"ui_theme": "mono", "dialog_placement": "inline"}


def test_a_project_file_cannot_set_them(isolated):
    _write(_project_file(isolated), {**CHOSEN, "thinking_effort": "high"})

    with structlog.testing.capture_logs() as logs:
        settings = BaseSettings()

    assert _chosen(settings) == {"ui_theme": "dark", "dialog_placement": "box"}
    assert settings.thinking_effort == "high"  # the file was read
    assert _dropped(logs, "project settings.json") == {"ui_theme", "dialog_placement"}


def test_a_cwd_dotenv_cannot_set_them(isolated):
    (isolated / "project" / ".env").write_text(
        "AGENTIC_UI_THEME=light\nAGENTIC_DIALOG_PLACEMENT=inline\nAGENTIC_THINKING_EFFORT=high\n"
    )

    class _Settings(BaseSettings):
        model_config = {**BaseSettings.model_config, "env_file": ".env"}

    with structlog.testing.capture_logs() as logs:
        settings = _Settings()

    assert _chosen(settings) == {"ui_theme": "dark", "dialog_placement": "box"}
    assert settings.thinking_effort == "high"  # the file was read
    assert _dropped(logs, "cwd .env") == {"ui_theme", "dialog_placement"}


# --- saving them ------------------------------------------------------------------------


def test_a_save_writes_them_to_the_user_file(isolated):
    settings = BaseSettings()
    for key, value in CHOSEN.items():
        settings.update_setting(key, value)

    result = SettingsPersistence(APP).save(settings, keys=set(CHOSEN))

    assert _read(_user_file(isolated)) == CHOSEN
    assert not _project_file(isolated).exists()
    assert result.project_keys == ()
    assert result.user_path == _user_file(isolated)
    assert _chosen(BaseSettings()) == CHOSEN


def test_one_set_back_to_its_default_leaves_the_user_file(isolated):
    _write(_user_file(isolated), {**CHOSEN, "other": 1})
    settings = BaseSettings()
    settings.update_setting("ui_theme", "dark")

    SettingsPersistence(APP).save(settings, keys={"ui_theme"})

    assert _read(_user_file(isolated)) == {"dialog_placement": "inline", "other": 1}
    assert BaseSettings().ui_theme == "dark"
