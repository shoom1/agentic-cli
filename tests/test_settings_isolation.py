"""Every test starts without global settings, whatever ran before it.

``set_settings()`` writes a process-wide singleton with no teardown, and
``get_settings()`` creates a default one the first time anything asks. A test
that left one behind answered ``get_settings()`` for every later test, so a
test's result depended on which tests ran before it (the research scenarios'
leak check, and ``TestFixtureIsolation`` after ``tests/workflow``).
"""

import pytest

from agentic_cli import config
from agentic_cli.config import BaseSettings, set_context_settings, set_settings


@pytest.fixture(scope="module")
def settings_left_behind(tmp_path_factory):
    """Global settings set before this module's tests, never cleared.

    A module-scoped fixture is set up before any function-scoped one, so this
    stands in for whatever an earlier test left behind.
    """
    workspace = tmp_path_factory.mktemp("leaked_ws")
    set_settings(BaseSettings(workspace_dir=workspace))
    set_context_settings(BaseSettings(workspace_dir=workspace))


def test_starts_without_global_settings(settings_left_behind):
    assert config._settings_instance is None


def test_starts_without_context_settings(settings_left_behind):
    assert config.get_context_settings() is None
