"""The CLI logs ADK's experimental-feature notices instead of printing them.

ADK 2 announces each experimental feature it turns on with a ``UserWarning``
("[EXPERIMENTAL] feature FeatureName.X is enabled."). Python prints warnings
to stderr, which in the CLI lands over the terminal UI. ``BaseCLIApp`` installs
a ``warnings.showwarning`` hook that logs those notices at debug and hands
every other warning to the hook it replaced.
"""

from __future__ import annotations

import sys
import warnings

import pytest
import structlog
from thinking_prompt import AppInfo

from agentic_cli.cli.app import BaseCLIApp, _install_adk_notice_hook
from agentic_cli.config import BaseSettings
from tests.demo_isolation import suppress_global_logging_config

NOTICE = "[EXPERIMENTAL] feature FeatureName.JSON_SCHEMA_FOR_FUNC_DECL is enabled."


@pytest.fixture
def previous_hook(monkeypatch) -> list[str]:
    """Stand in for Python's own hook: it prints, and records what it printed."""
    shown: list[str] = []

    def showwarning(message, category, filename, lineno, file=None, line=None):
        shown.append(str(message))
        sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))

    monkeypatch.setattr(warnings, "showwarning", showwarning)
    return shown


def _warn(message: str, category: type[Warning] = UserWarning) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn(message, category, stacklevel=2)


def test_an_experimental_notice_is_logged_not_printed(previous_hook, capsys):
    _install_adk_notice_hook()

    with structlog.testing.capture_logs() as logs:
        _warn(NOTICE)

    assert previous_hook == []
    assert capsys.readouterr().err == ""
    assert {
        "event": "adk_experimental_feature",
        "feature": "JSON_SCHEMA_FOR_FUNC_DECL",
        "log_level": "debug",
    } in logs


@pytest.mark.parametrize(
    "message, category",
    [
        ("Something else went wrong.", UserWarning),
        (NOTICE, DeprecationWarning),
        (f"Note: {NOTICE}", UserWarning),
    ],
)
def test_every_other_warning_reaches_the_previous_hook(previous_hook, capsys, message, category):
    _install_adk_notice_hook()

    _warn(message, category)

    assert previous_hook == [message]
    assert message in capsys.readouterr().err


def test_installing_twice_keeps_one_hook(previous_hook):
    _install_adk_notice_hook()
    installed = warnings.showwarning
    _install_adk_notice_hook()

    assert warnings.showwarning is installed


def test_the_cli_app_installs_the_hook(previous_hook, tmp_path, monkeypatch):
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    suppress_global_logging_config(monkeypatch)

    BaseCLIApp(
        app_info=AppInfo(name="Notice test", version="0"),
        agent_configs=[],
        settings=BaseSettings(workspace_dir=tmp_path / "workspace"),
    )
    _warn(NOTICE)

    assert previous_hook == []
