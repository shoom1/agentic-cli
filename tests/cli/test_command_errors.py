"""A slash command that raises: the user is told, and the failure is logged.

The line is "Error executing command: <error>", as before; the log has an
info record naming the command and the error's type (``command_failed``),
and the traceback at debug. Not a warning: at the default log level a
warning prints above the prompt, beside the line the user already sees.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_cli.cli import AppInfo
from agentic_cli.cli import app as app_module
from agentic_cli.cli.app import BaseCLIApp
from agentic_cli.cli.commands import Command
from agentic_cli.config import BaseSettings
from agentic_cli.workflow.config import AgentConfig
from tests.cli.dialog_fakes import autospec_session


class Broken(Command):
    def __init__(self) -> None:
        super().__init__(name="broken", description="Fails")

    async def execute(self, args: str, app) -> None:
        raise KeyError("missing")


@pytest.fixture
def logged(monkeypatch) -> list[dict]:
    """Every log call of ``cli/app.py``, with its level (recorded on the
    module's logger: ``capture_logs()`` misses a logger cached earlier)."""
    events: list[dict] = []

    class Recorder:
        def __getattr__(self, level: str):
            def log(event: str, **fields) -> None:
                events.append({"event": event, "log_level": level, **fields})

            return log

    monkeypatch.setattr(app_module, "logger", Recorder())
    return events


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BaseCLIApp:
    (tmp_path / "home").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(app_module, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "_install_adk_notice_hook", lambda: None)
    app = BaseCLIApp(
        app_info=AppInfo(name="Command test", version="0"),
        agent_configs=[AgentConfig(name="a", prompt="p")],
        settings=BaseSettings(session_store="memory"),
    )
    app.session = autospec_session()
    app.command_registry.register(Broken())
    return app


async def test_it_is_shown_as_before(app, logged):
    await app.process_input("/broken")

    errors = [c.args[0] for c in app.session.add_error.call_args_list]
    assert errors == ["Error executing command: 'missing'"]


async def test_it_is_logged_with_its_traceback_at_debug(app, logged):
    await app.process_input("/broken")

    (record,) = [e for e in logged if e["event"] == "command_failed" and "exc_info" not in e]
    assert record == {
        "event": "command_failed", "log_level": "info",
        "command": "broken", "error_type": "KeyError", "error": "'missing'",
    }
    assert not [e for e in logged if e["log_level"] in ("warning", "error")]
    (traceback,) = [e for e in logged if e["event"] == "command_failed" and "exc_info" in e]
    assert traceback["log_level"] == "debug"
    assert isinstance(traceback["exc_info"], KeyError)
