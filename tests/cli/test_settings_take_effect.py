"""When a ``/settings`` change takes effect, as the user is told.

Every key ``/settings`` may show (``PROJECT_SETTABLE_KEYS`` and
``USER_PREFERENCE_KEYS``) is placed in one of three sets beside them in
``settings_persistence``:

- ``REBUILD_KEYS``: read when the workflow builds its agents and their models
  (the retry and request-timeout settings among them). A change rebuilds the
  workflow, keeping its sessions, and its line is the plain "<Label>: <value>".
- ``RESTART_KEYS``: read only as the app or its session store starts. A
  change is saved and its line says "(takes effect at the next start)";
  nothing is rebuilt. A change of ``session_store`` waits for a start in
  progress, which reads it, as a rebuild does; ``log_level`` and
  ``log_format``, read by the app's constructor only, do not.
- ``LIVE_KEYS``: read each time they are used. A change applies at once.

A key placed in none fails the first test, so a new settable key is placed
when it is added. An app's own key, in none of them, applies at once, as
every unlisted key did.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agentic_cli.cli import AppInfo
from agentic_cli.cli import app as app_module
from agentic_cli.cli.app import BaseCLIApp
from agentic_cli.config import BaseSettings
from agentic_cli.settings_persistence import (
    LIVE_KEYS,
    PROJECT_SETTABLE_KEYS,
    REBUILD_KEYS,
    RESTART_KEYS,
    USER_PREFERENCE_KEYS,
)
from agentic_cli.workflow.config import AgentConfig
from tests.cli.dialog_fakes import autospec_session, opening_boxes
from tests.cli.test_settings_during_start import (
    Starts,
    apply_while_starting,
    starting_app,
    waited,
)

AT_THE_NEXT_START = "(takes effect at the next start)"


class Manager:
    """A running manager: what a rebuild touches."""

    backend_type = "adk"
    is_initialized = True
    model = "google:gemini-2.5-flash"
    agent_models: dict = {}

    def __init__(self) -> None:
        self.reinits: list[str | None] = []

    async def reinitialize(self, model=None, preserve_sessions=True) -> None:
        self.reinits.append(model)

    async def cleanup(self) -> None:
        pass


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BaseSettings:
    (tmp_path / "home").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(app_module, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "_install_adk_notice_hook", lambda: None)
    return BaseSettings()


def running_app(settings: BaseSettings) -> tuple[BaseCLIApp, Manager]:
    app = BaseCLIApp(
        app_info=AppInfo(name="Effect test", version="0"),
        agent_configs=[AgentConfig(name="a", prompt="p")],
        settings=settings,
    )
    app.session = opening_boxes(autospec_session())
    manager = Manager()
    app._workflow_controller._workflow = manager
    return app, manager


def successes(app: BaseCLIApp) -> list[str]:
    return [c.args[0] for c in app.session.add_success.call_args_list]


class TestEveryKeyIsPlaced:
    def test_each_settable_key_is_in_exactly_one_set(self):
        placed = [REBUILD_KEYS, RESTART_KEYS, LIVE_KEYS]
        settable = PROJECT_SETTABLE_KEYS | USER_PREFERENCE_KEYS

        unplaced = settable - REBUILD_KEYS - RESTART_KEYS - LIVE_KEYS
        assert not unplaced, f"place each in REBUILD_KEYS, RESTART_KEYS or LIVE_KEYS: {unplaced}"
        for i, one in enumerate(placed):
            for other in placed[i + 1:]:
                assert not one & other
        assert REBUILD_KEYS | RESTART_KEYS | LIVE_KEYS == settable

    def test_the_sets_name_settings_fields(self):
        fields = set(BaseSettings.model_fields)
        assert REBUILD_KEYS | RESTART_KEYS | LIVE_KEYS <= fields


class TestARebuildKey:
    @pytest.mark.parametrize("key, value, line", [
        ("retry_max_attempts", 9, "Max Retry Attempts: 9"),
        ("retry_initial_delay", 1.5, "Retry Initial Delay: 1.5"),
        ("retry_backoff_factor", 3.0, "Retry Backoff Factor: 3.0"),
        ("anthropic_request_timeout", 5, "Anthropic Request Timeout: 5"),
    ])
    async def test_rebuilds_the_workflow_and_says_so_plainly(self, settings, key, value, line):
        # Read when the models are built (adk/model_factory.py, langchain_models.py).
        app, manager = running_app(settings)

        assert await app.apply_settings({key: value}) is True

        assert manager.reinits == [None]  # same model, sessions kept
        assert successes(app) == [line]

    async def test_a_role_s_model_still_rebuilds(self, settings):
        app, manager = running_app(settings)

        assert await app.apply_settings({"model_role:fast": "google:gemini-2.5-flash-lite"})

        assert manager.reinits == [None]


class TestARestartKey:
    @pytest.mark.parametrize("key, value, line", [
        ("session_store", "memory", "Session Store: memory"),
        ("log_level", "debug", "Log Level: debug"),
        ("log_format", "json", "Log Format: json"),
    ])
    async def test_is_applied_and_said_to_take_effect_at_the_next_start(
        self, settings, key, value, line
    ):
        app, manager = running_app(settings)

        assert await app.apply_settings({key: value}) is True

        assert getattr(settings, key) == value  # saved by /settings as it is
        assert manager.reinits == []
        assert successes(app) == [f"{line} {AT_THE_NEXT_START}"]

    async def test_with_a_rebuild_key_only_the_restart_key_says_so(self, settings):
        app, manager = running_app(settings)

        await app.apply_settings({"retry_max_attempts": 4, "log_level": "info"})

        assert manager.reinits == [None]
        assert successes(app) == [
            "Max Retry Attempts: 4", f"Log Level: info {AT_THE_NEXT_START}",
        ]

    async def test_the_session_store_waits_for_a_start_in_progress(self, settings, monkeypatch):
        """The start reads it: once it is over, the line is true."""
        starts = Starts()
        app = starting_app(settings, monkeypatch, starts)

        applying = await apply_while_starting(app, starts, {"session_store": "memory"})
        assert not applying.done()
        starts.release()
        assert await applying is True

        assert waited(app)
        assert successes(app) == [f"Session Store: memory {AT_THE_NEXT_START}"]
        await app._workflow_controller.close()


    @pytest.mark.parametrize("key, value", [("log_level", "debug"), ("log_format", "json")])
    async def test_logging_does_not_wait_for_a_start_in_progress(
        self, settings, monkeypatch, key, value
    ):
        """Read by the app's constructor only: the start does not read it."""
        starts = Starts()
        app = starting_app(settings, monkeypatch, starts)

        applying = await apply_while_starting(app, starts, {key: value})
        assert await asyncio.wait_for(applying, timeout=2) is True

        assert not waited(app)
        (line,) = successes(app)
        assert line.endswith(AT_THE_NEXT_START)
        starts.release()
        await app._workflow_controller.close()


class TestALiveKey:
    @pytest.mark.parametrize("key, value", [
        ("verbose_thinking", False), ("python_executor_timeout", 10), ("sandbox_timeout", 30),
        ("search_backend", "brave"), ("webfetch_max_content_bytes", 1000),
        ("dialog_placement", "inline"),
    ])
    async def test_applies_at_once(self, settings, key, value):
        app, manager = running_app(settings)

        assert await app.apply_settings({key: value}) is True

        assert manager.reinits == []
        (line,) = successes(app)
        assert AT_THE_NEXT_START not in line

    async def test_does_not_wait_for_a_start_in_progress(self, settings, monkeypatch):
        starts = Starts()
        app = starting_app(settings, monkeypatch, starts)

        applying = await apply_while_starting(app, starts, {"verbose_thinking": False})
        await asyncio.wait_for(applying, timeout=2)

        assert not waited(app)
        starts.release()
        await app._workflow_controller.close()

    async def test_an_app_s_own_key_applies_at_once(self, tmp_path, monkeypatch):
        from pydantic import Field

        class AppSettings(BaseSettings):
            house_style: str = Field(default="plain", title="House Style")

        monkeypatch.setattr(app_module, "configure_logging", lambda *a, **k: None)
        monkeypatch.setattr(app_module, "_install_adk_notice_hook", lambda: None)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        app, manager = running_app(AppSettings())

        assert await app.apply_settings({"house_style": "fancy"}) is True

        assert manager.reinits == []
        assert successes(app) == ["House Style: fancy"]
