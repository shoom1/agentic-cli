"""``/settings`` while the workflow is still starting.

A start resolves its models from the settings first, then builds the rest
(services, the backend), so a change saved while it runs was reported and
saved but never run: the workflow kept the model it had resolved. A change
that needs a rebuild now waits for the start in progress, as a turn does
("Waiting for initialization...", Ctrl+C stops the wait and changes
nothing), then rebuilds as usual. A start that fails takes the change as a
failed workflow does: it is applied and saved, and the next start uses it.

A real app around a real ``WorkflowController``; only the managers are fakes,
each started by hand (``Starts.release``) so the test decides when a start ends.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agentic_cli.cli import AppInfo
from agentic_cli.cli import app as app_module
from agentic_cli.cli.app import BaseCLIApp
from agentic_cli.cli.message_processor import CANCELLED_LINE
from agentic_cli.cli.workflow_controller import WorkflowState
from agentic_cli.config import BaseSettings
from agentic_cli.workflow.config import AgentConfig
from tests.cli.dialog_fakes import autospec_session, opening_boxes

OLD, NEW = "google:gemini-2.5-flash", "google:gemini-2.5-pro"
WAITING = "Waiting for initialization..."


class Manager:
    """Resolves its model from the settings as it starts, then waits for the
    test to let the rest of its start finish (``gate``)."""

    def __init__(self, settings: BaseSettings, backend: str, starts: "Starts | None",
                 fails: bool) -> None:
        self._settings = settings
        self.backend_type = backend
        self.starts, self.fails = starts, fails
        self.model: str | None = None
        self.is_initialized = False
        self.agent_models: dict = {}
        self.reinits: list[str | None] = []
        self.loaded: list[str] = []

    async def initialize_services(self) -> None:
        self.model = self._settings.default_model  # _load_models, first
        if self.starts is not None:
            self.starts.resolved.set()
            await self.starts.gate.wait()  # services, the backend, MCP...
        if self.fails:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        self.is_initialized = True

    async def reinitialize(self, model=None, preserve_sessions=True) -> None:
        self.reinits.append(model)
        self.model = model or self._settings.default_model

    async def load_session(self, session_id, user_id=None) -> bool:
        self.loaded.append(session_id)
        return False

    async def cleanup(self) -> None:
        pass


class Starts:
    """The factory the controller builds managers with: the first start waits
    for ``release()``; a swap's start does not."""

    def __init__(self, *, fails: bool = False) -> None:
        self.resolved = asyncio.Event()  # the first manager has read its model
        self.gate = asyncio.Event()
        self.fails = fails
        self.built: list[Manager] = []

    def __call__(self, *, settings, **kwargs) -> Manager:
        first = not self.built
        backend = getattr(settings.orchestrator, "value", settings.orchestrator)
        manager = Manager(settings, backend, self if first else None, self.fails and first)
        self.built.append(manager)
        return manager

    def release(self) -> None:
        self.gate.set()


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BaseSettings:
    (tmp_path / "home").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(app_module, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "_install_adk_notice_hook", lambda: None)
    s = BaseSettings(session_store="memory")
    s.update_setting("model", OLD)
    return s


def starting_app(settings: BaseSettings, monkeypatch, starts: Starts) -> BaseCLIApp:
    monkeypatch.setattr(
        "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings", starts
    )
    app = BaseCLIApp(
        app_info=AppInfo(name="Start test", version="0"),
        agent_configs=[AgentConfig(name="a", prompt="p")],
        settings=settings,
    )
    app.session = opening_boxes(autospec_session())
    return app


async def apply_while_starting(app: BaseCLIApp, starts: Starts, changes: dict) -> asyncio.Task:
    """``apply_settings(changes)`` called while the start runs, as a task."""
    controller = app._workflow_controller
    await controller.start_background_init()
    await asyncio.wait_for(starts.resolved.wait(), timeout=5)
    assert controller.state is WorkflowState.INITIALIZING
    applying = asyncio.create_task(app.apply_settings(changes))
    for _ in range(5):
        await asyncio.sleep(0)
    return applying


def waited(app: BaseCLIApp) -> bool:
    return any(c.args[0]() == WAITING for c in app.session.start_thinking.call_args_list)


async def test_the_workflow_runs_the_model_the_settings_name(settings, monkeypatch):
    starts = Starts()
    app = starting_app(settings, monkeypatch, starts)

    applying = await apply_while_starting(app, starts, {"model": NEW})
    starts.release()
    assert await applying is True
    await app._workflow_controller.ensure_initialized()

    assert settings.default_model == NEW
    assert app._workflow_controller.workflow.model == NEW
    assert waited(app)
    await app._workflow_controller.close()


async def test_an_orchestrator_change_waits_for_the_start_then_starts_there(
    settings, monkeypatch
):
    """Whether the change starts a new conversation (one with nothing stored
    yet does) needs the start's outcome: it waits, then the conversation the
    start opened, still empty, is left for one on the new backend."""
    starts = Starts()
    app = starting_app(settings, monkeypatch, starts)

    applying = await apply_while_starting(app, starts, {"orchestrator": "deepagents"})
    starts.release()
    assert await applying is True
    controller = app._workflow_controller
    await controller.ensure_initialized()

    assert controller.workflow.backend_type == "deepagents"
    assert controller.backend == "deepagents"
    await controller.close()


async def test_ctrl_c_while_it_waits_changes_nothing(settings, monkeypatch):
    starts = Starts()
    app = starting_app(settings, monkeypatch, starts)

    applying = await apply_while_starting(app, starts, {"model": NEW})
    applying.cancel()
    with pytest.raises(asyncio.CancelledError):
        await applying

    assert settings.default_model == OLD
    assert app.session.add_success.call_count == 0
    messages = [c.args for c in app.session.add_message.call_args_list]
    assert messages == [("system", CANCELLED_LINE)]
    # The start goes on, on the settings as they were.
    starts.release()
    controller = app._workflow_controller
    assert await controller.ensure_initialized()
    assert controller.workflow.model == OLD
    await controller.close()


async def test_a_start_that_fails_takes_the_change_as_a_failed_workflow_does(
    settings, monkeypatch
):
    starts = Starts(fails=True)
    app = starting_app(settings, monkeypatch, starts)

    applying = await apply_while_starting(app, starts, {"model": NEW})
    starts.release()
    assert await applying is True

    controller = app._workflow_controller
    assert controller.state is WorkflowState.FAILED
    assert settings.default_model == NEW
    errors = [c.args[0] for c in app.session.add_error.call_args_list]
    assert errors == ["Initialization failed: GOOGLE_API_KEY is not set"]
    assert [c.args[0] for c in app.session.add_success.call_args_list] == [f"Model: {NEW}"]
    assert starts.built[0].reinits == []  # nothing to rebuild: it is not running
    await controller.close()
