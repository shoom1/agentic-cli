"""Ctrl+C anywhere in one ``/settings`` Save ends it the same way.

A Save can wait at several points: for a start in progress (a change the
start reads), while the workflow is rebuilt on the new values, for the turn
lock (an orchestrator change asks whether the conversation holds anything
yet), and while the backend of a new conversation starts. Ctrl+C at any of
them stops the request, as it stops a turn: "Operation cancelled..." is said
once, nothing of the Save stays (every setting restored, the workflow on the
model it ran), and nothing is saved. The wait for a start said so; the
rebuild and the wait for the turn lock said nothing.

A real app around a real ``WorkflowController``; only the managers are fakes,
and a test holds one at the point where Ctrl+C stops the Save.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agentic_cli.cli import AppInfo
from agentic_cli.cli import app as app_module
from agentic_cli.cli.app import BaseCLIApp
from agentic_cli.cli.message_processor import CANCELLED_LINE
from agentic_cli.cli.settings_command import SettingsCommand
from agentic_cli.cli.workflow_controller import WorkflowState
from agentic_cli.config import BaseSettings
from agentic_cli.workflow.config import AgentConfig
from tests.cli.dialog_fakes import autospec_session, edited_dialog, opening_boxes

OLD, NEW = "google:gemini-2.5-flash", "google:gemini-2.5-pro"


class Holds:
    """The points where a manager waits for the test, each once: ``hold(point)``
    arms one and returns the event set as a manager gets there;
    ``release(point)`` lets it go on."""

    def __init__(self) -> None:
        self._points: dict[str, tuple[asyncio.Event, asyncio.Event]] = {}

    def hold(self, point: str) -> asyncio.Event:
        reached = asyncio.Event()
        self._points[point] = (reached, asyncio.Event())
        return reached

    def release(self, point: str) -> None:
        self._points[point][1].set()

    async def at(self, point: str) -> None:
        held = self._points.get(point)
        if held is not None and not held[0].is_set():
            reached, released = held
            reached.set()
            await released.wait()


class Manager:
    """Resolves its model from the settings as it starts; a rebuild takes the
    model it is given, the manager down until it is done, as the real ones."""

    catalog = None  # the pickers' list comes from the settings

    def __init__(self, settings: BaseSettings, backend: str, holds: Holds) -> None:
        self._settings = settings
        self.backend_type = backend
        self.holds = holds
        self.model: str | None = None
        self.is_initialized = False
        self.agent_models: dict = {}
        self.reinits: list[str | None] = []

    async def initialize_services(self) -> None:
        self.model = self._settings.default_model
        await self.holds.at(f"start {self.backend_type}")
        self.is_initialized = True

    async def reinitialize(self, model=None, preserve_sessions=True) -> None:
        self.reinits.append(model)
        self.model = model or self._settings.default_model
        self.is_initialized = False
        await self.holds.at("rebuild")
        self.is_initialized = True

    async def load_session(self, session_id, user_id=None) -> bool:
        return False

    async def cleanup(self) -> None:
        self.is_initialized = False


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BaseSettings:
    (tmp_path / "home").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(app_module, "configure_logging", lambda *a, **k: None)
    s = BaseSettings(session_store="memory")
    s.update_setting("model", OLD)
    return s


@pytest.fixture
def holds() -> Holds:
    return Holds()


@pytest.fixture
def app(settings: BaseSettings, monkeypatch, holds: Holds) -> BaseCLIApp:
    """An app whose ``/settings`` shows the orchestrator too, and saves nothing."""

    def create(*, settings, orchestrator, **kwargs) -> Manager:
        return Manager(settings, orchestrator, holds)

    monkeypatch.setattr(
        "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings", create
    )

    class App(BaseCLIApp):
        def get_ui_setting_keys(self) -> list[str]:
            return [*super().get_ui_setting_keys(), "orchestrator", "ui_theme"]

    app = App(
        app_info=AppInfo(name="Stopped Save test", version="0"),
        agent_configs=[AgentConfig(name="a", prompt="p")],
        settings=settings,
    )
    app.session = opening_boxes(autospec_session())
    app.session.show_dialog.return_value = True  # Save
    app.save_settings = AsyncMock()
    return app


async def _running(app: BaseCLIApp):
    """The app's workflow, started."""
    controller = app._workflow_controller
    await controller.start_background_init()
    assert await controller.ensure_initialized()
    return controller


async def _stopped_save(app: BaseCLIApp, monkeypatch, edits: dict, waits) -> None:
    """``/settings`` saved with ``edits``, and Ctrl+C once ``waits()`` says it waits."""
    monkeypatch.setattr("agentic_cli.cli.settings_command.Dialog", edited_dialog(edits))
    saving = asyncio.create_task(SettingsCommand().execute("", app))
    await asyncio.wait_for(waits(), timeout=5)
    saving.cancel()
    with pytest.raises(asyncio.CancelledError):
        await saving


def _said_once_and_nothing_stayed(app: BaseCLIApp, settings: BaseSettings) -> None:
    messages = [c.args for c in app.session.add_message.call_args_list]
    assert messages == [("system", CANCELLED_LINE)]
    assert app.session.add_success.call_count == 0
    assert app.session.add_error.call_count == 0
    assert settings.default_model == OLD
    assert settings.orchestrator.value == "adk"
    assert settings.ui_theme == "dark"
    app.session.set_theme.assert_not_called()
    app.save_settings.assert_not_awaited()


async def test_while_it_waits_for_a_start_in_progress(app, settings, monkeypatch, holds):
    controller = app._workflow_controller
    starting = holds.hold("start adk")
    await controller.start_background_init()
    await asyncio.wait_for(starting.wait(), timeout=5)

    async def waits() -> None:  # the Save waits for the start ("Waiting for initialization...")
        while not app.session.start_thinking.called:
            await asyncio.sleep(0)

    await _stopped_save(app, monkeypatch, {"model": NEW, "ui_theme": "light"}, waits)

    _said_once_and_nothing_stayed(app, settings)
    holds.release("start adk")  # the start goes on, on the settings as they were
    assert await controller.ensure_initialized()
    assert controller.workflow.model == OLD
    await controller.close()


async def test_while_the_workflow_is_rebuilt(app, settings, monkeypatch, holds):
    controller = await _running(app)
    rebuilding = holds.hold("rebuild")

    await _stopped_save(app, monkeypatch, {"model": NEW, "ui_theme": "light"}, rebuilding.wait)

    _said_once_and_nothing_stayed(app, settings)
    assert controller.state is WorkflowState.READY
    assert controller.workflow.reinits == [NEW, OLD]  # back on the model it ran
    assert controller.workflow.model == OLD
    await controller.close()


async def test_while_it_waits_for_the_turn_lock(app, settings, monkeypatch, holds):
    """An orchestrator change reads, under the turn lock, whether the
    conversation holds anything yet; an app's task holds the lock."""
    controller = await _running(app)
    holding, gate = asyncio.Event(), asyncio.Event()

    async def an_apps_task() -> None:
        async with app._turn_lock:
            holding.set()
            await gate.wait()

    task = asyncio.create_task(an_apps_task())
    await holding.wait()

    async def waits() -> None:
        for _ in range(10):
            await asyncio.sleep(0)

    await _stopped_save(app, monkeypatch, {"orchestrator": "deepagents"}, waits)
    gate.set()
    await task

    _said_once_and_nothing_stayed(app, settings)
    assert controller.backend == "adk"
    await controller.close()


async def test_while_a_new_conversation_s_backend_starts(app, settings, monkeypatch, holds):
    """Said by the new conversation (``start_new_session``, as ``/new`` says
    it), not a second time by the Save."""
    pytest.importorskip("deepagents")
    controller = await _running(app)
    old = app.session_id
    starting = holds.hold("start deepagents")

    await _stopped_save(app, monkeypatch, {"orchestrator": "deepagents"}, starting.wait)

    _said_once_and_nothing_stayed(app, settings)
    assert controller.backend == "adk" and app.session_id == old
    await controller.close()
