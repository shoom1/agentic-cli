"""``/settings`` is all-or-nothing: a change that cannot be applied is undone.

``apply_settings`` set each field as it went. When a later field was invalid,
or the workflow could not be reinitialized with the new values, it reported the
error but left the earlier changes in memory, and ``/settings`` then saved them.
A model the workflow cannot start with was written to the config and applied
again on the next run, and after a failed in-place reinitialization the live
manager was left holding that model, so the session stayed broken.

These run a real ``WorkflowController`` and real settings; only the managers
are fakes, reinitializing transactionally as the real ones do (a failure leaves
the manager uninitialized, holding the model it was given).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agentic_cli.cli.workflow_controller import WorkflowController
from agentic_cli.config import BaseSettings
from agentic_cli.workflow.config import AgentConfig

OLD_MODEL = "gemini-2.5-pro"
BAD_MODEL = "gemini-2.5-flash"


class FakeManager:
    def __init__(self, model: str, backend: str = "adk", broken: tuple[str, ...] = ()):
        self.model = model
        self.backend_type = backend
        self.is_initialized = True
        self.broken = set(broken)
        self.reinits: list[str | None] = []
        self.cleanup = AsyncMock()

    async def reinitialize(self, model=None, preserve_sessions=True):
        self.reinits.append(model)
        if model is not None:
            self.model = model
        if self.model in self.broken:
            self.is_initialized = False
            raise RuntimeError(f"cannot start {self.model}")
        self.is_initialized = True


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BaseSettings:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    s = BaseSettings()
    s.update_setting("model", OLD_MODEL)
    return s


def _app(settings: BaseSettings, manager: FakeManager, dialog_result=None):
    from agentic_cli.cli.app import BaseCLIApp

    controller = WorkflowController([AgentConfig(name="a", prompt="p")], settings)
    controller._workflow = manager
    app = SimpleNamespace(settings=settings, _settings=settings, _workflow_controller=controller)
    app.errors, app.successes = [], []
    app.session = SimpleNamespace(
        show_dialog=AsyncMock(return_value=dialog_result),
        add_message=lambda *a: None,
        add_error=app.errors.append,
        add_success=app.successes.append,
        add_warning=lambda *a: None,
    )
    app._build_ui_items = lambda: ["item"]
    app.apply_settings = lambda changes: BaseCLIApp.apply_settings(app, changes)
    app.save_settings = AsyncMock(return_value=SimpleNamespace(
        project_path=Path("p"), user_path=None, user_scoped_keys=()))
    return app


class TestModelReinitializeFails:
    async def test_settings_are_restored(self, settings):
        app = _app(settings, FakeManager(OLD_MODEL, broken=(BAD_MODEL,)))

        applied = await app.apply_settings({"model": BAD_MODEL, "thinking_effort": "high"})

        assert applied is False
        assert settings.default_model == OLD_MODEL
        assert settings.thinking_effort != "high"
        assert any(BAD_MODEL in e for e in app.errors)

    async def test_the_workflow_is_brought_back_on_the_old_model(self, settings):
        manager = FakeManager(OLD_MODEL, broken=(BAD_MODEL,))
        app = _app(settings, manager)

        await app.apply_settings({"model": BAD_MODEL})

        assert manager.reinits == [BAD_MODEL, OLD_MODEL]
        assert manager.model == OLD_MODEL
        assert app._workflow_controller.is_ready

    async def test_a_failed_restore_is_reported(self, settings):
        manager = FakeManager(OLD_MODEL, broken=(BAD_MODEL, OLD_MODEL))
        app = _app(settings, manager)

        assert await app.apply_settings({"model": BAD_MODEL}) is False

        assert settings.default_model == OLD_MODEL
        assert len(app.errors) == 2
        assert not app._workflow_controller.is_ready

    async def test_nothing_is_saved(self, settings):
        from agentic_cli.cli.settings_command import SettingsCommand

        app = _app(settings, FakeManager(OLD_MODEL, broken=(BAD_MODEL,)), {"model": BAD_MODEL})
        with patch("agentic_cli.cli.settings_command.SettingsDialog", lambda **kw: None):
            await SettingsCommand().execute("", app)

        app.save_settings.assert_not_awaited()


class TestOrchestratorSwapFails:
    async def test_the_orchestrator_setting_is_restored(self, settings):
        manager = FakeManager(OLD_MODEL, backend="adk")
        app = _app(settings, manager)
        replacement = FakeManager(OLD_MODEL, backend="langgraph")
        replacement.initialize_services = AsyncMock(side_effect=RuntimeError("langgraph extra missing"))

        with patch(
            "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings",
            return_value=replacement,
        ):
            applied = await app.apply_settings({"orchestrator": "langgraph"})

        assert applied is False
        assert settings.orchestrator.value == "adk"
        # The old manager kept running; it is neither replaced nor reinitialized.
        assert app._workflow_controller._workflow is manager
        assert manager.reinits == []
        assert app._workflow_controller.is_ready


class TestInvalidValue:
    async def test_earlier_fields_in_the_same_change_are_undone(self, settings):
        app = _app(settings, FakeManager(OLD_MODEL))
        before = settings.thinking_effort

        applied = await app.apply_settings({"thinking_effort": "high", "log_level": "LOUD"})

        assert applied is False
        assert settings.thinking_effort == before
        assert app._workflow_controller._workflow.reinits == []

    async def test_nothing_is_saved(self, settings):
        from agentic_cli.cli.settings_command import SettingsCommand

        app = _app(
            settings, FakeManager(OLD_MODEL), {"thinking_effort": "high", "log_level": "LOUD"},
        )
        with patch("agentic_cli.cli.settings_command.SettingsDialog", lambda **kw: None):
            await SettingsCommand().execute("", app)

        app.save_settings.assert_not_awaited()


class TestSuccess:
    async def test_applies_reinitializes_and_reports(self, settings):
        manager = FakeManager(OLD_MODEL)
        app = _app(settings, manager)

        applied = await app.apply_settings({"model": BAD_MODEL.replace("flash", "flash-lite")})

        assert applied is True
        assert settings.default_model == "gemini-2.5-flash-lite"
        assert manager.reinits == ["gemini-2.5-flash-lite"]
        assert app.successes and not app.errors
