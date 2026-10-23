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

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from thinking_prompt import TextField

from agentic_cli.cli.workflow_controller import WorkflowController
from agentic_cli.config import BaseSettings
from agentic_cli.workflow.config import AgentConfig
from tests.cli.dialog_fakes import autospec_session, edited_dialog

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


def _app(settings: BaseSettings, manager: FakeManager, edits: dict | None = None):
    """An app around a real controller; its ``/settings`` dialog shows a field
    for each key in ``edits``, and Save sends them."""
    from agentic_cli.cli.app import BaseCLIApp

    controller = WorkflowController([AgentConfig(name="a", prompt="p")], settings)
    controller._workflow = manager
    app = SimpleNamespace(settings=settings, _settings=settings, _workflow_controller=controller)
    app.errors, app.successes = [], []
    app.session = autospec_session()
    app.session.show_dialog.return_value = True  # Save
    app.session.add_error.side_effect = app.errors.append
    app.session.add_success.side_effect = app.successes.append
    app.edits = edits or {}
    app._build_ui_items = lambda: [TextField(key, key) for key in app.edits]
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
        with patch("agentic_cli.cli.settings_command.Dialog", edited_dialog(app.edits)):
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


class InterruptedManager(FakeManager):
    """The Save's restart waits until it is cancelled (Ctrl+C), then the
    manager is down, holding the new model, as the real ones roll back."""

    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.restarting = asyncio.Event()

    async def reinitialize(self, model=None, preserve_sessions=True):
        if not self.reinits:
            self.reinits.append(model)
            if model is not None:
                self.model = model
            self.is_initialized = False
            self.restarting.set()
            await asyncio.Event().wait()
        await super().reinitialize(model, preserve_sessions)


async def _interrupt(task: asyncio.Task, restarting: asyncio.Event) -> None:
    """Ctrl+C while the workflow restarts: thinking-prompt cancels the command."""
    await asyncio.wait_for(restarting.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


class TestTheRestartIsInterrupted:
    """Ctrl+C stops ``/settings`` while the workflow restarts on the Save's
    values. Nothing of the Save stays, as when the restart fails, and nothing
    is said: Ctrl+C there is the session's interruption, not a cancel."""

    async def test_settings_are_restored(self, settings):
        manager = InterruptedManager(OLD_MODEL)
        app = _app(settings, manager)
        changes = {"model": BAD_MODEL, "thinking_effort": "high", "ui_theme": "light"}

        await _interrupt(asyncio.create_task(app.apply_settings(changes)), manager.restarting)

        assert settings.default_model == OLD_MODEL
        assert settings.thinking_effort != "high"
        assert settings.ui_theme == "dark"

    async def test_the_workflow_is_brought_back_on_the_old_model(self, settings):
        manager = InterruptedManager(OLD_MODEL)
        app = _app(settings, manager)

        await _interrupt(
            asyncio.create_task(app.apply_settings({"model": BAD_MODEL})), manager.restarting
        )

        assert manager.reinits == [BAD_MODEL, OLD_MODEL]
        assert manager.model == OLD_MODEL
        assert app._workflow_controller.is_ready

    async def test_a_failed_restore_still_lets_the_interruption_through(self, settings):
        """The cancel is what propagates, not the restore's error, and nothing
        is said; the workflow stays down on the old model, so the next request
        retries it (and reports a failure then)."""
        manager = InterruptedManager(OLD_MODEL)
        manager.broken = {OLD_MODEL}
        app = _app(settings, manager)

        await _interrupt(
            asyncio.create_task(app.apply_settings({"model": BAD_MODEL})), manager.restarting
        )

        assert settings.default_model == OLD_MODEL
        assert manager.model == OLD_MODEL
        assert not app._workflow_controller.is_ready
        assert app.session.method_calls == []

    async def test_nothing_is_said_or_repainted(self, settings):
        manager = InterruptedManager(OLD_MODEL)
        app = _app(settings, manager)

        await _interrupt(
            asyncio.create_task(app.apply_settings({"model": BAD_MODEL, "ui_theme": "light"})),
            manager.restarting,
        )

        assert app.session.method_calls == []

    async def test_nothing_is_saved(self, settings):
        from agentic_cli.cli.settings_command import SettingsCommand

        manager = InterruptedManager(OLD_MODEL)
        app = _app(settings, manager, {"model": BAD_MODEL})
        with patch("agentic_cli.cli.settings_command.Dialog", edited_dialog(app.edits)):
            await _interrupt(
                asyncio.create_task(SettingsCommand().execute("", app)), manager.restarting
            )

        app.save_settings.assert_not_awaited()
        assert settings.default_model == OLD_MODEL

    async def test_an_interrupted_orchestrator_switch_keeps_the_old_backend(self, settings):
        manager = FakeManager(OLD_MODEL, backend="adk")
        app = _app(settings, manager)
        replacement = FakeManager(OLD_MODEL, backend="langgraph")
        starting = asyncio.Event()

        async def initialize_services():
            starting.set()
            await asyncio.Event().wait()

        replacement.initialize_services = initialize_services
        with patch(
            "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings",
            return_value=replacement,
        ):
            await _interrupt(
                asyncio.create_task(app.apply_settings({"orchestrator": "langgraph"})), starting
            )

        assert settings.orchestrator.value == "adk"
        # The swap never published: the old manager runs on, untouched.
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
        with patch("agentic_cli.cli.settings_command.Dialog", edited_dialog(app.edits)):
            await SettingsCommand().execute("", app)

        app.save_settings.assert_not_awaited()


class TestTheLinesNameASettingAsTheDialogDoes:
    """A report or error line labels a setting as the dialog does: by its
    field's title, where it has one, rather than its key title-cased."""

    async def test_a_report_line_uses_the_field_s_title(self, settings):
        app = _app(settings, FakeManager(OLD_MODEL))

        await app.apply_settings({"retry_max_attempts": "5", "verbose_thinking": False})

        assert app.successes == ["Max Retry Attempts: 5", "Verbose Thinking: False"]

    async def test_an_error_line_does_too(self, settings):
        app = _app(settings, FakeManager(OLD_MODEL))

        assert await app.apply_settings({"retry_max_attempts": "many"}) is False

        (error,) = app.errors
        assert error.startswith("Failed to set Max Retry Attempts: ")

    async def test_the_model_pickers_keep_their_labels(self, settings):
        """``model`` is not a field; a role's picker is named after the role."""
        app = _app(settings, FakeManager(OLD_MODEL))

        await app.apply_settings(
            {"model": "gemini-2.5-flash-lite", "model_role:fast": "gemini-2.5-flash"}
        )

        assert app.successes == [
            "Model: gemini-2.5-flash-lite", "Fast Model: gemini-2.5-flash",
        ]


class TestSuccess:
    async def test_applies_reinitializes_and_reports(self, settings):
        manager = FakeManager(OLD_MODEL)
        app = _app(settings, manager)

        applied = await app.apply_settings({"model": BAD_MODEL.replace("flash", "flash-lite")})

        assert applied is True
        assert settings.default_model == "gemini-2.5-flash-lite"
        assert manager.reinits == ["gemini-2.5-flash-lite"]
        assert app.successes and not app.errors

    @pytest.mark.parametrize("key, value", [
        ("context_summarize_at", "0.7"), ("context_max_tokens", "50000"),
    ])
    async def test_when_to_summarize_rebuilds_the_agents(self, settings, key, value):
        # Each agent's thresholds are fixed when it is built.
        manager = FakeManager(OLD_MODEL)
        app = _app(settings, manager)

        assert await app.apply_settings({key: value}) is True

        assert manager.reinits == [None]  # same model, sessions kept
        assert str(getattr(settings, key)) == value
