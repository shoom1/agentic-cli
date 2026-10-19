"""A ``/settings`` switch to another provider's model says what it keeps and what it loses.

The conversation is kept, but no model is sent another provider's reasoning
(both backends leave it out), and the new provider has none of the old one's
prompt cache. That switch used to be silent. Now ``apply_settings`` adds one
system line per switch, naming the agents and both models; a switch within
one provider loses nothing and stays silent, and so does an orchestrator swap,
which starts a new manager rather than carrying the conversation over.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agentic_cli.cli.app import BaseCLIApp
from agentic_cli.cli.workflow_controller import WorkflowController
from agentic_cli.config import BaseSettings
from agentic_cli.models import ModelRef, ProviderKind, parse_ref
from agentic_cli.workflow.config import AgentConfig

KEY_VARIABLES = ("GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")
GEMINI = "google:gemini-2.5-flash"
GEMINI_PRO = "google:gemini-2.5-pro"
CLAUDE = "anthropic:claude-sonnet-4-5"
CONFIGS = [
    AgentConfig(name="coordinator", prompt="Coordinate.", sub_agents=["writer"]),
    AgentConfig(name="writer", prompt="Write.", model="fast"),
]


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch) -> Path:
    """A hermetic cwd and HOME, with Google and Anthropic keys."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for variable in KEY_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "g-test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-test-key")
    return tmp_path


def _settings() -> BaseSettings:
    return BaseSettings(
        session_store="memory",
        default_model=GEMINI,
        model_roles={"fast": "google:gemini-2.5-flash-lite"},
    )


def _app(settings: BaseSettings, workflow) -> BaseCLIApp:
    """A real app's apply path around a real controller."""
    controller = WorkflowController(CONFIGS, settings)
    controller._workflow = workflow
    app = BaseCLIApp.__new__(BaseCLIApp)
    app._settings = settings
    app._workflow_controller = controller
    app.notices, app.errors = [], []
    app.session = SimpleNamespace(
        add_message=lambda role, text: app.notices.append(text),
        add_error=app.errors.append,
        add_success=lambda *a: None,
        add_warning=lambda *a: None,
    )
    return app


async def _adk(settings: BaseSettings):
    pytest.importorskip("google.adk")
    from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager

    manager = GoogleADKWorkflowManager(agent_configs=CONFIGS, settings=settings)
    await manager.initialize_services()
    return manager


async def test_a_switch_to_another_provider_says_what_it_keeps_and_loses():
    settings = _settings()
    manager = await _adk(settings)
    try:
        app = _app(settings, manager)

        assert await app.apply_settings({"model": CLAUDE})

        assert app.notices == [
            f"coordinator: {GEMINI} → {CLAUDE}. The conversation is kept, but earlier "
            "reasoning from google is not sent to the new model, and the prompt cache "
            "starts over."
        ]
    finally:
        await manager.cleanup()


async def test_only_the_agents_that_switched_are_named():
    settings = _settings()
    manager = await _adk(settings)
    try:
        app = _app(settings, manager)

        assert await app.apply_settings({"model_role:fast": "anthropic:claude-haiku-4-5"})

        (notice,) = app.notices
        assert notice.startswith(
            "writer: google:gemini-2.5-flash-lite → anthropic:claude-haiku-4-5. "
        )
    finally:
        await manager.cleanup()


async def test_a_switch_within_one_provider_is_silent():
    settings = _settings()
    manager = await _adk(settings)
    try:
        app = _app(settings, manager)

        assert await app.apply_settings({"model": GEMINI_PRO})

        assert manager.agent_models["coordinator"].ref == ModelRef("google", "gemini-2.5-pro")
        assert app.notices == []
    finally:
        await manager.cleanup()


async def test_a_manager_reports_each_agent_s_model():
    settings = _settings()
    manager = await _adk(settings)
    try:
        assert {name: str(model.ref) for name, model in manager.agent_models.items()} == {
            "coordinator": GEMINI,
            "writer": "google:gemini-2.5-flash-lite",
        }
        with pytest.raises(TypeError):
            manager.agent_models["writer"] = None
    finally:
        await manager.cleanup()


class FakeManager:
    """Every agent runs on the manager's one model."""

    def __init__(self, model: str, backend: str = "adk", reports_models: bool = True) -> None:
        self.backend_type = backend
        self.is_initialized = True
        self.reports_models = reports_models
        self.cleanup = AsyncMock()
        self._use(model)

    def _use(self, model: str) -> None:
        self.model = model
        if not self.reports_models:
            return
        ref = parse_ref(model)
        resolved = SimpleNamespace(ref=ref, record=SimpleNamespace(kind=ProviderKind(ref.provider)))
        self.agent_models = {config.name: resolved for config in CONFIGS}

    async def initialize_services(self) -> None:
        pass

    async def reinitialize(self, model=None, preserve_sessions=True):
        self._use(model or self.model)


async def test_an_orchestrator_swap_is_silent():
    """A swap starts another manager: the conversation does not carry over to it."""
    settings = _settings()
    app = _app(settings, FakeManager(GEMINI))
    replacement = FakeManager(CLAUDE, backend="langgraph")

    with patch(
        "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings",
        lambda **kw: replacement,
    ):
        assert await app.apply_settings({"orchestrator": "langgraph", "model": CLAUDE})

    assert app._workflow_controller.workflow is replacement
    assert app.notices == []


async def test_a_manager_without_agent_models_is_silent():
    """An application's own manager may not report its agents' models."""
    settings = _settings()
    app = _app(settings, FakeManager(GEMINI, reports_models=False))

    assert await app.apply_settings({"model": CLAUDE})

    assert app.notices == []
    assert not app.errors
