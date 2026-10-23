"""The status bar: the main agent's model and how close it is to its next summary.

``ctx`` is the main agent's last prompt against its trigger, so the bar names
the main agent's own model, not the default model, when the root runs on a
model of its own. A ``/settings`` change that rebuilds the agents (a model,
``context_summarize_at``, ``context_max_tokens``) shows the new trigger at
once, before the next call; an orchestrator swap starts another manager,
whose conversation is another one, so the context is forgotten.

The app's apply path and controller are real; so are the ADK managers,
initialized offline.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agentic_cli.cli.app import BaseCLIApp
from agentic_cli.cli.usage_tracker import UsageTracker, format_tokens
from agentic_cli.cli.workflow_controller import WorkflowController
from agentic_cli.config import BaseSettings
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager
from agentic_cli.workflow.config import AgentConfig

SONNET = "anthropic:claude-sonnet-4-5"  # a 200,000-token window
OPUS = "anthropic:claude-opus-4-7"  # 1,000,000: the 200,000-token ceiling is its trigger
HAIKU = "anthropic:claude-haiku-4-5"


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch) -> Path:
    """A hermetic cwd and HOME, with a fake Claude key."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for variable in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-test-key")
    return tmp_path


def solo(model: str | None = None) -> list[AgentConfig]:
    return [AgentConfig(name="solo", prompt="Answer.", model=model)]


async def started(configs: list[AgentConfig], settings: BaseSettings) -> GoogleADKWorkflowManager:
    manager = GoogleADKWorkflowManager(agent_configs=configs, settings=settings)
    await manager.initialize_services()
    return manager


def app_for(configs: list[AgentConfig], settings: BaseSettings, manager) -> BaseCLIApp:
    """A real app's apply path around a real controller, recording the status bar."""
    controller = WorkflowController(configs, settings)
    controller._workflow = manager
    tracker = UsageTracker()
    controller.usage_tracker = tracker
    app = BaseCLIApp.__new__(BaseCLIApp)
    app._settings = settings
    app._workflow_controller = controller
    app._usage_tracker = tracker
    app.statuses = []
    app.session = SimpleNamespace(
        add_message=lambda *a: None, add_error=lambda *a: None,
        add_success=lambda *a: None, add_warning=lambda *a: None,
        set_status=app.statuses.append,
    )
    return app


def used(app: BaseCLIApp, prompt: int) -> None:
    """The main agent's last call: ``prompt`` tokens, against its trigger now."""
    budget = app._workflow_controller.workflow.context_budget()
    app._usage_tracker.record({"prompt_tokens": prompt, "completion_tokens": 10,
                               "context_trigger": budget.trigger,
                               "window_known": budget.window_known})


def share(prompt: int, trigger: int) -> str:
    return f"ctx {format_tokens(prompt)}/{format_tokens(trigger)} ({round(100 * prompt / trigger)}%)"


class TestAfterASave:
    async def test_a_model_switch_shows_the_new_trigger_at_once(self):
        settings = BaseSettings(session_store="memory", default_model=OPUS)
        manager = await started(solo(), settings)
        try:
            app = app_for(solo(), settings, manager)
            used(app, 150_000)

            assert await app.apply_settings({"model": HAIKU})
            trigger = manager.context_budget().trigger
        finally:
            await manager.cleanup()

        assert trigger < 150_000  # the next call is already past it
        bar = app.statuses[-1]
        assert bar.startswith(f"{HAIKU} | ")
        assert f" | {share(150_000, trigger)} | " in bar
        assert (app._usage_tracker.context_trigger, app._usage_tracker.window_known) == (
            trigger, True
        )

    async def test_a_new_ceiling_shows_at_once(self):
        settings = BaseSettings(session_store="memory", default_model=OPUS)
        manager = await started(solo(), settings)
        try:
            app = app_for(solo(), settings, manager)
            used(app, 150_000)

            assert await app.apply_settings({"context_max_tokens": 100_000})
        finally:
            await manager.cleanup()

        assert " | ctx 150k/100k (150%) | " in app.statuses[-1]

    async def test_an_orchestrator_swap_forgets_the_context(self):
        settings = BaseSettings(session_store="memory", default_model=OPUS)
        previous = SimpleNamespace(
            model=OPUS, backend_type="adk", is_initialized=True, cleanup=AsyncMock(),
            context_budget=lambda agent_name=None: SimpleNamespace(trigger=200_000,
                                                                   window_known=True),
        )
        replacement = SimpleNamespace(
            model=OPUS, backend_type="langgraph", is_initialized=True, cleanup=AsyncMock(),
            initialize_services=AsyncMock(),
        )
        app = app_for(solo(), settings, previous)
        used(app, 150_000)

        with patch(
            "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings",
            lambda **kw: replacement,
        ):
            assert await app.apply_settings({"orchestrator": "langgraph"})

        tracker = app._usage_tracker
        assert app._workflow_controller.workflow is replacement
        assert (tracker.last_prompt_tokens, tracker.context_trigger, tracker.window_known) == (
            0, None, None
        )
        assert "ctx" not in app.statuses[-1]
        assert tracker.prompt_tokens == 150_000  # the session's totals stay

    async def test_a_manager_without_a_budget_shows_the_prompt_alone(self):
        # An application's own manager may not have context_budget.
        settings = BaseSettings(session_store="memory", default_model=OPUS)
        manager = SimpleNamespace(
            model=SONNET, backend_type="adk", is_initialized=True,
            reinitialize=AsyncMock(),
        )
        app = app_for(solo(), settings, manager)
        app._usage_tracker.record({"prompt_tokens": 45_000, "context_trigger": 200_000,
                                   "window_known": True})

        assert await app.apply_settings({"context_max_tokens": 100_000})

        assert app.statuses[-1] == f"{SONNET} | Tokens: 45k in / 0 out | ctx 45k | " \
            "Ctrl+C: cancel | /help: commands"


class TestTheModelItNames:
    async def test_the_main_agents_own_model(self):
        # Its window and trigger are what ctx is measured against.
        settings = BaseSettings(session_store="memory", default_model=SONNET)
        manager = await started(solo(model=OPUS), settings)
        try:
            app = app_for(solo(model=OPUS), settings, manager)
            used(app, 45_000)
            app._workflow_controller.update_status_bar(app.session)
        finally:
            await manager.cleanup()

        assert manager.model == SONNET
        assert app.statuses[-1].startswith(f"{OPUS} | ")
        assert f" | {share(45_000, 200_000)} | " in app.statuses[-1]

    async def test_without_a_main_agent_the_default_model(self):
        # Not initialized (or LangGraph, or a native ADK app): no root's model.
        settings = BaseSettings(session_store="memory", default_model=SONNET)
        manager = GoogleADKWorkflowManager(agent_configs=solo(model=OPUS), settings=settings)
        app = app_for(solo(model=OPUS), settings, manager)

        app._workflow_controller.update_status_bar(app.session)

        assert app.statuses[-1].startswith(f"{SONNET} | ")
