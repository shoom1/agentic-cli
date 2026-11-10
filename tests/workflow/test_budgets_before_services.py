"""A context budget too small to work in is found before any service is built.

``BaseWorkflowManager._initialize_locked`` computes each agent's context
numbers (``context_budget``) once the models are resolved and before
``_construct_services``: they need only the resolved models, and a model
whose window is too small fails startup (``ContextBudgetError``) with
nothing allocated. Each backend used to compute them in ``_do_initialize``,
after the services, so an app whose agent used the knowledge base loaded its
embedding model, opened both knowledge bases and the jobs store, and only
then heard that its model could not work. Deep Agents also opened its
checkpointer first.

A backend takes part through ``_manages_context``: ADK (whether its agents
were built from configs or from a native ADK YAML) and Deep Agents say yes; a
backend that says nothing and LangGraph manage none and keep an empty
``_budgets``.

Offline: catalogs never list (``tests/conftest.py``), and no model is called.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import pytest

from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.config import AgentConfig, AgentGraph, validate_agent_graph
from agentic_cli.workflow.context import ContextBudgetError, context_budget
from agentic_cli.workflow.events import WorkflowEvent
from tests.conftest import MockContext

#: A model whose window the settings give as 16,000 tokens: under the window an
#: agent needs for its 16,000-token input budget (``MIN_BUDGET``). ROOMY is the
#: default model, with the table's window.
SMALL = "google:gemini-3.8-flash"
ROOMY = "google:gemini-3.8-pro"

BACKENDS = [
    pytest.param(
        "agentic_cli.workflow.adk.manager:GoogleADKWorkflowManager",
        "_make_session_service",
        id="adk",
    ),
    pytest.param(
        "agentic_cli.workflow.deepagents.manager:DeepAgentsWorkflowManager",
        "_open_session_store",
        id="deepagents",
    ),
]


@pytest.fixture
def ctx():
    models = {SMALL: {"context_window": 16_000}}
    with MockContext(google_api_key="test-key", models=models, default_model=ROOMY) as context:
        yield context


def _backend(path: str) -> type[BaseWorkflowManager]:
    module, _, name = path.partition(":")
    if "deepagents" in module:
        pytest.importorskip("deepagents")
    return getattr(importlib.import_module(module), name)


def _job_agent(model: str | None = None) -> list[AgentConfig]:
    """One agent whose tool needs a service (the jobs), on ``model`` (the default if None)."""
    from agentic_cli.tools.jobs.tools import job_status

    return [AgentConfig(name="a", prompt="p", tools=[job_status], model=model)]


class _Spy:
    """What a start built: the services (``_build_services`` calls) and the store openings."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, cls: type, store_opener: str) -> None:
        self.built: list[list[str]] = []
        self.stores: list[str] = []
        real_build = BaseWorkflowManager._build_services
        real_open = getattr(cls, store_opener)

        def build(manager, existing=frozenset()):
            services = real_build(manager, existing)
            self.built.append(sorted(services))
            return services

        def open_store(manager, *args, **kwargs):
            self.stores.append(store_opener)
            return real_open(manager, *args, **kwargs)

        monkeypatch.setattr(BaseWorkflowManager, "_build_services", build)
        monkeypatch.setattr(cls, store_opener, open_store)

    def reset(self) -> None:
        self.built.clear()
        self.stores.clear()


class TestEachBackendFailsBeforeBuildingAnything:
    @pytest.mark.parametrize(("path", "store_opener"), BACKENDS)
    async def test_a_window_too_small_to_work_in_builds_no_service(
        self, ctx, monkeypatch, path, store_opener
    ):
        cls = _backend(path)
        spy = _Spy(monkeypatch, cls, store_opener)
        manager = cls(_job_agent(SMALL), settings=ctx.settings)

        with pytest.raises(ContextBudgetError, match=r"'a'.*gemini-3\.8-flash"):
            await manager.initialize_services()

        # Neither a service (the jobs, the engine) nor the session store was
        # built, so the rollback had nothing but the catalog to release.
        assert (spy.built, spy.stores) == ([], [])
        assert not manager.is_initialized
        assert manager.services == {}
        assert manager.catalog is None
        assert manager._budgets == {}

    @pytest.mark.parametrize(("path", "store_opener"), BACKENDS)
    async def test_a_model_switch_to_one_too_small_builds_nothing_new(
        self, ctx, monkeypatch, path, store_opener
    ):
        cls = _backend(path)
        spy = _Spy(monkeypatch, cls, store_opener)
        manager = cls(_job_agent(), settings=ctx.settings)
        try:
            await manager.initialize_services()
            assert manager.context_budget().window > 16_000
            spy.reset()

            with pytest.raises(ContextBudgetError, match="gemini-3.8-flash"):
                await manager.reinitialize(model=SMALL)

            assert spy.built == []
            assert not manager.is_initialized
            assert manager._budgets == {}
        finally:
            await manager.cleanup()

    @pytest.mark.parametrize(("path", "store_opener"), BACKENDS)
    async def test_a_window_that_works_builds_services_and_keeps_its_budgets(
        self, ctx, monkeypatch, path, store_opener
    ):
        cls = _backend(path)
        manager = cls(_job_agent(), settings=ctx.settings)
        try:
            await manager.initialize_services()

            assert manager.is_initialized
            assert manager.context_budget() == context_budget(
                manager.agent_models["a"], ctx.settings
            )
        finally:
            await manager.cleanup()


class TestANativeAdkApp:
    async def test_the_configs_the_factory_passes_through_are_ignored(self, ctx, tmp_path):
        # ``create_workflow_manager_from_settings`` hands a native app's configs to
        # the manager too. ADK builds the agents from the YAML, so they are
        # ignored: a 16,000-token model on one of them is no reason to refuse
        # the start.
        path = tmp_path / "root_agent.yaml"
        path.write_text(
            "agent_class: LlmAgent\nname: root\nmodel: gemini-2.5-flash\n"
            "instruction: You are root.\n",
            encoding="utf-8",
        )
        cls = _backend(BACKENDS[0].values[0])
        manager = cls(_job_agent(SMALL), settings=ctx.settings, adk_config_path=str(path))
        try:
            await manager.initialize_services()

            assert manager.is_initialized
            assert "a" not in manager.agent_models
            with pytest.raises(KeyError, match="does not manage"):
                manager.context_budget("a")
        finally:
            await manager.cleanup()


# ---------------------------------------------------------------------------
# The base's own contract, on a backend written to the hooks alone
# ---------------------------------------------------------------------------


class _Backend(BaseWorkflowManager):
    """A backend of hooks only; it records what ``_budgets`` held at each step."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.steps: list[tuple[str, dict]] = []
        self.engine: object | None = None

    @property
    def backend_type(self) -> str:
        return "budgets-test"

    def _get_state_tools(self) -> list[Callable]:
        return []

    def _validate_agent_graph(self) -> AgentGraph:
        return validate_agent_graph(self._agent_configs, backend=self.backend_type)

    def _build_services(self, existing=frozenset()):
        self.steps.append(("services", dict(self._budgets)))
        return super()._build_services(existing)

    async def _do_initialize(self) -> None:
        self.steps.append(("do_initialize", dict(self._budgets)))
        self.engine = object()

    def _backend_ready(self) -> bool:
        return self._initialized and self.engine is not None

    def _detach_backend(self):
        self.engine = None
        return []

    async def process(self, message, user_id, session_id=None):
        async with self._turn_admission():
            yield WorkflowEvent.text("ok", session_id)


class _ManagesContext(_Backend):
    def _manages_context(self) -> bool:
        return True


class _OwnBudgets(_Backend):
    """A backend that computes its own numbers (a stand-in for each agent's)."""

    def _compute_budgets(self):
        return {name: "own" for name in self._resolved_models}


class TestTheBaseComputesThem:
    async def test_a_backend_that_manages_context_has_them_before_its_services(self, ctx):
        manager = _ManagesContext(_job_agent(), settings=ctx.settings)
        try:
            await manager.initialize_services()

            expected = {"a": context_budget(manager.agent_models["a"], ctx.settings)}
            assert manager.steps == [("services", expected), ("do_initialize", expected)]
            assert manager.context_budget() == expected["a"]
        finally:
            await manager.cleanup()

    async def test_one_that_cannot_work_on_its_model_is_never_built(self, ctx):
        manager = _ManagesContext(_job_agent(SMALL), settings=ctx.settings)

        with pytest.raises(ContextBudgetError, match="'a'"):
            await manager.initialize_services()

        assert manager.steps == []
        assert not manager.is_initialized
        assert manager.catalog is None

    async def test_a_backend_may_compute_its_own(self, ctx):
        manager = _OwnBudgets(_job_agent(SMALL), settings=ctx.settings)
        try:
            await manager.initialize_services()

            # Its numbers stand: the base's check does not run on top of them.
            assert manager.steps == [("services", {"a": "own"}), ("do_initialize", {"a": "own"})]
        finally:
            await manager.cleanup()

    async def test_a_backend_that_says_nothing_manages_none(self, ctx):
        manager = _Backend(_job_agent(SMALL), settings=ctx.settings)
        try:
            await manager.initialize_services()

            # No check on its model, and no number to show: as before.
            assert manager.steps == [("services", {}), ("do_initialize", {})]
            with pytest.raises(KeyError, match="does not manage"):
                manager.context_budget()
        finally:
            await manager.cleanup()

    async def test_the_numbers_are_released_with_the_agents_and_computed_again(self, ctx):
        manager = _ManagesContext(_job_agent(), settings=ctx.settings)
        await manager.initialize_services()
        await manager.cleanup()
        assert manager._budgets == {}

        await manager.initialize_services()
        try:
            assert manager.context_budget().window > 16_000
        finally:
            await manager.cleanup()
