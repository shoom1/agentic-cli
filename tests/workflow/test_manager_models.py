"""A workflow manager's models: the catalog it loads, and every agent resolved at startup.

The manager builds a ``ModelCatalog`` from settings (``workflow.model_services``)
when it initializes, loads it (listing only what the cache cannot answer),
resolves the default model and every agent in one all-or-nothing pass, and
owns the catalog until cleanup. Listings come from ``ProviderAPI``, a fake of
every provider's model list; nothing reaches the network.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

pytest.importorskip("google.adk")

from agentic_cli.config import BaseSettings, SettingsValidationError  # noqa: E402
from agentic_cli.models import ModelCatalog, ModelRef, ModelResolutionError  # noqa: E402
from agentic_cli.workflow import model_services  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.model_settings import ModelSettings, ThinkingSettings  # noqa: E402
from tests.models.fakes import ProviderAPI, hang, recorded, refuse  # noqa: E402

KEYS = {
    "GOOGLE_API_KEY": "g-test-key",
    "ANTHROPIC_API_KEY": "a-test-key",
    "OPENAI_API_KEY": "o-test-key",
    "OPENROUTER_API_KEY": "r-test-key",
}
REFRESH_TASK = "model-catalog-refresh"


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    """A hermetic cwd and HOME, and no provider key unless a test sets one."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for variable in KEYS:
        monkeypatch.delenv(variable, raising=False)
    return tmp_path


def keys(monkeypatch, *providers: str) -> None:
    """Set the keys of ``providers`` (``google``, ``anthropic``, …) in the environment."""
    for provider in providers:
        variable = f"{provider.upper()}_API_KEY"
        monkeypatch.setenv(variable, KEYS[variable])


def settings(**values) -> BaseSettings:
    return BaseSettings(session_store="memory", **values)


@pytest.fixture
def providers(home, monkeypatch) -> ProviderAPI:
    """Serve the recorded listings to every catalog a manager builds."""
    api = ProviderAPI()
    cache = home / "home" / ".agentic_cli" / "model_cache.json"

    def build_catalog(settings, endpoints, *, transport=None):
        return ModelCatalog(endpoints=endpoints, cache_path=cache, transport=api.transport)

    monkeypatch.setattr(model_services, "build_catalog", build_catalog)
    return api


def team(writer_model: str | None = None, **writer) -> list[AgentConfig]:
    return [
        AgentConfig(name="coordinator", prompt="Coordinate.", sub_agents=["writer"]),
        AgentConfig(name="writer", prompt="Write.", model=writer_model, **writer),
    ]


def adk(configs, s, **kwargs) -> GoogleADKWorkflowManager:
    return GoogleADKWorkflowManager(agent_configs=configs, settings=s, **kwargs)


def refreshing() -> list[asyncio.Task]:
    return [t for t in asyncio.all_tasks() if t.get_name() == REFRESH_TASK and not t.done()]


# --- a 0.6 configuration still starts (Review Focus 1) -------------------------------


async def test_a_0_6_settings_file_still_starts(providers, home, monkeypatch):
    """Bare model ids, as 0.6 wrote them, in a project settings file and an
    agent: the manager starts and resolves each to its provider."""
    keys(monkeypatch, "google", "anthropic")
    project = home / ".agentic_cli"
    project.mkdir()
    (project / "settings.json").write_text(json.dumps({"default_model": "gemini-2.5-flash"}))
    manager = adk(team("claude-sonnet-4-5"), settings())

    await manager.initialize_services()
    try:
        assert manager.is_initialized
        assert manager.model == "google:gemini-2.5-flash"
        resolved = {name: r.ref for name, r in manager._resolved_models.items()}
        assert resolved == {
            "coordinator": ModelRef("google", "gemini-2.5-flash"),
            "writer": ModelRef("anthropic", "claude-sonnet-4-5"),
        }
        assert manager.catalog.is_authoritative("google")
        assert manager.catalog.is_authoritative("anthropic")
        # The backend's model code still takes bare ids until its adapter lands.
        root = manager._root_agent
        assert root.model == "gemini-2.5-flash"
        assert root.sub_agents[0].model.model == "claude-sonnet-4-5"
    finally:
        await manager.cleanup()


# --- resolution: every agent, once, all or nothing --------------------------------------


class TestResolution:
    async def test_an_agent_without_a_model_runs_on_the_default(self, providers, monkeypatch):
        keys(monkeypatch, "anthropic")
        manager = adk(team(), settings())

        await manager.initialize_services()
        try:
            refs = {name: str(r.ref) for name, r in manager._resolved_models.items()}
            assert refs == {"coordinator": "anthropic:claude-sonnet-5-5",
                            "writer": "anthropic:claude-sonnet-5-5"}
        finally:
            await manager.cleanup()

    async def test_an_agent_naming_a_role(self, providers, monkeypatch):
        keys(monkeypatch, "google")
        manager = adk(team("fast"), settings(model_roles={"fast": "google:gemini-2.5-flash-lite"}))

        await manager.initialize_services()
        try:
            assert str(manager._resolved_models["writer"].ref) == "google:gemini-2.5-flash-lite"
            assert manager._root_agent.sub_agents[0].model == "gemini-2.5-flash-lite"
        finally:
            await manager.cleanup()

    async def test_every_problem_is_reported_together_and_nothing_is_kept(
        self, providers, monkeypatch
    ):
        keys(monkeypatch, "google")
        configs = [
            AgentConfig(name="coordinator", prompt="p", sub_agents=["a", "b"],
                        model="gemini-9-ultra"),
            AgentConfig(name="a", prompt="p", model="anthropic:claude-sonnet-4-5"),
            AgentConfig(name="b", prompt="p", model="google:gemini-2.5-pro"),
        ]
        manager = adk(configs, settings())

        with pytest.raises(SettingsValidationError) as caught:
            await manager.initialize_services(validate=False)

        message = str(caught.value)
        assert "Agent 'coordinator': Model google:gemini-9-ultra is not available." in message
        assert "Agent 'a': model 'anthropic:claude-sonnet-4-5' needs the anthropic" in message
        assert "Agent 'b'" not in message
        assert manager._resolved_models == {}
        assert not manager.is_initialized
        assert manager.services == {}
        assert manager.catalog is None

    async def test_a_static_problem_costs_no_listing(self, providers, monkeypatch):
        keys(monkeypatch, "google")
        manager = adk(team("nowhere:some-model"), settings())

        with pytest.raises(SettingsValidationError, match="Agent 'writer'.*'nowhere'"):
            await manager.initialize_services()

        assert providers.calls == []

    async def test_an_unknown_id_is_rejected_only_by_its_own_fresh_listing(
        self, providers, monkeypatch
    ):
        """An Anthropic outage neither stops Google rejecting an unknown Gemini
        id nor makes Anthropic's fallback list reject a real model."""
        keys(monkeypatch, "google", "anthropic")
        providers.answers["anthropic"] = 500

        rejected = adk(team("gemini-9-ultra"), settings())
        with pytest.raises(SettingsValidationError, match="Available google models"):
            await rejected.initialize_services()

        accepted = adk(team("claude-unknown-9"), settings())
        await accepted.initialize_services()
        try:
            assert str(accepted._resolved_models["writer"].ref) == "anthropic:claude-unknown-9"
        finally:
            await accepted.cleanup()

    async def test_offline_any_id_is_accepted_but_a_missing_key_still_fails(
        self, providers, monkeypatch
    ):
        keys(monkeypatch, "google")
        providers.answers.update(google=refuse, anthropic=refuse)

        offline = adk(team("gemini-9-ultra"), settings())
        await offline.initialize_services()
        await offline.cleanup()

        keyless = adk(team("anthropic:claude-sonnet-4-5"), settings())
        with pytest.raises(SettingsValidationError, match="set ANTHROPIC_API_KEY"):
            await keyless.initialize_services()

    async def test_an_unmapped_app_role_runs_on_the_default_model(self, providers, monkeypatch):
        keys(monkeypatch, "google")
        manager = adk(team("vision"), settings(default_model="gemini-2.5-pro"))

        await manager.initialize_services()
        try:
            assert str(manager._resolved_models["writer"].ref) == "google:gemini-2.5-pro"
        finally:
            await manager.cleanup()


# --- the manager's own model (R4) ------------------------------------------------------


class TestManagerModel:
    def test_before_initialization_the_model_comes_from_settings_without_the_network(
        self, providers, monkeypatch
    ):
        keys(monkeypatch, "google")
        assert adk(team(), settings()).model == "google:gemini-2.5-flash"
        assert adk(team(), settings(default_model="claude-opus-4-7")).model == (
            "anthropic:claude-opus-4-7"
        )
        keys(monkeypatch, "anthropic")
        assert adk(team(), settings()).model == "google:gemini-2.5-flash"  # Google first
        monkeypatch.delenv("GOOGLE_API_KEY")
        assert adk(team(), settings()).model == "anthropic:claude-sonnet-4-6"
        assert providers.calls == []

    def test_without_any_key_there_is_no_default_model(self, home):
        with pytest.raises(ModelResolutionError, match="No model for role 'default'"):
            adk(team(), settings()).model  # noqa: B018

    async def test_an_override_takes_the_default_role_for_this_manager_only(
        self, providers, monkeypatch
    ):
        keys(monkeypatch, "google", "anthropic")
        s = settings(default_model="gemini-2.5-flash")
        manager = adk(team(), s, model="claude-sonnet-4-5")

        await manager.initialize_services()
        try:
            assert manager.model == "anthropic:claude-sonnet-4-5"
            assert str(manager._resolved_models["writer"].ref) == "anthropic:claude-sonnet-4-5"
            assert adk(team(), s).model == "google:gemini-2.5-flash"
        finally:
            await manager.cleanup()

    async def test_an_override_is_a_model_never_a_role(self, providers, monkeypatch):
        keys(monkeypatch, "google")
        manager = adk(team(), settings(), model="strong")

        with pytest.raises(SettingsValidationError, match="manager's model"):
            await manager.initialize_services()

    async def test_an_override_is_checked_with_the_agents(self, providers, monkeypatch):
        keys(monkeypatch, "google")
        manager = adk(team("gemini-9-ultra"), settings(), model="anthropic:claude-sonnet-4-5")

        with pytest.raises(SettingsValidationError) as caught:
            await manager.initialize_services(validate=False)

        assert "Default model: model 'anthropic:claude-sonnet-4-5' needs the anthropic" in str(
            caught.value
        )
        assert "Agent 'writer'" in str(caught.value)

    async def test_reinitialize_switches_the_model_and_back(self, providers, monkeypatch):
        keys(monkeypatch, "google", "anthropic")
        manager = adk(team(), settings())
        await manager.initialize_services()
        try:
            # The newest default-tier model Google lists.
            assert manager.model == "google:gemini-3-flash-preview"

            await manager.reinitialize(model="anthropic:claude-opus-4-7")
            assert manager.model == "anthropic:claude-opus-4-7"
            assert str(manager._resolved_models["writer"].ref) == "anthropic:claude-opus-4-7"

            await manager.reinitialize()
            assert manager.model == "google:gemini-3-flash-preview"
        finally:
            await manager.cleanup()


# --- what each backend can build until its adapter lands (R8, R14) -----------------------


def _langgraph(configs, s):
    pytest.importorskip("langgraph")
    from agentic_cli.workflow.langgraph import LangGraphWorkflowManager

    return LangGraphWorkflowManager(agent_configs=configs, settings=s, checkpointer="memory")


class TestBackendGates:
    async def test_adk_builds_google_and_anthropic_models_only(self, providers, monkeypatch):
        keys(monkeypatch, "google", "openai")
        manager = adk(team("openai:gpt-5.5"), settings())

        with pytest.raises(SettingsValidationError) as caught:
            await manager.initialize_services()

        assert str(caught.value) == (
            "Agent 'writer': model openai:gpt-5.5 is not supported by the ADK backend yet; "
            "use a Google or Anthropic model."
        )

    async def test_the_adk_gate_covers_the_default_model(self, providers, monkeypatch):
        keys(monkeypatch, "openrouter")
        manager = adk(
            [AgentConfig(name="solo", prompt="p", model="openrouter:qwen/qwen3-235b-a22b")],
            settings(default_model="openrouter:qwen/qwen3-235b-a22b"),
        )

        with pytest.raises(SettingsValidationError, match="^Default model: model openrouter:"):
            await manager.initialize_services()

    async def test_langgraph_builds_google_anthropic_and_openai_models_only(
        self, providers, monkeypatch
    ):
        keys(monkeypatch, "google", "openrouter")
        manager = _langgraph(team("openrouter:qwen/qwen3-235b-a22b"), settings())

        with pytest.raises(SettingsValidationError) as caught:
            await manager.initialize_services()

        assert str(caught.value) == (
            "Agent 'writer': model openrouter:qwen/qwen3-235b-a22b is not supported by the "
            "LangGraph backend yet; use a Google, Anthropic or OpenAI model."
        )

    async def test_langgraph_takes_an_openai_model(self, providers, monkeypatch):
        pytest.importorskip("langchain_openai")
        keys(monkeypatch, "google", "openai")
        manager = _langgraph(team("openai:o3"), settings())

        await manager.initialize_services()
        try:
            assert str(manager._resolved_models["writer"].ref) == "openai:o3"
        finally:
            await manager.cleanup()

    @pytest.mark.parametrize("backend, label", [("adk", "ADK"), ("langgraph", "LangGraph")])
    async def test_a_native_thinking_level_waits_for_the_adapters(
        self, providers, monkeypatch, backend, label
    ):
        keys(monkeypatch, "anthropic")
        configs = team(
            "anthropic:claude-opus-4-7",
            model_settings=ModelSettings(thinking=ThinkingSettings(mode="xhigh")),
        )
        manager = adk(configs, settings()) if backend == "adk" else _langgraph(configs, settings())

        with pytest.raises(SettingsValidationError) as caught:
            await manager.initialize_services()

        assert str(caught.value) == (
            f"Agent 'writer': thinking mode 'xhigh' is not supported by the {label} backend yet."
        )

    async def test_budget_on_a_model_without_budget_thinking_is_an_error(
        self, providers, monkeypatch
    ):
        keys(monkeypatch, "google")
        configs = team(
            "gemini-3-flash-preview",
            model_settings=ModelSettings(
                thinking=ThinkingSettings(mode="budget", budget_tokens=4096)
            ),
        )

        manager = adk(configs, settings())

        with pytest.raises(SettingsValidationError, match="mode 'budget' is not available"):
            await manager.initialize_services()


# --- the catalog belongs to the manager -------------------------------------------------


def _stale_cache(home: Path) -> None:
    """A Google listing three days old, so startup uses it and refreshes it later."""
    cache = home / "home" / ".agentic_cli" / "model_cache.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    models = [{"id": m} for m in ("gemini-2.5-flash", "gemini-2.5-pro")]
    cache.write_text(json.dumps({"version": 1, "providers": {"google": {
        "fetched_at": time.time() - 3 * 86400, "ok": True, "models": models,
    }}}))


class TestCatalogLifecycle:
    async def test_cleanup_closes_the_catalog(self, providers, monkeypatch):
        keys(monkeypatch, "google")
        manager = adk(team(), settings())
        await manager.initialize_services()
        catalog = manager.catalog

        await manager.cleanup()
        await manager.cleanup()  # idempotent

        assert manager.catalog is None
        assert catalog.start_background_refresh() is None  # closed

    async def test_a_failed_initialization_releases_its_catalog(self, providers, monkeypatch):
        keys(monkeypatch, "google")
        built: list[ModelCatalog] = []
        build = model_services.build_catalog

        def recording(settings, endpoints, *, transport=None):
            built.append(build(settings, endpoints))
            return built[-1]

        monkeypatch.setattr(model_services, "build_catalog", recording)
        manager = adk(team("gemini-9-ultra"), settings())

        with pytest.raises(SettingsValidationError):
            await manager.initialize_services()

        assert manager.catalog is None
        assert built[0].start_background_refresh() is None  # closed

    async def test_a_stale_listing_is_refreshed_after_a_successful_start(
        self, providers, home, monkeypatch
    ):
        keys(monkeypatch, "google")
        _stale_cache(home)
        providers.answers["google"] = hang
        manager = adk(team(), settings())

        await manager.initialize_services()  # does not wait on the hung listing
        try:
            assert len(refreshing()) == 1
        finally:
            await manager.cleanup()
        assert refreshing() == []

    async def test_a_failed_start_leaves_no_refresh_running(self, providers, home, monkeypatch):
        keys(monkeypatch, "google")
        _stale_cache(home)
        providers.answers["google"] = hang
        manager = adk(team("anthropic:claude-sonnet-4-5"), settings())

        with pytest.raises(SettingsValidationError):
            await manager.initialize_services(validate=False)

        assert refreshing() == []

    async def test_reinitialize_cancels_the_background_refresh(self, providers, home, monkeypatch):
        """Review Focus 4, first half: no catalog task outlives a model switch
        and a cleanup."""
        keys(monkeypatch, "google")
        _stale_cache(home)
        providers.answers["google"] = hang
        manager = adk(team(), settings())
        await manager.initialize_services()
        first = refreshing()
        assert len(first) == 1

        await manager.reinitialize(model="gemini-2.5-pro")
        assert all(task.done() for task in first)
        assert manager.model == "google:gemini-2.5-pro"

        await manager.cleanup()
        assert refreshing() == []

    async def test_listing_at_startup_keeps_the_event_loop_free(self, providers, monkeypatch):
        keys(monkeypatch, "google")

        async def slow(request):
            await asyncio.sleep(0.2)
            return recorded("google")

        providers.answers["google"] = slow
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        counting = asyncio.create_task(ticker())
        manager = adk(team(), settings())
        try:
            await manager.initialize_services()
        finally:
            counting.cancel()
            await manager.cleanup()

        assert ticks >= 5
