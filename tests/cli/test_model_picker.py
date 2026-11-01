"""The ``/settings`` model picker offers the list (``ModelCatalog.offered``).

The ``models`` entries, then the models the settings and the agents name and
the default model, each once; without an entry that can run, each keyed
built-in provider's fast, default and strong models. Before the workflow is
ready the list comes from an unloaded catalog (the entries and the shipped
table); once it is ready, from the workflow's own catalog, listings included.
Either way the options are full ``provider:model`` references, and the
current value is the default model's.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentic_cli.cli.app import BaseCLIApp
from agentic_cli.config import BaseSettings
from agentic_cli.models import ModelCatalog
from agentic_cli.workflow import model_services
from agentic_cli.workflow.config import AgentConfig
from tests.models.fakes import ProviderAPI, openrouter_spec


@pytest.fixture(autouse=True)
def hermetic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for variable in ("GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(variable, raising=False)


def _app(settings: BaseSettings, workflow=None, agents=()) -> BaseCLIApp:
    class _App(BaseCLIApp):
        def get_ui_setting_keys(self) -> list[str]:
            return ["model"]

    app = _App.__new__(_App)
    app._settings = settings
    app._workflow_controller = SimpleNamespace(
        is_ready=workflow is not None, workflow=workflow, agent_configs=list(agents)
    )
    return app


def _model_item(app: BaseCLIApp):
    items = [item for item in app._build_ui_items() if item.key == "model"]
    return items[0] if items else None


def test_before_the_workflow_starts_each_keyed_provider_s_tiers_from_the_table():
    """No entry: the default model (computed), then each provider's newest
    fast, default and strong model among the table's."""
    settings = BaseSettings(google_api_key="k-g", anthropic_api_key="k-a")

    item = _model_item(_app(settings))

    assert item.options == [
        "google:gemini-2.5-flash",
        "google:gemini-3.1-flash-lite-preview",
        "google:gemini-3.1-pro-preview",
        "anthropic:claude-haiku-4-5",
        "anthropic:claude-sonnet-4-6",
        "anthropic:claude-opus-4-6",
    ]
    assert item.default == "google:gemini-2.5-flash"  # the default role, computed


def test_only_providers_with_a_key_are_offered():
    item = _model_item(_app(BaseSettings(google_api_key="k-g")))

    assert item.options == [
        "google:gemini-2.5-flash",
        "google:gemini-3.1-flash-lite-preview",
        "google:gemini-3.1-pro-preview",
    ]


def test_the_current_model_shows_as_its_reference():
    settings = BaseSettings(google_api_key="k-g", default_model="gemini-2.5-pro")

    assert _model_item(_app(settings)).default == "google:gemini-2.5-pro"


def test_the_list_is_the_entries_then_what_the_settings_and_the_agents_name(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k-r")
    settings = BaseSettings(
        google_api_key="k-g",
        anthropic_api_key="k-a",
        model_providers={"openrouter": openrouter_spec()},
        models={"openrouter:qwen/qwen3-235b-a22b": {}, "google:gemini-2.5-pro": {}},
        default_model="anthropic:claude-sonnet-4-6",
        model_roles={"fast": "google:gemini-2.5-flash-lite"},
    )
    agents = [AgentConfig(name="writer", prompt="Write.", model="anthropic:claude-haiku-4-5")]

    item = _model_item(_app(settings, agents=agents))

    # Not every model of every provider with a key: the list.
    assert item.options == [
        "openrouter:qwen/qwen3-235b-a22b",
        "google:gemini-2.5-pro",
        "anthropic:claude-sonnet-4-6",
        "google:gemini-2.5-flash-lite",
        "anthropic:claude-haiku-4-5",
    ]


def test_a_models_entry_on_a_provider_without_its_key_is_not_offered():
    settings = BaseSettings(
        google_api_key="k-g",
        model_providers={"openrouter": openrouter_spec()},
        models={"openrouter:qwen/qwen3-235b-a22b": {}},
    )

    assert "openrouter:qwen/qwen3-235b-a22b" not in _model_item(_app(settings)).options


def test_a_models_entry_on_a_provider_nobody_defined_is_not_offered():
    settings = BaseSettings(google_api_key="k-g", models={"together:deepseek-ai/DeepSeek-V3.1": {}})

    options = _model_item(_app(settings)).options

    assert "together:deepseek-ai/DeepSeek-V3.1" not in options
    assert "google:gemini-2.5-flash" in options  # no entry can run: the tiers


def test_without_any_provider_there_is_no_model_picker():
    assert _model_item(_app(BaseSettings())) is None


async def test_a_ready_workflow_offers_its_catalog_s_list():
    settings = BaseSettings(google_api_key="k-g")
    catalog = ModelCatalog(
        providers=model_services.providers_from_settings(settings),
        transport=ProviderAPI().transport,
    )
    await catalog.load()
    workflow = SimpleNamespace(catalog=catalog, model="google:gemini-3-flash-preview")

    item = _model_item(_app(settings, workflow))

    assert item.options == catalog.offered(extra=["google:gemini-3-flash-preview"])
    assert item.options == [
        "google:gemini-3-flash-preview",
        "google:gemini-2.5-flash-lite",  # the listing's newest fast model, not the table's
        "google:gemini-3.1-pro-preview",
    ]
    assert item.default == "google:gemini-3-flash-preview"
