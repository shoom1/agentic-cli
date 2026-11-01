"""The research demo ships its model list and the OpenRouter provider.

What an app built on agentic-cli does to offer a short list of models: it
overrides the defaults of ``models`` and ``model_providers`` in its settings
class. These tests pin the demo's list, that its settings class loads without
a warning, and that the list offers exactly the entries whose providers have
a key. Every key is fake, set in the environment (or written into the temp
home's ``.research_demo/.env``); none is read from the developer's files, and
no catalog is loaded, so nothing reaches a provider.
"""

from __future__ import annotations

import json
import os
import runpy
import warnings
from pathlib import Path

import pytest

from agentic_cli import BaseSettings
from agentic_cli.config import SettingsValidationError, validate_settings
from agentic_cli.models import ModelEntry, ProviderKind, ProviderSpec, load_table, parse_entry_key
from agentic_cli.workflow import model_services
from tests.demo_isolation import ResearchDemoSettings, isolated_env_file, make_isolated_settings

SETTINGS_MODULE = (
    Path(__file__).resolve().parents[2] / "examples" / "research_demo" / "settings.py"
)

#: The demo's list, in its order.
DEMO_MODELS = [
    "google:gemini-3.5-flash-lite",
    "google:gemini-3.8-flash",
    "anthropic:claude-haiku-5-5",
    "anthropic:claude-sonnet-5-5",
    "anthropic:claude-opus-5-5",
    "openrouter:qwen/qwen3.7-plus",
    "openrouter:deepseek/deepseek-v4-pro",
    "openrouter:moonshotai/kimi-k3",
]
#: The tier each built-in model has in the shipped table.
BUILTIN_TIERS = {
    "google:gemini-3.5-flash-lite": "fast",
    "google:gemini-3.8-flash": "default",
    "anthropic:claude-haiku-5-5": "fast",
    "anthropic:claude-sonnet-5-5": "default",
    "anthropic:claude-opus-5-5": "strong",
}
#: Each provider of the list and the variable its key is read from.
KEY_VARIABLES = {
    "google": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temp HOME and cwd, and no provider key or demo setting in the
    environment, whatever the process holds (the live-test conftest exports
    every line of ~/.research_demo/.env, such as RESEARCH_DEMO_DEFAULT_MODEL)."""
    home, project = tmp_path / "home", tmp_path / "project"
    for folder in (home, project):
        folder.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)
    for variable in ("OPENAI_API_KEY", *KEY_VARIABLES.values()):
        monkeypatch.delenv(variable, raising=False)
    for variable in [name for name in os.environ if name.startswith("RESEARCH_DEMO_")]:
        monkeypatch.delenv(variable)
    return home


def _settings(home: Path) -> ResearchDemoSettings:
    """The demo's settings, reading no file of the developer's."""
    return make_isolated_settings(home=home, workspace=home / "ws")


def _offered(settings: ResearchDemoSettings) -> list[str]:
    """The list the model pickers offer, before any listing: the demo's
    agents name no model, so it is the entries that can run."""
    from examples.research_demo.agents import AGENT_CONFIGS

    providers = model_services.providers_from_settings(settings)
    catalog = model_services.build_catalog(settings, providers, agent_configs=AGENT_CONFIGS)
    return catalog.offered()


def test_the_settings_class_is_defined_and_loaded_without_a_warning(home):
    """An app overriding the framework's model fields gets no pydantic
    "shadows an attribute in parent" warning, which would print over the UI
    at every start."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        namespace = runpy.run_path(str(SETTINGS_MODULE), run_name="research_demo_settings")
        namespace["ResearchDemoSettings"](
            workspace_dir=home / "ws", _env_file=str(isolated_env_file(home))
        )

    assert [str(warning.message) for warning in caught] == []


def test_it_ships_the_openrouter_provider(home):
    assert _settings(home).model_providers == {
        "openrouter": ProviderSpec(
            kind="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
        ),
    }


def test_it_ships_its_list_of_models_each_described_by_its_listing_and_the_table(home):
    models = _settings(home).models

    assert list(models) == DEMO_MODELS
    assert all(isinstance(entry, ModelEntry) for entry in models.values())
    assert {key: entry.model_fields_set for key, entry in models.items()} == {
        key: set() for key in DEMO_MODELS
    }


def test_the_fields_keep_the_framework_s_title_and_description():
    for name in ("models", "model_providers"):
        demo, base = ResearchDemoSettings.model_fields[name], BaseSettings.model_fields[name]
        assert (demo.title, demo.description) == (base.title, base.description), name


@pytest.mark.parametrize("key", list(BUILTIN_TIERS))
def test_each_built_in_model_is_in_the_shipped_table_with_its_tier(key):
    ref = parse_entry_key(key)
    record = load_table().record_for(ref.provider, ProviderKind(ref.provider), ref.model)

    assert record.tier == BUILTIN_TIERS[key]
    # The model's own numbers, not only its family's rule.
    assert record.sources["context_window"] == "table"


def test_another_source_adds_to_the_list_and_removes_nothing(home, monkeypatch):
    """The user's, the project's or (here) the environment's entries follow
    the demo's: none replaces its list."""
    monkeypatch.setenv("RESEARCH_DEMO_MODELS", '{"openai:gpt-5.5": {}}')

    assert list(_settings(home).models) == [*DEMO_MODELS, "openai:gpt-5.5"]


@pytest.mark.parametrize(
    "keyed",
    [
        ("google",),
        ("anthropic", "openrouter"),
        ("openrouter",),
        ("google", "anthropic", "openrouter"),
    ],
    ids="+".join,
)
def test_the_list_is_the_entries_whose_providers_have_a_key(home, monkeypatch, keyed):
    for provider in keyed:
        monkeypatch.setenv(KEY_VARIABLES[provider], f"fake-{provider}-key")
    settings = _settings(home)

    assert _offered(settings) == [
        key for key in DEMO_MODELS if parse_entry_key(key).provider in keyed
    ]
    providers = model_services.providers_from_settings(settings)
    catalog = model_services.build_catalog(settings, providers)
    assert catalog.not_offered() == {
        key: f"no key: set {KEY_VARIABLES[parse_entry_key(key).provider]}"
        for key in DEMO_MODELS
        if parse_entry_key(key).provider not in keyed
    }


def test_the_openrouter_key_is_read_from_the_demo_s_env_file(home):
    """The demo's docstring says to put OPENROUTER_API_KEY in
    ~/.research_demo/.env (here, the temp home's)."""
    env_file = isolated_env_file(home)
    env_file.parent.mkdir(parents=True)
    env_file.write_text("OPENROUTER_API_KEY=fake-openrouter-key\n")

    assert _offered(_settings(home)) == [
        key for key in DEMO_MODELS if key.startswith("openrouter:")
    ]


def test_without_a_key_nothing_is_offered(home):
    assert _offered(_settings(home)) == []


def test_with_an_anthropic_key_each_role_s_model_is_in_the_list(home, monkeypatch):
    """Haiku, Sonnet and Opus fill fast, default and strong: the pickers
    offer the list and nothing beside it."""
    from examples.research_demo.agents import AGENT_CONFIGS

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key")
    settings = _settings(home)
    providers = model_services.providers_from_settings(settings)
    catalog = model_services.build_catalog(settings, providers, agent_configs=AGENT_CONFIGS)

    assert model_services.role_models(settings, catalog, providers) == {
        "anthropic:claude-sonnet-5-5": ["default"],
        "anthropic:claude-haiku-5-5": ["fast"],
        "anthropic:claude-opus-5-5": ["strong"],
    }
    assert catalog.offered(extra=model_services.role_models(settings, catalog, providers)) == [
        key for key in DEMO_MODELS if key.startswith("anthropic:")
    ]


def test_with_an_openrouter_key_alone_the_default_model_is_chosen(home, monkeypatch):
    """What the docs tell a user with an OpenRouter key alone: an OpenRouter
    model has no tier, so the default model is set by hand."""
    from examples.research_demo.agents import AGENT_CONFIGS

    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-openrouter-key")
    with pytest.raises(SettingsValidationError, match="No model for role 'default'"):
        validate_settings(_settings(home), AGENT_CONFIGS)

    user_file = home / ".research_demo" / "settings.json"
    user_file.parent.mkdir(parents=True, exist_ok=True)
    user_file.write_text(json.dumps({"default_model": "openrouter:qwen/qwen3.7-plus"}))
    validate_settings(_settings(home), AGENT_CONFIGS)
