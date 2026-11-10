"""The ``/settings`` model picker offers the list (``ModelCatalog.offered``).

The ``models`` entries, then the models the settings and the agents name and
the default model, each once; without an entry that can run, each keyed
built-in provider's fast, default and strong models. Before the workflow is
ready the list comes from an unloaded catalog (the entries and the shipped
table); once it is ready, from the workflow's own catalog, listings included.
Either way the options are full ``provider:model`` references, and the
current value is the default model's: one the list leaves out is offered all
the same, marked, so a Save never changes a model the user did not touch.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

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


def _app(settings: BaseSettings, workflow=None, agents=(), keys=("model",)) -> BaseCLIApp:
    class _App(BaseCLIApp):
        def get_ui_setting_keys(self) -> list[str]:
            return list(keys)

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
        "google:gemini-3.8-flash",
        "google:gemini-3.5-flash-lite",
        "google:gemini-3.1-pro-preview",
        "anthropic:claude-haiku-5-5",
        "anthropic:claude-sonnet-5-5",
        "anthropic:claude-opus-5-5",
    ]
    assert item.default == "google:gemini-3.8-flash"  # the default role, computed


def test_only_providers_with_a_key_are_offered():
    item = _model_item(_app(BaseSettings(google_api_key="k-g")))

    assert item.options == [
        "google:gemini-3.8-flash",
        "google:gemini-3.5-flash-lite",
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


def test_every_role_s_model_is_offered_though_the_app_shows_one_picker():
    """The app shows the default model's picker only: the fast role's model,
    computed, is in the list all the same, as /models shows it."""
    settings = BaseSettings(google_api_key="k-g", models={"google:gemini-2.5-pro": {}})

    item = _model_item(_app(settings))

    assert item.options == [
        "google:gemini-2.5-pro",  # the entry, and the strong role's
        "google:gemini-3.8-flash",  # the default role's
        "google:gemini-3.5-flash-lite",  # the fast role's
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
    assert "google:gemini-3.8-flash" in options  # no entry can run: the tiers


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


# --- the current model, when the list leaves it out ---------------------------------


class TestTheCurrentModelIsAlwaysOffered:
    """A picker shows the model its role runs on now. When the list leaves that
    model out (its provider has no key, or does not serve it: a listing
    recorded it as absent, or this process's fresh listing lacks it), the
    picker still offers it, first, marked "(current)", and its description
    starts with "Not offered" and why. So a Save of the dialog as it opened
    changes no model, and a user who moved off it can pick it back. The
    marker is short, as the description says the rest: a long reference with
    a longer marker cut the picker's label, and the open menu cut the marker
    away."""

    MARKED = "{} (current)"

    def test_a_model_whose_provider_has_no_key(self):
        settings = BaseSettings(google_api_key="k-g", default_model="anthropic:claude-sonnet-4-6")

        item = _model_item(_app(settings))

        marked = self.MARKED.format("anthropic:claude-sonnet-4-6")
        assert item.options == [
            marked,
            "google:gemini-3.5-flash-lite",
            "google:gemini-3.1-pro-preview",
            "google:gemini-3.8-flash",
        ]
        assert item.default == marked
        assert item.description == (
            "Not offered (no key: set ANTHROPIC_API_KEY); "
            "agents without a model of their own run on it"
        )

    async def test_a_model_this_process_s_fresh_listing_lacks(self, monkeypatch):
        from tests.conftest import patch_catalogs

        patch_catalogs(monkeypatch, transport=ProviderAPI().transport)
        settings = BaseSettings(google_api_key="k-g", default_model="google:gemini-9-retired")
        catalog = model_services.build_catalog(
            settings, model_services.providers_from_settings(settings)
        )
        await catalog.load()
        workflow = SimpleNamespace(catalog=catalog, model="google:gemini-9-retired")

        item = _model_item(_app(settings, workflow))

        marked = self.MARKED.format("google:gemini-9-retired")
        assert catalog.not_offered() == {"google:gemini-9-retired": "not served by google"}
        assert item.options == [
            marked,
            "google:gemini-2.5-flash-lite",
            "google:gemini-3.1-pro-preview",
            "google:gemini-3-flash-preview",
        ]
        assert item.default == marked
        assert item.description.startswith("Not offered (not served by google); ")
        await catalog.aclose()

    def test_a_role_s_model(self):
        settings = BaseSettings(google_api_key="k-g", model_roles={"fast": "openai:gpt-5-mini"})
        app = _app(settings, keys=["model", "model_role:fast"])

        items = {item.key: item for item in app._build_ui_items()}

        fast = items["model_role:fast"]
        marked = self.MARKED.format("openai:gpt-5-mini")
        assert fast.options == [
            marked,
            "google:gemini-3.8-flash",
            "google:gemini-3.1-pro-preview",
            "google:gemini-3.5-flash-lite",
        ]
        assert fast.default == marked
        assert fast.description == (
            "Not offered (no key: set OPENAI_API_KEY); agents whose model is 'fast' run on it"
        )
        # The default model's picker offers the list as it is: its model is in it.
        assert items["model"].options == fast.options[1:]
        assert items["model"].default == "google:gemini-3.8-flash"

    def test_a_model_in_the_list_is_not_marked(self):
        settings = BaseSettings(google_api_key="k-g", default_model="gemini-2.5-pro")

        item = _model_item(_app(settings))

        assert item.default == "google:gemini-2.5-pro" == item.options[0]
        assert not any(option.endswith(" (current)") for option in item.options)
        assert item.description == "Agents without a model of their own run on it"

    def test_one_left_out_for_a_reason_the_catalog_does_not_give(self, monkeypatch):
        """The description still says it is not offered."""
        monkeypatch.setattr(ModelCatalog, "not_offered", lambda self: {})
        settings = BaseSettings(google_api_key="k-g", default_model="anthropic:claude-sonnet-4-6")

        item = _model_item(_app(settings))

        assert item.default == item.options[0] == self.MARKED.format("anthropic:claude-sonnet-4-6")
        assert item.description == "Not offered; agents without a model of their own run on it"

    @pytest.mark.parametrize("picked", ["as it opened", "picked again"])
    async def test_saving_it_changes_no_model(self, monkeypatch, picked):
        from agentic_cli.cli.settings_command import SettingsCommand
        from tests.cli.dialog_fakes import autospec_session, edited_dialog

        settings = BaseSettings(google_api_key="k-g", default_model="anthropic:claude-sonnet-4-6")
        app = _app(settings)
        app.session = autospec_session()
        app.session.show_dialog.return_value = True  # Save
        app.apply_settings = AsyncMock(return_value=True)
        edits = {}
        if picked == "picked again":  # moved off it, then back
            edits = {"model": self.MARKED.format("anthropic:claude-sonnet-4-6")}
        monkeypatch.setattr("agentic_cli.cli.settings_command.Dialog", edited_dialog(edits))

        await SettingsCommand().execute("", app)

        app.apply_settings.assert_not_awaited()
        app.session.add_message.assert_called_once_with("system", "Settings unchanged.")
        assert settings.default_model == "anthropic:claude-sonnet-4-6"
