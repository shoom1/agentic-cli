"""The ``/settings`` model picker offers what the model catalog can run.

Before the workflow is ready the options come from the shipped table (the
fallback models of each provider with a key); once it is ready, from the
workflow's own catalog, listings included. Either way they are full
``provider:model`` references, and the current value is the default model's.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentic_cli.cli.app import BaseCLIApp
from agentic_cli.config import BaseSettings
from agentic_cli.models import ModelCatalog, ProviderKind, load_table
from agentic_cli.workflow import model_services
from tests.models.fakes import ProviderAPI


@pytest.fixture(autouse=True)
def hermetic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for variable in ("GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(variable, raising=False)


def _app(settings: BaseSettings, workflow=None) -> BaseCLIApp:
    class _App(BaseCLIApp):
        def get_ui_setting_keys(self) -> list[str]:
            return ["model"]

    app = _App.__new__(_App)
    app._settings = settings
    app._workflow_controller = SimpleNamespace(is_ready=workflow is not None, workflow=workflow)
    return app


def _model_item(app: BaseCLIApp):
    items = [item for item in app._build_ui_items() if item.key == "model"]
    return items[0] if items else None


def _fallback_refs(kind: ProviderKind) -> set[str]:
    return {f"{kind.value}:{model}" for model in load_table().fallback_models(kind)}


def test_before_the_workflow_starts_the_table_s_models_of_each_keyed_provider():
    settings = BaseSettings(google_api_key="k-g", anthropic_api_key="k-a")

    item = _model_item(_app(settings))

    assert set(item.options) == (
        _fallback_refs(ProviderKind.GOOGLE) | _fallback_refs(ProviderKind.ANTHROPIC)
    )
    assert item.default == "google:gemini-2.5-flash"  # the default role, computed


def test_only_providers_with_a_key_are_offered():
    item = _model_item(_app(BaseSettings(google_api_key="k-g")))

    assert set(item.options) == _fallback_refs(ProviderKind.GOOGLE)


def test_the_current_model_shows_as_its_reference():
    settings = BaseSettings(google_api_key="k-g", default_model="gemini-2.5-pro")

    assert _model_item(_app(settings)).default == "google:gemini-2.5-pro"


def test_a_pinned_model_is_offered():
    settings = BaseSettings(
        google_api_key="k-g", openrouter_api_key="k-r",
        pinned_models=["openrouter:qwen/qwen3-235b-a22b"],
    )

    assert "openrouter:qwen/qwen3-235b-a22b" in _model_item(_app(settings)).options


def test_without_any_provider_there_is_no_model_picker():
    assert _model_item(_app(BaseSettings())) is None


async def test_a_ready_workflow_offers_its_catalog():
    settings = BaseSettings(google_api_key="k-g")
    catalog = ModelCatalog(
        endpoints=model_services.endpoints_from_settings(settings),
        transport=ProviderAPI().transport,
    )
    await catalog.load()
    workflow = SimpleNamespace(catalog=catalog, model="google:gemini-3-flash-preview")

    item = _model_item(_app(settings, workflow))

    assert item.options == catalog.picker_refs(referenced=["google:gemini-3-flash-preview"])
    assert "google:gemini-2.0-flash" in item.options  # from the listing, not the table
    assert item.default == "google:gemini-3-flash-preview"
