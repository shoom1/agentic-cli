"""workflow/model_services.py is where settings become models-package arguments."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import httpx
import pytest

from agentic_cli.config import BaseSettings
from agentic_cli.models import ModelRef, ModelSpecError, ProviderKind, RoleMap
from agentic_cli.workflow import model_services

# Bound before the test-suite guard (conftest ``_no_model_listings``) replaces
# the module attribute: these tests check the real builder.
from agentic_cli.workflow.model_services import build_catalog

_KEY_VARS = ("GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """A temp home (no real user settings), a temp project, and no keys in the env."""
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path / "project")
    for var in (*_KEY_VARS, "LAB_KEY", "AGENTIC_DEFAULT_MODEL"):
        monkeypatch.delenv(var, raising=False)


def _lab(**extra) -> dict:
    return {"lab": {"base_url": "http://gpu-box:8000/v1", "api_key_env": "LAB_KEY", **extra}}


def test_provider_keys_reads_the_four_key_settings():
    settings = BaseSettings(
        google_api_key="k-g", anthropic_api_key="k-a", openai_api_key="k-o", openrouter_api_key="k-r"
    )

    assert model_services.provider_keys(settings) == {
        "google": "k-g", "anthropic": "k-a", "openai": "k-o", "openrouter": "k-r",
    }


def test_provider_keys_without_keys():
    assert model_services.provider_keys(BaseSettings()) == dict.fromkeys(
        ("google", "anthropic", "openai", "openrouter")
    )


def test_endpoints_from_settings():
    settings = BaseSettings(anthropic_api_key="k-a", model_endpoints=_lab())

    endpoints = model_services.endpoints_from_settings(settings, env={"LAB_KEY": "k-lab"})

    assert set(endpoints) == {"google", "anthropic", "openai", "openrouter", "lab"}
    assert endpoints["anthropic"].api_key == "k-a"
    assert not endpoints["google"].has_credentials
    assert endpoints["lab"].kind is ProviderKind.OPENAI_COMPATIBLE
    assert endpoints["lab"].base_url == "http://gpu-box:8000/v1"
    assert endpoints["lab"].api_key == "k-lab"


def test_an_endpoint_key_comes_from_the_process_environment_by_default(monkeypatch):
    monkeypatch.setenv("LAB_KEY", "k-from-env")
    settings = BaseSettings(openai_api_key="k-o", model_endpoints=_lab())

    endpoints = model_services.endpoints_from_settings(settings)

    assert endpoints["lab"].api_key == "k-from-env"
    assert endpoints["openai"].api_key == "k-o"


def test_role_map_from_settings():
    settings = BaseSettings(default_model="gemini-2.5-flash", model_roles={"fast": "claude-haiku-4-5"})

    roles = model_services.role_map_from_settings(settings, manager_model="openai:gpt-5.5")

    assert isinstance(roles, RoleMap)
    assert roles.spec_for("default") == ModelRef("openai", "gpt-5.5")
    assert roles.spec_for("fast") == ModelRef("anthropic", "claude-haiku-4-5")
    assert model_services.role_map_from_settings(settings).spec_for("default") == ModelRef(
        "google", "gemini-2.5-flash"
    )


def test_a_default_model_that_names_a_role_is_an_error():
    """Settings reject it when they load; the role map rejects a manager's own."""
    with pytest.raises(ValueError, match="default_model"):
        BaseSettings(default_model="fast")
    with pytest.raises(ModelSpecError, match="manager's model"):
        model_services.role_map_from_settings(BaseSettings(), manager_model="fast")


def test_model_cache_path_is_in_the_user_folder(tmp_path):
    settings = BaseSettings(app_name="research_demo")

    assert model_services.model_cache_path(settings) == (
        Path(tmp_path / "home") / ".research_demo" / "model_cache.json"
    )


async def test_build_catalog_caches_listings_in_the_user_folder():
    settings = BaseSettings(google_api_key="k-g")
    endpoints = model_services.endpoints_from_settings(settings)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"models": []}))

    catalog = build_catalog(settings, endpoints, transport=transport)
    await catalog.load()

    cache = model_services.model_cache_path(settings)
    assert json.loads(cache.read_text())["providers"]["google"]["ok"] is True
    assert stat.S_IMODE(cache.stat().st_mode) == 0o600
    assert b"k-g" not in cache.read_bytes()
    assert catalog.is_authoritative("google")
