"""Providers: the three built-ins and the ones ``model_providers`` defines."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import agentic_cli.models as models
from agentic_cli.models import (
    PROVIDER_KEY_ENV,
    ModelSpecError,
    Provider,
    ProviderKind,
    ProviderSpec,
    build_providers,
)

_NO_KEYS = {"google": None, "anthropic": None, "openai": None}
_SRC = Path(models.__file__).resolve().parents[1]


def _keys(**given: str) -> dict[str, str | None]:
    return {**_NO_KEYS, **given}


def test_the_three_builtins_are_always_there():
    providers = build_providers(keys=_NO_KEYS, specs={}, env={})

    assert {name: (p.kind, p.base_url, p.key_env) for name, p in providers.items()} == {
        "google": (ProviderKind.GOOGLE, None, "GOOGLE_API_KEY"),
        "anthropic": (ProviderKind.ANTHROPIC, None, "ANTHROPIC_API_KEY"),
        "openai": (ProviderKind.OPENAI, None, "OPENAI_API_KEY"),
    }
    assert all(provider.builtin for provider in providers.values())
    assert dict(PROVIDER_KEY_ENV) == {
        "google": "GOOGLE_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
    }


def test_openrouter_is_not_built_in():
    """It is a provider an app or a user configures (kind openrouter); the
    framework holds neither its address nor its key variable."""
    assert not hasattr(models, "OPENROUTER_BASE_URL")
    shipped = [*_SRC.rglob("*.py"), *_SRC.rglob("*.json")]
    assert [path for path in shipped if "openrouter.ai" in path.read_text(encoding="utf-8")] == []
    assert "OPENROUTER_API_KEY" not in PROVIDER_KEY_ENV.values()


def test_a_builtin_has_credentials_only_with_its_key():
    providers = build_providers(keys=_keys(anthropic="k-anthropic"), specs={}, env={})

    assert providers["anthropic"].api_key == "k-anthropic"
    assert providers["anthropic"].has_credentials
    assert not providers["google"].has_credentials
    assert not providers["openai"].has_credentials


def test_a_spec_becomes_a_configured_provider():
    spec = ProviderSpec(base_url="http://localhost:8000/v1", api_key_env="LOCAL_VLLM_KEY")

    providers = build_providers(
        keys=_NO_KEYS, specs={"local-vllm": spec}, env={"LOCAL_VLLM_KEY": "k-vllm"}
    )

    local = providers["local-vllm"]
    assert (local.name, local.kind, local.base_url) == (
        "local-vllm", ProviderKind.OPENAI_COMPATIBLE, "http://localhost:8000/v1",
    )
    assert (local.api_key, local.key_env) == ("k-vllm", "LOCAL_VLLM_KEY")
    assert not local.builtin
    assert list(providers) == ["google", "anthropic", "openai", "local-vllm"]


def test_an_openrouter_provider_keeps_its_kind_under_any_name():
    spec = ProviderSpec(
        kind="openrouter", base_url="https://openrouter.ai/api/v1", api_key_env="OR_KEY"
    )

    provider = build_providers(keys=_NO_KEYS, specs={"or": spec}, env={"OR_KEY": "k-or"})["or"]

    assert (provider.kind, provider.base_url, provider.api_key) == (
        ProviderKind.OPENROUTER, "https://openrouter.ai/api/v1", "k-or",
    )
    assert not provider.builtin


def test_a_configured_key_comes_only_from_the_variable_it_names():
    spec = ProviderSpec(base_url="http://gpu-box:8000/v1", api_key_env="LAB_KEY")

    providers = build_providers(
        # A provider key under the provider's name, and other variables, are ignored.
        keys={**_NO_KEYS, "lab": "k-not-this"},
        specs={"lab": spec},
        env={"OPENAI_API_KEY": "k-openai", "lab": "k-nor-this", "LAB_KEY": ""},
    )

    assert providers["lab"].api_key is None


class TestCredentials:
    """A configured provider can be called when its key variable is set, or
    when it names none (a local server; the client sends a placeholder)."""

    def test_one_that_names_no_key_variable_has_credentials(self):
        provider = build_providers(
            keys=_NO_KEYS, specs={"lab": ProviderSpec(base_url="http://gpu-box:8000/v1")}, env={}
        )["lab"]

        assert (provider.api_key, provider.key_env) == (None, None)
        assert provider.has_credentials

    @pytest.mark.parametrize(("env", "callable_"), [({"LAB_KEY": "k"}, True), ({}, False)])
    def test_one_that_names_a_key_variable_needs_it_set(self, env, callable_):
        spec = ProviderSpec(base_url="http://gpu-box:8000/v1", api_key_env="LAB_KEY")

        provider = build_providers(keys=_NO_KEYS, specs={"lab": spec}, env=env)["lab"]

        assert provider.has_credentials is callable_

    def test_an_openrouter_provider_without_its_key_cannot_be_called(self):
        spec = ProviderSpec(
            kind="openrouter", base_url="https://openrouter.ai/api/v1", api_key_env="OR_KEY"
        )

        assert not build_providers(keys=_NO_KEYS, specs={"or": spec}, env={})["or"].has_credentials

    def test_a_builtin_without_its_key_never_counts_as_needing_none(self):
        provider = Provider(name="openai", kind=ProviderKind.OPENAI, base_url=None)

        assert not provider.has_credentials


def test_a_repr_never_shows_a_key():
    providers = build_providers(
        keys=_keys(google="k-secret-google"),
        specs={"lab": ProviderSpec(base_url="http://gpu-box:8000/v1", api_key_env="LAB_KEY")},
        env={"LAB_KEY": "k-secret-lab"},
    )

    assert "k-secret-google" not in repr(providers["google"])
    assert "k-secret-lab" not in repr(providers["lab"])
    assert "k-secret" not in repr(providers)


@pytest.mark.parametrize("name", ["google", "anthropic", "openai"])
def test_a_provider_named_like_a_builtin_is_an_error(name):
    with pytest.raises(ModelSpecError, match=f"'{name}'.*built-in"):
        build_providers(
            keys=_NO_KEYS, specs={name: ProviderSpec(base_url="http://x:1/v1")}, env={}
        )


def test_openrouter_is_an_ordinary_name():
    spec = ProviderSpec(base_url="http://x:1/v1")

    provider = build_providers(keys=_NO_KEYS, specs={"openrouter": spec}, env={})["openrouter"]

    assert provider.kind is ProviderKind.OPENAI_COMPATIBLE


def test_a_malformed_provider_name_is_an_error():
    with pytest.raises(ModelSpecError, match="Local VLLM"):
        build_providers(
            keys=_NO_KEYS, specs={"Local VLLM": ProviderSpec(base_url="http://x:1/v1")}, env={}
        )


def test_a_provider_is_listed_unless_its_spec_says_not():
    """``list_models: false``: a server with no /models route, or one whose
    listing leaves out models it serves. The built-ins are always listed."""
    specs = {
        "lab": ProviderSpec(base_url="http://gpu-box:8000/v1"),
        "local": ProviderSpec(base_url="http://localhost:8000/v1", list_models=False),
    }

    providers = build_providers(keys=_NO_KEYS, specs=specs, env={})

    assert {name: p.list_models for name, p in providers.items()} == {
        "google": True, "anthropic": True, "openai": True, "lab": True, "local": False,
    }


def test_a_provider_is_frozen():
    provider = Provider(name="google", kind=ProviderKind.GOOGLE, base_url=None, api_key="k")

    with pytest.raises(AttributeError):
        provider.api_key = "other"  # type: ignore[misc]


class TestProviderSpec:
    def test_a_trailing_slash_is_stripped(self):
        assert ProviderSpec(base_url="https://lab.example/v1/").base_url == "https://lab.example/v1"

    @pytest.mark.parametrize(
        "base_url",
        [
            "ftp://lab.example/v1",
            "file:///etc/passwd",
            "localhost:8000/v1",  # no scheme
            "http:///v1",  # no host
            "http://user:secret@lab.example/v1",  # userinfo
            "http://user@lab.example/v1",
            "http://lab.example/v1?key=1",  # query
            "http://lab.example/v1#frag",  # fragment
            "",
        ],
    )
    def test_a_base_url_must_be_plain_http_or_https(self, base_url):
        with pytest.raises(ValidationError):
            ProviderSpec(base_url=base_url)

    def test_a_base_url_is_required(self):
        with pytest.raises(ValidationError, match="base_url"):
            ProviderSpec(api_key_env="LAB_KEY")

    def test_a_userinfo_error_does_not_echo_the_password(self):
        with pytest.raises(ValidationError) as exc:
            ProviderSpec(base_url="http://user:hunter2@lab.example/v1")

        assert "hunter2" not in str(exc.value)

    @pytest.mark.parametrize("name", ["LOCAL_VLLM_KEY", "_KEY", "K1"])
    def test_an_api_key_env_name(self, name):
        assert ProviderSpec(base_url="http://x:1/v1", api_key_env=name).api_key_env == name

    @pytest.mark.parametrize("name", ["local_key", "1KEY", "MY-KEY", "MY KEY", ""])
    def test_a_malformed_api_key_env_name(self, name):
        with pytest.raises(ValidationError):
            ProviderSpec(base_url="http://x:1/v1", api_key_env=name)

    def test_the_kinds(self):
        assert ProviderSpec(base_url="http://x:1/v1").kind == "openai_compatible"
        assert ProviderSpec(
            kind="openrouter", base_url="https://openrouter.ai/api/v1", api_key_env="OR_KEY"
        ).kind == "openrouter"
        for kind in ("anthropic", "google", "openai"):
            with pytest.raises(ValidationError):
                ProviderSpec(kind=kind, base_url="http://x:1/v1")

    def test_an_openrouter_provider_names_its_key_variable(self):
        with pytest.raises(
            ValidationError,
            match="kind 'openrouter' needs api_key_env, the variable holding the OpenRouter key",
        ):
            ProviderSpec(kind="openrouter", base_url="https://openrouter.ai/api/v1")

    def test_unknown_fields_are_errors(self):
        with pytest.raises(ValidationError):
            ProviderSpec(base_url="http://x:1/v1", api_key="k-inline")

    def test_its_models_are_entries_in_models(self):
        """What a provider serves is said in the ``models`` setting, by reference."""
        with pytest.raises(ValidationError):
            ProviderSpec(base_url="http://x:1/v1", models={"Qwen/Qwen3-32B": {}})
