"""Providers: the four built-ins and the user's OpenAI-compatible endpoints."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentic_cli.models import (
    OPENROUTER_BASE_URL,
    EndpointSpec,
    ModelSpecError,
    Provider,
    ProviderKind,
    build_providers,
)

_NO_KEYS = {"google": None, "anthropic": None, "openai": None, "openrouter": None}


def _keys(**given: str) -> dict[str, str | None]:
    return {**_NO_KEYS, **given}


def test_the_four_builtins_are_always_there():
    providers = build_providers(keys=_NO_KEYS, specs={}, env={})

    assert {name: (e.kind, e.base_url) for name, e in providers.items()} == {
        "google": (ProviderKind.GOOGLE, None),
        "anthropic": (ProviderKind.ANTHROPIC, None),
        "openai": (ProviderKind.OPENAI, None),
        "openrouter": (ProviderKind.OPENROUTER, OPENROUTER_BASE_URL),
    }
    assert OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"


def test_a_builtin_has_credentials_only_with_its_key():
    providers = build_providers(keys=_keys(anthropic="k-anthropic"), specs={}, env={})

    assert providers["anthropic"].api_key == "k-anthropic"
    assert providers["anthropic"].has_credentials
    assert not providers["google"].has_credentials
    assert not providers["openai"].has_credentials
    assert not providers["openrouter"].has_credentials


def test_an_endpoint_spec_becomes_an_openai_compatible_endpoint():
    spec = EndpointSpec(base_url="http://localhost:8000/v1", api_key_env="LOCAL_VLLM_KEY")

    providers = build_providers(
        keys=_NO_KEYS, specs={"local-vllm": spec}, env={"LOCAL_VLLM_KEY": "k-vllm"}
    )

    local = providers["local-vllm"]
    assert local.name == "local-vllm"
    assert local.kind is ProviderKind.OPENAI_COMPATIBLE
    assert local.base_url == "http://localhost:8000/v1"
    assert local.api_key == "k-vllm"


def test_an_endpoint_key_comes_only_from_the_variable_it_names():
    spec = EndpointSpec(base_url="http://gpu-box:8000/v1", api_key_env="LAB_KEY")

    providers = build_providers(
        # A provider key under the endpoint's name, and other variables, are ignored.
        keys={**_NO_KEYS, "lab": "k-not-this"},
        specs={"lab": spec},
        env={"OPENAI_API_KEY": "k-openai", "lab": "k-nor-this"},
    )

    assert providers["lab"].api_key is None


def test_an_endpoint_without_a_key_still_has_credentials():
    """A local server often needs no key; the client sends a placeholder."""
    providers = build_providers(
        keys=_NO_KEYS, specs={"lab": EndpointSpec(base_url="http://gpu-box:8000/v1")}, env={}
    )

    assert providers["lab"].api_key is None
    assert providers["lab"].has_credentials


def test_a_repr_never_shows_a_key():
    providers = build_providers(
        keys=_keys(google="k-secret-google"),
        specs={"lab": EndpointSpec(base_url="http://gpu-box:8000/v1", api_key_env="LAB_KEY")},
        env={"LAB_KEY": "k-secret-lab"},
    )

    assert "k-secret-google" not in repr(providers["google"])
    assert "k-secret-lab" not in repr(providers["lab"])
    assert "k-secret" not in repr(providers)


def test_an_endpoint_named_like_a_builtin_is_an_error():
    with pytest.raises(ModelSpecError, match="openai"):
        build_providers(
            keys=_NO_KEYS, specs={"openai": EndpointSpec(base_url="http://x:1/v1")}, env={}
        )


def test_a_malformed_endpoint_name_is_an_error():
    with pytest.raises(ModelSpecError, match="Local VLLM"):
        build_providers(
            keys=_NO_KEYS, specs={"Local VLLM": EndpointSpec(base_url="http://x:1/v1")}, env={}
        )


def test_a_provider_is_frozen():
    provider = Provider(name="google", kind=ProviderKind.GOOGLE, base_url=None, api_key="k")

    with pytest.raises(AttributeError):
        provider.api_key = "other"  # type: ignore[misc]


class TestEndpointSpec:
    def test_a_trailing_slash_is_stripped(self):
        assert EndpointSpec(base_url="https://lab.example/v1/").base_url == "https://lab.example/v1"

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
            EndpointSpec(base_url=base_url)

    def test_a_userinfo_error_does_not_echo_the_password(self):
        with pytest.raises(ValidationError) as exc:
            EndpointSpec(base_url="http://user:hunter2@lab.example/v1")

        assert "hunter2" not in str(exc.value)

    @pytest.mark.parametrize("name", ["LOCAL_VLLM_KEY", "_KEY", "K1"])
    def test_an_api_key_env_name(self, name):
        assert EndpointSpec(base_url="http://x:1/v1", api_key_env=name).api_key_env == name

    @pytest.mark.parametrize("name", ["local_key", "1KEY", "MY-KEY", "MY KEY", ""])
    def test_a_malformed_api_key_env_name(self, name):
        with pytest.raises(ValidationError):
            EndpointSpec(base_url="http://x:1/v1", api_key_env=name)

    def test_the_kind_is_openai_compatible_only(self):
        assert EndpointSpec(base_url="http://x:1/v1").kind == "openai_compatible"
        with pytest.raises(ValidationError):
            EndpointSpec(kind="anthropic", base_url="http://x:1/v1")

    def test_unknown_fields_are_errors(self):
        with pytest.raises(ValidationError):
            EndpointSpec(base_url="http://x:1/v1", api_key="k-inline")

    def test_its_models_are_entries_in_models_now(self):
        """What an endpoint serves is said in the ``models`` setting, by reference."""
        with pytest.raises(ValidationError):
            EndpointSpec(base_url="http://x:1/v1", models={"Qwen/Qwen3-32B": {}})
