"""Provider endpoints: the four built-ins and the user's OpenAI-compatible ones."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentic_cli.models import (
    OPENROUTER_BASE_URL,
    DeclaredModel,
    Endpoint,
    EndpointSpec,
    ModelSpecError,
    ProviderKind,
    build_endpoints,
)

_NO_KEYS = {"google": None, "anthropic": None, "openai": None, "openrouter": None}


def _keys(**given: str) -> dict[str, str | None]:
    return {**_NO_KEYS, **given}


def test_the_four_builtins_are_always_there():
    endpoints = build_endpoints(keys=_NO_KEYS, specs={}, env={})

    assert {name: (e.kind, e.base_url) for name, e in endpoints.items()} == {
        "google": (ProviderKind.GOOGLE, None),
        "anthropic": (ProviderKind.ANTHROPIC, None),
        "openai": (ProviderKind.OPENAI, None),
        "openrouter": (ProviderKind.OPENROUTER, OPENROUTER_BASE_URL),
    }
    assert OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"


def test_a_builtin_has_credentials_only_with_its_key():
    endpoints = build_endpoints(keys=_keys(anthropic="k-anthropic"), specs={}, env={})

    assert endpoints["anthropic"].api_key == "k-anthropic"
    assert endpoints["anthropic"].has_credentials
    assert not endpoints["google"].has_credentials
    assert not endpoints["openai"].has_credentials
    assert not endpoints["openrouter"].has_credentials


def test_an_endpoint_spec_becomes_an_openai_compatible_endpoint():
    spec = EndpointSpec(
        base_url="http://localhost:8000/v1",
        api_key_env="LOCAL_VLLM_KEY",
        models={"Qwen/Qwen3-32B": DeclaredModel(context_window=32768, thinking="chat_template")},
    )

    endpoints = build_endpoints(
        keys=_NO_KEYS, specs={"local-vllm": spec}, env={"LOCAL_VLLM_KEY": "k-vllm"}
    )

    local = endpoints["local-vllm"]
    assert local.name == "local-vllm"
    assert local.kind is ProviderKind.OPENAI_COMPATIBLE
    assert local.base_url == "http://localhost:8000/v1"
    assert local.api_key == "k-vllm"
    assert local.models["Qwen/Qwen3-32B"].context_window == 32768


def test_an_endpoint_key_comes_only_from_the_variable_it_names():
    spec = EndpointSpec(base_url="http://gpu-box:8000/v1", api_key_env="LAB_KEY")

    endpoints = build_endpoints(
        # A provider key under the endpoint's name, and other variables, are ignored.
        keys={**_NO_KEYS, "lab": "k-not-this"},
        specs={"lab": spec},
        env={"OPENAI_API_KEY": "k-openai", "lab": "k-nor-this"},
    )

    assert endpoints["lab"].api_key is None


def test_an_endpoint_without_a_key_still_has_credentials():
    """A local server often needs no key; the client sends a placeholder."""
    endpoints = build_endpoints(
        keys=_NO_KEYS, specs={"lab": EndpointSpec(base_url="http://gpu-box:8000/v1")}, env={}
    )

    assert endpoints["lab"].api_key is None
    assert endpoints["lab"].has_credentials


def test_a_repr_never_shows_a_key():
    endpoints = build_endpoints(
        keys=_keys(google="k-secret-google"),
        specs={"lab": EndpointSpec(base_url="http://gpu-box:8000/v1", api_key_env="LAB_KEY")},
        env={"LAB_KEY": "k-secret-lab"},
    )

    assert "k-secret-google" not in repr(endpoints["google"])
    assert "k-secret-lab" not in repr(endpoints["lab"])
    assert "k-secret" not in repr(endpoints)


def test_an_endpoint_named_like_a_builtin_is_an_error():
    with pytest.raises(ModelSpecError, match="openai"):
        build_endpoints(
            keys=_NO_KEYS, specs={"openai": EndpointSpec(base_url="http://x:1/v1")}, env={}
        )


def test_a_malformed_endpoint_name_is_an_error():
    with pytest.raises(ModelSpecError, match="Local VLLM"):
        build_endpoints(
            keys=_NO_KEYS, specs={"Local VLLM": EndpointSpec(base_url="http://x:1/v1")}, env={}
        )


def test_the_declared_models_cannot_be_changed_through_the_endpoint():
    endpoints = build_endpoints(
        keys=_NO_KEYS,
        specs={"lab": EndpointSpec(base_url="http://x:1/v1", models={"m": DeclaredModel()})},
        env={},
    )

    with pytest.raises(TypeError):
        endpoints["lab"].models["other"] = DeclaredModel()  # type: ignore[index]


def test_an_endpoint_is_frozen():
    endpoint = Endpoint(name="google", kind=ProviderKind.GOOGLE, base_url=None, api_key="k")

    with pytest.raises(AttributeError):
        endpoint.api_key = "other"  # type: ignore[misc]


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

    def test_a_model_id_must_be_a_plain_name(self):
        with pytest.raises(ValidationError):
            EndpointSpec(base_url="http://x:1/v1", models={"has space": DeclaredModel()})
        with pytest.raises(ValidationError):
            EndpointSpec(base_url="http://x:1/v1", models={"": DeclaredModel()})


class TestDeclaredModel:
    def test_the_defaults(self):
        model = DeclaredModel()

        assert model.context_window is None
        assert model.max_output is None
        assert model.tools is True
        assert model.image_input is False
        assert model.thinking == "none"
        assert model.effort_levels == ["low", "medium", "high"]
        assert model.reasoning_text is True

    @pytest.mark.parametrize("field", ["context_window", "max_output"])
    def test_token_counts_are_positive(self, field):
        with pytest.raises(ValidationError):
            DeclaredModel(**{field: 0})

    def test_the_thinking_kinds(self):
        for kind in ("none", "chat_template", "effort"):
            assert DeclaredModel(thinking=kind).thinking == kind
        with pytest.raises(ValidationError):
            DeclaredModel(thinking="budget")

    @pytest.mark.parametrize("levels", [[], ["low", "low"], ["High"], ["very high"]])
    def test_malformed_effort_levels(self, levels):
        with pytest.raises(ValidationError):
            DeclaredModel(thinking="effort", effort_levels=levels)

    def test_unknown_fields_are_errors(self):
        with pytest.raises(ValidationError):
            DeclaredModel(context=1)
