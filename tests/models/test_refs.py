"""Model references: ``provider:model``, bare ids and role names."""

from __future__ import annotations

import pytest

from agentic_cli.models import (
    BUILTIN_PROVIDERS,
    BUILTIN_ROLES,
    ModelRecord,
    ModelRef,
    ModelSpecError,
    ProviderKind,
    RoleRef,
    parse_ref,
    parse_spec,
    validate_endpoint_name,
    validate_role_name,
)


def test_the_builtin_names():
    assert BUILTIN_PROVIDERS == ("google", "anthropic", "openai", "openrouter")
    assert BUILTIN_ROLES == ("default", "fast", "strong")


@pytest.mark.parametrize(
    ("spec", "provider", "model"),
    [
        ("anthropic:claude-opus-5-5", "anthropic", "claude-opus-5-5"),
        ("google:gemini-3.5-flash", "google", "gemini-3.5-flash"),
        ("local-vllm:Qwen/Qwen3-32B", "local-vllm", "Qwen/Qwen3-32B"),
        ("my_endpoint:some-model", "my_endpoint", "some-model"),
    ],
)
def test_a_provider_reference(spec, provider, model):
    assert parse_spec(spec) == ModelRef(provider, model)


def test_only_the_first_colon_splits_the_reference():
    ref = parse_spec("openrouter:qwen/qwen3-235b-a22b:free")

    assert ref == ModelRef("openrouter", "qwen/qwen3-235b-a22b:free")


@pytest.mark.parametrize(
    ("spec", "provider"),
    [
        ("gemini-2.5-flash", "google"),
        ("claude-sonnet-4-5", "anthropic"),
        ("gpt-5.5", "openai"),
        ("o4-mini", "openai"),
        ("o3x", "openai"),  # o<digit>: a bare OpenAI id, never a role
    ],
)
def test_a_bare_id_maps_to_its_provider(spec, provider):
    assert parse_spec(spec) == ModelRef(provider, spec)


@pytest.mark.parametrize("spec", ["default", "fast", "strong", "vision", "code_review", "my-role"])
def test_any_other_bare_name_is_a_role(spec):
    assert parse_spec(spec) == RoleRef(spec)


@pytest.mark.parametrize("spec", ["qwen3-32b", "gemma-3-27b-it", "deepseek-r1", "phi-4", "x-1"])
def test_a_bare_name_with_a_digit_is_a_model_id_missing_its_provider(spec):
    # As a role it would run on the default model, perhaps another provider's,
    # so a local model's prompts could go to the cloud.
    with pytest.raises(ModelSpecError, match=f"<endpoint>:{spec}"):
        parse_spec(spec)


@pytest.mark.parametrize(
    "spec",
    [
        # Not an id (no "-" after the family) and not a role (it starts like one).
        "gemini",
        "gpt5",
        "claude",
        "Fast",
        "fast!",
        "",
        "anthropic:",
        ":x",
        " gemini-2.5-flash",
        "gemini-2.5-flash ",
        "anthropic:claude sonnet",
        "Google:gemini-2.5-flash",
        "a" * 65,
    ],
)
def test_malformed_specs_are_errors(spec):
    with pytest.raises(ModelSpecError) as exc:
        parse_spec(spec)

    assert repr(spec) in str(exc.value)


def test_a_spec_error_is_a_value_error():
    assert issubclass(ModelSpecError, ValueError)


def test_parse_ref_rejects_a_role():
    with pytest.raises(ModelSpecError, match="role"):
        parse_ref("fast")


def test_parse_ref_accepts_a_reference_and_a_bare_id():
    assert parse_ref("google:gemini-2.5-flash") == ModelRef("google", "gemini-2.5-flash")
    assert parse_ref("claude-sonnet-4-5") == ModelRef("anthropic", "claude-sonnet-4-5")


def test_a_reference_prints_as_provider_colon_model():
    assert str(ModelRef("openrouter", "qwen/qwen3:free")) == "openrouter:qwen/qwen3:free"
    assert str(parse_spec("gpt-5.5")) == "openai:gpt-5.5"


@pytest.mark.parametrize("name", ["vision", "fast", "my_role-two", "a" * 64])
def test_a_valid_role_name(name):
    assert validate_role_name(name) == name


@pytest.mark.parametrize(
    "name", ["gemini-x", "claudette", "gpt_writer", "o1", "o9role", "Fast", "fast!", "", "1st", "a" * 65]
)
def test_a_role_name_cannot_look_like_a_model_id(name):
    with pytest.raises(ModelSpecError):
        validate_role_name(name)


@pytest.mark.parametrize("name", ["tier2", "my_role-2", "qwen3"])
def test_a_role_name_has_no_digits(name):
    with pytest.raises(ModelSpecError, match="digit"):
        validate_role_name(name)


@pytest.mark.parametrize("name", ["local-vllm", "lab_server", "vllm2"])
def test_a_valid_endpoint_name(name):
    assert validate_endpoint_name(name) == name


@pytest.mark.parametrize("name", list(BUILTIN_PROVIDERS))
def test_an_endpoint_cannot_reuse_a_builtin_provider_name(name):
    with pytest.raises(ModelSpecError, match="built-in"):
        validate_endpoint_name(name)


@pytest.mark.parametrize("name", ["Local", "local vllm", "", "-x", "x" * 65])
def test_a_malformed_endpoint_name(name):
    with pytest.raises(ModelSpecError):
        validate_endpoint_name(name)


def test_a_record_knows_its_reference():
    record = ModelRecord(provider="google", kind=ProviderKind.GOOGLE, model="gemini-2.5-flash")

    assert record.ref == ModelRef("google", "gemini-2.5-flash")
