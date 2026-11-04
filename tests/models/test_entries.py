"""``models`` entries: what an app, a user or a project says about one model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentic_cli.models import ModelEntry, ModelRef, ModelSpecError, parse_entry_key
from agentic_cli.models.entries import DEFAULT_EFFORT_LEVELS, THINKING_FIELDS, check_entries


def test_entry_keys():
    """Split on the first colon, as references are; a bare id needs its provider."""
    assert parse_entry_key("openrouter:qwen/qwen3-235b-a22b:free") == ModelRef(
        "openrouter", "qwen/qwen3-235b-a22b:free"
    )
    assert parse_entry_key("local-vllm:Qwen/Qwen3-32B") == ModelRef("local-vllm", "Qwen/Qwen3-32B")

    with pytest.raises(ModelSpecError) as caught:
        parse_entry_key("claude-sonnet-4-5")

    assert str(caught.value) == (
        "models key 'claude-sonnet-4-5' needs its provider: "
        "write it as anthropic:claude-sonnet-4-5."
    )


@pytest.mark.parametrize("key", ["fast", "qwen3-32b", "Local:x", "local:", "a b:c", ""])
def test_a_key_that_is_not_a_reference_is_refused(key):
    with pytest.raises(ModelSpecError) as caught:
        parse_entry_key(key)

    assert str(caught.value) == (
        f"models key {key!r} is not a provider:model reference, such as "
        "anthropic:claude-sonnet-5-5."
    )


class TestModelEntry:
    def test_every_field_is_unset_by_default(self):
        entry = ModelEntry()

        assert entry.model_fields_set == set()
        assert entry.model_dump() == dict.fromkeys(
            ("context_window", "max_output", "tools", "image_input", "thinking",
             "effort_levels", "reasoning_text")
        )

    @pytest.mark.parametrize("field", ["context_window", "max_output"])
    def test_token_counts_are_positive(self, field):
        with pytest.raises(ValidationError):
            ModelEntry(**{field: 0})

    def test_the_thinking_kinds(self):
        for kind in ("none", "chat_template", "effort"):
            assert ModelEntry(thinking=kind).thinking == kind
        with pytest.raises(ValidationError):
            ModelEntry(thinking="budget")

    @pytest.mark.parametrize("levels", [[], ["low", "low"], ["High"], ["very high"]])
    def test_malformed_effort_levels(self, levels):
        with pytest.raises(ValidationError):
            ModelEntry(effort_levels=levels)

    def test_unknown_fields_are_errors(self):
        with pytest.raises(ValidationError):
            ModelEntry(context=1)

    def test_the_constants(self):
        assert THINKING_FIELDS == ("thinking", "effort_levels", "reasoning_text")
        assert DEFAULT_EFFORT_LEVELS == ("low", "medium", "high")


class TestCheckEntries:
    @pytest.mark.parametrize("provider", ["google", "anthropic", "openai"])
    @pytest.mark.parametrize(
        "field", [{"thinking": "effort"}, {"effort_levels": ["low"]}, {"reasoning_text": True}]
    )
    def test_thinking_fields_on_a_builtin_model_are_refused(self, provider, field):
        key = f"{provider}:some-model"

        with pytest.raises(ValueError) as caught:
            check_entries({key: ModelEntry(**field)})

        assert str(caught.value) == (
            f"models[{key!r}]: thinking, effort_levels and reasoning_text are only for a model "
            "on a model_providers provider; the built-in providers' thinking comes from their "
            "listings and the table."
        )

    def test_the_other_fields_of_a_builtin_model_pass(self):
        entries = {
            "anthropic:claude-sonnet-4-5": ModelEntry(),
            "google:gemini-2.5-flash": ModelEntry(
                context_window=1_000_000, max_output=65_536, tools=True, image_input=True
            ),
        }

        assert check_entries(entries) == entries

    def test_thinking_fields_on_another_provider_s_model_pass(self):
        entries = {
            "local-vllm:Qwen/Qwen3-32B": ModelEntry(thinking="chat_template", reasoning_text=True),
            "openrouter:qwen/qwen3-235b-a22b": ModelEntry(effort_levels=["low", "high"]),
        }

        assert check_entries(entries) == entries

    def test_a_bare_key_is_refused_with_its_full_form(self):
        with pytest.raises(ValueError, match="write it as anthropic:claude-sonnet-4-5"):
            check_entries({"claude-sonnet-4-5": ModelEntry()})
