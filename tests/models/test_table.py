"""The shipped model table: family rules, per-model numbers and fallback lists."""

from __future__ import annotations

import copy
import json
from importlib import resources

import pytest

from agentic_cli.models import (
    CacheMechanism,
    ProviderKind,
    ShippedTable,
    ThinkingMechanism,
    load_table,
    version_key,
)

G, A, O = ProviderKind.GOOGLE, ProviderKind.ANTHROPIC, ProviderKind.OPENAI
_TRACKED = (
    "display_name", "tier", "context_window", "max_output", "tools", "image_input",
    "thinking", "reasoning_text", "cache", "sampling",
)


def _raw() -> dict:
    return json.loads(
        resources.files("agentic_cli.models").joinpath("data", "table.json").read_text("utf-8")
    )


def _record(kind: ProviderKind, model: str):
    return load_table().record_for(kind.value, kind, model)


# One case per family rule: the fields the rule decides.
@pytest.mark.parametrize(
    ("kind", "model", "tier", "mechanism", "cache", "sampling"),
    [
        (G, "gemini-2.5-flash-lite", "fast", ThinkingMechanism.GEMINI_BUDGET, "gemini", "always"),
        (G, "gemini-2.5-flash", "default", ThinkingMechanism.GEMINI_BUDGET, "gemini", "always"),
        (G, "gemini-2.5-pro", "strong", ThinkingMechanism.GEMINI_BUDGET, "gemini", "always"),
        (G, "gemini-3-pro-preview", "strong", ThinkingMechanism.GEMINI_LEVEL, "gemini", "always"),
        (G, "gemini-3.5-flash-lite", "fast", ThinkingMechanism.GEMINI_LEVEL, "gemini", "always"),
        (G, "gemini-3.5-flash", "default", ThinkingMechanism.GEMINI_LEVEL, "gemini", "always"),
        (G, "gemini-4-argon", None, ThinkingMechanism.GEMINI_LEVEL, "gemini", "never"),
        (A, "claude-fable-5", None, ThinkingMechanism.ANTHROPIC_ADAPTIVE, "anthropic", "never"),
        (A, "claude-haiku-5-5", "fast", ThinkingMechanism.ANTHROPIC_ADAPTIVE, "anthropic", "never"),
        (A, "claude-haiku-4-5", "fast", ThinkingMechanism.ANTHROPIC_BUDGET, "anthropic", "without_thinking"),
        (A, "claude-sonnet-4-6", "default", ThinkingMechanism.ANTHROPIC_ADAPTIVE, "anthropic", "without_thinking"),
        (A, "claude-sonnet-4-5", "default", ThinkingMechanism.ANTHROPIC_BUDGET, "anthropic", "without_thinking"),
        (A, "claude-opus-5-5", "strong", ThinkingMechanism.ANTHROPIC_ADAPTIVE, "anthropic", "never"),
        (A, "claude-opus-4-5", "strong", ThinkingMechanism.ANTHROPIC_BUDGET, "anthropic", "without_thinking"),
        (A, "claude-mythos-5", None, ThinkingMechanism.ANTHROPIC_ADAPTIVE, "anthropic", "without_thinking"),
        (A, "claude-3-7-sonnet", None, ThinkingMechanism.ANTHROPIC_BUDGET, "anthropic", "without_thinking"),
        (O, "gpt-5.4-mini", "fast", ThinkingMechanism.EFFORT, "none", "never"),
        (O, "gpt-5.5-pro", "strong", ThinkingMechanism.EFFORT, "none", "never"),
        (O, "gpt-5.5", "default", ThinkingMechanism.EFFORT, "none", "never"),
        (O, "gpt-6-sol", None, ThinkingMechanism.EFFORT, "none", "never"),
        (O, "gpt-4.1", None, ThinkingMechanism.NONE, "none", "always"),
    ],
)
def test_each_family_rule(kind, model, tier, mechanism, cache, sampling):
    record = _record(kind, model)

    assert (record.provider, record.kind, record.model) == (kind.value, kind, model)
    assert record.tier == tier
    assert record.thinking.mechanism is mechanism
    assert record.cache is CacheMechanism(cache)
    assert record.sampling == sampling
    assert record.sources["thinking"] == "table"


class TestGemini:
    @pytest.mark.parametrize(
        ("model", "budget_range", "can_disable"),
        [
            ("gemini-2.5-flash-lite", (512, 24576), True),
            ("gemini-2.5-flash", (1, 24576), True),
            ("gemini-2.5-pro", (128, 32768), False),
        ],
    )
    def test_gemini_2_5_budgets(self, model, budget_range, can_disable):
        rule = _record(G, model).thinking

        assert rule.budgets == (4096, 12288, 24576)
        assert rule.budget_range == budget_range
        assert rule.can_disable is can_disable

    def test_gemini_3_pro_maps_medium_to_high(self):
        rule = _record(G, "gemini-3-pro-preview").thinking

        assert rule.levels == ("low", "high")
        assert dict(rule.generic_map) == {"medium": "high"}
        assert rule.can_disable is False

    def test_a_ladder_that_has_medium_needs_no_mapping(self):
        """Gemini 3.1 Pro's profile ladder has medium, so the Pro rule's mapping goes."""
        rule = _record(G, "gemini-3.1-pro-preview").thinking

        assert rule.levels == ("low", "medium", "high")
        assert dict(rule.generic_map) == {}

    def test_gemini_3_flash_levels(self):
        for model in ("gemini-3-flash-preview", "gemini-3.1-flash-lite", "gemini-3.7-flash"):
            rule = _record(G, model).thinking
            assert rule.levels == ("minimal", "low", "medium", "high"), model
            assert rule.can_disable is False

    def test_gemini_reasoning_text(self):
        assert _record(G, "gemini-2.5-flash").reasoning_text is True
        assert _record(G, "gemini-3.5-flash").reasoning_text is True

    @pytest.mark.parametrize(
        "model", ["gemini-2.5-flash-image", "gemini-2.5-pro-preview-tts", "gemini-3.1-pro-preview-customtools"]
    )
    def test_variants_get_no_tier(self, model):
        assert _record(G, model).tier is None


class TestClaude:
    def test_budget_claude(self):
        rule = _record(A, "claude-sonnet-4-5").thinking

        assert rule.budgets == (4096, 10000, 32000)
        assert rule.budget_range[0] == 1024
        assert rule.levels == ()

    def test_adaptive_claude_takes_its_profile_ladder(self):
        assert _record(A, "claude-sonnet-4-6").thinking.levels == ("low", "medium", "high", "max")
        assert _record(A, "claude-opus-4-7").thinking.levels == (
            "low", "medium", "high", "xhigh", "max",
        )

    def test_adaptive_claude_without_a_profile_ladder(self):
        assert _record(A, "claude-sonnet-6").thinking.levels == ("low", "medium", "high", "max")
        assert _record(A, "claude-haiku-5-5").thinking.levels == ("low", "medium", "high", "max")

    @pytest.mark.parametrize(
        "model", ["claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5", "claude-fable-5-1"]
    )
    def test_the_models_that_cannot_turn_thinking_off(self, model):
        """Anthropic rejects ``thinking: {"type": "disabled"}`` on these models."""
        assert _record(A, model).thinking.can_disable is False

    @pytest.mark.parametrize(
        "model",
        [
            "claude-sonnet-4-6", "claude-sonnet-5", "claude-opus-4-7", "claude-opus-4-8",
            "claude-opus-5", "claude-haiku-5-5",
        ],
    )
    def test_the_models_that_can_turn_thinking_off(self, model):
        assert _record(A, model).thinking.can_disable is True

    @pytest.mark.parametrize("model", ["claude-sonnet-5-5", "claude-fable-5", "claude-fable-5-1"])
    def test_each_correction_cites_the_thinking_guide(self, model):
        source = _raw()["providers"]["anthropic"]["models"][model]["source"]

        assert "https://platform.claude.com/docs/en/build-with-claude/thinking" in source

    def test_the_version_decides_the_mechanism_dates_ignored(self):
        assert _record(A, "claude-opus-4-20250514").thinking.mechanism is ThinkingMechanism.ANTHROPIC_BUDGET
        assert _record(A, "claude-sonnet-4").thinking.mechanism is ThinkingMechanism.ANTHROPIC_BUDGET
        assert _record(A, "claude-opus-4-6").thinking.mechanism is ThinkingMechanism.ANTHROPIC_ADAPTIVE

    def test_claude_reasoning_text(self):
        assert _record(A, "claude-sonnet-4-5").reasoning_text is True


class TestOpenAI:
    def test_a_reasoning_model_takes_its_profile_ladder(self):
        assert _record(O, "gpt-5.5").thinking.levels == ("none", "low", "medium", "high", "xhigh")
        assert _record(O, "gpt-5").thinking.levels == ("minimal", "low", "medium", "high")

    def test_a_reasoning_model_without_a_profile_ladder(self):
        assert _record(O, "o4-mini").thinking.levels == ("low", "medium", "high")
        assert _record(O, "o4-mini").tier == "fast"

    def test_a_chat_alias_is_not_a_reasoning_model(self):
        assert _record(O, "gpt-5-chat-latest").thinking.mechanism is ThinkingMechanism.NONE

    def test_the_curated_list(self):
        curated = load_table().curated_openai()

        assert "gpt-5.5" in curated and "o3" in curated and "gpt-4.1" in curated
        assert len(curated) == len(set(curated))
        for model in curated:
            record = _record(O, model)
            assert record.tools, model
            assert record.sources["context_window"] == "table", model
            assert version_key(model) != () or model.startswith("o"), model
        assert not [m for m in curated if "latest" in m or "codex" in m or "-20" in m]


def test_per_model_numbers_come_from_the_profiles():
    flash = _record(G, "gemini-2.5-flash")
    opus = _record(A, "claude-opus-5-5")
    gpt = _record(O, "gpt-5.5")

    assert (flash.context_window, flash.max_output, flash.image_input) == (1048576, 65536, True)
    assert (opus.context_window, opus.max_output) == (1000000, 128000)
    assert (gpt.context_window, gpt.max_output) == (1050000, 128000)
    assert flash.display_name == "Gemini 2.5 Flash"
    assert flash.sources["context_window"] == "table"


@pytest.mark.parametrize(
    ("provider", "kind", "model"),
    [
        ("google", G, "gemma-4-31b-it"),
        ("openai", O, "text-embedding-3-large"),
        ("openrouter", ProviderKind.OPENROUTER, "qwen/qwen3-235b-a22b"),
        ("local-vllm", ProviderKind.OPENAI_COMPATIBLE, "Qwen/Qwen3-32B"),
    ],
)
def test_an_unknown_model_gets_the_conservative_record(provider, kind, model):
    record = load_table().record_for(provider, kind, model)

    assert (record.provider, record.kind, record.model) == (provider, kind, model)
    assert record.thinking.mechanism is ThinkingMechanism.NONE
    assert record.tools is True
    assert record.tier is None
    assert record.cache is CacheMechanism.NONE
    assert record.sampling == "always"
    assert record.context_window is None
    assert {field: record.sources[field] for field in _TRACKED} == dict.fromkeys(_TRACKED, "default")


@pytest.mark.parametrize(
    ("model_id", "key"),
    [
        ("claude-opus-4-20250514", (4,)),
        ("claude-sonnet-4-5", (4, 5)),
        ("claude-3-5-sonnet-20241022", (3, 5)),
        ("gemini-2.5-flash", (2, 5)),
        ("gemini-2.5-flash-preview-09-2025", (2, 5)),
        ("gemini-3-pro-preview", (3,)),
        ("gpt-5.5-pro", (5, 5)),
        ("gpt-5-2025-08-07", (5,)),
        ("gpt-4o-mini", (4,)),
        ("o3", (3,)),
        ("o4-mini", (4,)),
        ("gemini-flash-latest", ()),
    ],
)
def test_version_key_reads_the_version_and_ignores_dates(model_id, key):
    assert version_key(model_id) == key


def test_the_fallback_lists():
    table = load_table()

    assert table.fallback_models(G) == (
        "gemini-3.1-pro-preview", "gemini-3.1-flash-lite-preview", "gemini-2.5-pro", "gemini-2.5-flash",
    )
    assert table.fallback_models(A) == (
        "claude-sonnet-4-6", "claude-opus-4-6", "claude-sonnet-4-5", "claude-sonnet-4",
        "claude-haiku-4-5",
    )
    assert table.fallback_models(O) == table.curated_openai()
    assert table.fallback_models(ProviderKind.OPENROUTER) == ()
    assert table.fallback_models(ProviderKind.OPENAI_COMPATIBLE) == ()


def test_every_fallback_model_matches_a_rule():
    table = load_table()
    for kind in (G, A, O):
        for model in table.fallback_models(kind):
            assert table.record_for(kind.value, kind, model).sources["thinking"] == "table", model


def test_the_fallback_lists_fill_every_builtin_tier():
    table = load_table()
    for kind in (G, A, O):
        tiers = {table.record_for(kind.value, kind, m).tier for m in table.fallback_models(kind)}
        # Every built-in role finds a model while the provider's listing is
        # missing, so a failed listing never moves a role to another provider.
        assert {"fast", "default", "strong"} <= tiers, kind


def test_load_table_is_cached():
    assert load_table() is load_table()


class TestTableFile:
    def test_every_rule_and_model_entry_has_a_source_note(self):
        raw = _raw()

        assert raw["version"] == 1
        for kind, provider in raw["providers"].items():
            for rule in provider["rules"]:
                assert rule.get("source"), (kind, rule["match"])
            for model, entry in provider.get("models", {}).items():
                assert entry.get("source"), (kind, model)
            if provider.get("fallback_models"):
                assert provider.get("fallback_source"), kind
            if provider.get("curated"):
                assert provider.get("curated_source"), kind

    def test_the_file_is_valid_against_its_loader(self):
        ShippedTable(_raw())

    def test_a_rule_without_a_source_note_is_rejected(self):
        raw = _raw()
        del raw["providers"]["google"]["rules"][0]["source"]

        with pytest.raises(ValueError, match="source"):
            ShippedTable(raw)

    @pytest.mark.parametrize(
        "corrupt",
        [
            lambda raw: raw.update(version=2),
            lambda raw: raw["providers"]["google"]["rules"][0]["thinking"].update(mechanism="turbo"),
            lambda raw: raw["providers"]["google"]["rules"][0].update(match="(unclosed"),
            lambda raw: raw["providers"]["google"]["rules"][0]["thinking"].pop("budgets"),
            lambda raw: raw["providers"]["google"]["rules"][3]["thinking"].update(levels=[]),
            lambda raw: raw["providers"]["google"]["rules"][3]["thinking"].update(
                generic_map={"medium": "ultra"}
            ),
            lambda raw: raw["providers"]["google"]["rules"][0].update(colour="blue"),
        ],
    )
    def test_a_malformed_table_is_rejected(self, corrupt):
        raw = copy.deepcopy(_raw())
        corrupt(raw)

        with pytest.raises(ValueError):
            ShippedTable(raw)
