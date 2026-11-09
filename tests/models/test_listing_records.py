"""Listing records: what a provider's listing, the shipped table and the ``models``
entries make of its models.

``records_for`` is a pure function (no catalog, no network): the tests build
``Listing`` objects by hand and read what it serves, what it shows, and where
each value came from. The catalog's behaviour over these records (validation,
the list, the cache) is covered in ``test_catalog.py``.
"""

from __future__ import annotations

import dataclasses

import pytest

from agentic_cli.models import (
    ModelEntry,
    ModelRef,
    ProviderKind,
    ProviderSpec,
    ThinkingMechanism,
    ThinkingRule,
    load_table,
)
from agentic_cli.models.discovery import ListedModel, Listing
from agentic_cli.models.listing_records import records_for, with_entry
from tests.models.fakes import providers

TABLE = load_table()
ADAPTIVE = ThinkingMechanism.ANTHROPIC_ADAPTIVE
BUDGET = ThinkingMechanism.ANTHROPIC_BUDGET
PROVIDERS = providers(
    specs={"local-vllm": ProviderSpec(base_url="http://gpu-box:8000/v1")},
    google="g", anthropic="a", openai="o", openrouter="r",
)


def listing(provider: str, *models: ListedModel) -> Listing:
    return Listing(provider=provider, ok=True, models=models, fetched_at=1.0)


def served_and_shown(name: str, *models: ListedModel, entries=None):
    served, shown = records_for(PROVIDERS[name], listing(name, *models), TABLE, entries)
    return served, [record.model for record in shown]


# --- served and shown, by provider kind ----------------------------------------------


class TestGoogle:
    def test_every_id_is_served_and_only_the_shown_ones_are_shown(self):
        served, shown = served_and_shown(
            "google",
            ListedModel("gemini-2.5-flash"),
            ListedModel("gemini-flash-latest", shown=False),
            ListedModel("gemini-2.5-pro"),
        )

        assert list(served) == ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.5-pro"]
        assert shown == ["gemini-2.5-flash", "gemini-2.5-pro"]

    def test_the_listing_gives_the_numbers_and_the_table_the_rules(self):
        served, _ = served_and_shown(
            "google",
            ListedModel("gemini-2.5-flash", display_name="Gemini 2.5 Flash",
                        context_window=1048576, max_output=65536),
        )

        record = served["gemini-2.5-flash"]

        assert (record.display_name, record.context_window, record.max_output) == (
            "Gemini 2.5 Flash", 1048576, 65536,
        )
        assert record.thinking.mechanism is ThinkingMechanism.GEMINI_BUDGET
        assert record.sources["context_window"] == "listing"
        assert record.sources["thinking"] == "table"

    def test_a_model_the_listing_says_does_not_think_has_no_knob(self):
        served, _ = served_and_shown("google", ListedModel("gemini-2.5-flash", reasoning=False))

        record = served["gemini-2.5-flash"]

        assert record.thinking == ThinkingRule()
        assert record.sources["thinking"] == "listing"


class TestAnthropic:
    def test_a_dated_id_is_shown_as_its_alias_and_both_are_served(self):
        served, shown = served_and_shown(
            "anthropic",
            ListedModel("claude-sonnet-4-5-20250929", alias="claude-sonnet-4-5", shown=False),
            ListedModel("claude-opus-4-7"),
        )

        assert list(served) == ["claude-sonnet-4-5-20250929", "claude-sonnet-4-5", "claude-opus-4-7"]
        assert shown == ["claude-sonnet-4-5", "claude-opus-4-7"]

    def test_the_alias_takes_the_facts_of_the_first_id_listed_under_it(self):
        served, shown = served_and_shown(
            "anthropic",
            ListedModel("claude-sonnet-4-5-20250929", alias="claude-sonnet-4-5",
                        shown=False, context_window=200_000),
            ListedModel("claude-sonnet-4-5@2025-09-29", alias="claude-sonnet-4-5",
                        shown=False, context_window=100_000),
        )

        assert shown == ["claude-sonnet-4-5"]
        assert served["claude-sonnet-4-5"].context_window == 200_000
        assert served["claude-sonnet-4-5@2025-09-29"].context_window == 100_000

    def test_a_dated_id_is_described_by_its_alias_s_rule(self):
        served, _ = served_and_shown(
            "anthropic",
            ListedModel("claude-sonnet-5-5-20261001", alias="claude-sonnet-5-5", shown=False,
                        reasoning=True, thinking_types=("adaptive",)),
        )

        dated, alias = served["claude-sonnet-5-5-20261001"], served["claude-sonnet-5-5"]

        # The table's entry for the alias: no sampling parameters, thinking cannot be turned off.
        assert (dated.thinking.mechanism, dated.thinking.can_disable) == (ADAPTIVE, False)
        assert dated.sampling == "never"
        # Only the alias's name (its table entry's) is not the dated id's own.
        assert dataclasses.replace(dated, display_name=alias.display_name, sources=alias.sources) == (
            dataclasses.replace(alias, model=dated.model)
        )

    def test_the_listing_decides_the_mechanism_and_whether_it_can_be_turned_off(self):
        served, _ = served_and_shown(
            "anthropic",
            ListedModel("claude-opus-4-7", reasoning=True, thinking_types=("adaptive",),
                        effort_levels=("low", "high"), can_disable=True),
            ListedModel("claude-sonnet-4-5", reasoning=True, thinking_types=("enabled",),
                        can_disable=False),
            ListedModel("claude-3-5-haiku", reasoning=False),
        )

        opus, sonnet, haiku = (served[m] for m in ("claude-opus-4-7", "claude-sonnet-4-5",
                                                    "claude-3-5-haiku"))

        assert (opus.thinking.mechanism, opus.thinking.levels) == (ADAPTIVE, ("low", "high"))
        assert (sonnet.thinking.mechanism, sonnet.thinking.can_disable) == (BUDGET, False)
        assert haiku.thinking == ThinkingRule()
        assert opus.sources["thinking"] == "listing"

    def test_types_that_say_nothing_leave_the_tables_rule(self):
        served, _ = served_and_shown(
            "anthropic", ListedModel("claude-sonnet-4-6", reasoning=True),
        )

        record = served["claude-sonnet-4-6"]

        assert record.thinking.mechanism is ADAPTIVE
        assert record.sources["thinking"] == "table"


class TestClaudeThinkingAnUnexpectedTypeNames:
    """A listing can name a type the table's rule for the model does not expect:
    the rule is then the table's own for a Claude model it knows nothing more of
    (its ``^claude-`` rules), so the table is the one source of those numbers."""

    def test_adaptive_where_the_table_has_a_budget_takes_the_tables_adaptive_ladder(self):
        served, _ = served_and_shown(
            "anthropic",
            ListedModel("claude-sonnet-4-5", reasoning=True, thinking_types=("adaptive",)),
        )

        thinking = served["claude-sonnet-4-5"].thinking

        generic = TABLE.record_for("anthropic", ProviderKind.ANTHROPIC, "claude-unlisted-9").thinking
        assert generic.mechanism is ADAPTIVE
        assert thinking == ThinkingRule(ADAPTIVE, levels=generic.levels, can_disable=True)

    def test_a_budget_where_the_table_has_adaptive_takes_the_tables_budget_range(self):
        served, _ = served_and_shown(
            "anthropic",
            ListedModel("claude-sonnet-4-6", reasoning=True, thinking_types=("enabled",)),
        )

        thinking = served["claude-sonnet-4-6"].thinking

        generic = TABLE.record_for("anthropic", ProviderKind.ANTHROPIC, "claude-unlisted-1").thinking
        assert generic.mechanism is BUDGET
        assert thinking == ThinkingRule(
            BUDGET, budgets=generic.budgets, budget_range=generic.budget_range
        )

    def test_the_numbers_are_the_ones_the_table_ships(self):
        """What the catalog's own constants held before the table was the source."""
        adaptive = TABLE.record_for("anthropic", ProviderKind.ANTHROPIC, "claude-unlisted-9").thinking
        budget = TABLE.record_for("anthropic", ProviderKind.ANTHROPIC, "claude-unlisted-1").thinking

        assert adaptive.levels == ("low", "medium", "high", "max")
        assert (budget.budgets, budget.budget_range) == ((4096, 10000, 32000), (1024, 128000))


class TestOpenAI:
    def test_the_curated_ids_the_listing_has_come_first_then_other_reasoning_models(self):
        curated = TABLE.curated_openai()
        listed = [curated[2], "gpt-4.1", curated[0], "gpt-6.2-sol", "text-embedding-3-small"]

        served, shown = served_and_shown("openai", *(ListedModel(m) for m in listed))

        assert shown == [curated[0], curated[2], "gpt-6.2-sol"]
        assert set(served) == set(listed)

    def test_an_id_that_is_not_curated_nor_reasoning_is_served_not_shown(self):
        served, shown = served_and_shown("openai", ListedModel("gpt-4.1"))

        assert "gpt-4.1" in served and shown == []
        assert served["gpt-4.1"].thinking.mechanism is ThinkingMechanism.NONE

    def test_the_listing_gives_ids_only(self):
        curated = TABLE.curated_openai()[0]
        served, _ = served_and_shown(
            "openai", ListedModel(curated, display_name="Ignored", context_window=7),
        )

        assert served[curated] == TABLE.record_for("openai", ProviderKind.OPENAI, curated)


class TestOpenRouter:
    def test_every_listed_model_is_served_and_shown(self):
        served, shown = served_and_shown(
            "openrouter", ListedModel("qwen/qwen3-235b-a22b"), ListedModel("other/model"),
        )

        assert shown == ["qwen/qwen3-235b-a22b", "other/model"] == list(served)

    def test_the_listing_gives_tools_reasoning_and_the_numbers(self):
        served, _ = served_and_shown(
            "openrouter",
            ListedModel("qwen/qwen3-235b-a22b", context_window=131072, max_output=8192,
                        tools=True, reasoning=True, image_input=False),
            ListedModel("plain/model", reasoning=False, tools=False),
        )

        qwen, plain = served["qwen/qwen3-235b-a22b"], served["plain/model"]

        assert qwen.thinking == ThinkingRule(
            ThinkingMechanism.EFFORT, levels=("none", "minimal", "low", "medium", "high", "xhigh")
        )
        assert (qwen.tools, qwen.context_window, qwen.max_output) == (True, 131072, 8192)
        assert qwen.reasoning_text is True
        assert plain.thinking == ThinkingRule()
        assert plain.tools is False
        assert plain.sources["tools"] == "listing"


class TestOpenAICompatible:
    def test_each_served_name_is_one_conservative_record_with_its_context_length(self):
        served, shown = served_and_shown(
            "local-vllm",
            ListedModel("Qwen/Qwen3-32B", display_name="Qwen", context_window=32768,
                        tools=False, reasoning=True),
            ListedModel("Qwen/Qwen3-32B"),
        )

        record = served["Qwen/Qwen3-32B"]

        assert shown == ["Qwen/Qwen3-32B"]
        assert (record.display_name, record.context_window) == ("Qwen", 32768)
        # Tools are assumed, and the listing's reasoning flag is not read.
        assert record.tools is True and record.thinking == ThinkingRule()
        assert record.sources["tools"] == "default"


class TestWithoutAListing:
    @pytest.mark.parametrize("name", ["google", "anthropic", "openai"])
    def test_a_built_in_provider_stands_on_the_tables_fallback_models(self, name):
        served, shown = records_for(PROVIDERS[name], None, TABLE)

        kind = PROVIDERS[name].kind
        assert [r.model for r in shown] == list(TABLE.fallback_models(kind))
        assert list(served) == [r.model for r in shown]
        assert shown

    @pytest.mark.parametrize("name", ["openrouter", "local-vllm"])
    def test_a_configured_provider_has_none(self, name):
        assert records_for(PROVIDERS[name], None, TABLE) == ({}, ())


# --- the entries ------------------------------------------------------------------------


class TestEntries:
    def test_an_entry_beats_the_listing_which_beats_the_table(self):
        ref = ModelRef("google", "gemini-2.5-flash")
        listed = ListedModel(ref.model, context_window=1048576, max_output=65536)

        bare, _ = served_and_shown("google", listed)
        entry = ModelEntry(context_window=500_000)
        served, _ = served_and_shown("google", listed, entries={ref: entry})

        assert bare[ref.model].context_window == 1048576
        record = served[ref.model]
        assert record.context_window == 500_000
        assert record.sources["context_window"] == "config"
        assert record.max_output == 65536 and record.sources["max_output"] == "listing"
        assert record.sources["thinking"] == "table"

    def test_the_shown_record_is_the_served_one_with_its_entry(self):
        ref = ModelRef("google", "gemini-2.5-flash")
        served, shown = records_for(
            PROVIDERS["google"], listing("google", ListedModel(ref.model)), TABLE,
            {ref: ModelEntry(tools=False)},
        )

        assert shown == (served[ref.model],)
        assert shown[0].tools is False

    def test_only_this_providers_entries_apply_and_an_unlisted_one_adds_nothing(self):
        entries = {
            ModelRef("anthropic", "claude-opus-4-7"): ModelEntry(tools=False),
            ModelRef("google", "not-listed"): ModelEntry(tools=False),
        }

        served, shown = served_and_shown(
            "google", ListedModel("gemini-2.5-flash"), entries=entries
        )

        assert list(served) == shown == ["gemini-2.5-flash"]
        assert served["gemini-2.5-flash"].tools is True

    def test_an_entry_applies_to_a_model_a_provider_serves_but_does_not_show(self):
        ref = ModelRef("google", "gemini-flash-latest")

        served, shown = served_and_shown(
            "google", ListedModel(ref.model, shown=False), entries={ref: ModelEntry(tools=False)}
        )

        assert shown == [] and served[ref.model].tools is False

    def test_without_a_listing_the_fallback_records_take_their_entries_too(self):
        ref = ModelRef("anthropic", TABLE.fallback_models(ProviderKind.ANTHROPIC)[0])

        served, shown = records_for(PROVIDERS["anthropic"], None, TABLE, {ref: ModelEntry(tools=False)})

        assert served[ref.model].tools is False
        assert shown[0] is served[ref.model]


class TestWithEntry:
    RECORD = TABLE.record_for("local-vllm", ProviderKind.OPENAI_COMPATIBLE, "m")

    def test_no_entry_leaves_the_record(self):
        assert with_entry(self.RECORD, None) is self.RECORD

    def test_each_value_the_entry_sets_is_marked_config(self):
        record = with_entry(
            self.RECORD,
            ModelEntry(context_window=8192, max_output=1024, tools=False, image_input=True,
                       reasoning_text=True),
        )

        assert (record.context_window, record.max_output, record.tools, record.image_input,
                record.reasoning_text) == (8192, 1024, False, True, True)
        assert {name: record.sources[name] for name in (
            "context_window", "max_output", "tools", "image_input", "reasoning_text",
        )} == dict.fromkeys(
            ("context_window", "max_output", "tools", "image_input", "reasoning_text"), "config"
        )
        assert record.sources["display_name"] == "default"

    def test_thinking_effort_takes_the_entry_s_ladder_else_the_default(self):
        with_ladder = with_entry(self.RECORD, ModelEntry(thinking="effort", effort_levels=["low", "high"]))
        default = with_entry(self.RECORD, ModelEntry(thinking="effort"))

        assert with_ladder.thinking == ThinkingRule(ThinkingMechanism.EFFORT, levels=("low", "high"))
        assert default.thinking.mechanism is ThinkingMechanism.EFFORT and default.thinking.levels

    def test_effort_levels_replace_the_ladder_of_a_model_that_takes_an_effort(self):
        effort = dataclasses.replace(
            self.RECORD, thinking=ThinkingRule(ThinkingMechanism.EFFORT, levels=("low", "medium"))
        )

        record = with_entry(effort, ModelEntry(effort_levels=["high"]))

        assert record.thinking == ThinkingRule(ThinkingMechanism.EFFORT, levels=("high",))

    def test_effort_levels_alone_are_ignored_on_any_other_model(self):
        assert with_entry(self.RECORD, ModelEntry(effort_levels=["high"])) is self.RECORD

    def test_chat_template_and_none_set_the_mechanism(self):
        assert with_entry(self.RECORD, ModelEntry(thinking="chat_template")).thinking == ThinkingRule(
            ThinkingMechanism.CHAT_TEMPLATE
        )
        assert with_entry(self.RECORD, ModelEntry(thinking="none")).thinking == ThinkingRule()
