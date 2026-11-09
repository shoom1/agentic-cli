"""The catalog: records merged from ``models`` entries, listings and the table;
validation; the providers in use; the list the pickers offer; search; refresh.

Listings come from ``ProviderAPI``, a fake of every provider's model list
behind ``httpx.MockTransport``. The on-disk cache is covered in
``test_catalog_cache.py``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time

import httpx
import pytest

from agentic_cli.models import (
    ModelCatalog,
    ModelEntry,
    ModelRef,
    ModelResolutionError,
    NativeThinking,
    ProviderKind,
    ProviderSpec,
    ProviderState,
    ProviderStatus,
    RoleMap,
    ThinkingMechanism,
    ThinkingRule,
    load_table,
    resolve_model,
)
from tests.models.fakes import OPENROUTER_URL, Clock, ProviderAPI, hang, providers, recorded, refuse

KEYS = {"google": "g-test-key", "anthropic": "a-test-key", "openai": "o-test-key"}
VLLM = ProviderSpec(base_url="http://gpu-box:8000/v1")
#: What the user says about two of the server's models: one it serves, one it does not list.
ENTRIES = {
    ModelRef("local-vllm", "Qwen/Qwen3-32B"): ModelEntry(
        context_window=16384, thinking="chat_template"
    ),
    ModelRef("local-vllm", "entry-only"): ModelEntry(
        thinking="effort", effort_levels=["low", "high"]
    ),
}
GEMINI = ModelRef("google", "gemini-2.5-flash")
QWEN = ModelRef("openrouter", "qwen/qwen3-235b-a22b")
#: A setting names an OpenRouter model, so OpenRouter is in use (and listed).
ON_OPENROUTER = (QWEN,)


def catalog_for(
    api: ProviderAPI | None = None,
    *,
    specs=None,
    env=None,
    entries=None,
    references=(),
    computed_roles=(),
    clock=None,
    named_by=None,
    **keys,
) -> ModelCatalog:
    extra = {"named_by": named_by} if named_by is not None else {}
    return ModelCatalog(
        providers=providers(specs=specs or {}, env=env, **keys),
        entries=entries,
        references=references,
        computed_roles=computed_roles,
        transport=(api or ProviderAPI()).transport,
        clock=clock or Clock(),
        **extra,
    )


async def loaded(api: ProviderAPI | None = None, **kwargs) -> ModelCatalog:
    catalog = catalog_for(api, **kwargs)
    await catalog.load()
    return catalog


async def checked(
    api: ProviderAPI | None = None, *, catalog: ModelCatalog | None = None, **kwargs
) -> ModelCatalog:
    """A catalog loaded, then checked: its background refresh, which lists each
    provider in use the start did not wait on, has run."""
    if catalog is None:
        catalog = catalog_for(api, **kwargs)
    await catalog.load()
    task = catalog.start_background_refresh()
    if task is not None:
        await task
    return catalog


def ids(records) -> list[str]:
    return [record.model for record in records]


ADAPTIVE = ThinkingMechanism.ANTHROPIC_ADAPTIVE
BUDGET = ThinkingMechanism.ANTHROPIC_BUDGET


def claude_listing(model_id: str, *, disabled: bool | None, types=("adaptive",)) -> dict:
    """An Anthropic listing of one thinking model; ``disabled`` None leaves the field out."""
    listed_types = {name: {"supported": True} for name in types}
    if disabled is not None:
        listed_types["disabled"] = {"supported": disabled}
    entry = {
        "id": model_id,
        "capabilities": {"thinking": {"supported": True, "types": listed_types}},
    }
    return {"data": [entry], "has_more": False}


# --- records built from listings ---------------------------------------------


class TestGoogle:
    async def test_the_listing_gives_the_numbers_and_the_table_the_rules(self):
        catalog = await loaded(google=KEYS["google"])

        record = catalog.record(ModelRef("google", "gemini-2.5-flash"))

        assert (record.display_name, record.context_window, record.max_output) == (
            "Gemini 2.5 Flash", 1048576, 65536,
        )
        assert record.tier == "default"
        assert record.thinking.mechanism is ThinkingMechanism.GEMINI_BUDGET
        assert record.sources["context_window"] == "listing"
        assert record.sources["thinking"] == "table"

    async def test_the_records_are_the_filtered_listing(self):
        catalog = await loaded(google=KEYS["google"])

        assert ids(catalog.records("google")) == [
            "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite",
            "gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-2.0-flash",
        ]

    async def test_a_model_the_listing_says_does_not_think_has_no_knob(self):
        payload = recorded("google")
        payload["models"][0] = {**payload["models"][0], "thinking": False}
        catalog = await loaded(ProviderAPI(google=payload), google=KEYS["google"])

        record = catalog.record(ModelRef("google", "gemini-2.5-flash"))

        assert record.thinking == ThinkingRule()
        assert record.sources["thinking"] == "listing"

    async def test_a_model_no_rule_matches_gets_the_conservative_record(self):
        catalog = await loaded(google=KEYS["google"])

        record = catalog.record(ModelRef("google", "gemini-2.0-flash"))

        assert record.thinking.mechanism is ThinkingMechanism.NONE
        assert record.tier is None
        assert record.tools is True
        assert record.display_name == "Gemini 2.0 Flash"


class TestGoogleLatestAliases:
    """A pinned ``-latest`` alias is served (the list does not show it) and described as
    the family it points at: its listing says it thinks, and thinking is not silently off."""

    async def test_a_pinned_alias_thinks_as_its_family(self):
        catalog = await loaded(google=KEYS["google"])

        record = catalog.record(ModelRef("google", "gemini-flash-latest"))

        assert record.thinking.mechanism is ThinkingMechanism.GEMINI_LEVEL
        assert (record.tier, record.sources["thinking"]) == ("default", "table")
        assert record.context_window == 1048576 and record.sources["context_window"] == "listing"

    async def test_an_alias_the_listing_says_does_not_think_has_no_knob(self):
        payload = recorded("google")
        for model in payload["models"]:
            if model["name"] == "models/gemini-flash-latest":
                model["thinking"] = False
        catalog = await loaded(ProviderAPI(google=payload), google=KEYS["google"])

        record = catalog.record(ModelRef("google", "gemini-flash-latest"))

        assert record.thinking == ThinkingRule()
        assert record.sources["thinking"] == "listing"

    async def test_a_default_model_pinned_to_an_alias_resolves_with_its_thinking_level(self):
        catalog = await loaded(google=KEYS["google"])

        resolved = resolve_model(
            agent=None, spec="google:gemini-flash-latest", generation=None,
            thinking_effort="high", roles=RoleMap(default_model=None, model_roles={}),
            catalog=catalog, providers=providers(google=KEYS["google"]),
        )

        assert resolved.thinking is not None
        assert resolved.thinking.mechanism is ThinkingMechanism.GEMINI_LEVEL


class TestAnthropic:
    async def test_the_thinking_types_decide_the_mechanism(self):
        catalog = await loaded(anthropic=KEYS["anthropic"])

        def thinking(model: str) -> ThinkingRule:
            return catalog.record(ModelRef("anthropic", model)).thinking

        opus = thinking("claude-opus-4-7")
        assert opus.mechanism is ThinkingMechanism.ANTHROPIC_ADAPTIVE
        assert opus.levels == ("low", "medium", "high", "xhigh", "max")
        # Both types: adaptive is the one to use.
        assert thinking("claude-sonnet-4-6").mechanism is ThinkingMechanism.ANTHROPIC_ADAPTIVE
        assert thinking("claude-sonnet-4-6").levels == ("low", "medium", "high", "max")
        sonnet = thinking("claude-sonnet-4-5")
        assert sonnet.mechanism is ThinkingMechanism.ANTHROPIC_BUDGET
        assert (sonnet.budgets, sonnet.budget_range) == ((4096, 10000, 32000), (1024, 128000))
        # The listing says it does not think.
        assert thinking("claude-3-5-haiku-20241022") == ThinkingRule()
        record = catalog.record(ModelRef("anthropic", "claude-opus-4-7"))
        assert record.sources["thinking"] == "listing"
        assert (record.context_window, record.max_output) == (1000000, 128000)
        assert record.image_input is True

    async def test_the_listing_beats_the_version_check(self):
        payload = {"data": [{
            "id": "claude-sonnet-4-5", "display_name": "Claude Sonnet 4.5",
            "capabilities": {
                "effort": {
                    "supported": True, "low": {"supported": True}, "high": {"supported": True},
                },
                "thinking": {"supported": True, "types": {"adaptive": {"supported": True}}},
            },
        }], "has_more": False}
        catalog = await loaded(ProviderAPI(anthropic=payload), anthropic=KEYS["anthropic"])

        thinking = catalog.record(ModelRef("anthropic", "claude-sonnet-4-5")).thinking

        assert thinking.mechanism is ThinkingMechanism.ANTHROPIC_ADAPTIVE
        assert thinking.levels == ("low", "high")

    async def test_a_model_that_cannot_disable_thinking_keeps_that(self):
        """A listing that does not say whether thinking can be turned off
        leaves the table's answer."""
        payload = claude_listing("claude-opus-5-5", disabled=None)
        catalog = await loaded(ProviderAPI(anthropic=payload), anthropic=KEYS["anthropic"])

        thinking = catalog.record(ModelRef("anthropic", "claude-opus-5-5")).thinking

        assert thinking.mechanism is ThinkingMechanism.ANTHROPIC_ADAPTIVE
        assert thinking.can_disable is False

    async def test_the_recorded_listing_says_sonnet_5_5_cannot_turn_thinking_off(self):
        catalog = await loaded(anthropic=KEYS["anthropic"])

        def can_disable(model: str) -> bool:
            return catalog.record(ModelRef("anthropic", model)).thinking.can_disable

        assert can_disable("claude-sonnet-5-5") is False
        assert can_disable("claude-sonnet-4-6") is True
        assert can_disable("claude-opus-4-7") is True

    @pytest.mark.parametrize(
        ("model", "disabled", "types", "mechanism", "can_disable"),
        [
            # A listed false beats a table true, and a listed true a table false.
            ("claude-sonnet-4-6", False, ("adaptive",), ADAPTIVE, False),
            ("claude-opus-5-5", True, ("adaptive",), ADAPTIVE, True),
            ("claude-sonnet-4-5", False, ("enabled",), BUDGET, False),
            # Types that say nothing leave the table's mechanism, not its answer.
            ("claude-sonnet-4-6", False, (), ADAPTIVE, False),
            ("claude-opus-5-5", True, (), ADAPTIVE, True),
            # An absent field keeps the table's value.
            ("claude-sonnet-4-6", None, ("adaptive",), ADAPTIVE, True),
            ("claude-sonnet-5-5", None, ("adaptive",), ADAPTIVE, False),
            ("claude-sonnet-4-5", None, ("enabled",), BUDGET, True),
        ],
    )
    async def test_the_listing_decides_whether_thinking_can_be_turned_off(
        self, model, disabled, types, mechanism, can_disable
    ):
        payload = claude_listing(model, disabled=disabled, types=types)
        catalog = await loaded(ProviderAPI(anthropic=payload), anthropic=KEYS["anthropic"])

        thinking = catalog.record(ModelRef("anthropic", model)).thinking

        assert thinking.mechanism is mechanism
        assert thinking.can_disable is can_disable

    @pytest.mark.parametrize(
        ("model", "disabled", "expected"),
        [
            # It cannot be turned off: none is adaptive thinking at the lowest effort.
            ("claude-sonnet-5-5", False, NativeThinking(
                ADAPTIVE, True, level="low", label="adaptive, effort low",
            )),
            ("claude-sonnet-4-6", False, NativeThinking(
                ADAPTIVE, True, level="low", label="adaptive, effort low",
            )),
            # It can: none turns it off.
            ("claude-opus-5-5", True, NativeThinking(ADAPTIVE, False, label="off")),
        ],
    )
    async def test_thinking_none_where_it_cannot_be_turned_off(self, model, disabled, expected):
        payload = claude_listing(model, disabled=disabled)
        catalog = await loaded(ProviderAPI(anthropic=payload), anthropic=KEYS["anthropic"])

        resolved = resolve_model(
            agent="writer", spec=f"anthropic:{model}", generation=None, thinking_effort="none",
            roles=RoleMap(default_model=None, model_roles={}), catalog=catalog,
            providers=providers(anthropic=KEYS["anthropic"]),
        )

        assert resolved.thinking == expected

    async def test_without_capabilities_the_version_decides(self):
        catalog = await loaded(anthropic=KEYS["anthropic"])

        record = catalog.record(ModelRef("anthropic", "claude-3-haiku-20240307"))

        assert record.thinking.mechanism is ThinkingMechanism.ANTHROPIC_BUDGET
        assert record.sources["thinking"] == "table"

    async def test_after_a_failed_listing_the_version_decides(self):
        catalog = await loaded(ProviderAPI(anthropic=500), anthropic=KEYS["anthropic"])

        adaptive = catalog.record(ModelRef("anthropic", "claude-opus-4-7")).thinking
        budget = catalog.record(ModelRef("anthropic", "claude-sonnet-4-5")).thinking

        assert adaptive.mechanism is ThinkingMechanism.ANTHROPIC_ADAPTIVE
        assert budget.mechanism is ThinkingMechanism.ANTHROPIC_BUDGET


class TestOpenAI:
    async def test_a_fresh_listing_offers_the_curated_and_reasoning_ids_it_has(self):
        """R15: curated ids the listing contains, then listed ids outside the
        curated list that a reasoning-family rule matches."""
        listed = [
            "gpt-4.1", "gpt-5.6-luna", "gpt-5.5", "gpt-6.1-sol", "gpt-6-sol-pro",
            "text-embedding-3-small",
        ]
        api = ProviderAPI(openai={"object": "list", "data": [{"id": m, "object": "model"} for m in listed]})
        catalog = await loaded(api, openai=KEYS["openai"])

        assert ids(catalog.records("openai")) == [
            "gpt-6.1-sol", "gpt-5.6-luna", "gpt-5.5", "gpt-6-sol-pro",
        ]

    async def test_a_listed_id_no_rule_matches_is_accepted_but_not_offered(self):
        catalog = await loaded(openai=KEYS["openai"])

        record = catalog.record(ModelRef("openai", "text-embedding-3-small"))

        assert record.thinking.mechanism is ThinkingMechanism.NONE
        assert "text-embedding-3-small" not in ids(catalog.records("openai"))
        assert "gpt-4o-mini-tts" not in ids(catalog.records("openai"))

    async def test_an_unlisted_id_is_rejected_after_a_fresh_listing(self):
        catalog = await loaded(openai=KEYS["openai"])

        with pytest.raises(ModelResolutionError, match="openai:gpt-5.6 is not available"):
            catalog.record(ModelRef("openai", "gpt-5.6"))

    async def test_before_or_without_a_listing_the_curated_list(self):
        curated = list(load_table().curated_openai())
        unloaded = catalog_for(openai=KEYS["openai"])
        failed = await loaded(ProviderAPI(openai=401), openai=KEYS["openai"])

        assert ids(unloaded.records("openai")) == curated
        assert ids(failed.records("openai")) == curated
        assert failed.record(ModelRef("openai", "gpt-5.6")).model == "gpt-5.6"


class TestOpenRouter:
    async def test_records_from_the_listing(self):
        catalog = await loaded(references=ON_OPENROUTER, openrouter="r-test-key")

        qwen = catalog.record(ModelRef("openrouter", "qwen/qwen3-235b-a22b"))
        free = catalog.record(ModelRef("openrouter", "qwen/qwen3-235b-a22b:free"))
        llama = catalog.record(ModelRef("openrouter", "meta-llama/llama-3.3-70b-instruct"))

        assert qwen.thinking == ThinkingRule(
            ThinkingMechanism.EFFORT, levels=("none", "minimal", "low", "medium", "high", "xhigh")
        )
        assert (qwen.tools, qwen.context_window, qwen.max_output) == (True, 131072, 8192)
        assert qwen.reasoning_text is True
        assert free.tools is False
        assert llama.thinking.mechanism is ThinkingMechanism.NONE
        assert qwen.kind is ProviderKind.OPENROUTER
        assert qwen.sources["tools"] == "listing"

    async def test_no_listing_no_records(self):
        catalog = await loaded(
            ProviderAPI(openrouter=503), references=ON_OPENROUTER, openrouter="r-test-key"
        )

        assert catalog.records("openrouter") == []

    async def test_a_provider_of_kind_openrouter_under_any_name(self):
        """The kind, not the name, picks OpenRouter's lister and its records."""
        spec = ProviderSpec(kind="openrouter", base_url=OPENROUTER_URL, api_key_env="OR_KEY")
        api = ProviderAPI()

        catalog = await loaded(
            api,
            specs={"or": spec},
            env={"OR_KEY": "or-test-key"},
            references=[ModelRef("or", "qwen/qwen3-235b-a22b")],
        )

        qwen = catalog.record(ModelRef("or", "qwen/qwen3-235b-a22b"))
        assert (qwen.provider, qwen.kind) == ("or", ProviderKind.OPENROUTER)
        assert qwen.thinking == ThinkingRule(
            ThinkingMechanism.EFFORT, levels=("none", "minimal", "low", "medium", "high", "xhigh")
        )
        assert (qwen.tools, qwen.sources["tools"]) == (True, "listing")
        assert api.calls == ["openrouter"]
        assert catalog.is_authoritative("or")


class TestEntries:
    async def test_an_endpoint_s_entries_and_listed_models(self):
        catalog = await checked(specs={"local-vllm": VLLM}, entries=ENTRIES)

        def record(model: str):
            return catalog.record(ModelRef("local-vllm", model))

        qwen = record("Qwen/Qwen3-32B")
        assert qwen.context_window == 16384  # the entry beats the listing's 32768
        assert qwen.thinking.mechanism is ThinkingMechanism.CHAT_TEMPLATE
        assert qwen.sources["context_window"] == "config"
        assert qwen.sources["thinking"] == "config"
        assert qwen.kind is ProviderKind.OPENAI_COMPATIBLE
        lora = record("qwen3-sql-lora")  # listed, no entry
        assert (lora.tools, lora.context_window) == (True, 32768)
        assert lora.sources["context_window"] == "listing"
        # The provider's records are what it serves, each with its entry.
        assert set(ids(catalog.records("local-vllm"))) == {"Qwen/Qwen3-32B", "qwen3-sql-lora"}
        assert qwen in catalog.records("local-vllm")

    async def test_an_entry_without_a_number_takes_the_listing(self):
        entries = {ModelRef("local-vllm", "Qwen/Qwen3-32B"): ModelEntry()}
        catalog = await loaded(specs={"local-vllm": VLLM}, entries=entries)

        record = catalog.record(ModelRef("local-vllm", "Qwen/Qwen3-32B"))

        assert record.context_window == 32768
        assert record.sources["context_window"] == "listing"
        assert "config" not in record.sources.values()

    async def test_a_fresh_listing_rejects_an_id_it_lacks_but_not_an_entry(self):
        """On any provider: a server whose listing leaves out models it serves
        sets ``list_models: false`` instead. An entry is the settings' word, so
        it is accepted with what it says, and left out of the list."""
        catalog = await checked(specs={"local-vllm": VLLM}, entries=ENTRIES)

        with pytest.raises(ModelResolutionError, match="local-vllm:other is not available"):
            catalog.record(ModelRef("local-vllm", "other"))
        entry_only = catalog.record(ModelRef("local-vllm", "entry-only"))
        assert entry_only.thinking.levels == ("low", "high")
        assert catalog.not_offered() == {"local-vllm:entry-only": "not served by local-vllm"}

    async def test_a_server_that_is_down_still_takes_its_entries(self):
        catalog = await loaded(
            ProviderAPI(endpoint=refuse), specs={"local-vllm": VLLM}, entries=ENTRIES
        )

        assert catalog.records("local-vllm") == []
        qwen = catalog.record(ModelRef("local-vllm", "Qwen/Qwen3-32B"))
        assert (qwen.context_window, qwen.sources["context_window"]) == (16384, "config")
        assert catalog.record(ModelRef("local-vllm", "anything")).tools is True

    async def test_an_entry_its_fresh_listing_lacks_is_accepted_but_not_offered(self):
        typo = ModelRef("anthropic", "claude-sonnet-9-9")
        catalog = await checked(entries={typo: ModelEntry()}, anthropic=KEYS["anthropic"])

        assert catalog.is_authoritative("anthropic")
        assert catalog.record(typo).model == typo.model
        assert str(typo) not in catalog.offered()
        assert catalog.not_offered() == {str(typo): "not served by anthropic"}

    async def test_an_entry_beats_the_listing_which_beats_the_table(self):
        sonnet = ModelRef("anthropic", "claude-sonnet-4-6")
        entries = {sonnet: ModelEntry(context_window=200_000, image_input=False)}
        catalog = await checked(entries=entries, anthropic=KEYS["anthropic"])

        record = catalog.record(sonnet)

        assert (record.context_window, record.sources["context_window"]) == (200_000, "config")
        assert (record.image_input, record.sources["image_input"]) == (False, "config")
        assert (record.max_output, record.sources["max_output"]) == (64_000, "listing")
        assert record.sources["cache"] == "table"
        assert record.thinking.mechanism is ADAPTIVE  # the listing's: no entry may set it
        # /models reads the provider's records: the same record, entry and all.
        assert record in catalog.records("anthropic")

    async def test_a_model_no_listing_or_fallback_holds_takes_its_entry_over_the_table(self):
        sonnet = ModelRef("anthropic", "claude-sonnet-4-7")  # not among the fallbacks
        catalog = catalog_for(
            entries={sonnet: ModelEntry(max_output=32_000)}, anthropic=KEYS["anthropic"]
        )

        record = catalog.record(sonnet)

        assert (record.max_output, record.sources["max_output"]) == (32_000, "config")
        assert record.sources["tier"] == "table"

    async def test_thinking_effort_without_levels_takes_the_default_ladder(self):
        entries = {ModelRef("local-vllm", "Qwen/Qwen3-32B"): ModelEntry(thinking="effort")}
        catalog = await loaded(specs={"local-vllm": VLLM}, entries=entries)

        record = catalog.record(ModelRef("local-vllm", "Qwen/Qwen3-32B"))

        assert record.thinking == ThinkingRule(
            ThinkingMechanism.EFFORT, levels=("low", "medium", "high")
        )

    async def test_thinking_effort_on_a_model_that_takes_one_keeps_its_ladder(self):
        """Saying what the listing says changes nothing: ``none`` stays the
        lowest effort (thinking off), and minimal and xhigh stay."""
        qwen = ModelRef("openrouter", "qwen/qwen3-235b-a22b")
        catalog = await checked(
            entries={qwen: ModelEntry(thinking="effort")}, openrouter="r-test-key"
        )

        record = catalog.record(qwen)

        assert record.thinking == ThinkingRule(
            ThinkingMechanism.EFFORT, levels=("none", "minimal", "low", "medium", "high", "xhigh")
        )
        assert record.sources["thinking"] == "listing"

    async def test_effort_levels_alone_replace_an_effort_ladder_and_nothing_else(self):
        qwen = ModelRef("openrouter", "qwen/qwen3-235b-a22b")
        llama = ModelRef("openrouter", "meta-llama/llama-3.3-70b-instruct")
        entries = {
            qwen: ModelEntry(effort_levels=["low", "high"]),
            llama: ModelEntry(effort_levels=["low", "high"]),
        }
        catalog = await loaded(entries=entries, openrouter="r-test-key")

        reasoning, plain = catalog.record(qwen), catalog.record(llama)

        assert reasoning.thinking == ThinkingRule(ThinkingMechanism.EFFORT, levels=("low", "high"))
        assert reasoning.sources["thinking"] == "config"
        assert plain.thinking.mechanism is ThinkingMechanism.NONE  # no ladder to replace
        assert plain.sources["thinking"] != "config"

    async def test_an_entry_s_thinking_none_turns_a_listed_knob_off(self):
        qwen = ModelRef("openrouter", "qwen/qwen3-235b-a22b")
        catalog = await checked(
            entries={qwen: ModelEntry(thinking="none", reasoning_text=False)},
            openrouter="r-test-key",
        )

        record = catalog.record(qwen)

        assert record.thinking == ThinkingRule()
        assert (record.reasoning_text, record.sources["reasoning_text"]) == (False, "config")


# --- validation ----------------------------------------------------------------


class TestValidation:
    async def test_an_unknown_id_is_rejected_only_when_its_own_listing_is_fresh(self):
        """A failed Anthropic listing neither stops Google rejecting an unknown
        Gemini id nor makes Anthropic's fallbacks authoritative."""
        catalog = await loaded(
            ProviderAPI(anthropic=500), google=KEYS["google"], anthropic=KEYS["anthropic"]
        )

        with pytest.raises(ModelResolutionError) as caught:
            catalog.record(ModelRef("google", "gemini-9-ultra"))
        assert str(caught.value).startswith("Model google:gemini-9-ultra is not available.")
        assert "gemini-2.5-flash" in str(caught.value)
        assert catalog.record(ModelRef("anthropic", "claude-unknown-9")).model == "claude-unknown-9"

    async def test_a_provider_never_listed_accepts_any_id(self):
        catalog = await loaded()  # no keys: nothing listed

        record = catalog.record(ModelRef("google", "gemini-9-ultra"))

        assert record.model == "gemini-9-ultra"

    async def test_a_long_model_list_is_cut_short(self):
        payload = {"data": [{"id": f"vendor/model-{i}"} for i in range(60)]}
        catalog = await loaded(
            ProviderAPI(openrouter=payload), references=ON_OPENROUTER, openrouter="r-test-key"
        )

        with pytest.raises(ModelResolutionError, match=r"and 40 more"):
            catalog.record(ModelRef("openrouter", "vendor/unknown"))

    async def test_an_undefined_provider_is_an_error(self):
        catalog = await loaded()

        with pytest.raises(ModelResolutionError) as caught:
            catalog.record(ModelRef("openrouter", "qwen/qwen3-235b-a22b"))

        assert str(caught.value) == (
            "Model openrouter:qwen/qwen3-235b-a22b names 'openrouter', which is neither a "
            "built-in provider nor defined in model_providers."
        )


# --- served and shown: validation and search see every id, the list the shown ones ---

#: Ids the recorded listings serve but do not show: Google's alias, dated
#: snapshot, variant and point release; Anthropic's dated ids.
SERVED_NOT_SHOWN = (
    ModelRef("google", "gemini-flash-latest"),
    ModelRef("google", "gemini-2.5-flash-preview-09-2025"),
    ModelRef("google", "gemini-3.1-pro-preview-customtools"),
    ModelRef("google", "gemini-2.0-flash-001"),
    ModelRef("anthropic", "claude-sonnet-4-5-20250929"),
    ModelRef("anthropic", "claude-haiku-4-5-20251001"),
)


async def test_a_fresh_listing_accepts_every_id_its_provider_serves():
    """A pinned snapshot or alias the provider lists is no unknown id, though
    the list never offers it."""
    catalog = await loaded(google=KEYS["google"], anthropic=KEYS["anthropic"])

    assert catalog.is_authoritative("google") and catalog.is_authoritative("anthropic")
    assert [str(catalog.record(ref).ref) for ref in SERVED_NOT_SHOWN] == [
        str(ref) for ref in SERVED_NOT_SHOWN
    ]
    with pytest.raises(ModelResolutionError, match="google:gemini-9-ultra is not available"):
        catalog.record(ModelRef("google", "gemini-9-ultra"))


async def test_google_serves_for_chat_only_the_models_that_generate_content():
    """An embedding model (Imagen and Veo alike) is listed but takes no chat
    request: a fresh listing rejects it rather than the first turn failing."""
    catalog = await loaded(google=KEYS["google"])

    assert catalog.record(ModelRef("google", "gemma-3-27b-it")).model == "gemma-3-27b-it"
    with pytest.raises(ModelResolutionError, match="google:text-embedding-004 is not available"):
        catalog.record(ModelRef("google", "text-embedding-004"))


async def test_what_each_provider_shows():
    """Each provider's shown records, the tiers a list with no entry offers, the
    status counts and the "not available" message, for the recorded listings: a
    dated 3.x Claude id is shown as itself, since Anthropic serves no alias for it."""
    catalog = await loaded(references=ON_OPENROUTER, openrouter="r-test-key", **KEYS)

    assert {name: ids(catalog.records(name)) for name in (*KEYS, "openrouter")} == {
        "google": [
            "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite",
            "gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-2.0-flash",
        ],
        "anthropic": [
            "claude-sonnet-5-5", "claude-opus-4-7", "claude-sonnet-4-6", "claude-sonnet-4-5",
            "claude-haiku-4-5", "claude-3-5-haiku-20241022", "claude-3-haiku-20240307",
        ],
        "openai": [
            "gpt-5.5", "gpt-5.5-2026-03-05", "gpt-5.5-pro", "gpt-5.4-mini", "gpt-5-codex", "o3",
        ],
        "openrouter": [
            "qwen/qwen3-235b-a22b", "qwen/qwen3-235b-a22b:free", "anthropic/claude-sonnet-4.5",
            "meta-llama/llama-3.3-70b-instruct",
        ],
    }
    assert catalog.offered() == [
        "openrouter:qwen/qwen3-235b-a22b",
        "google:gemini-2.5-flash-lite",
        "google:gemini-3-flash-preview",
        "google:gemini-3.1-pro-preview",
        "anthropic:claude-haiku-4-5",
        "anthropic:claude-sonnet-5-5",
        "anthropic:claude-opus-4-7",
        "openai:gpt-5.4-mini",
        "openai:gpt-5.5",
        "openai:gpt-5.5-pro",
    ]
    assert [(s.provider, s.model_count) for s in catalog.status()] == [
        ("google", 6), ("anthropic", 7), ("openai", 6), ("openrouter", 4),
    ]
    with pytest.raises(ModelResolutionError) as caught:
        catalog.record(ModelRef("anthropic", "claude-sonnet-9-9"))
    assert str(caught.value).endswith(
        "Available anthropic models: claude-sonnet-5-5, claude-opus-4-7, claude-sonnet-4-6, "
        "claude-sonnet-4-5, claude-haiku-4-5, claude-3-5-haiku-20241022, claude-3-haiku-20240307"
    )


async def test_search_finds_an_id_its_provider_serves_but_does_not_show():
    catalog = await loaded(google=KEYS["google"], anthropic=KEYS["anthropic"])

    dated = await catalog.search("sonnet-4-5-2025")
    alias = await catalog.search("flash-latest")

    assert [str(record.ref) for record in dated] == ["anthropic:claude-sonnet-4-5-20250929"]
    assert [str(record.ref) for record in alias] == ["google:gemini-flash-latest"]


#: The date-stripped ids of the recorded listing's 3.x models: Anthropic's alias for 3.5
#: Haiku is ``claude-3-5-haiku-latest`` and 3 Haiku has none, so a request for these fails.
NOT_SERVED = (
    ModelRef("anthropic", "claude-3-5-haiku"),
    ModelRef("anthropic", "claude-3-haiku"),
)


async def test_an_id_the_provider_does_not_serve_is_not_accepted_shown_or_found():
    catalog = await loaded(anthropic=KEYS["anthropic"])

    for ref in NOT_SERVED:
        with pytest.raises(ModelResolutionError, match=f"{ref} is not available"):
            catalog.record(ref)
    assert not {ref.model for ref in NOT_SERVED} & set(ids(catalog.records("anthropic")))
    found = await catalog.search("haiku")
    assert [str(record.ref) for record in found] == [
        "anthropic:claude-3-5-haiku-20241022",
        "anthropic:claude-3-haiku-20240307",
        "anthropic:claude-haiku-4-5",
        "anthropic:claude-haiku-4-5-20251001",
    ]


async def test_a_dated_id_of_a_family_with_no_alias_is_accepted_and_shown_as_it_is():
    catalog = await loaded(anthropic=KEYS["anthropic"])

    for model in ("claude-3-5-haiku-20241022", "claude-3-haiku-20240307"):
        assert catalog.record(ModelRef("anthropic", model)).model == model
        assert model in ids(catalog.records("anthropic"))


async def test_the_aliases_the_table_lists_are_accepted_and_shown():
    catalog = await loaded(anthropic=KEYS["anthropic"])

    for model in ("claude-sonnet-4-5", "claude-haiku-4-5"):
        assert catalog.record(ModelRef("anthropic", model)).model == model
        assert model in ids(catalog.records("anthropic"))


def anthropic_shaped(*model_ids: str) -> dict:
    """An Anthropic listing of ``model_ids``, each with the recorded Sonnet 4.5's capabilities."""
    shape = next(m for m in recorded("anthropic")["data"] if m["id"] == "claude-sonnet-4-5-20250929")
    data = [dict(shape, id=model_id, display_name=model_id) for model_id in model_ids]
    return {"data": data, "has_more": False, "first_id": model_ids[0], "last_id": model_ids[-1]}


#: What Anthropic serves, of the ids a listing of its dated 4.x models and two 3.x ones
#: suggests: its documented aliases (``claude-opus-4-0`` and ``claude-sonnet-4-0`` for the
#: 20250514 models, the rest the date-stripped id), and not the date-stripped ids
#: ``claude-opus-4`` and ``claude-sonnet-4``, nor any 3.x one.
ALIASES_SERVED = (
    "claude-opus-4-0", "claude-sonnet-4-0", "claude-opus-4-1", "claude-opus-4-5",
    "claude-sonnet-4-5", "claude-haiku-4-5",
)
ALIASES_NOT_SERVED = (
    "claude-opus-4", "claude-sonnet-4", "claude-3-7-sonnet", "claude-3-7-sonnet-latest",
    "claude-3-5-haiku", "claude-3-5-haiku-latest", "claude-3-haiku",
)
DATED_4X = (
    "claude-opus-4-5-20251101", "claude-haiku-4-5-20251001", "claude-sonnet-4-5-20250929",
    "claude-opus-4-1-20250805", "claude-opus-4-20250514", "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022", "claude-3-haiku-20240307",
)


async def test_the_aliases_are_those_the_table_lists_not_every_date_stripped_id():
    """Anthropic's aliases for Opus 4 and Sonnet 4 end in -4-0: the date-stripped ids are
    not served, and the aliases that are must not be rejected as "not available"."""
    payload = anthropic_shaped("claude-opus-4-7", "claude-sonnet-4-6", *DATED_4X)
    catalog = await loaded(ProviderAPI(anthropic=payload), anthropic=KEYS["anthropic"])
    shown = ids(catalog.records("anthropic"))

    for model in ALIASES_SERVED:
        assert catalog.record(ModelRef("anthropic", model)).model == model
        assert model in shown
    for model in ALIASES_NOT_SERVED:
        with pytest.raises(ModelResolutionError, match=f"anthropic:{model} is not available"):
            catalog.record(ModelRef("anthropic", model))
        assert model not in shown
    for model in DATED_4X:
        assert catalog.record(ModelRef("anthropic", model)).model == model
    # A dated id whose alias is shown is not; the 3.x ones, with none, are shown as they are.
    assert shown == [
        "claude-opus-4-7", "claude-sonnet-4-6", "claude-opus-4-5", "claude-haiku-4-5",
        "claude-sonnet-4-5", "claude-opus-4-1", "claude-opus-4-0", "claude-sonnet-4-0",
        "claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022", "claude-3-haiku-20240307",
    ]


async def test_an_alias_that_is_not_its_date_stripped_id_is_described_by_its_family():
    payload = anthropic_shaped("claude-opus-4-20250514", "claude-sonnet-4-20250514")
    catalog = await loaded(ProviderAPI(anthropic=payload), anthropic=KEYS["anthropic"])

    opus = catalog.record(ModelRef("anthropic", "claude-opus-4-0"))
    sonnet = catalog.record(ModelRef("anthropic", "claude-sonnet-4-0"))

    assert (opus.tier, sonnet.tier) == ("strong", "default")
    assert (opus.thinking.mechanism, sonnet.thinking.mechanism) == (BUDGET, BUDGET)
    assert opus.context_window == 200_000 and opus.sources["context_window"] == "listing"


async def test_a_listed_dated_claude_id_is_described_as_its_alias_is():
    catalog = await loaded(anthropic=KEYS["anthropic"])

    for alias, dated in [
        ("claude-sonnet-4-5", "claude-sonnet-4-5-20250929"),
        ("claude-haiku-4-5", "claude-haiku-4-5-20251001"),
    ]:
        record = catalog.record(ModelRef("anthropic", dated))
        expected = catalog.record(ModelRef("anthropic", alias))
        assert record == dataclasses.replace(expected, model=dated)


async def test_a_dated_claude_id_is_offered_as_its_alias_described_by_the_first_listed():
    """As in 0.6, the alias takes the facts of the first id listed under it;
    each id the provider serves keeps its own."""
    data = recorded("anthropic")["data"]
    dated = next(m for m in data if m["id"] == "claude-sonnet-4-5-20250929")
    page = {"data": [
        dated,
        {**dated, "id": "claude-sonnet-4-5", "display_name": "The alias"},
        {**dated, "id": "claude-sonnet-4-5@2025-09-29", "max_input_tokens": 100_000},
    ], "has_more": False}
    catalog = await loaded(ProviderAPI(anthropic=page), anthropic=KEYS["anthropic"])

    assert ids(catalog.records("anthropic")) == ["claude-sonnet-4-5"]
    alias = catalog.record(ModelRef("anthropic", "claude-sonnet-4-5"))
    vertex = catalog.record(ModelRef("anthropic", "claude-sonnet-4-5@2025-09-29"))
    assert (alias.display_name, alias.context_window) == ("Claude Sonnet 4.5", 200_000)
    assert (vertex.display_name, vertex.context_window) == ("Claude Sonnet 4.5", 100_000)


@pytest.mark.parametrize("listed", [True, False], ids=["listed", "not listed"])
async def test_a_dated_claude_id_has_its_alias_s_thinking_and_sampling(listed):
    """Claude Sonnet 5.5 cannot turn thinking off and takes no sampling
    parameters (its table entry), under a dated id too: with a listing that
    does not say, and with none."""
    dated = ModelRef("anthropic", "claude-sonnet-5-5-20261001")
    api = ProviderAPI(anthropic=claude_listing(dated.model, disabled=None))
    catalog = catalog_for(api, anthropic=KEYS["anthropic"])
    if listed:
        await catalog.load()
        assert catalog.is_authoritative("anthropic")

    record = catalog.record(dated)

    assert record.thinking.mechanism is ADAPTIVE
    assert record.thinking.can_disable is False
    assert record.sampling == "never"


# --- status and authority ------------------------------------------------------


class TestStatus:
    async def test_each_provider_in_use_and_its_state_after_load(self):
        clock = Clock(5000.0)
        catalog = await loaded(
            ProviderAPI(anthropic=500), specs={"local-vllm": VLLM, "lab": VLLM}, clock=clock,
            references=[ModelRef("openai", "gpt-5.5"), ModelRef("local-vllm", "Qwen/Qwen3-32B")],
            google=KEYS["google"], anthropic=KEYS["anthropic"],
        )

        status = {s.provider: s for s in catalog.status()}

        assert list(status) == ["google", "anthropic", "openai", "local-vllm"]
        assert status["google"] == ProviderStatus("google", True, ProviderState.FRESH, 5000.0, 6)
        assert status["anthropic"].state is ProviderState.DEGRADED
        assert status["anthropic"].error == "HTTP 500 Internal Server Error"
        assert status["anthropic"].model_count == 5  # the fallbacks
        assert status["openai"] == ProviderStatus(
            "openai", False, ProviderState.UNATTEMPTED, None, len(load_table().curated_openai())
        )
        assert status["local-vllm"].state is ProviderState.FRESH
        assert [catalog.is_authoritative(p) for p in status] == [True, False, False, True]

    async def test_before_load_nothing_is_authoritative(self):
        catalog = catalog_for(google=KEYS["google"])

        assert {s.state for s in catalog.status()} == {ProviderState.UNATTEMPTED}
        assert not catalog.is_authoritative("google")


# --- roles over the catalog (R16) ----------------------------------------------


class TestRoles:
    async def test_fallback_records_fill_the_roles_on_an_offline_first_run(self):
        catalog = await loaded(ProviderAPI(google=refuse), google=KEYS["google"])
        roles = RoleMap(default_model=None, model_roles={})
        keyed = providers(google=KEYS["google"])

        assert catalog.status()[0].state is ProviderState.DEGRADED
        resolved = {
            role: str(roles.resolve(role, catalog=catalog, providers=keyed))
            for role in ("default", "fast", "strong")
        }
        assert resolved == {
            "default": "google:gemini-3.8-flash",
            "fast": "google:gemini-3.5-flash-lite",
            "strong": "google:gemini-3.1-pro-preview",
        }

    async def test_the_list_s_entries_fill_the_roles_first(self):
        """Not the listing's newest: the list's own model of each tier, when it has one."""
        entries = {GEMINI: ModelEntry(), ModelRef("google", "gemini-2.5-pro"): ModelEntry()}
        catalog = await loaded(
            entries=entries, computed_roles=["default", "fast", "strong"], google=KEYS["google"]
        )
        roles = RoleMap(default_model=None, model_roles={})
        keyed = providers(google=KEYS["google"])

        def role(name: str) -> str:
            return str(roles.resolve(name, catalog=catalog, providers=keyed))

        assert role("default") == "google:gemini-2.5-flash"  # not gemini-3-flash-preview
        assert role("strong") == "google:gemini-2.5-pro"  # not gemini-3.1-pro-preview
        assert role("fast") == "google:gemini-2.5-flash-lite"  # none in the list: the listing's

    async def test_a_fresh_listing_fills_the_roles_with_its_newest(self):
        catalog = await loaded(anthropic=KEYS["anthropic"])
        roles = RoleMap(default_model=None, model_roles={})
        keyed = providers(anthropic=KEYS["anthropic"])

        def role(name: str) -> str:
            return str(roles.resolve(name, catalog=catalog, providers=keyed))

        assert role("default") == "anthropic:claude-sonnet-5-5"
        assert role("strong") == "anthropic:claude-opus-4-7"


# --- the providers in use ------------------------------------------------------------


class TestInUse:
    """Which providers the catalog lists: known before any listing."""

    def test_the_providers_of_the_entries_that_can_run(self):
        catalog = catalog_for(
            entries={QWEN: ModelEntry(), GEMINI: ModelEntry()},
            google=KEYS["google"], anthropic=KEYS["anthropic"], openrouter="r-test-key",
        )

        assert catalog.in_use == ("google", "openrouter")  # catalog order

    def test_an_entry_on_an_undefined_provider_or_without_its_key_puts_none_in_use(self):
        """No entry can run, so every built-in provider with a key is in use."""
        catalog = catalog_for(
            entries={ModelRef("together", "x"): ModelEntry(), QWEN: ModelEntry()},
            anthropic=KEYS["anthropic"], openrouter=None,
        )

        assert catalog.in_use == ("anthropic",)

    def test_the_providers_of_the_references_with_a_key_or_not(self):
        catalog = catalog_for(
            entries={GEMINI: ModelEntry()},
            references=[ModelRef("openai", "gpt-5.5"), QWEN, ModelRef("together", "x")],
            google=KEYS["google"], openrouter=None,
        )

        assert catalog.in_use == ("google", "openai", "openrouter")

    def test_without_an_entry_every_built_in_provider_with_a_key(self):
        catalog = catalog_for(
            specs={"local-vllm": VLLM}, google=KEYS["google"], openai=KEYS["openai"]
        )

        assert catalog.in_use == ("google", "openai")

    def test_a_computed_role_puts_the_built_in_providers_with_a_key_in_use(self):
        """An OpenRouter model has no tier, so it cannot fill the role."""
        catalog = catalog_for(
            entries={QWEN: ModelEntry()},
            computed_roles=["fast"],
            google=KEYS["google"], anthropic=KEYS["anthropic"], openrouter="r-test-key",
        )

        assert catalog.in_use == ("google", "anthropic", "openrouter")

    def test_a_computed_role_an_entry_can_fill_puts_no_built_in_provider_in_use(self):
        catalog = catalog_for(
            entries={ModelRef("anthropic", "claude-sonnet-4-6"): ModelEntry(), QWEN: ModelEntry()},
            computed_roles=["default"],
            google=KEYS["google"], anthropic=KEYS["anthropic"], openrouter="r-test-key",
        )

        assert catalog.in_use == ("anthropic", "openrouter")

    def test_one_computed_role_no_entry_can_fill_puts_them_all_in_use(self):
        catalog = catalog_for(
            entries={ModelRef("anthropic", "claude-sonnet-4-6"): ModelEntry()},
            computed_roles=["default", "fast"],
            google=KEYS["google"], anthropic=KEYS["anthropic"],
        )

        assert catalog.in_use == ("google", "anthropic")

    async def test_when_the_listings_leave_a_computed_role_no_entry_the_built_ins_are_listed(
        self,
    ):
        """Rule 4 again, once the listings have spoken: Google is listed for an
        entry nothing describes, and its listing lacks the entry of the default
        role's tier, so the role falls back to the built-ins, which are listed for
        it in a second round."""
        api = ProviderAPI()
        retired, gemma = ModelRef("google", "gemini-9-flash"), ModelRef("google", "gemma-3-27b-it")

        catalog = await loaded(
            api, entries={retired: ModelEntry(), gemma: ModelEntry()}, computed_roles=["default"],
            google=KEYS["google"], anthropic=KEYS["anthropic"],
        )

        assert api.calls == ["google", "anthropic"]
        assert catalog.in_use == ("google", "anthropic")
        roles = RoleMap(default_model=None, model_roles={})
        keyed = providers(google=KEYS["google"], anthropic=KEYS["anthropic"])
        default = roles.resolve("default", catalog=catalog, providers=keyed)
        assert default == ModelRef("google", "gemini-3-flash-preview")  # listed, not the table's

    async def test_load_lists_the_providers_in_use_only(self):
        api = ProviderAPI()

        catalog = await loaded(
            api, entries={QWEN: ModelEntry()}, specs={"local-vllm": VLLM},
            google=KEYS["google"], openrouter="r-test-key",
        )

        assert api.calls == ["openrouter"]
        assert [s.provider for s in catalog.status()] == ["openrouter"]
        # A provider not in use was never listed: it rejects no id.
        assert catalog.record(ModelRef("google", "gemini-9-ultra")).model == "gemini-9-ultra"

    async def test_when_the_listings_leave_no_entry_the_keyed_built_ins_are_listed(self):
        """Rule 3 again, once the listings have spoken: the one entry that could
        run, which nothing describes, is not in its provider's listing, so the list
        offers the tiers, from listed models."""
        api = ProviderAPI()

        catalog = await loaded(
            api, entries={ModelRef("google", "gemini-2.0-pro-exp"): ModelEntry()},
            google=KEYS["google"], anthropic=KEYS["anthropic"],
        )

        assert api.calls == ["google", "anthropic"]
        assert catalog.in_use == ("google", "anthropic")
        assert catalog.is_authoritative("google")
        assert catalog.offered() == [
            "google:gemini-2.5-flash-lite",  # the listing's, not the table's
            "google:gemini-3-flash-preview",
            "google:gemini-3.1-pro-preview",
            "anthropic:claude-haiku-4-5",
            "anthropic:claude-sonnet-5-5",
            "anthropic:claude-opus-4-7",
        ]


class TestUnlistedProvider:
    """``list_models: false``: a server with no /models route, or one whose
    listing leaves out models it serves."""

    QWEN3 = ModelRef("local", "qwen3-32b")

    @staticmethod
    def specs() -> dict[str, ProviderSpec]:
        return {"local": ProviderSpec(base_url="http://gpu-box:8000/v1", list_models=False)}

    async def test_it_is_in_use_but_never_listed_and_takes_every_id(self):
        api = ProviderAPI()
        entries = {self.QWEN3: ModelEntry(context_window=32768)}

        catalog = await loaded(api, specs=self.specs(), entries=entries)

        assert catalog.in_use == ("local",)
        assert api.calls == []
        assert catalog.status() == [
            ProviderStatus("local", True, ProviderState.UNATTEMPTED, None, 0)
        ]
        record = catalog.record(self.QWEN3)
        assert (record.context_window, record.sources["context_window"]) == (32768, "config")
        assert catalog.record(ModelRef("local", "anything")).tools is True
        assert catalog.offered() == ["local:qwen3-32b"]

    async def test_refresh_and_search_list_it_neither(self):
        api = ProviderAPI()
        catalog = await loaded(
            api, specs=self.specs(), entries={self.QWEN3: ModelEntry()}
        )

        (status,) = await catalog.refresh()

        assert status.state is ProviderState.UNATTEMPTED
        assert await catalog.search("qwen") == []
        assert catalog.start_background_refresh() is None
        assert api.calls == []


# --- the list the pickers offer ---------------------------------------------------------


class TestOffered:
    async def test_the_entries_then_the_references_then_extra_each_once(self):
        catalog = await loaded(
            entries={
                QWEN: ModelEntry(),
                ModelRef("anthropic", "claude-sonnet-4-6"): ModelEntry(),
                GEMINI: ModelEntry(),
            },
            references=[ModelRef("google", "gemini-2.5-pro"), QWEN],
            google=KEYS["google"], anthropic=KEYS["anthropic"], openrouter="r-test-key",
        )

        offered = catalog.offered(
            extra=["anthropic:claude-opus-4-7", "google:gemini-2.5-pro", "fast", "anthropic:"]
        )

        # Not a provider's whole listing, nor the tiers' fallback: an entry is offered.
        assert offered == [
            "openrouter:qwen/qwen3-235b-a22b",
            "anthropic:claude-sonnet-4-6",
            "google:gemini-2.5-flash",
            "google:gemini-2.5-pro",
            "anthropic:claude-opus-4-7",
        ]

    async def test_with_no_entry_each_keyed_built_in_s_newest_fast_default_and_strong(self):
        catalog = await loaded(
            google=KEYS["google"], anthropic=KEYS["anthropic"], openai=KEYS["openai"]
        )

        assert catalog.offered() == [
            "google:gemini-2.5-flash-lite",
            "google:gemini-3-flash-preview",
            "google:gemini-3.1-pro-preview",
            "anthropic:claude-haiku-4-5",
            "anthropic:claude-sonnet-5-5",
            "anthropic:claude-opus-4-7",
            "openai:gpt-5.4-mini",
            "openai:gpt-5.5",
            "openai:gpt-5.5-pro",
        ]

    def test_before_any_listing_the_tiers_come_from_the_table_s_fallback_models(self):
        catalog = catalog_for(
            google=KEYS["google"], anthropic=KEYS["anthropic"], openai=KEYS["openai"]
        )

        assert catalog.offered() == [
            "google:gemini-3.5-flash-lite",
            "google:gemini-3.8-flash",
            "google:gemini-3.1-pro-preview",
            "anthropic:claude-haiku-5-5",
            "anthropic:claude-sonnet-5-5",
            "anthropic:claude-opus-5-5",
            "openai:gpt-6-luna",
            "openai:gpt-6.1-terra",
            "openai:gpt-6.1-sol",
        ]

    async def test_the_tiers_come_when_no_entry_can_run_not_only_when_there_is_none(self):
        catalog = await loaded(
            entries={QWEN: ModelEntry()},  # OpenRouter has no key
            references=[ModelRef("google", "gemini-2.5-pro")],
            google=KEYS["google"], openrouter=None,
        )

        assert catalog.offered() == [
            "google:gemini-2.5-pro",
            "google:gemini-2.5-flash-lite",
            "google:gemini-3-flash-preview",
            "google:gemini-3.1-pro-preview",
        ]

    async def test_a_reference_is_offered_only_when_it_can_run(self):
        catalog = await checked(
            entries={GEMINI: ModelEntry()},
            references=[
                ModelRef("anthropic", "claude-sonnet-4-6"),  # no key
                ModelRef("together", "x"),  # no such provider
                ModelRef("google", "gemini-9-ultra"),  # the fresh listing lacks it
                ModelRef("google", "gemini-2.5-pro"),
            ],
            google=KEYS["google"],
        )

        assert catalog.offered() == ["google:gemini-2.5-flash", "google:gemini-2.5-pro"]

    async def test_an_entry_its_fresh_listing_lacks_is_not_offered_on_any_provider(self):
        catalog = await checked(
            specs={"local-vllm": VLLM},
            entries={
                ModelRef("local-vllm", "entry-only"): ModelEntry(),
                ModelRef("local-vllm", "Qwen/Qwen3-32B"): ModelEntry(),
                ModelRef("anthropic", "claude-sonnet-9-9"): ModelEntry(),
            },
            anthropic=KEYS["anthropic"],
        )

        assert catalog.offered() == ["local-vllm:Qwen/Qwen3-32B"]


class TestNotOffered:
    async def test_why_an_entry_is_not_offered(self):
        catalog = await checked(
            entries={
                ModelRef("together", "deepseek-ai/DeepSeek-V3.1"): ModelEntry(),
                QWEN: ModelEntry(),
                ModelRef("anthropic", "claude-sonnet-4-6"): ModelEntry(),
                ModelRef("google", "gemini-9-ultra"): ModelEntry(),
                GEMINI: ModelEntry(),
            },
            google=KEYS["google"], openrouter=None,
        )

        assert catalog.not_offered() == {
            "together:deepseek-ai/DeepSeek-V3.1": (
                "provider 'together' is not defined in model_providers"
            ),
            "openrouter:qwen/qwen3-235b-a22b": "no key: set OPENROUTER_API_KEY",
            "anthropic:claude-sonnet-4-6": "no key: set ANTHROPIC_API_KEY",
            "google:gemini-9-ultra": "not served by google",
        }
        assert catalog.offered() == ["google:gemini-2.5-flash"]

    async def test_why_a_model_a_setting_names_is_not_offered(self):
        """The references (``default_model``, a role, an agent, the manager's model) are
        left out of the list by the entries' rule, so ``not_offered`` says why for them
        too; one that is also an entry is one row."""
        unserved = ModelRef("google", "gemini-9-ultra")
        catalog = await checked(
            entries={
                ModelRef("anthropic", "claude-sonnet-4-6"): ModelEntry(),
                unserved: ModelEntry(),
                GEMINI: ModelEntry(),
            },
            references=[
                ModelRef("together", "deepseek-ai/DeepSeek-V3.1"), QWEN, unserved, GEMINI,
                ModelRef("anthropic", "claude-sonnet-4-6"),
            ],
            google=KEYS["google"], openrouter=None,
        )

        assert catalog.not_offered() == {
            "anthropic:claude-sonnet-4-6": "no key: set ANTHROPIC_API_KEY",
            "google:gemini-9-ultra": "not served by google",
            "together:deepseek-ai/DeepSeek-V3.1": (
                "provider 'together' is not defined in model_providers"
            ),
            "openrouter:qwen/qwen3-235b-a22b": "no key: set OPENROUTER_API_KEY",
        }
        assert catalog.offered() == ["google:gemini-2.5-flash"]

    async def test_a_model_a_setting_names_is_warned_of_once_and_runs(self, caplog):
        """A reference a fresh listing lacks is accepted (the settings decide what
        runs) and warned of once; one on a provider that is not defined is not
        (resolution names it, and the start fails on it)."""
        missing = ModelRef("google", "gemini-9-ultra")
        catalog = catalog_for(references=[missing, ModelRef("together", "x")], google=KEYS["google"])

        with caplog.at_level(logging.WARNING, logger="agentic_cli.models.catalog"):
            await catalog.load()
            await catalog.load()

        assert catalog.record(missing).model == missing.model
        warned = [r.getMessage() for r in caplog.records if r.name == "agentic_cli.models.catalog"]
        assert len(warned) == 1 and warned[0].startswith(
            "Model google:gemini-9-ultra (a setting) is not served by google"
        )

    async def test_the_offered_entries_records_in_list_order(self):
        catalog = await checked(
            entries={
                QWEN: ModelEntry(context_window=100_000),
                ModelRef("google", "gemini-9-ultra"): ModelEntry(),
                GEMINI: ModelEntry(max_output=1_000),
            },
            google=KEYS["google"], openrouter="r-test-key",
        )

        records = catalog.entry_records()

        assert [record.ref for record in records] == [QWEN, GEMINI]
        assert (records[0].context_window, records[1].max_output) == (100_000, 1_000)

    async def test_an_entry_that_cannot_be_offered_warns_once_unless_it_lacks_a_key(
        self, caplog
    ):
        catalog = catalog_for(
            entries={
                ModelRef("together", "deepseek-ai/DeepSeek-V3.1"): ModelEntry(),
                QWEN: ModelEntry(),  # no key: /models says so, no warning
                ModelRef("google", "gemini-9-ultra"): ModelEntry(),
                GEMINI: ModelEntry(),
            },
            google=KEYS["google"], openrouter=None,
        )

        with caplog.at_level(logging.WARNING, logger="agentic_cli.models.catalog"):
            await catalog.load()
            await catalog.start_background_refresh()  # Google's listing lacks gemini-9-ultra
            await catalog.load()  # once per catalog

        assert [r.getMessage() for r in caplog.records if r.name == "agentic_cli.models.catalog"] == [
            "Model together:deepseek-ai/DeepSeek-V3.1 in models is not offered: "
            "provider 'together' is not defined in model_providers",
            "Model google:gemini-9-ultra in models is not offered: not served by google. "
            "Remove it from models, or name a model google serves "
            "(`/models google:` lists what it serves).",
        ]

    async def test_a_warning_comes_once_per_process_not_per_catalog(self, caplog):
        """A manager builds a catalog at every reinitialize, and /models one
        while the workflow starts: the user was told already."""
        entries = {ModelRef("together", "deepseek-ai/DeepSeek-V3.1"): ModelEntry(), GEMINI: ModelEntry()}

        with caplog.at_level(logging.WARNING, logger="agentic_cli.models.catalog"):
            for _ in range(3):
                await checked(entries=entries, google=KEYS["google"])
            await checked(
                entries={ModelRef("google", "gemini-9-ultra"): ModelEntry()}, google=KEYS["google"]
            )

        warned = [r.getMessage() for r in caplog.records if r.name == "agentic_cli.models.catalog"]
        assert len(warned) == 2
        assert "together:" in warned[0] and "gemini-9-ultra" in warned[1]


class TestAModelItsProviderDoesNotServe:
    """The settings decide what runs: a model they name that its provider does not
    serve is used as they name it, left out of the list, and warned of once, with
    the setting that names it and what to do."""

    @staticmethod
    def warnings(caplog) -> list[str]:
        return [r.getMessage() for r in caplog.records if r.name == "agentic_cli.models.catalog"]

    async def test_the_background_refresh_warns_of_a_reference(self, caplog):
        retired = ModelRef("anthropic", "claude-sonnet-3-7")
        catalog = catalog_for(
            entries={ModelRef("anthropic", "claude-sonnet-4-6"): ModelEntry()},
            references=[retired], named_by={retired: "default_model"},
            anthropic=KEYS["anthropic"],
        )

        with caplog.at_level(logging.WARNING, logger="agentic_cli.models.catalog"):
            await catalog.load()
            assert self.warnings(caplog) == []  # nothing listed yet, nothing to say
            await catalog.start_background_refresh()

        assert self.warnings(caplog) == [
            "Model anthropic:claude-sonnet-3-7 (default_model) is not served by anthropic: "
            "its listing lacks it, so a turn on it will fail at the provider. Choose a model "
            "anthropic serves for default_model (`/models anthropic:` lists what it serves)."
        ]
        assert catalog.not_offered() == {str(retired): "not served by anthropic"}
        assert catalog.record(retired).model == retired.model

    async def test_a_listing_the_start_needed_anyway_warns_at_load(self, caplog):
        """No entry, so Google is listed at the start for the list's tiers: its
        listing lacks the agent's model, which no longer stops the start."""
        missing = ModelRef("google", "gemini-9-ultra")
        catalog = catalog_for(
            references=[missing], named_by={missing: "agent 'writer'"}, google=KEYS["google"]
        )

        with caplog.at_level(logging.WARNING, logger="agentic_cli.models.catalog"):
            await catalog.load()

        assert self.warnings(caplog) == [
            "Model google:gemini-9-ultra (agent 'writer') is not served by google: its listing "
            "lacks it, so a turn on it will fail at the provider. Choose a model google serves "
            "for agent 'writer' (`/models google:` lists what it serves)."
        ]

    async def test_a_model_nothing_describes_runs_on_the_conservative_record(self, caplog):
        """Nothing describes it and its provider's listing lacks it, so the app cannot
        know its limits: it runs as the settings name it, on the conservative record,
        and the warning says so."""
        typo = ModelRef("openrouter", "qwen/qwen3-235b-a2")
        catalog = catalog_for(
            references=[typo], named_by={typo: "default_model"}, openrouter="r-test-key"
        )

        with caplog.at_level(logging.WARNING, logger="agentic_cli.models.catalog"):
            await catalog.load()

        record = catalog.record(typo)
        assert (record.thinking, record.context_window, record.tools) == (ThinkingRule(), None, True)
        assert set(record.sources.values()) == {"default"}
        (warning,) = self.warnings(caplog)
        assert warning.endswith(
            "(`/models openrouter:` lists what it serves). Nothing describes it either: it runs "
            "without thinking and with a 128,000-token window (the floor for an unknown one), "
            "unless a models entry says otherwise."
        )

    async def test_a_model_an_entry_describes_says_nothing_of_its_limits(self, caplog):
        typo = ModelRef("openrouter", "qwen/qwen3-235b-a2")
        catalog = catalog_for(
            entries={typo: ModelEntry(context_window=32768, thinking="effort")},
            references=[typo], named_by={typo: "default_model"}, openrouter="r-test-key",
        )

        with caplog.at_level(logging.WARNING, logger="agentic_cli.models.catalog"):
            await checked(catalog=catalog)

        (warning,) = self.warnings(caplog)
        assert warning.startswith("Model openrouter:qwen/qwen3-235b-a2 (default_model) is not")
        assert "Nothing" not in warning
        assert catalog.record(typo).context_window == 32768

    async def test_a_built_in_model_nothing_describes_is_told_what_an_entry_can_set(
        self, caplog
    ):
        """No family rule matches a Gemini 2.0 id: its thinking comes only from the
        listing (an entry on a built-in provider may not set it), its window from an
        entry too."""
        typo = ModelRef("google", "gemini-2.0-flash-typo")
        catalog = catalog_for(
            references=[typo], named_by={typo: "default_model"}, google=KEYS["google"]
        )

        with caplog.at_level(logging.WARNING, logger="agentic_cli.models.catalog"):
            await catalog.load()

        (warning,) = self.warnings(caplog)
        assert warning.endswith(
            "Nothing describes it either: it runs without thinking and with a 128,000-token "
            "window (the floor for an unknown one), unless a models entry sets its "
            "context_window."
        )

    async def test_the_warning_names_only_what_nothing_gives(self, caplog):
        """The entry gives the window, not the thinking: the warning says the model
        runs without thinking, and nothing of the window."""
        typo = ModelRef("openrouter", "qwen/qwen3-235b-a2")
        catalog = catalog_for(
            entries={typo: ModelEntry(context_window=32768)}, references=[typo],
            named_by={typo: "default_model"}, openrouter="r-test-key",
        )

        with caplog.at_level(logging.WARNING, logger="agentic_cli.models.catalog"):
            await catalog.load()

        (warning,) = self.warnings(caplog)
        assert warning.endswith(
            "Nothing gives its thinking either: it runs without thinking, unless its models "
            "entry sets thinking."
        )

    async def test_once_per_process(self, caplog):
        missing = ModelRef("google", "gemini-9-ultra")

        with caplog.at_level(logging.WARNING, logger="agentic_cli.models.catalog"):
            for _ in range(2):
                await loaded(references=[missing], google=KEYS["google"])

        assert len(self.warnings(caplog)) == 1


# --- search: a provider's full listing ----------------------------------------------------


class TestSearch:
    async def test_a_provider_not_in_use_is_listed_the_first_time_it_is_searched(self):
        api = ProviderAPI()
        catalog = await checked(
            api, entries={GEMINI: ModelEntry()},
            google=KEYS["google"], anthropic=KEYS["anthropic"], openrouter="r-test-key",
        )
        assert api.calls == ["google"]

        found = await catalog.search("SONNET")

        assert [str(record.ref) for record in found] == [
            "anthropic:claude-sonnet-4-5",
            "anthropic:claude-sonnet-4-5-20250929",
            "anthropic:claude-sonnet-4-6",
            "anthropic:claude-sonnet-5-5",
            "openrouter:anthropic/claude-sonnet-4.5",
        ]
        assert sorted(api.calls) == ["anthropic", "google", "openrouter"]
        await catalog.search("qwen")
        assert len(api.calls) == 3  # listed once, then searched
        assert [s.provider for s in catalog.status()] == ["google", "anthropic", "openrouter"]
        api.calls.clear()
        await catalog.refresh()
        assert sorted(api.calls) == ["anthropic", "google", "openrouter"]

    async def test_it_finds_ids_the_list_and_the_records_leave_out(self):
        """An OpenAI id no rule knows is never offered, but it can be found."""
        catalog = await loaded(openai=KEYS["openai"])

        found = await catalog.search("embedding")

        assert [str(record.ref) for record in found] == ["openai:text-embedding-3-small"]

    async def test_a_provider_without_a_key_is_neither_listed_nor_searched(self):
        """Not even the table's models of a built-in without its key."""
        api = ProviderAPI()
        catalog = await loaded(api, google=KEYS["google"], openrouter=None)
        api.calls.clear()

        assert await catalog.search("qwen") == []
        assert await catalog.search("claude") == []
        assert api.calls == []
        assert [s.provider for s in catalog.status()] == ["google"]

    async def test_a_provider_whose_listing_failed_is_listed_again(self):
        api = ProviderAPI(google=refuse)
        catalog = await loaded(api, google=KEYS["google"])
        api.answers["google"] = recorded("google")

        found = await catalog.search("2.0-flash")

        assert [str(record.ref) for record in found] == [
            "google:gemini-2.0-flash",
            "google:gemini-2.0-flash-001",
            "google:gemini-2.0-flash-exp-image-generation",
        ]
        assert catalog.is_authoritative("google")


# --- load: the first-run wait (R10, Review Focus 2) --------------------------------


class TestLoad:
    async def test_offline_startup_does_not_hang(self):
        """Review Focus 2: no network means a quick, degraded start that
        accepts any id; a hung network costs one timeout, not one per provider."""
        refused = await loaded(
            ProviderAPI(google=refuse, anthropic=refuse),
            google=KEYS["google"], anthropic=KEYS["anthropic"],
        )
        assert {s.state for s in refused.status() if s.has_credentials} == {ProviderState.DEGRADED}
        assert refused.record(ModelRef("google", "gemini-9-ultra")).model == "gemini-9-ultra"

        hung = ModelCatalog(
            providers=providers(google=KEYS["google"], anthropic=KEYS["anthropic"]),
            transport=ProviderAPI(google=hang, anthropic=hang).transport,
            timeout=0.3,
        )
        started = time.monotonic()
        await hung.load()
        assert time.monotonic() - started < 1.5
        assert [s.error for s in hung.status() if s.has_credentials] == ["timed out after 0.3s"] * 2
        assert hung.record(ModelRef("anthropic", "claude-unknown-9")).model == "claude-unknown-9"

    async def test_only_providers_in_use_with_credentials_are_listed(self):
        api = ProviderAPI()

        await loaded(
            api, specs={"local-vllm": VLLM, "lab": VLLM},
            references=[ModelRef("local-vllm", "Qwen/Qwen3-32B"), ModelRef("openai", "gpt-5.5")],
            google=KEYS["google"], anthropic=KEYS["anthropic"],
        )

        # Not lab (not in use), nor OpenAI (in use, but no key).
        assert sorted(api.calls) == ["anthropic", "endpoint", "google"]

    async def test_load_never_raises_for_a_broken_response(self):
        catalog = await loaded(ProviderAPI(google={"models": "nope"}), google=KEYS["google"])

        assert catalog.status()[0].state is ProviderState.DEGRADED


# --- background refresh, refresh, aclose ---------------------------------------------


class TestRefresh:
    async def test_refresh_lists_again_and_swaps_in(self):
        api = ProviderAPI(google=503)
        catalog = await loaded(api, google=KEYS["google"])
        assert catalog.status()[0].state is ProviderState.DEGRADED

        api.answers["google"] = recorded("google")
        statuses = await catalog.refresh()

        # The providers in use: Anthropic and OpenAI have no key.
        assert [(s.provider, s.state) for s in statuses] == [("google", ProviderState.FRESH)]
        assert catalog.is_authoritative("google")

    async def test_refresh_of_named_providers(self):
        api = ProviderAPI()
        catalog = await loaded(api, google=KEYS["google"], anthropic=KEYS["anthropic"])
        api.calls.clear()

        statuses = await catalog.refresh(["anthropic"])

        assert api.calls == ["anthropic"]
        assert [s.provider for s in statuses] == ["anthropic"]

    async def test_a_failed_refresh_keeps_the_listing_and_marks_the_provider_degraded(self):
        api = ProviderAPI()
        catalog = await loaded(api, google=KEYS["google"])

        api.answers["google"] = 500
        (status,) = await catalog.refresh(["google"])

        assert status.state is ProviderState.DEGRADED
        assert status.error == "HTTP 500 Internal Server Error"
        assert not catalog.is_authoritative("google")
        assert "gemini-2.0-flash" in ids(catalog.records("google"))  # the listing kept

    async def test_nothing_due_means_no_background_task(self):
        catalog = await loaded(google=KEYS["google"])

        assert catalog.start_background_refresh() is None

    async def test_each_provider_swaps_in_as_its_listing_arrives(self):
        release = asyncio.Event()

        async def slow_anthropic(request):
            await release.wait()
            return recorded("anthropic")

        api = ProviderAPI(anthropic=slow_anthropic)
        catalog = catalog_for(api, google=KEYS["google"], anthropic=KEYS["anthropic"])
        refreshing = asyncio.create_task(catalog.refresh())
        for _ in range(100):
            if catalog.is_authoritative("google"):
                break
            await asyncio.sleep(0.01)

        assert catalog.is_authoritative("google")
        assert not catalog.is_authoritative("anthropic")
        release.set()
        await refreshing
        assert catalog.is_authoritative("anthropic")

    async def test_aclose_cancels_the_background_refresh(self, tmp_path):
        clock, cache = Clock(), tmp_path / "model_cache.json"
        keyed = providers(google=KEYS["google"])
        await ModelCatalog(providers=keyed, cache_path=cache, transport=ProviderAPI().transport,
                           clock=clock).load()
        clock.now += 2 * 86400  # the cached listing is now stale
        catalog = ModelCatalog(providers=keyed, cache_path=cache,
                               transport=ProviderAPI(google=hang).transport, clock=clock)
        await catalog.load()
        task = catalog.start_background_refresh()
        assert task is not None
        await asyncio.sleep(0.01)

        await catalog.aclose()
        await catalog.aclose()  # idempotent

        assert task.done()
        assert catalog.start_background_refresh() is None


# --- a listing that failed at startup is tried once more --------------------------


def fails_first(payload):
    """An answer that fails the first request (a network blip) and serves ``payload`` after."""
    asked: list[httpx.Request] = []

    def answer(request):
        asked.append(request)
        return refuse(request) if len(asked) == 1 else payload

    return answer


class TestAFailedStartupListing:
    """Without a listing an OpenRouter model has no thinking knob and no window
    (the shipped table's conservative record): the background refresh lists the
    provider once more, so a blip at startup does not last the session."""

    async def test_the_background_refresh_lists_it_again_and_the_record_gets_the_listing_s_facts(
        self,
    ):
        api = ProviderAPI(openrouter=fails_first(recorded("openrouter")))
        catalog = await loaded(api, references=ON_OPENROUTER, openrouter="r-test-key")
        assert catalog.status()[0].state is ProviderState.DEGRADED
        before = catalog.record(QWEN)
        assert (before.thinking.mechanism, before.context_window) == (ThinkingMechanism.NONE, None)

        task = catalog.start_background_refresh()
        assert task is not None
        await task

        assert api.calls == ["openrouter", "openrouter"]
        assert catalog.status()[0].state is ProviderState.FRESH
        after = catalog.record(QWEN)
        assert (after.thinking.mechanism, after.context_window) == (
            ThinkingMechanism.EFFORT, 131072,
        )

    async def test_it_is_tried_once_more_and_no_more(self, tmp_path):
        cache = tmp_path / "model_cache.json"
        api = ProviderAPI(openrouter=500)
        keyed = providers(openrouter="r-test-key")
        catalog = ModelCatalog(providers=keyed, references=ON_OPENROUTER, cache_path=cache,
                               transport=api.transport, clock=Clock())
        await catalog.load()

        task = catalog.start_background_refresh()
        assert task is not None
        await task

        assert api.calls == ["openrouter", "openrouter"]
        assert catalog.status()[0].state is ProviderState.DEGRADED
        assert catalog.start_background_refresh() is None  # no loop
        # The failure is cached, so the next start tries again, as before.
        again = ModelCatalog(providers=keyed, references=ON_OPENROUTER, cache_path=cache,
                             transport=ProviderAPI(openrouter=500).transport, clock=Clock())
        await again.load()
        assert again.start_background_refresh() is not None
        await again.aclose()

    async def test_a_refresh_or_search_that_fails_is_not_tried_again_by_itself(self):
        api = ProviderAPI()
        catalog = await loaded(api, google=KEYS["google"])
        api.answers["google"] = 500
        await catalog.refresh()

        assert catalog.start_background_refresh() is None

    async def test_the_warning_says_what_it_means_for_a_built_in_provider_s_models(self, caplog):
        with caplog.at_level(logging.WARNING, logger="agentic_cli.models.catalog"):
            await loaded(ProviderAPI(google=refuse), google=KEYS["google"])

        (warning,) = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert warning.startswith("Could not list google models (")
        assert "ids are not checked" in warning
        assert "limits and thinking come from the shipped table" in warning
        # The agents are resolved once, at startup; a later listing reaches the rest.
        assert "this session's agents" in warning and "next start" in warning

    async def test_the_warning_says_what_it_means_for_an_openrouter_provider_s_models(self, caplog):
        """The table knows no OpenRouter model: without its listing a model has no thinking
        and the unknown-window floor, for this session's agents."""
        with caplog.at_level(logging.WARNING, logger="agentic_cli.models.catalog"):
            await loaded(
                ProviderAPI(openrouter=fails_first(recorded("openrouter"))),
                references=ON_OPENROUTER, openrouter="r-test-key",
            )

        (warning,) = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert warning.startswith("Could not list openrouter models (")
        assert "ids are not checked" in warning
        assert "without thinking" in warning and "128,000-token window" in warning
        assert "shipped table" not in warning
        assert "this session's agents" in warning and "next start" in warning
