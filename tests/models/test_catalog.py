"""The catalog: records merged from listings, the table and declarations; validation; refresh.

Listings come from ``ProviderAPI``, a fake of every provider's model list
behind ``httpx.MockTransport``. The on-disk cache is covered in
``test_catalog_cache.py``.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from agentic_cli.models import (
    DeclaredModel,
    EndpointSpec,
    ModelCatalog,
    ModelRef,
    ModelResolutionError,
    NativeThinking,
    ProviderKind,
    ProviderState,
    ProviderStatus,
    RoleMap,
    ThinkingMechanism,
    ThinkingRule,
    load_table,
    resolve_model,
)
from tests.models.fakes import Clock, ProviderAPI, endpoints, hang, recorded, refuse

KEYS = {"google": "g-test-key", "anthropic": "a-test-key", "openai": "o-test-key"}
VLLM = EndpointSpec(
    base_url="http://gpu-box:8000/v1",
    models={
        "Qwen/Qwen3-32B": DeclaredModel(context_window=16384, thinking="chat_template"),
        "declared-only": DeclaredModel(thinking="effort", effort_levels=["low", "high"]),
    },
)


def catalog_for(api: ProviderAPI | None = None, *, specs=None, clock=None, **keys) -> ModelCatalog:
    return ModelCatalog(
        endpoints=endpoints(specs=specs or {}, **keys),
        transport=(api or ProviderAPI()).transport,
        clock=clock or Clock(),
    )


async def loaded(api: ProviderAPI | None = None, **kwargs) -> ModelCatalog:
    catalog = catalog_for(api, **kwargs)
    await catalog.load()
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
        assert thinking("claude-3-5-haiku") == ThinkingRule()
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
            endpoints=endpoints(anthropic=KEYS["anthropic"]),
        )

        assert resolved.thinking == expected

    async def test_without_capabilities_the_version_decides(self):
        catalog = await loaded(anthropic=KEYS["anthropic"])

        record = catalog.record(ModelRef("anthropic", "claude-3-haiku"))

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
        catalog = await loaded(openai=KEYS["openai"])

        assert ids(catalog.records("openai")) == [
            "gpt-5.5", "gpt-5.5-pro", "gpt-5.4-mini", "o3", "gpt-4.1",
            "gpt-5.5-2026-03-05", "gpt-5-codex",
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
        catalog = await loaded(openrouter="r-test-key")

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
        catalog = await loaded(ProviderAPI(openrouter=503), openrouter="r-test-key")

        assert catalog.records("openrouter") == []


class TestEndpoints:
    async def test_declared_and_listed_models(self):
        catalog = await loaded(specs={"local-vllm": VLLM})

        def record(model: str):
            return catalog.record(ModelRef("local-vllm", model))

        qwen = record("Qwen/Qwen3-32B")
        assert qwen.context_window == 16384  # declared beats the listing's 32768
        assert qwen.thinking.mechanism is ThinkingMechanism.CHAT_TEMPLATE
        assert qwen.sources["context_window"] == "declared"
        assert qwen.kind is ProviderKind.OPENAI_COMPATIBLE
        lora = record("qwen3-sql-lora")  # listed, not declared
        assert (lora.tools, lora.context_window) == (True, 32768)
        assert lora.sources["context_window"] == "listing"
        effort = record("declared-only")  # declared, not served
        assert effort.thinking == ThinkingRule(ThinkingMechanism.EFFORT, levels=("low", "high"))
        assert set(ids(catalog.records("local-vllm"))) == {
            "Qwen/Qwen3-32B", "declared-only", "qwen3-sql-lora",
        }

    async def test_a_declaration_without_a_number_takes_the_listing(self):
        spec = EndpointSpec(
            base_url="http://gpu-box:8000/v1", models={"Qwen/Qwen3-32B": DeclaredModel()}
        )
        catalog = await loaded(specs={"local-vllm": spec})

        record = catalog.record(ModelRef("local-vllm", "Qwen/Qwen3-32B"))

        assert record.context_window == 32768
        assert record.sources["context_window"] == "listing"
        assert record.sources["tools"] == "declared"

    async def test_a_listing_is_authoritative_for_undeclared_ids_only(self):
        catalog = await loaded(specs={"local-vllm": VLLM})

        with pytest.raises(ModelResolutionError, match="local-vllm:other is not available"):
            catalog.record(ModelRef("local-vllm", "other"))
        assert catalog.record(ModelRef("local-vllm", "declared-only")).model == "declared-only"

    async def test_a_server_that_is_down_still_has_its_declared_models(self):
        catalog = await loaded(ProviderAPI(endpoint=refuse), specs={"local-vllm": VLLM})

        assert set(ids(catalog.records("local-vllm"))) == {"Qwen/Qwen3-32B", "declared-only"}
        assert catalog.record(ModelRef("local-vllm", "anything")).tools is True


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
        catalog = await loaded(ProviderAPI(openrouter=payload), openrouter="r-test-key")

        with pytest.raises(ModelResolutionError, match=r"and 40 more"):
            catalog.record(ModelRef("openrouter", "vendor/unknown"))

    async def test_an_undefined_endpoint_is_an_error(self):
        catalog = await loaded()

        with pytest.raises(ModelResolutionError, match="nowhere"):
            catalog.record(ModelRef("nowhere", "model"))


# --- status and authority ------------------------------------------------------


class TestStatus:
    async def test_each_provider_s_state_after_load(self):
        clock = Clock(5000.0)
        catalog = await loaded(
            ProviderAPI(anthropic=500), specs={"local-vllm": VLLM}, clock=clock,
            google=KEYS["google"], anthropic=KEYS["anthropic"],
        )

        status = {s.provider: s for s in catalog.status()}

        assert list(status) == ["google", "anthropic", "openai", "openrouter", "local-vllm"]
        assert status["google"] == ProviderStatus("google", True, ProviderState.FRESH, 5000.0, 6)
        assert status["anthropic"].state is ProviderState.DEGRADED
        assert status["anthropic"].error == "HTTP 500 Internal Server Error"
        assert status["anthropic"].model_count == 5  # the fallbacks
        assert status["openai"] == ProviderStatus(
            "openai", False, ProviderState.UNATTEMPTED, None, len(load_table().curated_openai())
        )
        assert status["local-vllm"].state is ProviderState.FRESH
        assert [catalog.is_authoritative(p) for p in status] == [True, False, False, False, True]

    async def test_before_load_nothing_is_authoritative(self):
        catalog = catalog_for(google=KEYS["google"])

        assert {s.state for s in catalog.status()} == {ProviderState.UNATTEMPTED}
        assert not catalog.is_authoritative("google")


# --- roles over the catalog (R16) ----------------------------------------------


class TestRoles:
    async def test_fallback_records_fill_the_roles_on_an_offline_first_run(self):
        catalog = await loaded(ProviderAPI(google=refuse), google=KEYS["google"])
        roles = RoleMap(default_model=None, model_roles={})
        eps = endpoints(google=KEYS["google"])

        assert catalog.status()[0].state is ProviderState.DEGRADED
        resolved = {
            role: str(roles.resolve(role, catalog=catalog, endpoints=eps))
            for role in ("default", "fast", "strong")
        }
        assert resolved == {
            "default": "google:gemini-2.5-flash",
            "fast": "google:gemini-3.1-flash-lite-preview",
            "strong": "google:gemini-3.1-pro-preview",
        }

    async def test_a_fresh_listing_fills_the_roles_with_its_newest(self):
        catalog = await loaded(anthropic=KEYS["anthropic"])
        roles = RoleMap(default_model=None, model_roles={})
        eps = endpoints(anthropic=KEYS["anthropic"])

        def role(name: str) -> str:
            return str(roles.resolve(name, catalog=catalog, endpoints=eps))

        assert role("default") == "anthropic:claude-sonnet-5-5"
        assert role("strong") == "anthropic:claude-opus-4-7"


# --- the picker ------------------------------------------------------------------


class TestPickerRefs:
    async def test_the_runnable_models_by_provider_tier_and_newest(self):
        catalog = await loaded(
            specs={"local-vllm": VLLM},
            google=KEYS["google"], anthropic=KEYS["anthropic"], openrouter="r-test-key",
        )

        refs = catalog.picker_refs(
            referenced=[
                "openrouter:qwen/qwen3-235b-a22b", "fast", "google:gemini-2.5-flash", "anthropic:",
            ],
            pinned=["openrouter:anthropic/claude-sonnet-4.5"],
        )

        assert refs == [
            "google:gemini-2.5-flash-lite",
            "google:gemini-3-flash-preview",
            "google:gemini-2.5-flash",
            "google:gemini-3.1-pro-preview",
            "google:gemini-2.5-pro",
            "google:gemini-2.0-flash",
            "anthropic:claude-haiku-4-5",
            "anthropic:claude-sonnet-5-5",
            "anthropic:claude-sonnet-4-6",
            "anthropic:claude-sonnet-4-5",
            "anthropic:claude-opus-4-7",
            "anthropic:claude-3-5-haiku",
            "anthropic:claude-3-haiku",
            # version_key reads "a22b" as 22: across vendors "newest" means little.
            "openrouter:qwen/qwen3-235b-a22b",
            "openrouter:anthropic/claude-sonnet-4.5",
            "local-vllm:Qwen/Qwen3-32B",
            "local-vllm:qwen3-sql-lora",
            "local-vllm:declared-only",
        ]

    async def test_openrouter_needs_a_reference_and_a_key(self):
        keyless = await loaded(google=KEYS["google"])
        keyed = await loaded(openrouter="r-test-key")

        assert keyless.picker_refs(pinned=["openrouter:qwen/qwen3-235b-a22b"]) == [
            r for r in keyless.picker_refs() if r.startswith("google:")
        ]
        assert keyed.picker_refs() == []

    async def test_a_provider_without_a_key_is_not_offered(self):
        catalog = await loaded(google=KEYS["google"])

        assert not any(r.startswith(("anthropic:", "openai:")) for r in catalog.picker_refs())

    async def test_a_reference_the_fresh_listing_lacks_is_not_offered(self):
        catalog = await loaded(google=KEYS["google"])

        refs = catalog.picker_refs(referenced=["google:gemini-9-ultra"])

        assert "google:gemini-9-ultra" not in refs


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
            endpoints=endpoints(google=KEYS["google"], anthropic=KEYS["anthropic"]),
            transport=ProviderAPI(google=hang, anthropic=hang).transport,
            timeout=0.3,
        )
        started = time.monotonic()
        await hung.load()
        assert time.monotonic() - started < 1.5
        assert [s.error for s in hung.status() if s.has_credentials] == ["timed out after 0.3s"] * 2
        assert hung.record(ModelRef("anthropic", "claude-unknown-9")).model == "claude-unknown-9"

    async def test_only_providers_with_credentials_are_listed(self):
        api = ProviderAPI()

        await loaded(api, google=KEYS["google"], specs={"local-vllm": VLLM})

        assert sorted(api.calls) == ["endpoint", "google"]

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

        assert [(s.provider, s.state) for s in statuses] == [
            ("google", ProviderState.FRESH),
            ("anthropic", ProviderState.UNATTEMPTED),
            ("openai", ProviderState.UNATTEMPTED),
            ("openrouter", ProviderState.UNATTEMPTED),
        ]
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
        eps = endpoints(google=KEYS["google"])
        await ModelCatalog(endpoints=eps, cache_path=cache, transport=ProviderAPI().transport,
                           clock=clock).load()
        clock.now += 2 * 86400  # the cached listing is now stale
        catalog = ModelCatalog(endpoints=eps, cache_path=cache,
                               transport=ProviderAPI(google=hang).transport, clock=clock)
        await catalog.load()
        task = catalog.start_background_refresh()
        assert task is not None
        await asyncio.sleep(0.01)

        await catalog.aclose()
        await catalog.aclose()  # idempotent

        assert task.done()
        assert catalog.start_background_refresh() is None
