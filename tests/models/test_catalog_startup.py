"""Startup: what ``ModelCatalog.load`` waits on.

The settings name the models, and a start uses them at once: ``load`` waits on
a listing only to fill a gap (``models.startup``), every one it needs at once,
and the background refresh lists each other provider in use. Each listing here
takes ``DELAY`` seconds, as in the review's probe of the start before this
(``p3_sequential_rounds.py``: four providers listed in four sequential rounds).
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time

from agentic_cli.models import (
    ModelCatalog,
    ModelEntry,
    ModelRef,
    ProviderState,
    RoleMap,
    ThinkingMechanism,
)
from tests.models.fakes import Clock, ProviderAPI, providers, recorded

DELAY = 0.3
KEYS = {"google": "g-test-key", "anthropic": "a-test-key", "openai": "o-test-key",
        "openrouter": "r-test-key"}
QWEN = ModelRef("openrouter", "qwen/qwen3-235b-a22b")
ALL = ["anthropic", "google", "openai", "openrouter"]


class Slow(ProviderAPI):
    """Each listing takes ``DELAY`` seconds; ``peak`` is how many were in flight at once."""

    def __init__(self, **answers) -> None:
        super().__init__(**answers)
        self.in_flight = 0
        self.peak = 0

    async def _handle(self, request):
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(DELAY)
            return await super()._handle(request)
        finally:
            self.in_flight -= 1


def make(api, *, cache=None, clock=None, entries=None, references=(), computed_roles=()):
    return ModelCatalog(
        providers=providers(**KEYS),
        entries=entries,
        references=references,
        computed_roles=computed_roles,
        cache_path=cache,
        transport=api.transport,
        clock=clock or Clock(),
    )


async def timed(catalog: ModelCatalog) -> float:
    start = time.monotonic()
    await catalog.load()
    return time.monotonic() - start


def states(catalog: ModelCatalog) -> dict[str, ProviderState]:
    return {status.provider: status.state for status in catalog.status()}


async def test_settings_that_name_described_models_wait_on_no_listing():
    """Every model is named and something describes each (a table rule, or a models
    entry that gives its window and thinking), and the default role is mapped (a
    computed one has no model of its own: see below): the start lists nothing, and
    the background refresh then lists each provider in use, whose listing reaches
    the records."""
    api = Slow()
    entries = {
        ModelRef("anthropic", "claude-sonnet-4-6"): ModelEntry(),
        QWEN: ModelEntry(context_window=131072, thinking="none"),
    }
    references = (ModelRef("openai", "gpt-5.5"), ModelRef("google", "gemini-2.5-flash"))
    catalog = make(api, entries=entries, references=references)

    elapsed = await timed(catalog)

    assert (api.peak, api.calls) == (0, [])
    assert elapsed < DELAY
    assert catalog.in_use == ("google", "anthropic", "openai", "openrouter")
    named = [*map(str, entries), *map(str, references)]
    assert [str(catalog.record(ref).ref) for ref in (*entries, *references)] == named
    assert catalog.offered() == named
    assert catalog.record(QWEN).max_output is None  # no listing yet

    await catalog.start_background_refresh()

    assert sorted(api.calls) == ALL
    assert set(states(catalog).values()) == {ProviderState.FRESH}
    assert catalog.offered() == named
    assert (catalog.record(QWEN).max_output, catalog.record(QWEN).sources["max_output"]) == (
        8192, "listing",
    )
    assert catalog.record(QWEN).thinking.mechanism is ThinkingMechanism.NONE  # the entry wins
    await catalog.aclose()


async def test_an_entry_that_leaves_out_the_window_or_the_thinking_describes_nothing():
    """A configured provider's model is described only by an entry that gives what a
    run needs, its window and its thinking (the table knows nothing of it): one that
    sets only ``tools`` waits on the listing, which gives the model's reasoning and
    window, rather than run the session without thinking on the window floor."""
    api = Slow()
    catalog = make(api, entries={QWEN: ModelEntry(tools=True)}, references=(QWEN,))

    await catalog.load()

    assert api.calls == ["openrouter"]
    record = catalog.record(QWEN)
    assert (record.thinking.mechanism, record.context_window) == (ThinkingMechanism.EFFORT, 131072)
    assert record.sources["tools"] == "config"


async def test_the_listings_a_start_needs_run_at_once():
    """No entry (the list's tiers come from Google, Anthropic and OpenAI) and an
    OpenRouter model nothing describes: four listings, in flight together, so the
    start takes one listing's time, not four's."""
    api = Slow()
    catalog = make(api, references=(QWEN,), computed_roles=("default",))

    elapsed = await timed(catalog)

    assert sorted(api.calls) == ALL
    assert api.peak == 4
    assert elapsed < 2 * DELAY
    assert catalog.start_background_refresh() is None  # nothing left to check


async def test_the_probe_s_start_waits_on_what_it_needs_at_once(tmp_path):
    """The probe's start: an OpenRouter entry nothing describes, an Anthropic entry
    and an OpenAI reference a listing cached an hour ago lacks (each described by a
    table rule), and a computed default role the Anthropic entry would fill. It took
    four rounds. Now the first lists OpenRouter, for its entry, and Anthropic, for the
    role's entry, at once; Anthropic does not serve that model, so a second round
    lists Google for the role. The background refresh confirms the OpenAI reference."""
    cache, clock = tmp_path / "model_cache.json", Clock()
    seed = ModelCatalog(
        providers=providers(anthropic=KEYS["anthropic"], openai=KEYS["openai"]),
        cache_path=cache, transport=ProviderAPI().transport, clock=clock,
    )
    await seed.load()
    clock.now += 3600
    api = Slow()
    sonnet, terra = ModelRef("anthropic", "claude-sonnet-5-6"), ModelRef("openai", "gpt-6.2-terra")
    catalog = make(
        api, cache=cache, clock=clock, entries={QWEN: ModelEntry(), sonnet: ModelEntry()},
        references=(terra,), computed_roles=("default",),
    )

    elapsed = await timed(catalog)

    assert sorted(api.calls[:2]) == ["anthropic", "openrouter"] and api.calls[2:] == ["google"]
    assert api.peak == 2
    assert elapsed < 3 * DELAY  # two rounds, not four
    roles = RoleMap(default_model=None, model_roles={})
    assert str(roles.resolve("default", catalog=catalog, providers=providers(**KEYS))) == (
        "google:gemini-3-flash-preview"
    )
    assert catalog.record(terra).model == terra.model
    assert catalog.offered() == [str(QWEN), str(terra)]
    assert await catalog.confirm_rejections() is False

    await catalog.start_background_refresh()

    assert sorted(api.calls) == ["anthropic", "google", "openai", "openrouter"]
    # The recorded listings serve neither: the list leaves both out, and says why.
    assert catalog.offered() == [str(QWEN)]
    assert catalog.not_offered() == {
        str(sonnet): "not served by anthropic", str(terra): "not served by openai",
    }
    await catalog.aclose()


async def test_a_second_round_lists_the_built_ins_the_first_one_made_needed():
    """Google is listed for an entry nothing describes, and its listing lacks the
    entry of the default tier: the computed role then needs the built-ins, and
    Anthropic is listed in a second round. There is no third."""
    api = ProviderAPI()
    entries = {
        ModelRef("google", "gemini-2.0-flash"): ModelEntry(),
        ModelRef("google", "gemini-9-flash"): ModelEntry(),
    }
    catalog = ModelCatalog(
        providers=providers(google=KEYS["google"], anthropic=KEYS["anthropic"]),
        entries=entries, computed_roles=("default",), transport=api.transport, clock=Clock(),
    )

    assert catalog.in_use == ("google",)
    await catalog.load()

    assert api.calls == ["google", "anthropic"]
    assert catalog.in_use == ("google", "anthropic")
    assert catalog.not_offered() == {"google:gemini-9-flash": "not served by google"}
    assert catalog.start_background_refresh() is None


async def test_a_provider_in_use_nothing_waits_for_is_still_listed_once():
    """An app's list of described models: nothing is listed at the start, and the
    background refresh lists each provider of the list once, no more."""
    api = ProviderAPI()
    catalog = ModelCatalog(
        providers=providers(**KEYS),
        entries={ModelRef("anthropic", "claude-sonnet-4-6"): ModelEntry()},
        transport=api.transport, clock=Clock(),
    )
    await catalog.load()
    assert api.calls == []

    await catalog.start_background_refresh()
    assert catalog.start_background_refresh() is None

    assert api.calls == ["anthropic"]
    assert states(catalog) == {"anthropic": ProviderState.FRESH}


# --- a computed role has no model of its own --------------------------------------------
#
# ``default`` without ``default_model``: the role takes the list's newest model of its
# tier. A models entry is only the settings' word that the model exists, so before a
# role runs on one, a listing must serve it: the start lists the entry's provider unless
# a listing in hand (fresh or stale) serves it.

SONNET_47 = ModelRef("anthropic", "claude-sonnet-4-7")  # a typo: Anthropic serves no such model
TWO_KEYS = {"google": KEYS["google"], "anthropic": KEYS["anthropic"]}


def computed(api, entries, *, cache=None, clock=None) -> ModelCatalog:
    return ModelCatalog(
        providers=providers(**TWO_KEYS), entries=entries, computed_roles=("default",),
        cache_path=cache, transport=api.transport, clock=clock or Clock(),
    )


def default_role(catalog: ModelCatalog) -> str:
    roles = RoleMap(default_model=None, model_roles={})
    return str(roles.resolve("default", catalog=catalog, providers=providers(**TWO_KEYS)))


async def _cached(cache, clock, api=None) -> None:
    """An earlier start, which named no model of Anthropic's, cached its listing."""
    seed = ModelCatalog(
        providers=providers(anthropic=KEYS["anthropic"]), cache_path=cache,
        transport=(api or ProviderAPI()).transport, clock=clock,
    )
    await seed.load()


async def test_a_computed_role_waits_on_the_listing_of_the_entry_it_would_take():
    """The list's one default-tier model is one Anthropic does not serve: the start
    lists Anthropic, finds it absent, and the role takes the newest model the
    built-ins serve, listed in a second round, not the unserved entry."""
    api = ProviderAPI()
    catalog = computed(api, {SONNET_47: ModelEntry()})

    await catalog.load()

    assert api.calls == ["anthropic", "google"]
    assert default_role(catalog) == "google:gemini-3-flash-preview"
    assert catalog.not_offered() == {str(SONNET_47): "not served by anthropic"}


async def test_a_warm_start_whose_cache_serves_the_role_s_entry_waits_on_nothing(tmp_path):
    cache, clock = tmp_path / "model_cache.json", Clock()
    await _cached(cache, clock)
    clock.now += 3600
    api = ProviderAPI()
    catalog = computed(
        api, {ModelRef("anthropic", "claude-sonnet-4-6"): ModelEntry()}, cache=cache, clock=clock
    )

    await catalog.load()

    assert api.calls == []
    assert default_role(catalog) == "anthropic:claude-sonnet-4-6"
    assert catalog.start_background_refresh() is None


async def test_a_cache_that_lacks_the_role_s_entry_is_listed_again(tmp_path):
    """A listing cached an hour ago by a start that did not name the model may
    predate it: the start lists Anthropic again, and takes a served model."""
    cache, clock = tmp_path / "model_cache.json", Clock()
    await _cached(cache, clock)
    clock.now += 3600
    api = ProviderAPI()
    catalog = computed(api, {SONNET_47: ModelEntry()}, cache=cache, clock=clock)

    await catalog.load()

    assert api.calls == ["anthropic", "google"]
    assert default_role(catalog) == "google:gemini-3-flash-preview"


async def test_a_role_that_runs_on_an_entry_its_provider_dropped_is_named_in_the_warning(
    tmp_path, caplog
):
    """A stale cached listing served the entry, so the start took it at once; the
    background refresh finds that Anthropic no longer serves it. The warning says
    that the default role runs on it, and what to do."""
    cache, clock = tmp_path / "model_cache.json", Clock()
    then = copy.deepcopy(recorded("anthropic"))
    then["data"].insert(0, dict(then["data"][0], id=SONNET_47.model, display_name="Sonnet 4.7"))
    await _cached(cache, clock, ProviderAPI(anthropic=then))
    clock.now += 2 * 86400  # stale
    api = ProviderAPI()
    catalog = computed(api, {SONNET_47: ModelEntry()}, cache=cache, clock=clock)

    with caplog.at_level(logging.WARNING, logger="agentic_cli.models.catalog"):
        await catalog.load()
        assert api.calls == []
        assert default_role(catalog) == str(SONNET_47)
        await catalog.start_background_refresh()

    warned = [r.getMessage() for r in caplog.records if r.name == "agentic_cli.models.catalog"]
    assert warned == [
        "Model anthropic:claude-sonnet-4-7 in models is not served by anthropic: its listing "
        "lacks it, yet the default role, which nothing maps, runs on it (the list's newest "
        "default-tier model), so a turn on it will fail at the provider. Set default_model, "
        "or restart to take another (`/models anthropic:` lists what it serves)."
    ]
