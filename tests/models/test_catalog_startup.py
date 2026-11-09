"""Startup: what ``ModelCatalog.load`` waits on.

The settings name the models, and a start uses them at once: ``load`` waits on
a listing only to fill a gap (``models.startup``), every one it needs at once,
and the background refresh lists each other provider in use. Each listing here
takes ``DELAY`` seconds, as in the review's probe of the start before this
(``p3_sequential_rounds.py``: four providers listed in four sequential rounds).
"""

from __future__ import annotations

import asyncio
import time

from agentic_cli.models import (
    ModelCatalog,
    ModelEntry,
    ModelRef,
    ProviderState,
    ThinkingMechanism,
)
from tests.models.fakes import Clock, ProviderAPI, providers

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
    entry's facts): the start lists nothing, and the background refresh then lists
    each provider in use, whose listing reaches the records."""
    api = Slow()
    entries = {
        ModelRef("anthropic", "claude-sonnet-4-6"): ModelEntry(),
        QWEN: ModelEntry(context_window=131072),
    }
    references = (ModelRef("openai", "gpt-5.5"), ModelRef("google", "gemini-2.5-flash"))
    catalog = make(api, entries=entries, references=references, computed_roles=("default",))

    elapsed = await timed(catalog)

    assert (api.peak, api.calls) == (0, [])
    assert elapsed < DELAY
    assert catalog.in_use == ("google", "anthropic", "openai", "openrouter")
    named = [*map(str, entries), *map(str, references)]
    assert [str(catalog.record(ref).ref) for ref in (*entries, *references)] == named
    assert catalog.offered() == named
    assert catalog.record(QWEN).thinking.mechanism is ThinkingMechanism.NONE  # no listing yet

    await catalog.start_background_refresh()

    assert sorted(api.calls) == ALL
    assert set(states(catalog).values()) == {ProviderState.FRESH}
    assert catalog.offered() == named
    assert catalog.record(QWEN).thinking.mechanism is ThinkingMechanism.EFFORT
    assert catalog.record(QWEN).context_window == 131072  # the entry still wins
    await catalog.aclose()


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


async def test_the_probe_s_start_waits_on_the_one_listing_it_needs(tmp_path):
    """The probe's start: an OpenRouter entry nothing describes, an Anthropic entry
    and an OpenAI reference a listing cached an hour ago lacks (each described by a
    table rule), and a computed default role the Anthropic entry fills. It took four
    rounds; now it lists OpenRouter alone, and the background refresh confirms the
    other two against new listings."""
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

    assert api.calls == ["openrouter"]
    assert elapsed < 2 * DELAY
    assert catalog.record(sonnet).model == sonnet.model
    assert catalog.record(terra).model == terra.model
    assert catalog.offered() == [str(QWEN), str(sonnet), str(terra)]
    assert await catalog.confirm_rejections() is False

    await catalog.start_background_refresh()

    assert sorted(api.calls) == ["anthropic", "openai", "openrouter"]
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
