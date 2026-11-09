"""The model-list cache: ``~/.<app>/model_cache.json``, read at startup, written after listings.

Fresh entries (under 24 hours) spare startup the network; stale ones are used
while a background refresh runs; a broken file, a file from before version 3,
or an entry for another address or kind is ignored and listed again (R9). A
catalog writes only the providers it listed and keeps every other entry as the
file holds it (another catalog may have written it, another release too: an
entry this version cannot read is carried over as it is, and a file of a newer
version is never written), and a search adopts one.
The file is private and holds no credential.
"""

from __future__ import annotations

import copy
import json
import logging
import stat
import time

import pytest

from agentic_cli.models import (
    ModelCatalog,
    ModelEntry,
    ModelRef,
    ModelResolutionError,
    ProviderSpec,
    ProviderState,
    RoleMap,
)
from tests.models.fakes import (
    OPENROUTER_URL,
    Clock,
    ProviderAPI,
    hang,
    providers,
    recorded,
    refuse,
)

DAY = 86400.0
KEYS = {
    "google": "g-SECRET-test-key",
    "anthropic": "a-SECRET-test-key",
    "openai": "o-SECRET-test-key",
    "openrouter": "r-SECRET-test-key",
}
ENDPOINT_KEY = "v-SECRET-endpoint-key"
VLLM = ProviderSpec(base_url="http://gpu-box:8000/v1", api_key_env="VLLM_KEY")
#: Settings that name a model of each, so the provider is in use (and listed).
ON_VLLM = (ModelRef("local-vllm", "Qwen/Qwen3-32B"),)
ON_OPENROUTER = (ModelRef("openrouter", "qwen/qwen3-235b-a22b"),)


def make(
    cache, api=None, *, clock=None, specs=None, references=(), entries=None, **keys
) -> ModelCatalog:
    return ModelCatalog(
        providers=providers(specs=specs or {}, env={"VLLM_KEY": ENDPOINT_KEY}, **keys),
        entries=entries,
        references=references,
        cache_path=cache,
        transport=(api or ProviderAPI()).transport,
        clock=clock or Clock(),
    )


@pytest.fixture
def cache(tmp_path):
    return tmp_path / "agentic_cli" / "model_cache.json"


def read(cache) -> dict:
    return json.loads(cache.read_text(encoding="utf-8"))


def states(catalog: ModelCatalog) -> dict[str, ProviderState]:
    return {s.provider: s.state for s in catalog.status() if s.has_credentials}


class NoNetwork(ProviderAPI):
    """A transport whose every request fails the test."""

    async def _handle(self, request):
        raise AssertionError(f"unexpected request to {request.url}")


# --- what is written ---------------------------------------------------------------


async def test_the_file_holds_each_listing_versioned(cache):
    await make(cache, google=KEYS["google"], clock=Clock(1234.5)).load()

    document = read(cache)

    assert document["version"] == 3
    entry = document["providers"]["google"]
    assert (entry["fetched_at"], entry["ok"], entry["error"]) == (1234.5, True, None)
    assert (entry["base_url"], entry["kind"]) == (None, "google")
    assert [m["id"] for m in entry["models"]][:2] == ["gemini-2.5-flash", "gemini-2.5-pro"]
    assert entry["models"][0]["context_window"] == 1048576


async def test_the_file_is_private_and_holds_no_key(cache):
    await make(
        cache, specs={"local-vllm": VLLM}, references=[*ON_VLLM, *ON_OPENROUTER], **KEYS
    ).load()

    assert stat.S_IMODE(cache.stat().st_mode) == 0o600
    raw = cache.read_bytes()
    for key in (*KEYS.values(), ENDPOINT_KEY):
        assert key.encode() not in raw
    assert set(read(cache)["providers"]) == {*KEYS, "local-vllm"}


async def test_what_a_listing_says_about_turning_thinking_off_survives_the_cache(cache):
    clock = Clock()
    await make(cache, anthropic=KEYS["anthropic"], clock=clock).load()
    models = {m["id"]: m for m in read(cache)["providers"]["anthropic"]["models"]}
    assert models["claude-sonnet-5-5"]["can_disable"] is False
    assert models["claude-haiku-4-5-20251001"]["can_disable"] is None
    clock.now += 60

    catalog = make(cache, NoNetwork(), anthropic=KEYS["anthropic"], clock=clock)
    await catalog.load()

    assert states(catalog) == {"anthropic": ProviderState.FRESH}
    sonnet = catalog.record(ModelRef("anthropic", "claude-sonnet-5-5")).thinking
    assert sonnet.can_disable is False


async def test_what_a_listing_serves_and_what_it_shows_survive_the_cache(cache):
    clock = Clock()
    await make(cache, google=KEYS["google"], anthropic=KEYS["anthropic"], clock=clock).load()
    written = read(cache)["providers"]
    google = {m["id"]: m for m in written["google"]["models"]}
    anthropic = {m["id"]: m for m in written["anthropic"]["models"]}
    assert (google["gemini-flash-latest"]["shown"], google["gemini-flash-latest"]["alias"]) == (
        False, None,
    )
    dated = anthropic["claude-sonnet-4-5-20250929"]
    assert (dated["shown"], dated["alias"]) == (False, "claude-sonnet-4-5")
    clock.now += 60

    catalog = make(
        cache, NoNetwork(), google=KEYS["google"], anthropic=KEYS["anthropic"], clock=clock
    )
    await catalog.load()

    assert states(catalog) == {"google": ProviderState.FRESH, "anthropic": ProviderState.FRESH}
    for ref in (
        ModelRef("google", "gemini-flash-latest"),
        ModelRef("anthropic", "claude-sonnet-4-5-20250929"),
    ):
        assert catalog.record(ref).ref == ref
    assert [r.model for r in catalog.records("google")] == [
        "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite",
        "gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-2.0-flash",
    ]
    assert [r.model for r in catalog.records("anthropic")] == [
        "claude-sonnet-5-5", "claude-opus-4-7", "claude-sonnet-4-6", "claude-sonnet-4-5",
        "claude-haiku-4-5", "claude-3-5-haiku", "claude-3-haiku",
    ]


async def test_a_configured_provider_s_entry_records_its_base_url_and_kind(cache):
    await make(cache, specs={"local-vllm": VLLM}, references=ON_VLLM).load()

    entry = read(cache)["providers"]["local-vllm"]
    assert (entry["base_url"], entry["kind"]) == ("http://gpu-box:8000/v1", "openai_compatible")


async def test_no_cache_path_means_no_file(tmp_path):
    keyed = providers(google=KEYS["google"])
    catalog = ModelCatalog(providers=keyed, transport=ProviderAPI().transport)

    await catalog.load()

    assert list(tmp_path.iterdir()) == []
    assert catalog.is_authoritative("google")


# --- what is read ------------------------------------------------------------------


async def test_a_fresh_entry_spares_startup_the_network(cache):
    clock = Clock()
    await make(cache, google=KEYS["google"], clock=clock).load()
    clock.now += DAY - 60

    catalog = make(cache, NoNetwork(), google=KEYS["google"], clock=clock)
    await catalog.load()

    assert states(catalog) == {"google": ProviderState.FRESH}
    assert catalog.is_authoritative("google")
    assert catalog.start_background_refresh() is None


async def test_an_entry_a_day_old_is_stale_and_refreshed_in_the_background(cache):
    clock = Clock()
    await make(cache, google=KEYS["google"], clock=clock).load()
    clock.now += DAY + 1
    api = ProviderAPI()

    catalog = make(cache, api, google=KEYS["google"], clock=clock)
    await catalog.load()

    assert api.calls == []  # startup did not wait
    assert states(catalog) == {"google": ProviderState.STALE}
    assert not catalog.is_authoritative("google")
    assert catalog.record(ModelRef("google", "gemini-9-ultra")).model == "gemini-9-ultra"
    assert "gemini-2.0-flash" in [r.model for r in catalog.records("google")]

    task = catalog.start_background_refresh()
    await task

    assert api.calls == ["google"]
    assert states(catalog) == {"google": ProviderState.FRESH}
    assert read(cache)["providers"]["google"]["fetched_at"] == clock.now


async def test_a_stale_cache_with_the_network_down_does_not_wait(cache):
    """Review Focus 2, second half: a stale cache plus a hung network starts at
    once, and the stale entries are not authoritative."""
    clock = Clock()
    await make(cache, google=KEYS["google"], anthropic=KEYS["anthropic"], clock=clock).load()
    clock.now += 3 * DAY

    catalog = make(cache, ProviderAPI(google=hang, anthropic=hang), clock=clock,
                   google=KEYS["google"], anthropic=KEYS["anthropic"])
    started = time.monotonic()
    await catalog.load()

    assert time.monotonic() - started < 0.5
    assert states(catalog) == {"google": ProviderState.STALE, "anthropic": ProviderState.STALE}
    assert not catalog.is_authoritative("google") and not catalog.is_authoritative("anthropic")
    await catalog.aclose()


async def test_a_failed_first_listing_is_remembered_so_the_next_start_does_not_wait(cache):
    clock = Clock()
    first = make(cache, ProviderAPI(google=refuse), google=KEYS["google"], clock=clock)
    await first.load()
    entry = read(cache)["providers"]["google"]
    assert (entry["ok"], entry["models"]) == (False, [])
    assert entry["error"].startswith("connection failed")

    api = ProviderAPI(google=hang)
    catalog = make(cache, api, google=KEYS["google"], clock=clock)
    await catalog.load()

    assert api.calls == []
    assert states(catalog) == {"google": ProviderState.DEGRADED}
    assert catalog.status()[0].error.startswith("connection failed")
    task = catalog.start_background_refresh()
    assert task is not None
    await catalog.aclose()


async def test_a_failed_refresh_never_overwrites_a_good_entry(cache):
    clock = Clock()
    await make(cache, google=KEYS["google"], clock=clock).load()
    good = read(cache)["providers"]["google"]
    clock.now += 2 * DAY

    catalog = make(cache, ProviderAPI(google=500), google=KEYS["google"], clock=clock)
    await catalog.load()
    await catalog.start_background_refresh()

    assert states(catalog) == {"google": ProviderState.DEGRADED}
    assert read(cache)["providers"]["google"] == good


async def test_entries_of_providers_without_a_key_are_kept(cache):
    clock = Clock()
    await make(cache, anthropic=KEYS["anthropic"], clock=clock).load()

    # No Anthropic key now: its entry is not used, and not dropped either.
    catalog = make(cache, google=KEYS["google"], clock=clock)
    await catalog.load()

    assert set(read(cache)["providers"]) == {"google", "anthropic"}
    assert not catalog.is_authoritative("anthropic")


async def test_entries_of_providers_not_in_use_are_kept_as_they_are(cache):
    clock = Clock()
    await make(
        cache, references=ON_OPENROUTER, google=KEYS["google"], openrouter=KEYS["openrouter"],
        clock=clock,
    ).load()
    kept = read(cache)["providers"]["openrouter"]
    clock.now += 2 * DAY  # stale, but not in use: not listed again
    api = ProviderAPI()

    catalog = make(
        cache, api, google=KEYS["google"], openrouter=KEYS["openrouter"], clock=clock
    )
    await catalog.load()
    await catalog.start_background_refresh()

    assert api.calls == ["google"]
    assert read(cache)["providers"]["openrouter"] == kept
    assert [s.provider for s in catalog.status()] == ["google"]


async def test_entries_of_providers_this_catalog_does_not_define_are_kept(cache):
    """Another project of the same app may define the provider."""
    await make(cache, specs={"local-vllm": VLLM}, references=ON_VLLM).load()

    await make(cache, google=KEYS["google"]).load()

    assert set(read(cache)["providers"]) == {"google", "local-vllm"}


# --- several catalogs, one file ----------------------------------------------------------
# Two terminals, or two projects of one app, share the file: each catalog writes
# the providers it listed, and leaves every other entry as the file holds it.


async def test_a_write_keeps_the_entries_another_catalog_wrote(cache):
    clock = Clock()
    first = make(cache, google=KEYS["google"], openrouter=KEYS["openrouter"], clock=clock)
    await first.load()
    await make(
        cache, references=ON_OPENROUTER, google=KEYS["google"], openrouter=KEYS["openrouter"],
        clock=clock,
    ).load()
    written = read(cache)["providers"]["openrouter"]

    await first.refresh(["google"])

    assert read(cache)["providers"]["openrouter"] == written


async def test_a_listing_read_from_the_file_never_replaces_a_newer_one(cache):
    clock = Clock()
    await make(cache, google=KEYS["google"], anthropic=KEYS["anthropic"], clock=clock).load()
    clock.now += 2 * DAY
    first = make(cache, google=KEYS["google"], anthropic=KEYS["anthropic"], clock=clock)
    await first.load()
    second = make(cache, google=KEYS["google"], anthropic=KEYS["anthropic"], clock=clock)
    await second.load()
    await second.refresh(["google"])

    await first.refresh(["anthropic"])

    assert read(cache)["providers"]["google"]["fetched_at"] == clock.now


async def test_a_failed_refresh_keeps_its_earlier_listing_when_the_file_lost_it(cache):
    """With no listing in the file (deleted, or a failure written since), a
    failed refresh writes the listing this catalog had, not the failure."""
    clock = Clock()
    await make(cache, google=KEYS["google"], clock=clock).load()
    good = read(cache)["providers"]["google"]
    clock.now += 2 * DAY
    catalog = make(cache, ProviderAPI(google=500), google=KEYS["google"], clock=clock)
    await catalog.load()
    cache.unlink()

    await catalog.start_background_refresh()

    assert read(cache)["providers"]["google"] == good


async def test_a_failed_listing_never_replaces_one_another_catalog_wrote(cache):
    clock = Clock()
    failing = make(cache, ProviderAPI(google=500), google=KEYS["google"], clock=clock)
    await failing.load()
    other = make(cache, google=KEYS["google"], clock=clock)
    await other.load()
    await other.refresh(["google"])

    await failing.refresh(["google"])

    entry = read(cache)["providers"]["google"]
    assert (entry["ok"], entry["error"]) == (True, None)
    assert entry["models"]


async def test_a_missing_cache_folder_is_created(tmp_path):
    cache = tmp_path / "nested" / "deeper" / "model_cache.json"

    await make(cache, google=KEYS["google"]).load()

    assert cache.is_file()


async def test_an_unwritable_cache_never_fails_startup(tmp_path):
    cache = tmp_path / "model_cache.json"
    cache.mkdir()  # a folder where the file should be

    catalog = make(cache, google=KEYS["google"])
    await catalog.load()

    assert catalog.is_authoritative("google")


# --- a broken file is ignored (R9, Review Focus 3) -------------------------------------


BROKEN = {
    "invalid JSON": lambda doc: b"{not json",
    "not an object": lambda doc: b"[1, 2, 3]",
    "an older version": lambda doc: json.dumps({**doc, "version": 2}).encode(),
    "no version": lambda doc: json.dumps({"providers": doc["providers"]}).encode(),
    "fetched_at a string": lambda doc: _with_google(doc, fetched_at="yesterday"),
    "ok a number": lambda doc: _with_google(doc, ok=1),
    "models not a list": lambda doc: _with_google(doc, models="gemini-2.5-flash"),
    "a model id a number": lambda doc: _with_google(doc, models=[{"id": 5}]),
    "a context window a string": lambda doc: _with_google(
        doc, models=[{"id": "gemini-2.5-flash", "context_window": "1048576"}]
    ),
    "an unknown field": lambda doc: _with_google(doc, api_key="g-SECRET-test-key"),
    "not UTF-8": lambda doc: b"\xff\xfe\x00",
}


def _with_google(doc: dict, **fields) -> bytes:
    providers = dict(doc["providers"])
    providers["google"] = {**providers["google"], **fields}
    return json.dumps({**doc, "providers": providers}).encode()


@pytest.mark.parametrize("damage", list(BROKEN))
async def test_a_broken_cache_file_is_ignored(cache, damage):
    """Review Focus 3: whatever is wrong with the file, startup lists again
    and goes on; nothing raises."""
    clock = Clock()
    await make(cache, google=KEYS["google"], clock=clock).load()
    cache.write_bytes(BROKEN[damage](read(cache)))
    api = ProviderAPI()

    catalog = make(cache, api, google=KEYS["google"], clock=clock)
    await catalog.load()

    assert api.calls == ["google"]  # listed again
    assert states(catalog) == {"google": ProviderState.FRESH}
    assert read(cache)["version"] == 3  # and rewritten


async def test_a_provider_whose_base_url_changed_is_listed_again(cache):
    clock = Clock()
    await make(
        cache, specs={"local-vllm": VLLM}, references=ON_VLLM, google=KEYS["google"], clock=clock
    ).load()
    moved = ProviderSpec(base_url="http://other-box:8000/v1", api_key_env="VLLM_KEY")
    api = ProviderAPI()

    catalog = make(
        cache, api, specs={"local-vllm": moved}, references=ON_VLLM, google=KEYS["google"],
        clock=clock,
    )
    await catalog.load()

    assert api.calls == ["endpoint"]  # Google's entry still served from the cache
    assert read(cache)["providers"]["local-vllm"]["base_url"] == "http://other-box:8000/v1"


async def test_a_provider_whose_kind_changed_is_listed_again(cache):
    """The same address, now spoken to as OpenRouter: another lister, other facts."""
    clock = Clock()
    gateway = ProviderSpec(base_url=OPENROUTER_URL, api_key_env="VLLM_KEY")
    references = [ModelRef("gateway", "qwen/qwen3-235b-a22b")]
    await make(cache, specs={"gateway": gateway}, references=references, clock=clock).load()
    assert read(cache)["providers"]["gateway"]["kind"] == "openai_compatible"
    api = ProviderAPI()

    as_openrouter = ProviderSpec(kind="openrouter", base_url=OPENROUTER_URL, api_key_env="VLLM_KEY")
    catalog = make(cache, api, specs={"gateway": as_openrouter}, references=references, clock=clock)
    await catalog.load()

    assert api.calls == ["openrouter"]
    assert read(cache)["providers"]["gateway"]["kind"] == "openrouter"
    assert catalog.record(references[0]).sources["tools"] == "listing"  # OpenRouter's


async def test_a_version_1_cache_is_listed_again(cache):
    """Review Focus 2: a file from before the cache knew a provider's kind (an
    OpenRouter entry without one) is ignored, without an error, and every
    provider in use is listed again and written with its kind."""
    clock = Clock()
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"version": 1, "providers": {
        "openrouter": {
            "fetched_at": clock.now - 60, "ok": True, "error": None, "base_url": OPENROUTER_URL,
            "models": [{"id": "qwen/qwen3-235b-a22b"}],
        },
        "google": {"fetched_at": clock.now - 60, "ok": True, "models": [{"id": "gemini-2.5-flash"}]},
    }}))
    api = ProviderAPI()

    catalog = make(
        cache, api, references=ON_OPENROUTER, google=KEYS["google"],
        openrouter=KEYS["openrouter"], clock=clock,
    )
    await catalog.load()

    assert sorted(api.calls) == ["google", "openrouter"]
    assert states(catalog) == {"google": ProviderState.FRESH, "openrouter": ProviderState.FRESH}
    document = read(cache)
    assert document["version"] == 3
    assert {name: entry["kind"] for name, entry in document["providers"].items()} == {
        "google": "google", "openrouter": "openrouter",
    }


async def test_a_version_2_cache_is_listed_again(cache):
    """A file from before the listers kept every id a provider serves holds
    Google's listing without its aliases and Anthropic's without its dated
    ids: fresh, it would reject those until it aged. It is ignored, and every
    provider in use is listed again."""
    clock = Clock()
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"version": 2, "providers": {
        "google": {
            "fetched_at": clock.now - 60, "ok": True, "kind": "google",
            "models": [{"id": "gemini-2.5-flash"}, {"id": "gemini-2.0-flash"}],
        },
        "anthropic": {
            "fetched_at": clock.now - 60, "ok": True, "kind": "anthropic",
            "models": [{"id": "claude-sonnet-4-5", "display_name": "Claude Sonnet 4.5"}],
        },
    }}))
    api = ProviderAPI()

    catalog = make(cache, api, google=KEYS["google"], anthropic=KEYS["anthropic"], clock=clock)
    await catalog.load()

    assert sorted(api.calls) == ["anthropic", "google"]
    assert states(catalog) == {"google": ProviderState.FRESH, "anthropic": ProviderState.FRESH}
    for ref in (
        ModelRef("google", "gemini-flash-latest"),
        ModelRef("anthropic", "claude-sonnet-4-5-20250929"),
    ):
        assert catalog.record(ref).ref == ref
    assert read(cache)["version"] == 3


# --- a file this version cannot fully read ------------------------------------------------
# Two releases of one app (a dev checkout beside the installed app) share the file. What
# one cannot read, the other has written: a catalog keeps every entry it did not list as the
# file holds it, readable or not, and never writes a file of a newer version.

GEMINI_PRO = ModelRef("google", "gemini-3.1-pro-preview")


async def _shared_file(cache, clock) -> dict:
    """A healthy file holding Google, Anthropic and OpenRouter; the document as read."""
    await make(
        cache, references=ON_OPENROUTER, google=KEYS["google"], anthropic=KEYS["anthropic"],
        openrouter=KEYS["openrouter"], clock=clock,
    ).load()
    document = read(cache)
    assert set(document["providers"]) == {"google", "anthropic", "openrouter"}
    return document


async def _listing_only_google(cache, clock, api=None) -> tuple[ModelCatalog, ProviderAPI]:
    """A catalog whose one entry is a Google model: Google is the one provider in use."""
    api = api or ProviderAPI()
    catalog = make(
        cache, api, entries={GEMINI_PRO: ModelEntry()}, google=KEYS["google"],
        anthropic=KEYS["anthropic"], openrouter=KEYS["openrouter"], clock=clock,
    )
    await catalog.load()
    return catalog, api


async def test_a_cached_model_with_a_field_this_version_does_not_know_is_read(cache):
    """A release that adds a field to a cached model needs no version bump: an
    older one reads the model without it."""
    clock = Clock()
    document = await _shared_file(cache, clock)
    document["providers"]["openrouter"]["models"][0]["pricing"] = 1.0
    cache.write_text(json.dumps(document))

    catalog = make(cache, NoNetwork(), references=ON_OPENROUTER, openrouter=KEYS["openrouter"],
                   clock=clock)
    await catalog.load()

    assert states(catalog) == {"openrouter": ProviderState.FRESH}
    assert catalog.record(ON_OPENROUTER[0]).ref == ON_OPENROUTER[0]


async def test_a_cached_model_with_an_unknown_field_costs_no_other_entry(cache):
    clock = Clock()
    document = await _shared_file(cache, clock)
    del document["providers"]["google"]
    document["providers"]["openrouter"]["models"][0]["pricing"] = 1.0
    cache.write_text(json.dumps(document))

    _, api = await _listing_only_google(cache, clock)

    assert api.calls == ["google"]
    written = read(cache)
    assert written["version"] == 3
    assert set(written["providers"]) == {"google", "anthropic", "openrouter"}
    assert written["providers"]["anthropic"] == document["providers"]["anthropic"]
    assert written["providers"]["openrouter"] == document["providers"]["openrouter"]  # the field too


DAMAGE = {
    "ok a string": lambda entry: {**entry, "ok": "yes"},
    "no fetched_at": lambda entry: {key: v for key, v in entry.items() if key != "fetched_at"},
    "an unknown field": lambda entry: {**entry, "api_key": "x"},
    "not an object": lambda entry: [entry],
}


@pytest.mark.parametrize("damage", list(DAMAGE))
async def test_an_entry_this_version_cannot_read_survives_a_write_of_another(cache, damage):
    clock = Clock()
    document = await _shared_file(cache, clock)
    del document["providers"]["google"]
    document["providers"]["openrouter"] = DAMAGE[damage](document["providers"]["openrouter"])
    cache.write_text(json.dumps(document))

    _, api = await _listing_only_google(cache, clock)

    assert api.calls == ["google"]
    written = read(cache)["providers"]
    assert set(written) == {"google", "anthropic", "openrouter"}
    assert written["openrouter"] == document["providers"]["openrouter"]
    assert written["anthropic"] == document["providers"]["anthropic"]


async def test_a_damaged_entry_is_replaced_by_its_own_provider_s_listing(cache):
    clock = Clock()
    document = await _shared_file(cache, clock)
    document["providers"]["google"]["ok"] = "yes"
    cache.write_text(json.dumps(document))

    _, api = await _listing_only_google(cache, clock)

    assert api.calls == ["google"]
    written = read(cache)["providers"]
    assert written["google"]["ok"] is True and written["google"]["models"]
    assert written["anthropic"] == document["providers"]["anthropic"]


async def test_a_file_of_a_newer_version_is_listed_around_and_never_written(cache):
    clock = Clock()
    document = await _shared_file(cache, clock)
    cache.write_text(json.dumps({**document, "version": 4}))
    before = cache.read_bytes()

    catalog, api = await _listing_only_google(cache, clock)
    await catalog.refresh()
    await catalog.search("gemini")

    assert "google" in api.calls  # nothing in the file was used
    assert states(catalog)["google"] is ProviderState.FRESH  # the listings are kept in memory
    assert cache.read_bytes() == before


async def test_a_newer_file_is_not_used_by_this_version(cache):
    clock = Clock()
    document = await _shared_file(cache, clock)
    cache.write_text(json.dumps({**document, "version": 4}))
    api = ProviderAPI()

    catalog = make(cache, api, google=KEYS["google"], anthropic=KEYS["anthropic"], clock=clock)
    await catalog.load()

    assert sorted(api.calls) == ["anthropic", "google"]


# --- one cache, several projects: what each catalog says is absent --------------------------


async def test_two_projects_with_different_retired_entries_list_once_each(cache):
    """Two projects of one app share the file, and each names a model the
    provider has retired. Each is confirmed by one listing, and no start then
    lists for it while the listing is fresh: neither project's record of what
    is absent replaces the other's."""
    clock = Clock()
    good = ModelRef("anthropic", "claude-sonnet-4-6")
    retired_here, retired_there = RETIRED, ModelRef("anthropic", "claude-opus-3")
    project_a = {retired_here: ModelEntry(), good: ModelEntry()}
    project_b = {retired_there: ModelEntry(), good: ModelEntry()}
    listings = []

    for entries in (project_a, project_a, project_b, project_a, project_b, project_a):
        api = ProviderAPI()
        catalog = make(cache, api, entries=entries, anthropic=KEYS["anthropic"], clock=clock)
        await catalog.load()
        listings.append(len(api.calls))
        clock.now += 600  # still fresh

    assert listings == [1, 0, 1, 0, 0, 0]
    assert read(cache)["providers"]["anthropic"]["absent"] == [
        retired_there.model, retired_here.model,
    ]


async def test_a_model_the_provider_serves_leaves_the_recorded_absent(cache):
    """The file's record of what is absent is another catalog's too, from a listing
    that may predate a model: a new listing that serves it drops it from the record."""
    clock = Clock()
    served = "claude-sonnet-4-6"
    await make(cache, entries={RETIRED: ModelEntry()}, anthropic=KEYS["anthropic"], clock=clock).load()
    document = read(cache)
    document["providers"]["anthropic"]["absent"] = ["claude-opus-3", served, RETIRED.model]
    cache.write_text(json.dumps(document))
    catalog = make(
        cache, entries={RETIRED: ModelEntry()}, anthropic=KEYS["anthropic"], clock=clock
    )
    await catalog.load()

    await catalog.refresh(["anthropic"])

    assert read(cache)["providers"]["anthropic"]["absent"] == ["claude-opus-3", RETIRED.model]


# --- search: a provider not in use, from the cache or listed --------------------------


async def test_a_search_adopts_the_cached_listing_of_a_provider_not_in_use(cache):
    clock = Clock()
    await make(
        cache, references=ON_OPENROUTER, openrouter=KEYS["openrouter"], google=KEYS["google"],
        clock=clock,
    ).load()  # an earlier start used OpenRouter
    clock.now += 60
    api = ProviderAPI()
    catalog = make(cache, api, google=KEYS["google"], openrouter=KEYS["openrouter"], clock=clock)
    await catalog.load()

    found = await catalog.search("llama")

    assert api.calls == []
    assert [str(record.ref) for record in found] == [
        "openrouter:meta-llama/llama-3.3-70b-instruct"
    ]
    assert states(catalog) == {"google": ProviderState.FRESH, "openrouter": ProviderState.FRESH}


async def test_a_search_lists_a_provider_whose_cached_listing_is_stale(cache):
    """Weeks after a start used OpenRouter, a search finds a model it has
    released since: a stale listing is listed again first."""
    clock = Clock()
    older = recorded("openrouter")
    older["data"] = [model for model in older["data"] if "llama" not in model["id"]]
    await make(
        cache, ProviderAPI(openrouter=older), references=ON_OPENROUTER,
        openrouter=KEYS["openrouter"], clock=clock,
    ).load()
    clock.now += 40 * DAY
    api = ProviderAPI()
    catalog = make(cache, api, google=KEYS["google"], openrouter=KEYS["openrouter"], clock=clock)
    await catalog.load()
    api.calls.clear()

    found = await catalog.search("llama")

    assert api.calls == ["openrouter"]
    assert [str(record.ref) for record in found] == [
        "openrouter:meta-llama/llama-3.3-70b-instruct"
    ]


async def test_a_search_lists_a_provider_whose_cached_listing_failed(cache):
    clock = Clock()
    await make(
        cache, ProviderAPI(openrouter=401), references=ON_OPENROUTER,
        openrouter=KEYS["openrouter"], clock=clock,
    ).load()
    clock.now += 60  # the key has been fixed since
    api = ProviderAPI()
    catalog = make(cache, api, google=KEYS["google"], openrouter=KEYS["openrouter"], clock=clock)
    await catalog.load()
    api.calls.clear()

    found = await catalog.search("llama")

    assert api.calls == ["openrouter"]
    assert len(found) == 1


async def test_a_search_lists_a_provider_whose_cached_listing_is_for_another_address(cache):
    clock = Clock()
    await make(cache, specs={"local-vllm": VLLM}, references=ON_VLLM, clock=clock).load()
    moved = ProviderSpec(base_url="http://other-box:8000/v1", api_key_env="VLLM_KEY")
    api = ProviderAPI()
    catalog = make(cache, api, specs={"local-vllm": moved}, google=KEYS["google"], clock=clock)
    await catalog.load()

    found = await catalog.search("qwen3")

    assert sorted(api.calls) == ["endpoint", "google"]
    assert [str(record.ref) for record in found] == [
        "local-vllm:Qwen/Qwen3-32B", "local-vllm:qwen3-sql-lora",
    ]
    assert read(cache)["providers"]["local-vllm"]["base_url"] == "http://other-box:8000/v1"


async def test_a_timestamp_from_the_future_is_stale(cache):
    clock = Clock()
    await make(cache, google=KEYS["google"], clock=clock).load()
    clock.now -= 3600  # the clock went back

    catalog = make(cache, ProviderAPI(google=hang), google=KEYS["google"], clock=clock)
    await catalog.load()

    assert states(catalog) == {"google": ProviderState.STALE}
    await catalog.aclose()


# --- a cached listing that rejects an id ---------------------------------------------
#
# A listing read from the cache can predate a model the provider now serves (a
# release, a plan upgrade, another account's key). Before such a listing
# rejects an id for good, the provider is listed again, once.

NEW = ModelRef("anthropic", "claude-sonnet-5-6")


def _serving(model_id: str) -> dict:
    """Anthropic's recorded listing, plus a model released after it was cached."""
    payload = copy.deepcopy(recorded("anthropic"))
    payload["data"].insert(0, dict(payload["data"][0], id=model_id, display_name=model_id))
    return payload


async def _cached_an_hour_ago(cache, clock) -> None:
    await make(cache, anthropic=KEYS["anthropic"], clock=clock).load()
    clock.now += 3600


async def test_a_cached_listing_that_rejects_an_id_is_listed_again_once(cache):
    clock = Clock()
    await _cached_an_hour_ago(cache, clock)
    api = ProviderAPI(anthropic=_serving(NEW.model))
    catalog = make(cache, api, anthropic=KEYS["anthropic"], clock=clock)
    await catalog.load()
    assert api.calls == []

    with pytest.raises(ModelResolutionError, match="not available"):
        catalog.record(NEW)
    assert await catalog.confirm_rejections() is True

    assert api.calls == ["anthropic"]
    assert catalog.record(NEW).model == NEW.model
    assert await catalog.confirm_rejections() is False
    assert api.calls == ["anthropic"]
    # The new listing is cached: the next start knows the model.
    assert NEW.model in [m["id"] for m in read(cache)["providers"]["anthropic"]["models"]]


async def test_an_id_the_provider_still_lacks_is_rejected_by_the_new_listing(cache):
    clock = Clock()
    await _cached_an_hour_ago(cache, clock)
    api = ProviderAPI()
    catalog = make(cache, api, anthropic=KEYS["anthropic"], clock=clock)
    await catalog.load()
    with pytest.raises(ModelResolutionError):
        catalog.record(NEW)

    assert await catalog.confirm_rejections() is True

    with pytest.raises(ModelResolutionError, match="not available"):
        catalog.record(NEW)
    assert await catalog.confirm_rejections() is False
    assert api.calls == ["anthropic"]


async def test_when_listing_again_fails_the_id_is_accepted(cache):
    clock = Clock()
    await _cached_an_hour_ago(cache, clock)
    catalog = make(cache, ProviderAPI(anthropic=500), anthropic=KEYS["anthropic"], clock=clock)
    await catalog.load()
    with pytest.raises(ModelResolutionError):
        catalog.record(NEW)

    assert await catalog.confirm_rejections() is True

    # As for any failed listing: the provider's ids are not checked.
    assert catalog.record(NEW).model == NEW.model


async def test_a_rejection_by_a_listing_from_this_start_is_not_listed_again(cache):
    api = ProviderAPI()
    catalog = make(cache, api, anthropic=KEYS["anthropic"])
    await catalog.load()
    with pytest.raises(ModelResolutionError):
        catalog.record(NEW)

    assert await catalog.confirm_rejections() is False
    assert api.calls == ["anthropic"]


# The same holds for a models entry: before a cached listing keeps it out of
# the list, with a warning, the provider is listed again, once, while loading.


def _catalog_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "agentic_cli.models.catalog"]


async def test_an_entry_a_cached_listing_lacks_is_listed_again_once(cache, caplog):
    clock = Clock()
    await _cached_an_hour_ago(cache, clock)
    api = ProviderAPI(anthropic=_serving(NEW.model))
    catalog = make(cache, api, entries={NEW: ModelEntry()}, anthropic=KEYS["anthropic"], clock=clock)

    with caplog.at_level(logging.WARNING, logger="agentic_cli.models.catalog"):
        await catalog.load()

    assert api.calls == ["anthropic"]
    assert catalog.offered() == [str(NEW)]
    assert _catalog_warnings(caplog) == []
    assert await catalog.confirm_rejections() is False


RETIRED = ModelRef("anthropic", "claude-sonnet-3-7")


async def test_a_confirmed_absence_is_trusted_at_the_next_start(cache):
    """An app's list names a model its provider has retired. The listing that
    found it missing says so in the cache, so the next starts (within a day)
    do not list again for it: no wait at every start."""
    clock = Clock()
    entries = {RETIRED: ModelEntry(), ModelRef("anthropic", "claude-sonnet-4-6"): ModelEntry()}
    calls = []
    for _ in range(3):
        api = ProviderAPI()
        catalog = make(cache, api, entries=entries, anthropic=KEYS["anthropic"], clock=clock)
        await catalog.load()
        calls.append(api.calls)
        assert catalog.not_offered() == {str(RETIRED): "not served by anthropic"}
        clock.now += 600

    assert calls == [["anthropic"], [], []]
    assert read(cache)["providers"]["anthropic"]["absent"] == [RETIRED.model]


async def test_a_new_entry_is_still_confirmed_once(cache):
    """A model added to the list after the cached listing may be one released
    since: its provider is listed once more, and then trusted."""
    clock = Clock()
    await make(cache, entries={RETIRED: ModelEntry()}, anthropic=KEYS["anthropic"], clock=clock).load()
    entries = {RETIRED: ModelEntry(), NEW: ModelEntry()}
    calls = []
    for _ in range(2):
        clock.now += 600
        api = ProviderAPI(anthropic=_serving(NEW.model))
        catalog = make(cache, api, entries=entries, anthropic=KEYS["anthropic"], clock=clock)
        await catalog.load()
        calls.append(api.calls)
        assert catalog.offered() == [str(NEW)]

    assert calls == [["anthropic"], []]


async def test_a_stale_listing_keeps_out_an_entry_it_recorded_as_absent(cache):
    """A start more than a day after the listing that found an entry's model
    retired uses that listing while a refresh runs: the entry stays out of
    the list, so no role is computed onto a model the provider lacks."""
    clock = Clock()
    retired = ModelRef("anthropic", "claude-sonnet-9-9")  # the list's newest Sonnet
    entries = {retired: ModelEntry(), ModelRef("anthropic", "claude-haiku-4-5"): ModelEntry()}
    await make(cache, entries=entries, anthropic=KEYS["anthropic"], clock=clock).load()
    clock.now += DAY + 600

    catalog = make(cache, NoNetwork(), entries=entries, anthropic=KEYS["anthropic"], clock=clock)
    await catalog.load()

    assert states(catalog)["anthropic"] is ProviderState.STALE
    assert catalog.not_offered() == {str(retired): "not served by anthropic"}
    assert retired not in [record.ref for record in catalog.entry_records()]
    default = RoleMap(default_model=None, model_roles={}).resolve(
        "default", catalog=catalog, providers=providers(anthropic=KEYS["anthropic"])
    )
    assert default != retired
    assert default.model in {record.model for record in catalog.records("anthropic")}


async def test_offline_a_recorded_absence_keeps_the_listing_authoritative(cache):
    """With the network down, a start does not try the listing again for the
    retired model, so the fresh cached listing still rejects a typo."""
    clock = Clock()
    entries = {RETIRED: ModelEntry(), ModelRef("anthropic", "claude-sonnet-4-6"): ModelEntry()}
    await make(cache, entries=entries, anthropic=KEYS["anthropic"], clock=clock).load()
    clock.now += 600
    api = ProviderAPI(anthropic=refuse)

    catalog = make(cache, api, entries=entries, anthropic=KEYS["anthropic"], clock=clock)
    await catalog.load()

    assert api.calls == []
    assert catalog.is_authoritative("anthropic")
    assert catalog.offered() == ["anthropic:claude-sonnet-4-6"]
    with pytest.raises(ModelResolutionError, match="not available"):
        catalog.record(ModelRef("anthropic", "claude-typo-9"))


async def test_an_entry_the_provider_still_lacks_warns_after_one_listing(cache, caplog):
    clock = Clock()
    await _cached_an_hour_ago(cache, clock)
    api = ProviderAPI()
    catalog = make(cache, api, entries={NEW: ModelEntry()}, anthropic=KEYS["anthropic"], clock=clock)

    with caplog.at_level(logging.WARNING, logger="agentic_cli.models.catalog"):
        await catalog.load()

    assert api.calls == ["anthropic"]
    assert catalog.not_offered() == {str(NEW): "not served by anthropic"}
    assert _catalog_warnings(caplog) == [
        f"Model {NEW} in models is not offered: not served by anthropic"
    ]
