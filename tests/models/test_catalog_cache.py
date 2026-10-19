"""The model-list cache: ``~/.<app>/model_cache.json``, read at startup, written after listings.

Fresh entries (under 24 hours) spare startup the network; stale ones are used
while a background refresh runs; a broken file is ignored and listed again
(R9). The file is private and holds no credential.
"""

from __future__ import annotations

import copy
import json
import stat
import time

import pytest

from agentic_cli.models import (
    EndpointSpec,
    ModelCatalog,
    ModelRef,
    ModelResolutionError,
    ProviderState,
)
from tests.models.fakes import Clock, ProviderAPI, endpoints, hang, recorded, refuse

DAY = 86400.0
KEYS = {
    "google": "g-SECRET-test-key",
    "anthropic": "a-SECRET-test-key",
    "openai": "o-SECRET-test-key",
    "openrouter": "r-SECRET-test-key",
}
ENDPOINT_KEY = "v-SECRET-endpoint-key"
VLLM = EndpointSpec(base_url="http://gpu-box:8000/v1", api_key_env="VLLM_KEY")


def make(cache, api=None, *, clock=None, specs=None, **keys) -> ModelCatalog:
    return ModelCatalog(
        endpoints=endpoints(specs=specs or {}, env={"VLLM_KEY": ENDPOINT_KEY}, **keys),
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

    assert document["version"] == 1
    entry = document["providers"]["google"]
    assert (entry["fetched_at"], entry["ok"], entry["error"]) == (1234.5, True, None)
    assert [m["id"] for m in entry["models"]][:2] == ["gemini-2.5-flash", "gemini-2.5-pro"]
    assert entry["models"][0]["context_window"] == 1048576


async def test_the_file_is_private_and_holds_no_key(cache):
    await make(cache, specs={"local-vllm": VLLM}, **KEYS).load()

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
    assert models["claude-haiku-4-5"]["can_disable"] is None
    clock.now += 60

    catalog = make(cache, NoNetwork(), anthropic=KEYS["anthropic"], clock=clock)
    await catalog.load()

    assert states(catalog) == {"anthropic": ProviderState.FRESH}
    sonnet = catalog.record(ModelRef("anthropic", "claude-sonnet-5-5")).thinking
    assert sonnet.can_disable is False


async def test_an_endpoint_entry_records_its_base_url(cache):
    await make(cache, specs={"local-vllm": VLLM}).load()

    assert read(cache)["providers"]["local-vllm"]["base_url"] == "http://gpu-box:8000/v1"


async def test_no_cache_path_means_no_file(tmp_path):
    eps = endpoints(google=KEYS["google"])
    catalog = ModelCatalog(endpoints=eps, transport=ProviderAPI().transport)

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


async def test_entries_of_providers_not_listed_are_kept(cache):
    clock = Clock()
    await make(cache, anthropic=KEYS["anthropic"], clock=clock).load()

    # No Anthropic key now: its entry is not used, and not dropped either.
    catalog = make(cache, google=KEYS["google"], clock=clock)
    await catalog.load()

    assert set(read(cache)["providers"]) == {"google", "anthropic"}
    assert catalog.status()[1].state is ProviderState.UNATTEMPTED


async def test_entries_of_endpoints_no_longer_defined_are_dropped(cache):
    await make(cache, specs={"local-vllm": VLLM}).load()

    await make(cache, google=KEYS["google"]).load()

    assert set(read(cache)["providers"]) == {"google"}


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
    "wrong version": lambda doc: json.dumps({**doc, "version": 2}).encode(),
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
    assert read(cache)["version"] == 1  # and rewritten


async def test_an_endpoint_whose_base_url_changed_is_listed_again(cache):
    clock = Clock()
    await make(cache, specs={"local-vllm": VLLM}, google=KEYS["google"], clock=clock).load()
    moved = EndpointSpec(base_url="http://other-box:8000/v1", api_key_env="VLLM_KEY")
    api = ProviderAPI()

    catalog = make(cache, api, specs={"local-vllm": moved}, google=KEYS["google"], clock=clock)
    await catalog.load()

    assert api.calls == ["endpoint"]  # Google's entry still served from the cache
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
