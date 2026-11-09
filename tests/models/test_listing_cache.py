"""The listing cache file: ``ListingCache`` reads and writes ``~/.<app>/model_cache.json``.

No catalog, no network: the tests build ``Listing`` objects by hand and read the file
as JSON. What the catalog does with the entries (fresh, stale, listed again) is
covered in ``test_catalog_cache.py``.

The contract here: a write puts one provider's entry in and carries every other
entry over as the file holds it, read or not; a file of a newer version is never
written; the format (version 3) does not change.
"""

from __future__ import annotations

import json
import logging
import stat

import pytest

from agentic_cli.models import ProviderSpec
from agentic_cli.models.discovery import ListedModel, Listing
from agentic_cli.models.listing_cache import CACHE_VERSION, ListingCache
from tests.models.fakes import OPENROUTER_URL, providers

PROVIDERS = providers(
    specs={"local-vllm": ProviderSpec(base_url="http://gpu-box:8000/v1")},
    google="g", anthropic="a", openai="o", openrouter="r",
)
GOOGLE, ANTHROPIC, OPENROUTER, VLLM = (
    PROVIDERS[name] for name in ("google", "anthropic", "openrouter", "local-vllm")
)


@pytest.fixture
def path(tmp_path):
    return tmp_path / "app" / "model_cache.json"


def ok(provider, *models: ListedModel, at: float = 100.0) -> Listing:
    return Listing(provider=provider.name, ok=True, models=models or (ListedModel("m"),), fetched_at=at)


def failed(provider, at: float = 200.0) -> Listing:
    return Listing(provider=provider.name, ok=False, models=(), fetched_at=at, error="HTTP 500")


def document(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def entry_of(path, name: str) -> dict:
    return document(path)["providers"][name]


# --- reading and writing one entry ---------------------------------------------------------


class TestRoundTrip:
    def test_a_missing_file_holds_nothing(self, path):
        assert ListingCache(path).read().entry("google", GOOGLE) is None

    def test_what_is_put_is_read_back_as_the_listing_it_was(self, path):
        listing = Listing(
            provider="anthropic", ok=True, fetched_at=1234.5,
            models=(
                ListedModel("claude-sonnet-4-5-20250929", display_name="Sonnet", context_window=1,
                            max_output=2, image_input=True, tools=False, reasoning=True,
                            thinking_types=("adaptive", "enabled"), effort_levels=("low", "high"),
                            can_disable=False, shown=False, alias="claude-sonnet-4-5"),
                ListedModel("claude-opus-4-7"),
            ),
        )
        cache = ListingCache(path)

        cache.put("anthropic", ANTHROPIC, listing, absent={"claude-sonnet-3-7"})
        entry = cache.read().entry("anthropic", ANTHROPIC)

        assert entry is not None
        assert entry.to_listing("anthropic") == listing
        assert entry.absent == ("claude-sonnet-3-7",)
        assert (entry.fetched_at, entry.ok, entry.error) == (1234.5, True, None)

    def test_the_file_is_version_3_private_and_has_the_fields_of_0_7_0(self, path):
        ListingCache(path).put("openrouter", OPENROUTER, ok(OPENROUTER))

        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert document(path)["version"] == CACHE_VERSION == 3
        assert entry_of(path, "openrouter") == {
            "fetched_at": 100.0, "ok": True, "error": None, "base_url": OPENROUTER_URL,
            "kind": "openrouter", "absent": [],
            "models": [{
                "id": "m", "display_name": "", "context_window": None, "max_output": None,
                "image_input": None, "tools": None, "reasoning": None, "thinking_types": [],
                "effort_levels": [], "can_disable": None, "shown": True, "alias": None,
            }],
        }

    def test_a_missing_folder_is_created(self, tmp_path):
        nested = tmp_path / "a" / "b" / "model_cache.json"

        ListingCache(nested).put("google", GOOGLE, ok(GOOGLE))

        assert nested.is_file()

    def test_without_a_path_nothing_is_read_or_written(self):
        cache = ListingCache(None)

        cache.put("google", GOOGLE, ok(GOOGLE))

        assert cache.read().entry("google", GOOGLE) is None

    def test_an_unwritable_path_never_raises(self, path):
        path.mkdir(parents=True)  # a folder where the file should be

        cache = ListingCache(path)
        cache.put("google", GOOGLE, ok(GOOGLE))

        assert cache.read().entry("google", GOOGLE) is None


class TestWhoseEntryItIs:
    def test_an_entry_describes_the_provider_of_its_address_and_kind_only(self, path):
        cache = ListingCache(path)
        cache.put("local-vllm", VLLM, ok(VLLM))
        moved = providers(specs={"local-vllm": ProviderSpec(base_url="http://other:8000/v1")})
        gateway = providers(
            specs={"local-vllm": ProviderSpec(kind="openrouter", base_url="http://gpu-box:8000/v1",
                                              api_key_env="K")}
        )

        contents = cache.read()

        assert contents.entry("local-vllm", VLLM) is not None
        assert contents.entry("local-vllm", moved["local-vllm"]) is None
        assert contents.entry("local-vllm", gateway["local-vllm"]) is None
        assert contents.entry("google", GOOGLE) is None


# --- a write keeps what it did not list -------------------------------------------------------


class TestPutKeepsTheRest:
    def test_every_other_entry_is_kept_as_the_file_holds_it(self, path):
        cache = ListingCache(path)
        cache.put("google", GOOGLE, ok(GOOGLE, at=1.0))
        cache.put("anthropic", ANTHROPIC, ok(ANTHROPIC, at=2.0))
        google = entry_of(path, "google")

        cache.put("openrouter", OPENROUTER, ok(OPENROUTER, at=3.0))

        assert entry_of(path, "google") == google
        assert set(document(path)["providers"]) == {"google", "anthropic", "openrouter"}

    def test_a_field_this_version_does_not_know_survives_where_it_is_carried(self, path):
        cache = ListingCache(path)
        cache.put("google", GOOGLE, ok(GOOGLE))
        raw = document(path)
        raw["providers"]["google"]["models"][0]["pricing"] = 1.0
        raw["providers"]["google"]["fetched_at"] = 7.0
        path.write_text(json.dumps(raw))

        cache.put("anthropic", ANTHROPIC, ok(ANTHROPIC))

        assert entry_of(path, "google") == raw["providers"]["google"]

    def test_a_cached_model_with_an_unknown_field_is_read_without_it(self, path):
        ListingCache(path).put("google", GOOGLE, ok(GOOGLE, ListedModel("gemini-2.5-flash")))
        raw = document(path)
        raw["providers"]["google"]["models"][0]["pricing"] = 1.0
        path.write_text(json.dumps(raw))

        entry = ListingCache(path).read().entry("google", GOOGLE)

        assert entry is not None
        assert entry.to_listing("google").models == (ListedModel("gemini-2.5-flash"),)

    @pytest.mark.parametrize("damaged", [
        {"fetched_at": "yesterday", "ok": True, "models": []},
        {"ok": True, "models": []},
        {"fetched_at": 1.0, "ok": 1, "models": []},
        {"fetched_at": 1.0, "ok": True, "models": [{"id": 5}]},
        {"fetched_at": 1.0, "ok": True, "models": [], "api_key": "k"},
        [1, 2],
        "text",
        None,
    ], ids=["fetched_at a string", "no fetched_at", "ok a number", "a model id a number",
            "an unknown entry field", "a list", "a string", "null"])
    def test_an_entry_that_does_not_parse_is_unread_and_carried(self, path, damaged):
        ListingCache(path).put("google", GOOGLE, ok(GOOGLE))
        raw = document(path)
        raw["providers"]["anthropic"] = damaged
        path.write_text(json.dumps(raw))
        cache = ListingCache(path)
        assert cache.read().entry("anthropic", ANTHROPIC) is None

        cache.put("openrouter", OPENROUTER, ok(OPENROUTER))

        assert document(path)["providers"]["anthropic"] == damaged
        assert entry_of(path, "google") == raw["providers"]["google"]

    def test_the_entry_a_provider_is_put_replaces_a_damaged_one_of_its_own(self, path):
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(
            {"version": 3, "providers": {"google": {"ok": "yes"}, "anthropic": {"ok": "yes"}}}
        ))

        ListingCache(path).put("google", GOOGLE, ok(GOOGLE))

        written = document(path)["providers"]
        assert written["google"]["ok"] is True
        assert written["anthropic"] == {"ok": "yes"}


class TestAFileItCannotUse:
    @pytest.mark.parametrize("content", [
        b"{not json", b"[1, 2, 3]", b"\xff\xfe\x00", b"", b"null",
        json.dumps({"providers": {}}).encode(),
        json.dumps({"version": 3}).encode(),
        json.dumps({"version": 3, "providers": []}).encode(),
        json.dumps({"version": "3", "providers": {}}).encode(),
        json.dumps({"version": 3.5, "providers": {}}).encode(),
        json.dumps({"version": True, "providers": {}}).encode(),
        json.dumps({"version": 2, "providers": {"google": {"fetched_at": 1.0, "ok": True}}}).encode(),
        json.dumps({"version": 1, "providers": {"google": {"fetched_at": 1.0, "ok": True}}}).encode(),
        json.dumps({"version": 0, "providers": {}}).encode(),
    ], ids=["not json", "a list", "not utf-8", "empty", "null", "no version", "no providers",
            "providers a list", "version a string", "version a float", "version a bool",
            "version 2", "version 1", "version 0"])
    def test_a_broken_or_older_file_is_ignored_and_rewritten_with_this_entry_alone(
        self, path, content
    ):
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
        cache = ListingCache(path)
        assert cache.read().entry("google", GOOGLE) is None

        cache.put("anthropic", ANTHROPIC, ok(ANTHROPIC))

        assert document(path)["version"] == 3
        assert set(document(path)["providers"]) == {"anthropic"}

    @pytest.mark.parametrize("version", [4, 5, 100])
    def test_a_file_of_a_newer_version_is_neither_used_nor_written(self, path, version):
        path.parent.mkdir(parents=True)
        newer = json.dumps({
            "version": version, "providers": {"google": {"fetched_at": 1.0, "ok": True,
                                                         "kind": "google", "models": []}},
        }).encode()
        path.write_bytes(newer)
        cache = ListingCache(path)

        assert cache.read().entry("google", GOOGLE) is None
        cache.put("google", GOOGLE, ok(GOOGLE))
        cache.put("anthropic", ANTHROPIC, failed(ANTHROPIC))

        assert path.read_bytes() == newer


class TestANewerFileIsSaidOnce:
    """A release that cannot use a newer cache lists at each start: it tells the user
    why, once, naming the file and both versions."""

    LOGGER = "agentic_cli.models.listing_cache"

    @staticmethod
    def newer(path, version: int = 4) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": version, "providers": {}}))

    def infos(self, caplog) -> list[str]:
        return [
            r.getMessage() for r in caplog.records
            if r.name == self.LOGGER and r.levelno == logging.INFO
        ]

    def test_one_line_at_info_whatever_is_read_or_written_after(self, path, caplog):
        self.newer(path)
        cache = ListingCache(path)

        with caplog.at_level(logging.INFO, logger=self.LOGGER):
            cache.read()
            cache.put("google", GOOGLE, ok(GOOGLE))
            cache.read()
            ListingCache(path).read()  # another catalog of the same process

        assert self.infos(caplog) == [
            f"The model cache {path} is of version 4, newer than this release's (3): "
            "this release does not use it, and lists the providers in use at each start"
        ]

    def test_another_version_of_the_file_is_said_again(self, path, caplog):
        self.newer(path, 5)
        with caplog.at_level(logging.INFO, logger=self.LOGGER):
            ListingCache(path).read()
            self.newer(path, 6)
            ListingCache(path).read()

        assert [line.split(" is of ")[1].split(",")[0] for line in self.infos(caplog)] == [
            "version 5", "version 6",
        ]

    @pytest.mark.parametrize("content", [
        json.dumps({"version": 3, "providers": {}}).encode(),
        json.dumps({"version": 2, "providers": {}}).encode(),
        b"{not json",
    ], ids=["this version", "an older one", "broken"])
    def test_a_file_this_release_can_use_or_replace_is_not_said(self, path, caplog, content):
        path.parent.mkdir(parents=True)
        path.write_bytes(content)

        with caplog.at_level(logging.INFO, logger=self.LOGGER):
            ListingCache(path).read()
            ListingCache(path).put("google", GOOGLE, ok(GOOGLE))

        assert self.infos(caplog) == []


# --- a failure never replaces a listing --------------------------------------------------------


class TestAFailure:
    def test_never_replaces_an_ok_entry_of_the_same_address_and_kind(self, path):
        cache = ListingCache(path)
        cache.put("google", GOOGLE, ok(GOOGLE, at=1.0))
        before = path.read_bytes()

        cache.put("google", GOOGLE, failed(GOOGLE), after_failure=True)

        assert path.read_bytes() == before

    def test_never_replaces_an_ok_entry_even_when_not_told_it_follows_a_failure(self, path):
        cache = ListingCache(path)
        cache.put("google", GOOGLE, ok(GOOGLE, at=1.0))
        before = path.read_bytes()

        cache.put("google", GOOGLE, failed(GOOGLE))

        assert path.read_bytes() == before

    def test_is_written_when_the_file_has_no_listing_for_the_provider(self, path):
        cache = ListingCache(path)

        cache.put("google", GOOGLE, failed(GOOGLE), after_failure=True)

        entry = entry_of(path, "google")
        assert (entry["ok"], entry["error"], entry["models"], entry["absent"]) == (
            False, "HTTP 500", [], [],
        )

    def test_replaces_a_failure(self, path):
        cache = ListingCache(path)
        cache.put("google", GOOGLE, failed(GOOGLE, at=1.0), after_failure=True)

        cache.put("google", GOOGLE, failed(GOOGLE, at=2.0), after_failure=True)

        assert entry_of(path, "google")["fetched_at"] == 2.0

    def test_replaces_a_listing_of_another_address(self, path):
        cache = ListingCache(path)
        cache.put("local-vllm", VLLM, ok(VLLM, at=1.0))
        moved = providers(specs={"local-vllm": ProviderSpec(base_url="http://other:8000/v1")})

        cache.put("local-vllm", moved["local-vllm"], failed(moved["local-vllm"]), after_failure=True)

        assert entry_of(path, "local-vllm")["ok"] is False

    def test_an_earlier_listing_kept_after_a_failure_is_written_when_the_file_lost_it(self, path):
        cache = ListingCache(path)

        cache.put("google", GOOGLE, ok(GOOGLE, at=1.0), after_failure=True)

        assert entry_of(path, "google")["ok"] is True

    def test_an_earlier_listing_kept_after_a_failure_does_not_replace_the_files_own(self, path):
        cache = ListingCache(path)
        cache.put("google", GOOGLE, ok(GOOGLE, at=5.0))

        cache.put("google", GOOGLE, ok(GOOGLE, at=1.0), after_failure=True)

        assert entry_of(path, "google")["fetched_at"] == 5.0

    def test_a_success_replaces_an_ok_entry(self, path):
        cache = ListingCache(path)
        cache.put("google", GOOGLE, ok(GOOGLE, at=1.0))

        cache.put("google", GOOGLE, ok(GOOGLE, at=2.0))

        assert entry_of(path, "google")["fetched_at"] == 2.0


# --- what is absent ---------------------------------------------------------------------------


class TestAbsent:
    def test_is_this_catalog_s_ids_joined_to_the_files_for_the_same_listing(self, path):
        cache = ListingCache(path)
        cache.put("anthropic", ANTHROPIC, ok(ANTHROPIC), absent={"claude-a"})

        cache.put("anthropic", ANTHROPIC, ok(ANTHROPIC), absent={"claude-b"})

        assert entry_of(path, "anthropic")["absent"] == ["claude-a", "claude-b"]

    def test_an_id_the_listing_serves_is_not_absent(self, path):
        cache = ListingCache(path)
        cache.put("anthropic", ANTHROPIC, ok(ANTHROPIC, ListedModel("old")), absent={"claude-a", "new"})

        cache.put(
            "anthropic", ANTHROPIC,
            ok(ANTHROPIC, ListedModel("new"), ListedModel("dated-1", alias="the-alias", shown=False)),
            absent={"the-alias", "dated-1", "claude-b"},
        )

        assert entry_of(path, "anthropic")["absent"] == ["claude-a", "claude-b"]

    def test_what_the_catalog_serves_replaces_the_ids_and_aliases_the_listing_names(self, path):
        """A dated id's alias in the listing is no served id when the catalog does not
        accept it (the table names no such family): the file's record of it stays."""
        cache = ListingCache(path)
        listing = ok(
            ANTHROPIC, ListedModel("claude-3-haiku-20240307", alias="claude-3-haiku", shown=False),
        )
        cache.put("anthropic", ANTHROPIC, ok(ANTHROPIC), absent={"claude-3-haiku", "claude-old"})

        cache.put(
            "anthropic", ANTHROPIC, listing, absent={"claude-x"},
            served={"claude-3-haiku-20240307", "claude-old"},
        )

        assert entry_of(path, "anthropic")["absent"] == ["claude-3-haiku", "claude-x"]

    def test_the_files_record_of_another_address_is_not_joined(self, path):
        cache = ListingCache(path)
        cache.put("local-vllm", VLLM, ok(VLLM), absent={"other-box-model"})
        moved = providers(specs={"local-vllm": ProviderSpec(base_url="http://other:8000/v1")})

        cache.put("local-vllm", moved["local-vllm"], ok(moved["local-vllm"]), absent={"mine"})

        assert entry_of(path, "local-vllm")["absent"] == ["mine"]

    def test_the_record_of_a_failed_entry_is_not_joined_and_a_failure_records_none(self, path):
        cache = ListingCache(path)
        cache.put("google", GOOGLE, failed(GOOGLE), absent={"x"}, after_failure=True)
        assert entry_of(path, "google")["absent"] == []

        cache.put("google", GOOGLE, ok(GOOGLE), absent={"y"})

        assert entry_of(path, "google")["absent"] == ["y"]

    def test_an_absent_list_in_the_file_that_is_not_a_list_of_strings_makes_the_entry_unread(self, path):
        cache = ListingCache(path)
        cache.put("google", GOOGLE, ok(GOOGLE))
        raw = document(path)
        raw["providers"]["google"]["absent"] = "claude-a"
        path.write_text(json.dumps(raw))
        assert cache.read().entry("google", GOOGLE) is None

        cache.put("google", GOOGLE, ok(GOOGLE), absent={"y"})

        assert entry_of(path, "google")["absent"] == ["y"]
