"""The listing cache: ``~/.<app>/model_cache.json`` in the workflow.

One entry per provider: fetch time, outcome, address, kind, the models (each
with whether it is shown) and the entries' models the listing lacked
(``absent``), never a key. ``ListingCache`` reads the file and writes one
provider's entry at a time; what a cached listing means (fresh, stale, listed
again) is the catalog's.

Several catalogs share the file: two terminals, two projects of one app, two
releases of one app side by side. So a write reads the file as it is then and
changes one provider's entry only:

- every other entry is carried over as the file holds it, **including one
  this version cannot read** (a field it does not know, a shape it does not
  expect): each entry is validated on its own, and one that does not parse is
  left out of what the catalog reads and kept as it is on write;
- a model's fields this version does not know are ignored when reading
  (``CachedModel``), so a release can add one without a version bump (an
  entry's own fields are strict: it cannot be misread);
- a file of a **newer** version is neither read nor written: this catalog
  lists as if there were none and keeps its listings in memory, so the newer
  release's entries are not lost (it says so once, at info: the user sees why
  every start lists). A broken file, or one of an older version (its models
  meant something else), is ignored and rewritten;
- a failed listing never replaces a listing of the provider's address and
  kind, and ``absent`` is the file's record of that listing joined to this
  catalog's (two projects name different retired models), less every id the
  listing now serves.

A format an older release cannot read goes under a new file name, so each
release keeps its own cache (see ``CACHE_VERSION``).

The format (``CACHE_VERSION``) is unchanged: a file this module writes is one
the reader it replaced accepts, and this module reads what that one wrote.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import tempfile
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentic_cli.models.discovery import ListedModel, Listing
from agentic_cli.models.providers import Provider

logger = logging.getLogger(__name__)

#: Version 3 holds every model a provider serves, and which are shown
#: (version 2 held only those shown, so it rejected the others while fresh);
#: a file of an older version is listed again, one of a newer version is left
#: as it is. A format an older release cannot read goes under a new file name
#: (``model_cache.v4.json``, say), so each release keeps its own cache and
#: none rewrites another's. Within a version a field is added only to a cached
#: model (``CachedModel`` ignores what it does not know): a field added to an
#: entry makes the entry unreadable to the releases before it.
CACHE_VERSION = 3

#: The (file, version) pairs of newer caches already said to be unusable: once
#: per process, though a manager builds a catalog at every reinitialize.
_NEWER_SAID: set[tuple[str, int]] = set()


# --- the schema: strict, so a damaged entry is ignored, not misread ---------------------------


class CachedModel(BaseModel):
    """A ``ListedModel``, as written. A field this version does not know is ignored."""

    model_config = ConfigDict(strict=True, extra="ignore", frozen=True)

    id: str = Field(min_length=1)
    display_name: str = ""
    context_window: int | None = None
    max_output: int | None = None
    image_input: bool | None = None
    tools: bool | None = None
    reasoning: bool | None = None
    thinking_types: tuple[str, ...] = ()
    effort_levels: tuple[str, ...] = ()
    can_disable: bool | None = None
    shown: bool = True
    alias: str | None = Field(default=None, min_length=1)


class CacheEntry(BaseModel):
    """One provider's cached listing."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    fetched_at: float
    ok: bool
    error: str | None = None
    base_url: str | None = None
    #: The provider's kind (``ProviderKind``'s value): another kind lists again.
    kind: str | None = None
    models: tuple[CachedModel, ...] = ()
    #: The entries' models the listing lacked (``ModelCatalog``'s ``absent``).
    absent: tuple[str, ...] = ()

    def describes(self, provider: Provider) -> bool:
        """Whether this is a listing of ``provider``: of its address and kind."""
        return self.base_url == provider.base_url and self.kind == provider.kind.value

    def to_listing(self, provider: str) -> Listing:
        """The listing this entry holds."""
        return Listing(
            provider=provider,
            ok=self.ok,
            models=tuple(ListedModel(**model.model_dump()) for model in self.models),
            fetched_at=self.fetched_at,
            error=self.error,
        )


def _parse(raw: Any) -> CacheEntry | None:
    """A cache entry from its JSON value; None when it does not parse.

    Validated as JSON text, where strict mode takes an array for a tuple.
    """
    try:
        return CacheEntry.model_validate_json(json.dumps(raw))
    except ValidationError as exc:
        logger.debug(
            "Ignoring a model cache entry: %d problem(s), first: %s",
            exc.error_count(),
            exc.errors(include_input=False)[0]["msg"],
        )
        return None


@dataclass(frozen=True)
class CacheContents:
    """The entries a cache file held when it was read: those that parse."""

    entries: Mapping[str, CacheEntry]

    def entry(self, name: str, provider: Provider) -> CacheEntry | None:
        """``name``'s entry when it is a listing of ``provider`` (its address
        and kind), else None."""
        found = self.entries.get(name)
        return found if found is not None and found.describes(provider) else None


class ListingCache:
    """The cache file, read whole and written one provider's entry at a time.

    Args:
        path: The file; None keeps nothing (every read is empty, every write
            a no-op).
    """

    def __init__(self, path: Path | None) -> None:
        self._path = Path(path) if path is not None else None

    def read(self) -> CacheContents:
        """The file's entries that parse; empty for a missing, unreadable,
        broken, older or newer file."""
        raw, _ = self._load()
        parsed = {name: _parse(value) for name, value in raw.items()}
        return CacheContents({name: entry for name, entry in parsed.items() if entry is not None})

    def put(
        self,
        name: str,
        provider: Provider,
        listing: Listing,
        absent: Iterable[str] = (),
        *,
        after_failure: bool = False,
        served: Collection[str] | None = None,
    ) -> None:
        """Write ``name``'s entry, keeping every other entry as the file holds
        it now; a failure is logged, never raised, and a file of a newer
        version is left alone.

        A failure never replaces a listing: when the file holds an ok entry
        for ``provider``'s address and kind (another catalog may have written
        it since this one read the file), a failed ``listing`` is not written,
        nor is the earlier one a catalog keeps after a failed attempt
        (``after_failure``).

        Args:
            name: The provider's name in the file.
            provider: The provider, for the address and kind written.
            listing: The listing to write: the new one, or the failure.
            absent: The entries' models this catalog's listing lacked. The
                entry holds them with the file's own record of the same
                address and kind, less every id the listing serves (a model
                a provider serves again is no longer absent).
            after_failure: ``listing`` follows a failed attempt: the failure
                itself, or the listing kept through it.
            served: Every id the catalog accepts from ``listing``; by default
                the ids it lists and each model's alias. A catalog passes its
                own, since it decides which of the aliases are served.
        """
        if self._path is None:
            return
        entries, writable = self._load()
        if not writable:
            logger.debug("Not writing the model cache %s: a newer version's", self._path)
            return
        held = _parse(entries[name]) if name in entries else None
        held_here = held if held is not None and held.ok and held.describes(provider) else None
        if held_here is not None and (after_failure or not listing.ok):
            return
        entries[name] = self._written(provider, listing, absent, held_here, served)
        try:
            _atomic_write_json(self._path, {"version": CACHE_VERSION, "providers": entries})
        except OSError as exc:
            logger.debug("Could not write the model cache %s: %s", self._path, exc)

    def _load(self) -> tuple[dict[str, Any], bool]:
        """The file's entries as JSON, and whether this version may write it.

        The entries are those of a file of this version; a missing, unreadable,
        broken or older file gives none, and may be written (replaced), a
        newer one gives none and may not.
        """
        if self._path is None:
            return {}, True
        try:
            text = self._path.read_bytes()
        except FileNotFoundError:
            return {}, True
        except OSError as exc:
            logger.debug("Ignoring the model cache %s: %s", self._path, exc)
            return {}, True
        try:
            document = json.loads(text)
        except (ValueError, RecursionError):
            logger.debug("Ignoring the model cache %s: not JSON", self._path)
            return {}, True
        if not isinstance(document, dict):
            logger.debug("Ignoring the model cache %s: not an object", self._path)
            return {}, True
        version, providers = document.get("version"), document.get("providers")
        if type(version) is int and version > CACHE_VERSION:
            if (str(self._path), version) not in _NEWER_SAID:
                _NEWER_SAID.add((str(self._path), version))
                logger.info(
                    "The model cache %s is of version %d, newer than this release's (%d): "
                    "this release does not use it, and lists the providers in use at each start",
                    self._path,
                    version,
                    CACHE_VERSION,
                )
            return {}, False
        if type(version) is not int or version != CACHE_VERSION or not isinstance(providers, dict):
            logger.debug("Ignoring the model cache %s: not a version %d file", self._path, CACHE_VERSION)
            return {}, True
        return providers, True

    @staticmethod
    def _written(
        provider: Provider,
        listing: Listing,
        absent: Iterable[str],
        held: CacheEntry | None,
        served: Collection[str] | None,
    ) -> dict[str, Any]:
        """An entry as written."""
        if listing.ok:
            if served is None:
                served = {model.id for model in listing.models}
                served.update(model.alias for model in listing.models if model.alias)
            joined = {*absent, *(held.absent if held is not None else ())} - set(served)
        else:
            joined = set()
        return {
            "fetched_at": listing.fetched_at,
            "ok": listing.ok,
            "error": listing.error,
            "base_url": provider.base_url,
            "kind": provider.kind.value,
            "models": [dataclasses.asdict(model) for model in listing.models],
            "absent": sorted(joined),
        }


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write ``data`` as JSON through a private (0600) temp file and a rename.

    The same contract as ``agentic_cli.file_utils.atomic_write_json``; the
    package keeps its own copy so it imports nothing from agentic-cli.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=1)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
