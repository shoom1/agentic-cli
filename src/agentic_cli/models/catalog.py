"""The catalog: every model the workflow can name, and what is known about each.

Three sources, highest precedence first; each value in a record keeps the name
of its source (``ModelRecord.sources``):

1. the ``models`` entries (``config``): what an app, a user or a project says
   about a model, field by field;
2. live listings (``listing``): Google's token limits and thinking flag,
   Anthropic's limits, thinking types, effort levels and whether thinking
   can be turned off, OpenAI's ids, an OpenRouter provider's limits and
   tool and reasoning support, an OpenAI-compatible provider's served names
   and context length;
3. the shipped table (``table``): family rules, tiers, the curated OpenAI
   list and the fallback models used without a listing.

**The providers in use** (``in_use``), known before any listing: the
providers of the ``models`` entries that can run (defined, with
credentials); the providers of the models a setting, an agent or the manager
names (``references``); and every built-in provider with a key when no entry
can run, or when a role is computed (``computed_roles``) that no entry can
fill (a built-in model's tier comes from the table's family rules). ``load``
asks the last two rules again once the listings have rejected what they
lack, and lists the built-ins they then need. Only these are listed at
startup and refreshed in the background; one whose ``list_models`` is False
is in use but never listed. ``search`` lists any other provider with
credentials the first time it is searched, and any whose listing is not
fresh.

**The list** (``offered``) is what the model pickers offer: the entries that
can run, then the models the references and the caller name, each once; when
no entry can run, each built-in provider's newest fast, default and strong
models, of those it shows. Listings never add to it. ``not_offered`` says why
an entry is left out, and ``load`` warns, once per process, of an entry whose
provider is not defined or whose listing lacks its model.

**Served and shown.** A listing holds every model the provider serves, and
the catalog accepts each of their ids (``record()``, ``search``). A provider
shows fewer (``records()``, the tier picks above, the models an "is not
available" message names, the status count): Google its text models, without
aliases, snapshots or variants; Anthropic each dated id as its alias, which
takes the first one listed's facts; OpenAI its curated and reasoning models.
``listing_records.records_for`` makes both maps, each record with its entry.

**Validation.** An unknown id is rejected only when its provider's listing is
fresh (``ProviderState.FRESH``); a stale, failed or untried listing cannot
prove a model does not exist, so the id is accepted with the table's record.
A provider's state is derived each time it is asked, from its listing's age,
its last failure and the clock: a listing that outlives the TTL in a long
session is stale, whether the cache or this process made it.
A fresh listing read from the cache can predate a model, so before it
rejects one the provider is listed again, once: while loading for an entry's
model, through ``confirm_rejections`` for a referenced one. A listing records
the entries' models it lacked (``absent``, in the cache too), and a later
start trusts that record: a model a provider retired costs one listing, not
one at every start, whichever project's list names it.

**Cache.** Listings are kept in a JSON file (``~/.<app>/model_cache.json`` in
the workflow; ``listing_cache.ListingCache`` reads and writes it), one entry
per provider: fetch time, outcome, address, kind, models (each with whether
it is shown) and the entries' models it lacked, never a key. Startup reads it
and waits on the network only for a provider in use that has credentials and
no entry for its address and kind; a fresh entry needs no listing, and an
older one (or a cached failure) is used while ``start_background_refresh``
lists it again. A broken file, one from an older version, an entry this
version cannot read, or an entry for another address or kind is ignored and
listed again; a file of a newer version is not used, and never written (the
listings stay in memory). A listing writes its own provider's entry and
leaves every other entry as the file then holds it, one this version cannot
read included: several catalogs share the file (two terminals, two projects
of one app, two releases of it), and a catalog's own copy of an entry may be
older than the file's. A failed listing never replaces a listing of the
provider's address and kind, and the models recorded absent are the file's
and this catalog's together, less those the listing serves.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType

import httpx

from agentic_cli.models.discovery import DEFAULT_TIMEOUT, Listing, list_models
from agentic_cli.models.entries import ModelEntry
from agentic_cli.models.listing_cache import (  # noqa: F401 - CACHE_VERSION stays importable from here
    CACHE_VERSION,
    CacheContents,
    CacheEntry,
    ListingCache,
)
from agentic_cli.models.listing_records import records_for, with_entry
from agentic_cli.models.providers import Provider
from agentic_cli.models.records import ModelRecord
from agentic_cli.models.refs import BUILTIN_PROVIDERS, ModelRef, ModelSpecError, parse_ref
from agentic_cli.models.resolve import ModelResolutionError
from agentic_cli.models.table import ShippedTable, load_table, newest

logger = logging.getLogger(__name__)

#: Seconds a listing stays fresh.
DEFAULT_TTL = 86400.0
# The tiers the list offers of each built-in provider when no entry can run.
_TIERS = ("fast", "default", "strong")
# How many models an "is not available" message names.
_MESSAGE_MODELS = 20
#: The (reference, reason) pairs already warned of as not offered. Once per
#: process: a manager builds a new catalog at every reinitialize, and /models
#: one while the workflow starts, and the user was told already.
_WARNED_ENTRIES: set[tuple[str, str]] = set()


class ProviderState(str, Enum):
    """How much the catalog knows about one provider's models.

    - ``UNATTEMPTED``: not listed (not in use, no credentials, never listed,
      or not loaded yet); the table's fallback models stand in.
    - ``FRESH``: a listing under the TTL old, from the network or the cache.
      Authoritative: an id it lacks is rejected.
    - ``STALE``: an older listing from the cache, in use until a refresh
      replaces it. Not authoritative.
    - ``DEGRADED``: the last listing failed. An earlier listing is kept when
      there was one, else the fallback models stand in. Not authoritative.
    """

    UNATTEMPTED = "unattempted"
    FRESH = "fresh"
    STALE = "stale"
    DEGRADED = "degraded"


@dataclass(frozen=True)
class ProviderStatus:
    """One provider's listing, for the UI.

    Attributes:
        provider: The provider's name.
        has_credentials: Whether requests can be sent (a key, or a configured
            provider that needs none).
        state: How much is known (see ``ProviderState``).
        fetched_at: When the listing in use was fetched, else when the last
            attempt failed; None when never listed.
        model_count: How many models the catalog offers for it.
        error: Why the last listing failed, if it did.
    """

    provider: str
    has_credentials: bool
    state: ProviderState
    fetched_at: float | None
    model_count: int
    error: str | None = None


@dataclass
class _ProviderState:
    """What the catalog knows of one provider's models."""

    provider: Provider
    #: The last successful listing; kept through a failed refresh.
    listing: Listing | None = None
    #: The last attempt, while it is a failure.
    failure: Listing | None = None
    #: A cached entry that a background refresh should replace.
    due: bool = False
    #: The listing was read from the cache, not listed by this catalog.
    from_cache: bool = False
    #: The entries' models the listing lacked when it was made: a cached
    #: listing that lacks one of these is not listed again for it.
    absent: frozenset[str] = frozenset()
    #: Every id the provider serves, which ``record()`` accepts and ``search``
    #: finds; and the records it shows, which ``records()`` offers.
    accepted: Mapping[str, ModelRecord] = field(default_factory=dict)
    offered: tuple[ModelRecord, ...] = ()

    @property
    def listable(self) -> bool:
        """Whether a listing can be asked for: credentials, and ``list_models``."""
        return self.provider.has_credentials and self.provider.list_models


class ModelCatalog:
    """Records for every provider's models, from entries, listings and the table.

    Args:
        providers: Every provider, built in and configured, by name
            (``build_providers``).
        entries: The ``models`` entries, by reference, in the list's order:
            each wins, field by field, over its model's listing and table
            values.
        references: The models a setting, an agent or the manager names:
            their providers are in use, and the list offers them.
        computed_roles: The built-in roles (``default``, ``fast``,
            ``strong``) nothing maps, which take a model of their tier: from
            the entries, else from the built-in providers with a key, which
            are then in use too.
        cache_path: The listing cache; None keeps listings in memory only.
        table: The shipped table (the packaged one by default).
        ttl_seconds: How long a listing stays fresh.
        transport: The HTTP transport for listings (a fake one in tests); the
            caller keeps ownership.
        timeout: Seconds one provider's listing may take.
        clock: Gives fetch times and ages.
    """

    def __init__(
        self,
        *,
        providers: Mapping[str, Provider],
        entries: Mapping[ModelRef, ModelEntry] | None = None,
        references: Iterable[ModelRef] = (),
        computed_roles: Iterable[str] = (),
        cache_path: Path | None = None,
        table: ShippedTable | None = None,
        ttl_seconds: float = DEFAULT_TTL,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._cache = ListingCache(cache_path)
        self._table = table or load_table()
        self._ttl = ttl_seconds
        self._transport = transport
        self._timeout = timeout
        self._clock = clock
        self._entries: Mapping[ModelRef, ModelEntry] = MappingProxyType(dict(entries or {}))
        self._references = tuple(dict.fromkeys(references))
        self._states = {name: _ProviderState(provider) for name, provider in providers.items()}
        for state in self._states.values():
            self._rebuild(state)
        self._computed_roles = tuple(computed_roles)
        self._in_use = self._providers_in_use()
        # Providers a search listed (or read from the cache), beside those in use.
        self._searched: set[str] = set()
        self._refresh_task: asyncio.Task[None] | None = None
        self._closed = False
        # Providers whose cached listing rejected an id (``confirm_rejections``).
        self._cached_rejections: set[str] = set()

    @property
    def in_use(self) -> tuple[str, ...]:
        """The providers in use, in catalog order (the module docstring says
        which): known from the settings, and widened by ``load``."""
        return self._in_use

    # --- loading and refreshing --------------------------------------------

    async def load(self) -> None:
        """Read the cache, then list the providers in use it cannot answer for.

        Each provider in use that can be listed and has no cache entry for
        its address and kind is listed now, concurrently, each within the
        timeout: the first run still validates ids. A cached listing that
        lacks an entry's model is listed again, once, unless it recorded
        that model as absent. When the listings leave no entry that can run,
        or none of a computed role's tier, the built-in providers with a key
        are in use after all, and those not yet loaded are. Then each entry
        left out of the list is warned of, once per process, unless only its
        key is missing (``not_offered`` says that). Never raises for a
        network or cache problem.
        """
        cached = self._cache.read()
        to_list: list[str] = []
        for name, state in self._states.items():
            if name not in self._in_use or not state.listable:
                continue
            entry = cached.entry(name, state.provider)
            if entry is None:
                to_list.append(name)
            else:
                self._adopt(state, entry)
        await self._list_now(to_list)
        # A listing read from the cache can predate a model an entry names,
        # unless it was made knowing the entry and lacked it then.
        await self._list_now(
            ref.provider
            for ref in self._entries
            if self._rejected(ref)
            and self._states[ref.provider].from_cache
            and ref.model not in self._states[ref.provider].absent
        )
        await self._load_builtins_now_needed(cached)
        self._warn_of_entries_left_out()

    def start_background_refresh(self) -> "asyncio.Task[None] | None":
        """List again, in a task, each provider in use or searched whose cached
        listing is not fresh.

        Each result is swapped in as it arrives. Returns the task, or None
        when nothing is due or the catalog is closed. ``aclose()`` cancels it.
        """
        if self._closed:
            return None
        if self._refresh_task is not None and not self._refresh_task.done():
            return self._refresh_task
        due = [
            name
            for name in self._listed_names()
            if self._states[name].due and self._states[name].listable
        ]
        if not due:
            return None
        self._refresh_task = asyncio.get_running_loop().create_task(
            self._refresh_in_background(due), name="model-catalog-refresh"
        )
        return self._refresh_task

    async def refresh(self, providers: Iterable[str] | None = None) -> list[ProviderStatus]:
        """List ``providers`` again now: by default, those in use or searched.

        Providers without credentials, or never listed (``list_models``), are
        not listed. A failed listing keeps the earlier one and marks the
        provider degraded.

        Returns:
            The status of each provider asked about, in catalog order by
            default.
        """
        if providers is None:
            names = self._listed_names()
        else:
            names = [name for name in dict.fromkeys(providers) if name in self._states]
        await asyncio.gather(*(self._list(name) for name in names if self._states[name].listable))
        return [self._status(name) for name in names]

    async def confirm_rejections(self) -> bool:
        """List again each provider whose cached listing rejected an id.

        A listing read from the cache can predate a model the provider now
        serves (a release, a plan upgrade, another account's key). A caller
        about to refuse to start lists those providers again, once, and asks
        ``record()`` again: the new listing decides, and a failed one accepts
        the id, as any failed listing does.

        Returns:
            Whether any provider was listed again.
        """
        names = [name for name in self._cached_rejections if self._states[name].from_cache]
        self._cached_rejections.clear()
        if not names:
            return False
        await self._list_now(names)
        return True

    async def search(self, text: str) -> list[ModelRecord]:
        """Every model a provider with credentials lists whose reference
        contains ``text``, ignoring case: how a user finds an exact id.

        A provider not listed yet (one not in use) is read from the cache
        when it holds a listing for its address and kind, and is searched
        from then on: ``status``, ``refresh`` and the background refresh
        cover it too. Any provider whose listing is not fresh then (stale,
        failed, or never made) is listed now, so a search never answers from
        an old listing. A listing never adds a model to the list
        (``offered``).

        Returns:
            The matches, by provider in catalog order, then by model id.
        """
        listable = [name for name, state in self._states.items() if state.listable]
        unlisted = [
            n for n in listable if self._state_of(self._states[n]) is ProviderState.UNATTEMPTED
        ]
        if unlisted:
            cached = self._cache.read()
            for name in unlisted:
                state = self._states[name]
                entry = cached.entry(name, state.provider)
                if entry is not None:
                    self._adopt(state, entry)
            self._searched.update(unlisted)
        stale = [n for n in listable if self._state_of(self._states[n]) is not ProviderState.FRESH]
        await asyncio.gather(*(self._list(name) for name in stale))
        needle = text.casefold()
        found: list[ModelRecord] = []
        for state in self._states.values():
            if state.provider.has_credentials:
                matches = [r for r in state.accepted.values() if needle in str(r.ref).casefold()]
                found.extend(sorted(matches, key=lambda r: r.model))
        return found

    async def aclose(self) -> None:
        """Cancel and await the background refresh. Idempotent."""
        self._closed = True
        task, self._refresh_task = self._refresh_task, None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.wait({task})

    # --- what the catalog knows ---------------------------------------------

    def record(self, ref: ModelRef) -> ModelRecord:
        """The model's record, with its ``models`` entry when it has one.

        Raises:
            ModelResolutionError: If the provider is neither built in nor
                configured, or its fresh listing lacks the model (the message
                lists the provider's models).
        """
        state = self._states.get(ref.provider)
        if state is None:
            raise ModelResolutionError(
                f"Model {ref} names {ref.provider!r}, which is neither a built-in provider "
                "nor defined in model_providers."
            )
        found = state.accepted.get(ref.model)
        if found is not None:
            return found
        if self._state_of(state) is ProviderState.FRESH:
            if state.from_cache:
                self._cached_rejections.add(ref.provider)
            models = [record.model for record in state.offered]
            shown = ", ".join(models[:_MESSAGE_MODELS]) or "none"
            if len(models) > _MESSAGE_MODELS:
                shown += f", and {len(models) - _MESSAGE_MODELS} more"
            raise ModelResolutionError(
                f"Model {ref} is not available. Available {ref.provider} models: {shown}"
            )
        record = self._table.record_for(ref.provider, state.provider.kind, ref.model)
        return with_entry(record, self._entries.get(ref))

    def records(self, provider: str) -> list[ModelRecord]:
        """The models offered for ``provider``: those its listing shows, else
        the fallbacks."""
        found = self._states.get(provider)
        return list(found.offered) if found is not None else []

    def is_authoritative(self, provider: str) -> bool:
        """Whether ``provider``'s listing can reject an unknown id (FRESH only)."""
        found = self._states.get(provider)
        return found is not None and self._state_of(found) is ProviderState.FRESH

    def status(self) -> list[ProviderStatus]:
        """The status of each provider in use or searched, in catalog order
        (the built-ins first, then the configured ones)."""
        return [self._status(name) for name in self._listed_names()]

    def offered(self, extra: Iterable[str] = ()) -> list[str]:
        """The list: the references the model pickers offer, each once, where
        it first comes.

        1. The ``models`` entries that can run, in their order (``not_offered``
           says why the others cannot).
        2. The references, then ``extra`` (the roles' models, say), that can
           run: a provider with credentials, and a model its fresh listing
           does not lack. A string that is no reference is skipped.
        3. Only when no entry can run: each built-in provider with a key, in
           the order Google, Anthropic, OpenAI, its newest fast, default and
           strong model, from its listing or else the table's fallback models.
        """
        chosen = [ref for ref in self._entries if self._entry_left_out(ref) is None]
        tiers = not chosen
        chosen += [ref for ref in (*self._references, *_parsed(extra)) if self._runs(ref)]
        if tiers:
            chosen += self._tier_models()
        return [str(ref) for ref in dict.fromkeys(chosen)]

    def not_offered(self) -> dict[str, str]:
        """The ``models`` entries the list leaves out, each with why: its
        provider is not defined, has no key, or does not serve it (a fresh
        listing lacks it, or the listing in use recorded it as absent)."""
        left_out = ((ref, self._entry_left_out(ref)) for ref in self._entries)
        return {str(ref): reason for ref, reason in left_out if reason is not None}

    def entry_records(self) -> list[ModelRecord]:
        """The records of the ``models`` entries the list offers, in its order."""
        return [self.record(ref) for ref in self._entries if self._entry_left_out(ref) is None]

    # --- the providers in use and the list ------------------------------------

    def _providers_in_use(self) -> tuple[str, ...]:
        """The providers in use, from the settings alone (no listing)."""
        can_run = [ref for ref in self._entries if self._can_run(ref.provider)]
        used = {ref.provider for ref in can_run}
        used.update(ref.provider for ref in self._references if ref.provider in self._states)
        if self._needs_builtins(can_run):
            used.update(self._keyed_builtins())
        return tuple(name for name in self._states if name in used)

    def _needs_builtins(self, entries: list[ModelRef]) -> bool:
        """Whether the built-in providers with a key are in use, given the
        entries that can run: none can, or a role is computed that none of
        them can fill (none is of its tier)."""
        tiers = {self._tier(ref) for ref in entries}
        return not entries or any(role not in tiers for role in self._computed_roles)

    def _tier(self, ref: ModelRef) -> str | None:
        """A model's tier, from the table's family rules: a built-in model's only."""
        kind = self._states[ref.provider].provider.kind
        return self._table.record_for(ref.provider, kind, ref.model).tier

    def _keyed_builtins(self) -> list[str]:
        return [
            name
            for name, state in self._states.items()
            if state.provider.builtin and state.provider.has_credentials
        ]

    async def _load_builtins_now_needed(self, cached: CacheContents) -> None:
        """Ask the built-ins' rules again with the entries the listings left
        offered, and load the built-ins they now need: from the cache when it
        holds their listing, else listed once (a retired entry may have been
        the only one that could run, or the one of a computed role's tier)."""
        offered = [ref for ref in self._entries if self._entry_left_out(ref) is None]
        if not self._needs_builtins(offered):
            return
        needed = [name for name in self._keyed_builtins() if name not in self._in_use]
        if not needed:
            return
        self._in_use = tuple(n for n in self._states if n in self._in_use or n in needed)
        to_list: list[str] = []
        for name in needed:
            state = self._states[name]
            entry = cached.entry(name, state.provider)
            if entry is None:
                to_list.append(name)
            else:
                self._adopt(state, entry)
        await self._list_now(to_list)

    def _listed_names(self) -> list[str]:
        """The providers in use or searched, in catalog order."""
        return [name for name in self._states if name in self._in_use or name in self._searched]

    def _can_run(self, provider: str) -> bool:
        state = self._states.get(provider)
        return state is not None and state.provider.has_credentials

    def _runs(self, ref: ModelRef) -> bool:
        """Whether ``ref`` can run: a provider with credentials whose fresh
        listing, if it has one, serves the model (``record()`` would not raise)."""
        return self._can_run(ref.provider) and not self._rejected(ref)

    def _rejected(self, ref: ModelRef) -> bool:
        """Whether a fresh listing of ``ref``'s provider lacks the model."""
        state = self._states.get(ref.provider)
        return (
            state is not None
            and self._state_of(state) is ProviderState.FRESH
            and ref.model not in state.accepted
        )

    def _entry_left_out(self, ref: ModelRef) -> str | None:
        """Why the list leaves out the entry ``ref``; None when it offers it."""
        state = self._states.get(ref.provider)
        if state is None:
            return f"provider {ref.provider!r} is not defined in model_providers"
        if not state.provider.has_credentials:
            return f"no key: set {state.provider.key_env}"
        if self._rejected(ref) or self._known_absent(ref):
            return f"not served by {ref.provider}"
        return None

    def _known_absent(self, ref: ModelRef) -> bool:
        """Whether the listing in use, though not fresh (a stale one, or one
        kept through a failed refresh), recorded the entry ``ref`` as absent.

        It cannot reject an id (``record()`` still accepts it), but the list
        leaves the entry out: a role computed from the list must not run on a
        model the provider has retired while a refresh is under way.
        """
        state = self._states[ref.provider]
        return (
            self._state_of(state) is not ProviderState.FRESH
            and state.listing is not None
            and ref.model in state.absent
        )

    def _tier_models(self) -> list[ModelRef]:
        """Each built-in provider's newest model of each tier, for a list with no entry."""
        models: list[ModelRef] = []
        for name in BUILTIN_PROVIDERS:
            if self._can_run(name):
                for tier in _TIERS:
                    candidates = [r for r in self._states[name].offered if r.tier == tier]
                    if candidates:
                        models.append(newest(candidates).ref)
        return models

    def _warn_of_entries_left_out(self) -> None:
        """One warning per entry the list leaves out because its provider is
        not defined or does not serve it, once per process. None for a
        missing key: an app's list names models of providers a user may hold
        no key for, and ``not_offered`` says so."""
        for ref in self._entries:
            state = self._states.get(ref.provider)
            if state is not None and not state.provider.has_credentials:
                continue
            reason = self._entry_left_out(ref)
            if reason is not None and (str(ref), reason) not in _WARNED_ENTRIES:
                _WARNED_ENTRIES.add((str(ref), reason))
                logger.warning("Model %s in models is not offered: %s", ref, reason)

    # --- listings ------------------------------------------------------------

    async def _refresh_in_background(self, names: list[str]) -> None:
        try:
            await self.refresh(names)
        except Exception:  # noqa: BLE001 - a background refresh never surfaces
            logger.warning("Background model-list refresh failed", exc_info=True)

    async def _list_now(self, names: Iterable[str]) -> None:
        """List ``names`` concurrently, as startup does (a failure is a warning)."""
        await asyncio.gather(*(self._list(name, startup=True) for name in dict.fromkeys(names)))

    async def _list(self, name: str, *, startup: bool = False) -> None:
        state = self._states[name]
        listing = await list_models(
            state.provider, transport=self._transport, timeout=self._timeout, clock=self._clock
        )
        if listing.ok:
            state.listing, state.failure = listing, None
        else:
            state.failure = listing
            # At startup this is the first run's validation going missing, as
            # 0.6 reported it; later it only means a refresh will retry.
            logger.log(
                logging.WARNING if startup else logging.INFO,
                "Could not list %s models (%s); its model ids are not checked "
                "until a listing succeeds",
                name,
                listing.error,
            )
        state.due = False
        state.from_cache = False
        self._rebuild(state)
        if listing.ok:
            state.absent = frozenset(
                ref.model
                for ref in self._entries
                if ref.provider == name and ref.model not in state.accepted
            )
        # A failed attempt writes the listing kept through it, else the failure;
        # the cache never lets either replace a listing of the same address and kind.
        latest = state.listing or state.failure
        if latest is not None:
            self._cache.put(
                name, state.provider, latest, state.absent, after_failure=state.failure is not None
            )

    def _adopt(self, state: _ProviderState, entry: CacheEntry) -> None:
        """Use a cache entry: fresh, stale, or a cached failure."""
        listing = entry.to_listing(state.provider.name)
        if listing.ok:
            state.listing, state.failure = listing, None
            state.due = not self._is_fresh(listing)
            state.from_cache = True
            state.absent = frozenset(entry.absent)
        else:
            state.listing, state.failure = None, listing
            state.due = True
        self._rebuild(state)

    def _state_of(self, state: _ProviderState) -> ProviderState:
        """What the catalog knows of the provider now, from its listing, its last
        failure and the clock: a listing that ages past the TTL is stale
        whether the cache or this process made it."""
        if state.failure is not None:
            return ProviderState.DEGRADED
        if state.listing is None:
            return ProviderState.UNATTEMPTED
        return ProviderState.FRESH if self._is_fresh(state.listing) else ProviderState.STALE

    def _is_fresh(self, listing: Listing) -> bool:
        """Whether ``listing`` is under the TTL old; one stamped in the future is not."""
        return 0 <= self._clock() - listing.fetched_at < self._ttl

    def _status(self, name: str) -> ProviderStatus:
        state = self._states[name]
        latest = state.listing or state.failure
        return ProviderStatus(
            provider=name,
            has_credentials=state.provider.has_credentials,
            state=self._state_of(state),
            fetched_at=latest.fetched_at if latest is not None else None,
            model_count=len(state.offered),
            error=state.failure.error if state.failure is not None else None,
        )

    # --- records ---------------------------------------------------------------

    def _rebuild(self, state: _ProviderState) -> None:
        """Recompute a provider's records from what it now knows; swapped in whole.

        Each record takes its model's entry (``records_for``), so ``records()``
        and ``record()`` give the same record.
        """
        state.accepted, state.offered = records_for(
            state.provider, state.listing, self._table, self._entries
        )


def _parsed(specs: Iterable[str]) -> Iterator[ModelRef]:
    """Each of ``specs`` that is a reference or a bare model id; the rest skipped."""
    for spec in specs:
        try:
            yield parse_ref(spec)
        except ModelSpecError:
            continue
