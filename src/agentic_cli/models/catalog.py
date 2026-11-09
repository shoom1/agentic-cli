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
fill (a built-in model's tier comes from the table's family rules).
``startup.plan`` applies these rules, and ``load`` asks it again once the
listings have shown what their providers lack. Only these are listed, at
startup or in the background; one whose ``list_models`` is False is in use
but never listed. ``search`` lists any other provider with credentials the
first time it is searched, and any whose listing is not fresh.

**Startup: the settings first.** The settings name the models, and ``load``
uses them at once. It waits on a listing only to fill a gap they leave
(``startup``): no entry can run (the list's tier models come from the
built-in providers' listings), a computed role has no entry of its tier, or
nothing describes a model they name (no ``models`` entry sets a fact of it, no
table rule or entry matches it, and no listing in hand serves it). Those
listings run concurrently, each within the timeout, and a second round lists
the built-in providers the first one's results put in use; there is no
third. ``start_background_refresh`` lists every other provider in use: one
nothing is known of, one whose cached listing is not fresh or failed, and one
whose cached listing lacks a model the settings name (it may predate the
model). That is how the models the settings name are checked.

**The list** (``offered``) is what the model pickers offer: the entries that
can run, then the models the references and the caller name that can run (one
rule for all of them), each once; when no entry can run, each built-in
provider's newest fast, default and strong models, of those it shows.
Listings never add to it. ``not_offered`` says why a model the settings name
(an entry or a reference) is left out. Once a listing shows that a provider
does not serve a model the settings name (at startup, or in the background),
the catalog warns, once per process, naming the model, the setting that names
it (``named_by``) and what to do; and of an entry whose provider is not
defined (a reference to one fails to resolve, and says so).

**Served and shown.** A listing holds every model the provider serves, and
the catalog accepts each id it lists (``record()``, ``search``), and for
Anthropic the undated alias the table lists for a dated id (``claude-sonnet-4-5``
for ``claude-sonnet-4-5-20250929``, ``claude-opus-4-0`` for
``claude-opus-4-20250514``: ``ShippedTable.served_alias``). No other
date-stripped id is accepted: ``claude-opus-4`` and ``claude-3-5-haiku`` are not
served. A
provider shows fewer (``records()``, the tier picks above, the models an "is
not available" message names, the status count ``ProviderStatus.model_count``):
Google its text models, without aliases, snapshots or variants; Anthropic each
dated id as its alias where the table lists one, which takes the first one
listed's facts, else as itself; OpenAI its curated and reasoning models.
``listing_records.records_for`` makes both maps, each record with its entry.

**Validation.** A model the settings name is never refused (``record()``):
the settings decide what runs, a turn on a model its provider does not serve
fails at the provider, and the list and the warnings say so beforehand; one
nothing describes runs on the conservative record. Any other id is rejected
only when its provider's listing is fresh (``ProviderState.FRESH``); a stale,
failed or untried listing cannot prove a model does not exist, so the id is
accepted with the table's record. A provider's state is derived each time it
is asked, from its listing's age, its last failure and the clock: a listing
that outlives the TTL in a long session is stale, whether the cache or this
process made it. A fresh listing read from the cache can predate a model:
before it rejects an id no setting names (a ``/models`` lookup), the provider
is listed again, once, through ``confirm_rejections``, and it leaves a model
the settings name out of the list only once a listing this process made, or
its own record of the model as absent, says so. A listing records the models
the settings name that it lacks (``absent``, in the cache too), and a later
start trusts that record at any age: a model a provider retired costs one
listing, not one at every start, whichever project's settings name it.

**Cache.** Listings are kept in a JSON file (``~/.<app>/model_cache.json`` in
the workflow; ``listing_cache.ListingCache`` reads and writes it), one entry
per provider: fetch time, outcome, address, kind, models (each with whether
it is shown) and the models the settings named that it lacked, never a key.
Startup reads it first, and waits on a provider with an entry for its address
and kind only when the entry is a listing that lacks a model nothing else
describes, without having recorded it as absent; a fresh entry needs no
listing, and an older one (or a cached failure) is used while
``start_background_refresh`` lists it again. A listing that fails at startup
is listed once more there (a blip should not last the session): until a
listing succeeds, the provider's ids are not checked and its models' limits
and thinking come from the table (an OpenRouter or configured provider's,
which the table does not know: no thinking and the 128,000-token window
floor); the session's agents are resolved once, at startup, and keep that,
and a later listing reaches ``/models``, the pickers, side calls and the next
start. If it fails too, the next start tries again. A broken file, one from
an older version, an entry this version cannot read, or an entry for another
address or kind is ignored and listed again; a file of a newer version is not
used, and never written (the listings stay in memory). A listing writes its own provider's entry and
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
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType

import httpx

from agentic_cli.models.discovery import DEFAULT_TIMEOUT, Listing, list_models
from agentic_cli.models.entries import ModelEntry
from agentic_cli.models.listing_cache import (  # noqa: F401 - CACHE_VERSION stays importable from here
    CACHE_VERSION,
    CacheEntry,
    ListingCache,
)
from agentic_cli.models.listing_records import records_for, with_entry
from agentic_cli.models.providers import Provider, ProviderKind
from agentic_cli.models.records import ModelRecord
from agentic_cli.models.refs import BUILTIN_PROVIDERS, ModelRef, ModelSpecError, parse_ref
from agentic_cli.models.resolve import ModelResolutionError
from agentic_cli.models.startup import Known, NamedModel, StartupPlan, plan
from agentic_cli.models.table import ShippedTable, load_table, newest

logger = logging.getLogger(__name__)

#: Seconds a listing stays fresh.
DEFAULT_TTL = 86400.0
# The tiers the list offers of each built-in provider when no entry can run.
_TIERS = ("fast", "default", "strong")
# How many models an "is not available" message names.
_MESSAGE_MODELS = 20
# How a model nothing describes runs: the conservative record. (The window is the
# workflow's floor for an unknown one, ``workflow.context.UNKNOWN_WINDOW``.)
_CONSERVATIVE = "without thinking and with a 128,000-token window (the floor for an unknown one)"
# What a provider's models are without its listing, for the warning that a listing
# failed. The agents are resolved once at startup and keep this for the session; a
# later listing reaches /models, the pickers, side calls and the next start.
_WITHOUT_LISTING = {
    kind: (
        "and its models' limits and thinking come from the shipped table for this "
        "session's agents (a later listing reaches /models and the next start)"
    )
    for kind in (ProviderKind.GOOGLE, ProviderKind.ANTHROPIC, ProviderKind.OPENAI)
} | {
    kind: (
        f"and its models run {_CONSERVATIVE} for this session's agents, unless a models "
        "entry says otherwise (a later listing reaches /models and the next start)"
    )
    for kind in (ProviderKind.OPENROUTER, ProviderKind.OPENAI_COMPATIBLE)
}
#: The (model, reason) pairs already warned of: a model the settings name that
#: the list leaves out. Once per process: a manager builds a new catalog at every
#: reinitialize, and /models one while the workflow starts, and the user was
#: told already.
_WARNED_MODELS: set[tuple[str, str]] = set()


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
        model_count: How many models the catalog shows for it: the records
            ``ModelCatalog.records`` returns, not every id it accepts, and
            not the models the list offers.
        error: Why the last listing failed, if it did.
    """

    provider: str
    has_credentials: bool
    state: ProviderState
    fetched_at: float | None
    model_count: int
    error: str | None = None


@dataclass
class _ProviderKnowledge:
    """What the catalog knows of one provider's models: the facts its state
    (``ProviderState``) is derived from, and the records built from them."""

    provider: Provider
    #: The last successful listing; kept through a failed refresh.
    listing: Listing | None = None
    #: The last attempt, while it is a failure.
    failure: Listing | None = None
    #: A listing the background refresh should make: the provider is in use
    #: and the start did not list it (nothing is known of it, or its cached
    #: listing is not fresh, failed, or lacks a model the settings name), or
    #: its listing failed at startup.
    due: bool = False
    #: The listing was read from the cache, not listed by this catalog.
    from_cache: bool = False
    #: The models the settings named that the listing lacked when it was made
    #: (this catalog's, or, from the cache, any catalog's sharing the file): a
    #: listing that lacks one of these is not listed again for it.
    absent: frozenset[str] = frozenset()
    #: Every id the provider serves, which ``record()`` accepts and ``search``
    #: finds; and the records it shows, which ``records()`` returns.
    accepted: Mapping[str, ModelRecord] = field(default_factory=dict)
    shown: tuple[ModelRecord, ...] = ()

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
        named_by: Where each reference is named (``default_model``, ``agent
            'writer'``), for the warning that its provider does not serve it;
            "a setting" for one it leaves out.
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
        named_by: Mapping[ModelRef, str] | None = None,
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
        # Every model the settings name: the entries, then the references, each once.
        self._named = dict.fromkeys((*self._entries, *self._references))
        self._named_by = dict(named_by or {})
        self._states = {name: _ProviderKnowledge(provider) for name, provider in providers.items()}
        for state in self._states.values():
            self._rebuild(state)
        self._computed_roles = tuple(computed_roles)
        self._in_use = self._plan().in_use
        # Providers a search listed (or read from the cache), beside those in use.
        self._searched: set[str] = set()
        self._refresh_task: asyncio.Task[None] | None = None
        self._closed = False
        # Providers whose cached listing rejected an id no setting names
        # (``record()`` notes them for ``confirm_rejections``).
        self._cached_rejections: set[str] = set()

    @property
    def in_use(self) -> tuple[str, ...]:
        """The providers in use, in catalog order (the module docstring says
        which): known from the settings, and widened by ``load``."""
        return self._in_use

    # --- loading and refreshing --------------------------------------------

    async def load(self) -> None:
        """Read the cache, then list only what fills a gap the settings leave.

        Each provider in use takes its cache entry for its address and kind,
        when there is one. Then ``startup.plan`` says which providers to list
        now (no entry can run, a computed role has none of its tier, or
        nothing describes a model the settings name), and they are listed
        concurrently, each within the timeout; when those listings put a
        built-in provider in use (they found absent the entries that could
        run), it is planned in a second round. Every model the settings name
        is used as they name it: each provider in use that was not listed
        now, nothing being known of it or its cached listing lacking a model
        the settings name, is listed by ``start_background_refresh``. Then
        each model the settings name that the list leaves out is warned of,
        once per process, unless only its key is missing (``not_offered``
        says that). Never raises for a network or cache problem.
        """
        cached = self._cache.read()
        read: set[str] = set()
        listed: set[str] = set()
        while True:
            startup = self._plan(listed)
            self._in_use = tuple(n for n in self._states if n in self._in_use or n in startup.in_use)
            # A provider newly in use takes its cache entry before it is planned.
            unread = [name for name in self._in_use if name not in read]
            if unread:
                read.update(unread)
                for name in unread:
                    state = self._states[name]
                    entry = cached.entry(name, state.provider) if state.listable else None
                    if entry is not None:
                        self._adopt(state, entry)
                continue
            if not startup.to_list:
                break
            await self._list_now(startup.to_list)
            listed.update(startup.to_list)
        self._check_later(listed)
        self._warn_of_models_left_out()

    def start_background_refresh(self) -> "asyncio.Task[None] | None":
        """List, in a task, each provider in use that ``load`` left to it: one
        the start did not list (nothing is known of it, or its cached listing is
        not fresh, failed, or lacks a model the settings name), or whose listing
        failed at startup.

        This is how the models the settings name are checked: once the
        listings are in, each model they show its provider does not serve is
        warned of, once per process.
        Each provider is listed once: if that fails, only the next start tries
        again. Each result is swapped in as it arrives. Returns the task, or
        None when nothing is due or the catalog is closed. ``aclose()``
        cancels it.
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
        serves (a release, a plan upgrade, another account's key).
        ``record()`` notes each provider whose cached listing rejected an id,
        which can only be one no setting names (a lookup in ``/models``, say):
        a caller about to act on that rejection lists those providers again,
        once, and asks ``record()`` again. The new listing decides, and a
        failed one accepts the id, as any failed listing does. A model the
        settings name is never rejected (``load`` and the background refresh
        check those, and nothing waits on this), so after ``load`` there is
        nothing to confirm until such a lookup.

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

        A model the settings name (an entry, or a reference) is never refused:
        the settings decide what runs, and a provider that does not serve it
        is said by the list (``not_offered``) and the warnings, not here. Its
        record is then its entry's and the table's. A rejection by a listing
        read from the cache is noted for ``confirm_rejections``.

        Raises:
            ModelResolutionError: If the provider is neither built in nor
                configured, or, for a model the settings do not name, its
                fresh listing lacks the model (the message lists the
                provider's models).
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
        if ref not in self._named and self._state_of(state) is ProviderState.FRESH:
            if state.from_cache:
                self._cached_rejections.add(ref.provider)
            models = [record.model for record in state.shown]
            shown = ", ".join(models[:_MESSAGE_MODELS]) or "none"
            if len(models) > _MESSAGE_MODELS:
                shown += f", and {len(models) - _MESSAGE_MODELS} more"
            raise ModelResolutionError(
                f"Model {ref} is not available. Available {ref.provider} models: {shown}"
            )
        record = self._table.record_for(ref.provider, state.provider.kind, ref.model)
        return with_entry(record, self._entries.get(ref))

    def records(self, provider: str) -> list[ModelRecord]:
        """The models shown for ``provider``: those its listing shows, else the
        fallbacks. Not every id ``record()`` accepts (a provider serves more
        than it shows), and not the list the pickers offer (``offered``)."""
        found = self._states.get(provider)
        return list(found.shown) if found is not None else []

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
           run, by the same rule: a provider with credentials, and a model it
           is not known not to serve (no listing recorded it as absent, and no
           fresh listing lacks it; for a model the settings name, a fresh
           listing this process made). A string that is no reference is
           skipped.
        3. Only when no entry can run: each built-in provider with a key, in
           the order Google, Anthropic, OpenAI, its newest fast, default and
           strong model, from its listing or else the table's fallback models.
        """
        chosen = [ref for ref in self._entries if self._runs(ref)]
        tiers = not chosen
        chosen += [ref for ref in (*self._references, *_parsed(extra)) if self._runs(ref)]
        if tiers:
            chosen += self._tier_models()
        return [str(ref) for ref in dict.fromkeys(chosen)]

    def not_offered(self) -> dict[str, str]:
        """The models the settings name that the list leaves out, each with why:
        the ``models`` entries, then the references (``default_model``, a role,
        an agent, the manager's own model), each once. Its provider is not
        defined, has no key, or does not serve it (a listing recorded it as
        absent, or this process's fresh listing lacks it)."""
        left_out = ((ref, self._left_out(ref)) for ref in self._named)
        return {str(ref): reason for ref, reason in left_out if reason is not None}

    def entry_records(self) -> list[ModelRecord]:
        """The records of the ``models`` entries the list offers, in its order."""
        return [self.record(ref) for ref in self._entries if self._runs(ref)]

    # --- the providers in use and the list ------------------------------------

    def _plan(self, listed: Collection[str] = ()) -> StartupPlan:
        """``startup.plan`` of what the catalog knows now; ``listed``: the
        providers this start has listed already."""
        named = [
            NamedModel(
                ref=ref,
                entry=ref in self._entries,
                reference=ref in self._references,
                tier=self._tier(ref) if ref.provider in self._states else None,
                described=ref.provider in self._states and self._described(ref),
                not_served=ref.provider in self._states and self._not_served(ref),
            )
            for ref in self._named
        ]
        known = {
            name: Known(
                listing=state.listing is not None,
                failed=state.failure is not None,
                listed=name in listed,
            )
            for name, state in self._states.items()
        }
        providers = {name: state.provider for name, state in self._states.items()}
        return plan(named, self._computed_roles, providers, known)

    def _tier(self, ref: ModelRef) -> str | None:
        """A model's tier, from the table's family rules: a built-in model's only."""
        kind = self._states[ref.provider].provider.kind
        return self._table.record_for(ref.provider, kind, ref.model).tier

    def _described(self, ref: ModelRef) -> bool:
        """Whether something describes ``ref`` without a new listing: the listing
        in hand serves it, or its ``models`` entry or the table gives a fact of it."""
        state = self._states[ref.provider]
        if state.listing is not None and ref.model in state.accepted:
            return True
        table = self._table.record_for(ref.provider, state.provider.kind, ref.model)
        record = with_entry(table, self._entries.get(ref))
        return any(source != "default" for source in record.sources.values())

    def _check_later(self, listed: Collection[str]) -> None:
        """Mark for the background refresh each provider in use this start did
        not list whose listing could check a model the settings name: one
        nothing is known of, or one whose cached listing lacks such a model
        without having recorded it as absent (it may predate the model). A
        stale or failed cached listing is marked already (``_adopt``)."""
        for name in self._in_use:
            state = self._states[name]
            if name in listed or not state.listable:
                continue
            unknown = state.listing is None and state.failure is None
            if unknown or any(self._unconfirmed(ref) for ref in self._named if ref.provider == name):
                state.due = True

    def _unconfirmed(self, ref: ModelRef) -> bool:
        """Whether a listing read from the cache lacks ``ref``, a model the
        settings name, without having recorded it as absent."""
        state = self._states[ref.provider]
        return (
            state.listing is not None
            and state.from_cache
            and ref.model not in state.accepted
            and ref.model not in state.absent
        )

    def _listed_names(self) -> list[str]:
        """The providers in use or searched, in catalog order."""
        return [name for name in self._states if name in self._in_use or name in self._searched]

    def _can_run(self, provider: str) -> bool:
        state = self._states.get(provider)
        return state is not None and state.provider.has_credentials

    def _runs(self, ref: ModelRef) -> bool:
        """Whether ``ref`` can run: the one rule the list uses for the entries,
        the references and the roles' models alike (``_left_out`` says why not)."""
        return self._left_out(ref) is None

    def _left_out(self, ref: ModelRef) -> str | None:
        """Why the list leaves out ``ref``: its provider is not defined, has no
        key, or does not serve the model (``_not_served``); None when it can run."""
        state = self._states.get(ref.provider)
        if state is None:
            return f"provider {ref.provider!r} is not defined in model_providers"
        if not state.provider.has_credentials:
            return f"no key: set {state.provider.key_env}"
        if self._not_served(ref):
            return f"not served by {ref.provider}"
        return None

    def _not_served(self, ref: ModelRef) -> bool:
        """Whether ``ref``'s provider does not serve it, as far as the catalog knows.

        The listing in hand lacks the model, and either recorded it as absent
        (at any age: a model these settings or a catalog sharing the cache file
        named; a role computed from the list must not run on a model the
        provider has retired while a refresh is under way), or is fresh. A
        fresh listing read from the cache is not enough for a model the
        settings name, since it may predate the model: the background refresh
        lists that provider again (``_check_later``) and its listing decides.
        """
        state = self._states[ref.provider]
        if state.listing is None or ref.model in state.accepted:
            return False
        if ref.model in state.absent:
            return True
        return self._is_fresh(state.listing) and not (state.from_cache and ref in self._named)

    def _tier_models(self) -> list[ModelRef]:
        """Each built-in provider's newest model of each tier, for a list with no entry."""
        models: list[ModelRef] = []
        for name in BUILTIN_PROVIDERS:
            if self._can_run(name):
                for tier in _TIERS:
                    candidates = [r for r in self._states[name].shown if r.tier == tier]
                    if candidates:
                        models.append(newest(candidates).ref)
        return models

    def _warn_of_models_left_out(self) -> None:
        """One warning per model the settings name that the list leaves out
        because its provider does not serve it, or (an entry) is not defined,
        once per process, saying what to do. None for a missing key (an app's
        list names models of providers a user may hold no key for, and
        ``not_offered`` says so), nor for a reference to an undefined
        provider (resolving it fails, and says so)."""
        for ref in self._named:
            state = self._states.get(ref.provider)
            if state is None and ref not in self._entries:
                continue
            if state is not None and not state.provider.has_credentials:
                continue
            reason = self._left_out(ref)
            if reason is None or (str(ref), reason) in _WARNED_MODELS:
                continue
            _WARNED_MODELS.add((str(ref), reason))
            if state is None:
                logger.warning("Model %s in models is not offered: %s", ref, reason)
            else:
                logger.warning("%s", self._not_served_message(ref))

    def _not_served_message(self, ref: ModelRef) -> str:
        """What to say of a model the settings name that its provider does not
        serve: an entry only leaves the list; a model a setting, a role, an agent
        or the manager names runs (a turn on it fails at the provider), and when
        nothing describes it, on the conservative record."""
        provider = ref.provider
        find = f"/models {provider}: lists what it serves"
        if ref not in self._references:
            return (
                f"Model {ref} in models is not offered: not served by {provider}. "
                f"Remove it from models, or name a model {provider} serves ({find})."
            )
        where = self._named_by.get(ref, "a setting")
        message = (
            f"Model {ref} ({where}) is not served by {provider}: its listing lacks it, so a "
            f"turn on it will fail at the provider. Choose a model {provider} serves for "
            f"{where} ({find})."
        )
        if not self._described(ref):
            message += (
                f" Nothing describes it either: it runs {_CONSERVATIVE}, unless a models "
                "entry says otherwise."
            )
        return message

    # --- listings ------------------------------------------------------------

    async def _refresh_in_background(self, names: list[str]) -> None:
        try:
            await self.refresh(names)
        except Exception:  # noqa: BLE001 - a background refresh never surfaces
            logger.warning("Background model-list refresh failed", exc_info=True)
        # The listings that check the models the settings name: say what they found.
        self._warn_of_models_left_out()

    async def _list_now(self, names: Iterable[str]) -> None:
        """List ``names`` concurrently, as startup does: a failure is a warning,
        and the background refresh lists that provider again."""
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
            # 0.6 reported it, and the models' limits and thinking with it;
            # later it only means a refresh will retry.
            logger.log(
                logging.WARNING if startup else logging.INFO,
                "Could not list %s models (%s); its model ids are not checked, %s",
                name,
                listing.error,
                _WITHOUT_LISTING[state.provider.kind],
            )
        # A failure at startup is listed once more by the background refresh
        # (a blip should not last the session); any other attempt is not
        # retried by itself.
        state.due = startup and not listing.ok
        state.from_cache = False
        self._rebuild(state)
        if listing.ok:
            state.absent = frozenset(
                ref.model
                for ref in self._named
                if ref.provider == name and ref.model not in state.accepted
            )
        # A failed attempt writes the listing kept through it, else the failure;
        # the cache never lets either replace a listing of the same address and kind.
        latest = state.listing or state.failure
        if latest is not None:
            self._cache.put(
                name,
                state.provider,
                latest,
                state.absent,
                after_failure=state.failure is not None,
                served=state.accepted,
            )

    def _adopt(self, state: _ProviderKnowledge, entry: CacheEntry) -> None:
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

    def _state_of(self, state: _ProviderKnowledge) -> ProviderState:
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
            model_count=len(state.shown),
            error=state.failure.error if state.failure is not None else None,
        )

    # --- records ---------------------------------------------------------------

    def _rebuild(self, state: _ProviderKnowledge) -> None:
        """Recompute a provider's records from what it now knows; swapped in whole.

        Each record takes its model's entry (``records_for``), so ``records()``
        and ``record()`` give the same record.
        """
        state.accepted, state.shown = records_for(
            state.provider, state.listing, self._table, self._entries
        )


def _parsed(specs: Iterable[str]) -> Iterator[ModelRef]:
    """Each of ``specs`` that is a reference or a bare model id; the rest skipped."""
    for spec in specs:
        try:
            yield parse_ref(spec)
        except ModelSpecError:
            continue
