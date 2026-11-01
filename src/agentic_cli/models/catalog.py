"""The catalog: every model the workflow can name, and what is known about each.

Three sources, highest precedence first; each value in a record keeps the name
of its source (``ModelRecord.sources``):

1. the ``models`` entries (``config``): what an app, a user or a project says
   about a model, field by field;
2. live listings (``listing``): Google's token limits and thinking flag,
   Anthropic's limits, thinking types, effort levels and whether thinking
   can be turned off, OpenAI's ids, OpenRouter's limits and tool and
   reasoning support, an endpoint's served names and context length;
3. the shipped table (``table``): family rules, tiers, the curated OpenAI
   list and the fallback models used without a listing.

**Validation.** An unknown id is rejected only when its provider's listing is
fresh (``ProviderState.FRESH``); a stale, failed or untried listing cannot
prove a model does not exist, so the id is accepted with the table's record.
A model with an entry on a user endpoint is always accepted.

**Cache.** Listings are kept in a JSON file (``~/.<app>/model_cache.json`` in
the workflow), one entry per provider or endpoint: fetch time, outcome and
models, never a key. Startup reads it and waits on the network only for a
provider that has credentials and no entry; a fresh entry needs no listing,
and an older one (or a cached failure) is used while
``start_background_refresh`` lists it again. A broken file, or an endpoint
entry for another ``base_url``, is ignored and listed again.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentic_cli.models.discovery import DEFAULT_TIMEOUT, ListedModel, Listing, list_models
from agentic_cli.models.entries import DEFAULT_EFFORT_LEVELS, ModelEntry
from agentic_cli.models.providers import Provider, ProviderKind
from agentic_cli.models.records import ModelRecord, ThinkingMechanism, ThinkingRule
from agentic_cli.models.refs import ModelRef, ModelSpecError, parse_ref
from agentic_cli.models.resolve import ModelResolutionError
from agentic_cli.models.table import ShippedTable, load_table, version_key

logger = logging.getLogger(__name__)

#: Seconds a listing stays fresh.
DEFAULT_TTL = 86400.0
CACHE_VERSION = 1

# OpenRouter's ``reasoning.effort`` values, lowest first.
_OPENROUTER_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")
# Claude thinking for a listing that names a type the table did not expect.
_CLAUDE_ADAPTIVE_LEVELS = ("low", "medium", "high", "max")
_CLAUDE_BUDGETS = (4096, 10000, 32000)
_CLAUDE_BUDGET_RANGE = (1024, 128000)
_ENTRY_THINKING = {
    "none": ThinkingMechanism.NONE,
    "chat_template": ThinkingMechanism.CHAT_TEMPLATE,
    "effort": ThinkingMechanism.EFFORT,
}
# The picker lists each provider's models fast, default, strong, then the rest.
_TIER_ORDER = {"fast": 0, "default": 1, "strong": 2}
# How many models an "is not available" message names.
_MESSAGE_MODELS = 20


class ProviderState(str, Enum):
    """How much the catalog knows about one provider's models.

    - ``UNATTEMPTED``: not listed (no credentials, or not loaded yet); the
      table's fallback models stand in.
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
        provider: The provider or endpoint name.
        has_credentials: Whether requests can be sent (a key, or an endpoint).
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
class _Provider:
    provider: Provider
    state: ProviderState = ProviderState.UNATTEMPTED
    #: The last successful listing; kept through a failed refresh.
    listing: Listing | None = None
    #: The last attempt, while it is a failure.
    failure: Listing | None = None
    #: A cached entry that a background refresh should replace.
    due: bool = False
    #: The fresh listing was read from the cache, not listed by this catalog.
    from_cache: bool = False
    #: Every id ``record()`` knows, and the records ``records()`` offers.
    accepted: Mapping[str, ModelRecord] = field(default_factory=dict)
    offered: tuple[ModelRecord, ...] = ()


class ModelCatalog:
    """Records for every provider's models, from entries, listings and the table.

    Args:
        providers: Every provider and user endpoint, by name
            (``build_providers``).
        entries: The ``models`` entries, by reference: each wins, field by
            field, over its model's listing and table values.
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
        cache_path: Path | None = None,
        table: ShippedTable | None = None,
        ttl_seconds: float = DEFAULT_TTL,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._cache_path = Path(cache_path) if cache_path is not None else None
        self._table = table or load_table()
        self._ttl = ttl_seconds
        self._transport = transport
        self._timeout = timeout
        self._clock = clock
        self._entries: Mapping[ModelRef, ModelEntry] = MappingProxyType(dict(entries or {}))
        self._providers = {name: _Provider(provider) for name, provider in providers.items()}
        for provider in self._providers.values():
            self._rebuild(provider)
        # Cache entries this catalog does not use (a provider without
        # credentials now), written back unchanged.
        self._kept: dict[str, _CacheEntry] = {}
        self._refresh_task: asyncio.Task[None] | None = None
        self._closed = False
        # Providers whose cached listing rejected an id (``confirm_rejections``).
        self._cached_rejections: set[str] = set()

    # --- loading and refreshing --------------------------------------------

    async def load(self) -> None:
        """Read the cache, then list what it cannot answer for.

        Every provider with credentials and no usable cache entry is listed
        now, concurrently, each within the timeout: the first run still
        validates ids. Never raises for a network or cache problem.
        """
        cached = self._read_cache()
        to_list: list[str] = []
        for name, provider in self._providers.items():
            entry = cached.get(name)
            if not provider.provider.has_credentials:
                if entry is not None:
                    self._kept[name] = entry
                continue
            if entry is None or entry.base_url != provider.provider.base_url:
                to_list.append(name)
            else:
                self._adopt(provider, entry)
        if to_list:
            await asyncio.gather(*(self._list(name, startup=True) for name in to_list))

    def start_background_refresh(self) -> "asyncio.Task[None] | None":
        """List again, in a task, every provider whose cached listing is not fresh.

        Each result is swapped in as it arrives. Returns the task, or None
        when nothing is due or the catalog is closed. ``aclose()`` cancels it.
        """
        if self._closed:
            return None
        if self._refresh_task is not None and not self._refresh_task.done():
            return self._refresh_task
        due = [
            name
            for name, provider in self._providers.items()
            if provider.due and provider.provider.has_credentials
        ]
        if not due:
            return None
        self._refresh_task = asyncio.get_running_loop().create_task(
            self._refresh_in_background(due), name="model-catalog-refresh"
        )
        return self._refresh_task

    async def refresh(self, providers: Iterable[str] | None = None) -> list[ProviderStatus]:
        """List ``providers`` (every one by default) again now.

        Providers without credentials are not listed. A failed listing keeps
        the earlier one and marks the provider degraded.

        Returns:
            The status of each provider asked about, in catalog order when
            all are.
        """
        names = list(self._providers) if providers is None else list(dict.fromkeys(providers))
        names = [name for name in names if name in self._providers]
        listable = [name for name in names if self._providers[name].provider.has_credentials]
        await asyncio.gather(*(self._list(name) for name in listable))
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
        names = [
            name for name in self._cached_rejections if self._providers[name].from_cache
        ]
        self._cached_rejections.clear()
        if not names:
            return False
        await asyncio.gather(*(self._list(name, startup=True) for name in names))
        return True

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
            ModelResolutionError: If the provider is neither built in nor a
                defined endpoint, or its fresh listing lacks the model and the
                model has no entry on an endpoint (the message lists the
                provider's models).
        """
        provider = self._providers.get(ref.provider)
        if provider is None:
            raise ModelResolutionError(
                f"Model {ref} names {ref.provider!r}, which is neither a provider nor an "
                "endpoint defined in model_endpoints."
            )
        found = provider.accepted.get(ref.model)
        if found is not None:
            return found
        entry = self._entries.get(ref)
        # A model with an entry on an endpoint is taken at its word: a
        # server's listing can leave out a model it serves.
        vouched = entry is not None and provider.provider.kind is ProviderKind.OPENAI_COMPATIBLE
        if provider.state is ProviderState.FRESH and not vouched:
            if provider.from_cache:
                self._cached_rejections.add(ref.provider)
            models = [record.model for record in provider.offered]
            shown = ", ".join(models[:_MESSAGE_MODELS]) or "none"
            if len(models) > _MESSAGE_MODELS:
                shown += f", and {len(models) - _MESSAGE_MODELS} more"
            raise ModelResolutionError(
                f"Model {ref} is not available. Available {ref.provider} models: {shown}"
            )
        record = self._table.record_for(ref.provider, provider.provider.kind, ref.model)
        return _with_entry(record, entry)

    def records(self, provider: str) -> list[ModelRecord]:
        """The models offered for ``provider``: its listing's, else the fallbacks."""
        found = self._providers.get(provider)
        return list(found.offered) if found is not None else []

    def is_authoritative(self, provider: str) -> bool:
        """Whether ``provider``'s listing can reject an unknown id (FRESH only)."""
        found = self._providers.get(provider)
        return found is not None and found.state is ProviderState.FRESH

    def status(self) -> list[ProviderStatus]:
        """Every provider's status, built-ins first, then the user's endpoints."""
        return [self._status(name) for name in self._providers]

    def picker_refs(
        self, *, referenced: Iterable[str] = (), entries: Iterable[str] = ()
    ) -> list[str]:
        """The references a model picker offers: runnable models only.

        The records of every provider with credentials except OpenRouter,
        whose catalog runs to hundreds of models: an OpenRouter model (or any
        other) is offered when it is ``referenced`` (by a role, the default
        model or an agent) or has an entry in ``models`` (``entries``).
        Sorted by provider, then tier (fast, default, strong, none), then
        newest first.
        """
        chosen: dict[ModelRef, ModelRecord] = {}
        for provider in self._providers.values():
            if (
                provider.provider.has_credentials
                and provider.provider.kind is not ProviderKind.OPENROUTER
            ):
                for record in provider.offered:
                    chosen.setdefault(record.ref, record)
        for spec in (*referenced, *entries):
            try:
                ref = parse_ref(spec)
            except ModelSpecError:
                continue
            provider = self._providers.get(ref.provider)
            if ref in chosen or provider is None or not provider.provider.has_credentials:
                continue
            try:
                chosen[ref] = self.record(ref)
            except ModelResolutionError:
                continue

        position = {name: index for index, name in enumerate(self._providers)}
        ordered = sorted(chosen.values(), key=lambda r: r.model)
        ordered.sort(key=lambda r: (version_key(r.model), "preview" not in r.model), reverse=True)
        ordered.sort(key=lambda r: (position[r.provider], _TIER_ORDER.get(r.tier, 3)))
        return [str(record.ref) for record in ordered]

    # --- listings ------------------------------------------------------------

    async def _refresh_in_background(self, names: list[str]) -> None:
        try:
            await self.refresh(names)
        except Exception:  # noqa: BLE001 - a background refresh never surfaces
            logger.warning("Background model-list refresh failed", exc_info=True)

    async def _list(self, name: str, *, startup: bool = False) -> None:
        provider = self._providers[name]
        listing = await list_models(
            provider.provider, transport=self._transport, timeout=self._timeout, clock=self._clock
        )
        if listing.ok:
            provider.listing, provider.failure = listing, None
            provider.state = ProviderState.FRESH
        else:
            provider.failure = listing
            provider.state = ProviderState.DEGRADED
            # At startup this is the first run's validation going missing, as
            # 0.6 reported it; later it only means a refresh will retry.
            logger.log(
                logging.WARNING if startup else logging.INFO,
                "Could not list %s models (%s); its model ids are not checked "
                "until a listing succeeds",
                name,
                listing.error,
            )
        provider.due = False
        provider.from_cache = False
        self._rebuild(provider)
        self._write_cache()

    def _adopt(self, provider: _Provider, entry: _CacheEntry) -> None:
        """Use a cache entry: fresh, stale, or a cached failure."""
        listing = Listing(
            provider=provider.provider.name,
            ok=entry.ok,
            models=tuple(ListedModel(**model.model_dump()) for model in entry.models),
            fetched_at=entry.fetched_at,
            error=entry.error,
        )
        if listing.ok:
            age = self._clock() - listing.fetched_at
            fresh = 0 <= age < self._ttl
            provider.listing, provider.failure = listing, None
            provider.state = ProviderState.FRESH if fresh else ProviderState.STALE
            provider.due = not fresh
            provider.from_cache = fresh
        else:
            provider.listing, provider.failure = None, listing
            provider.state = ProviderState.DEGRADED
            provider.due = True
        self._rebuild(provider)

    def _status(self, name: str) -> ProviderStatus:
        provider = self._providers[name]
        latest = provider.listing or provider.failure
        return ProviderStatus(
            provider=name,
            has_credentials=provider.provider.has_credentials,
            state=provider.state,
            fetched_at=latest.fetched_at if latest is not None else None,
            model_count=len(provider.offered),
            error=provider.failure.error if provider.failure is not None else None,
        )

    # --- records ---------------------------------------------------------------

    def _rebuild(self, provider: _Provider) -> None:
        """Recompute a provider's records from what it now knows; swapped in whole.

        Each record takes its model's entry here, so ``records()`` and
        ``record()`` give the same record.
        """
        accepted, offered = self._records_for(provider)
        name = provider.provider.name
        accepted = {
            model: _with_entry(record, self._entries.get(ModelRef(name, model)))
            for model, record in accepted.items()
        }
        provider.accepted = MappingProxyType(accepted)
        provider.offered = tuple(accepted[record.model] for record in offered)

    def _records_for(self, provider: _Provider) -> tuple[dict[str, ModelRecord], list[ModelRecord]]:
        listing = provider.listing
        name, kind = provider.provider.name, provider.provider.kind
        if kind is ProviderKind.OPENAI_COMPATIBLE:
            return self._endpoint_records(provider.provider, listing)
        if listing is None:
            models = self._table.fallback_models(kind)
            fallback = [self._table.record_for(name, kind, model) for model in models]
            return {record.model: record for record in fallback}, fallback

        build = {
            ProviderKind.GOOGLE: self._google,
            ProviderKind.ANTHROPIC: self._anthropic,
            ProviderKind.OPENAI: self._openai,
            ProviderKind.OPENROUTER: self._openrouter,
        }[kind]
        accepted: dict[str, ModelRecord] = {}
        for listed in listing.models:
            if listed.id not in accepted:
                accepted[listed.id] = build(name, listed)
        if kind is not ProviderKind.OPENAI:
            return accepted, list(accepted.values())
        # R15: the curated ids the account has, then any other reasoning
        # model it lists. Ids no rule knows (embeddings, TTS) are accepted by
        # record() but never offered.
        curated = self._table.curated_openai()
        offered = [accepted[model] for model in curated if model in accepted]
        offered += [
            record
            for model, record in accepted.items()
            if model not in curated and record.thinking.mechanism is ThinkingMechanism.EFFORT
        ]
        return accepted, offered

    def _google(self, name: str, listed: ListedModel) -> ModelRecord:
        record = self._table.record_for(name, ProviderKind.GOOGLE, listed.id)
        thinking = None
        if listed.reasoning is False and record.thinking.mechanism is not ThinkingMechanism.NONE:
            thinking = ThinkingRule()
        return _merge(
            record,
            "listing",
            display_name=listed.display_name or None,
            context_window=listed.context_window,
            max_output=listed.max_output,
            thinking=thinking,
        )

    def _anthropic(self, name: str, listed: ListedModel) -> ModelRecord:
        record = self._table.record_for(name, ProviderKind.ANTHROPIC, listed.id)
        return _merge(
            record,
            "listing",
            display_name=listed.display_name or None,
            context_window=listed.context_window,
            max_output=listed.max_output,
            image_input=listed.image_input,
            thinking=_claude_thinking(record.thinking, listed),
        )

    def _openai(self, name: str, listed: ListedModel) -> ModelRecord:
        # Ids only: the listing confirms the model exists; the table says what it is.
        return self._table.record_for(name, ProviderKind.OPENAI, listed.id)

    def _openrouter(self, name: str, listed: ListedModel) -> ModelRecord:
        record = self._table.record_for(name, ProviderKind.OPENROUTER, listed.id)
        reasons = listed.reasoning is True
        effort = ThinkingRule(ThinkingMechanism.EFFORT, levels=_OPENROUTER_EFFORTS)
        return _merge(
            record,
            "listing",
            display_name=listed.display_name or None,
            context_window=listed.context_window,
            max_output=listed.max_output,
            image_input=listed.image_input,
            tools=listed.tools,
            thinking=effort if reasons else None,
            reasoning_text=True if reasons else None,
        )

    def _endpoint_records(
        self, provider: Provider, listing: Listing | None
    ) -> tuple[dict[str, ModelRecord], list[ModelRecord]]:
        """The served models: each the conservative record (tools assumed),
        with its served name and context length."""
        records: dict[str, ModelRecord] = {}
        for listed in listing.models if listing is not None else ():
            if listed.id not in records:
                record = self._table.record_for(provider.name, provider.kind, listed.id)
                records[listed.id] = _merge(
                    record,
                    "listing",
                    display_name=listed.display_name or None,
                    context_window=listed.context_window,
                )
        return records, list(records.values())

    # --- the cache file ---------------------------------------------------------

    def _read_cache(self) -> dict[str, _CacheEntry]:
        """The cache's entries; empty for a missing, unreadable or broken file."""
        if self._cache_path is None:
            return {}
        try:
            raw = self._cache_path.read_bytes()
        except FileNotFoundError:
            return {}
        except OSError as exc:
            logger.debug("Ignoring the model cache %s: %s", self._cache_path, exc)
            return {}
        try:
            document = _CacheFile.model_validate_json(raw)
        except ValidationError as exc:
            logger.debug(
                "Ignoring the model cache %s: %d problem(s), first: %s",
                self._cache_path,
                exc.error_count(),
                exc.errors(include_input=False)[0]["msg"],
            )
            return {}
        return dict(document.providers)

    def _write_cache(self) -> None:
        """Write every entry the catalog holds; a failure is logged, never raised."""
        if self._cache_path is None:
            return
        providers: dict[str, Any] = {
            name: entry.model_dump() for name, entry in self._kept.items()
        }
        for name, provider in self._providers.items():
            # A failed refresh keeps the earlier listing; a failure is written
            # only when there is no listing to keep.
            listing = provider.listing or provider.failure
            if listing is not None:
                providers[name] = {
                    "fetched_at": listing.fetched_at,
                    "ok": listing.ok,
                    "error": listing.error,
                    "base_url": provider.provider.base_url,
                    "models": [dataclasses.asdict(model) for model in listing.models],
                }
        try:
            _atomic_write_json(self._cache_path, {"version": CACHE_VERSION, "providers": providers})
        except OSError as exc:
            logger.debug("Could not write the model cache %s: %s", self._cache_path, exc)


def _claude_thinking(table: ThinkingRule, listed: ListedModel) -> ThinkingRule | None:
    """Claude's thinking from the listing; None leaves the table's rule.

    The listing's types decide the mechanism (adaptive wins when both are
    listed, with the listing's effort ladder, else the table's); without
    either, the table's (version) rule stands. Whether thinking can be turned
    off is the listing's answer when it gives one (its ``disabled`` type),
    else the table's.
    """
    if listed.reasoning is False:
        return ThinkingRule() if table.mechanism is not ThinkingMechanism.NONE else None
    rule = _claude_mechanism(table, listed)
    if listed.can_disable is None:
        return rule
    decided = rule if rule is not None else table
    if decided.mechanism is ThinkingMechanism.NONE or decided.can_disable is listed.can_disable:
        return rule
    return dataclasses.replace(decided, can_disable=listed.can_disable)


def _claude_mechanism(table: ThinkingRule, listed: ListedModel) -> ThinkingRule | None:
    """The rule the listing's thinking types name; None when they name neither."""
    if "adaptive" in listed.thinking_types:
        table_adaptive = table.mechanism is ThinkingMechanism.ANTHROPIC_ADAPTIVE
        default_levels = table.levels if table_adaptive else _CLAUDE_ADAPTIVE_LEVELS
        levels = listed.effort_levels or default_levels
        return ThinkingRule(
            ThinkingMechanism.ANTHROPIC_ADAPTIVE,
            levels=tuple(levels),
            can_disable=table.can_disable if table_adaptive else True,
        )
    if "enabled" in listed.thinking_types:
        if table.mechanism is ThinkingMechanism.ANTHROPIC_BUDGET:
            return table
        return ThinkingRule(
            ThinkingMechanism.ANTHROPIC_BUDGET,
            budgets=_CLAUDE_BUDGETS,
            budget_range=_CLAUDE_BUDGET_RANGE,
        )
    return None


def _with_entry(record: ModelRecord, entry: ModelEntry | None) -> ModelRecord:
    """``record`` with each value its ``models`` entry sets, from ``config``.

    ``thinking`` sets the mechanism (an ``effort`` ladder is the entry's
    ``effort_levels``, else ``DEFAULT_EFFORT_LEVELS``). On a model that
    already takes an effort, ``thinking: effort`` changes nothing and
    ``effort_levels`` replaces its ladder; on any other, ``effort_levels``
    alone is ignored.
    """
    if entry is None:
        return record
    takes_effort = record.thinking.mechanism is ThinkingMechanism.EFFORT
    thinking = None
    if entry.thinking is not None and not (entry.thinking == "effort" and takes_effort):
        mechanism = _ENTRY_THINKING[entry.thinking]
        if mechanism is ThinkingMechanism.EFFORT:
            levels = tuple(entry.effort_levels or DEFAULT_EFFORT_LEVELS)
            thinking = ThinkingRule(mechanism, levels=levels)
        else:
            thinking = ThinkingRule(mechanism)
    elif entry.effort_levels is not None and takes_effort:
        thinking = dataclasses.replace(record.thinking, levels=tuple(entry.effort_levels))
    return _merge(
        record,
        "config",
        context_window=entry.context_window,
        max_output=entry.max_output,
        tools=entry.tools,
        image_input=entry.image_input,
        thinking=thinking,
        reasoning_text=entry.reasoning_text,
    )


def _merge(record: ModelRecord, source: str, **values: Any) -> ModelRecord:
    """``record`` with the values that are not None, each marked as from ``source``."""
    changes = {name: value for name, value in values.items() if value is not None}
    if not changes:
        return record
    sources = {**record.sources, **dict.fromkeys(changes, source)}
    return dataclasses.replace(record, sources=MappingProxyType(sources), **changes)


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


# --- the cache file's schema: strict, so a damaged file is ignored, not misread ---


class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class _CachedModel(_Strict):
    """A ``ListedModel``, as written."""

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


class _CacheEntry(_Strict):
    fetched_at: float
    ok: bool
    error: str | None = None
    base_url: str | None = None
    models: tuple[_CachedModel, ...] = ()


class _CacheFile(_Strict):
    version: Literal[1]
    providers: dict[str, _CacheEntry]
