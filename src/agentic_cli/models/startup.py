"""Startup planning: the providers in use, and the listings a start waits on.

The settings name the models (the ``models`` entries, ``default_model``, the
roles, the agents and a manager's own model), and a start uses them at once.
It waits on a provider's listing only to fill a gap the settings leave:

- **no entry can run** (none is configured, or none has a provider that can be
  called and serves it): the list offers each built-in provider's newest fast,
  default and strong models, which come from its listing;
- **a role is computed** (``default`` without ``default_model``, or a
  built-in role an agent names that ``model_roles`` does not map) and no
  entry that can run is of its tier: the role takes the newest model of its
  tier from the built-in providers with a key;
- **nothing describes a model the settings name**: no ``models`` entry sets a
  fact of it, no table rule or entry matches it, and no listing in hand
  serves it, so its provider's listing is the only source of its limits and
  thinking.

The first two wait on each built-in provider with a key that nothing is known
of; the third on the model's provider. A provider something is known of is not
waited on for the first two (a listing, fresh or stale, or a failure, read
from the cache: the background refresh lists it again), nor for the third when
its last listing failed (one that is down would cost every start a timeout),
and a model a listing recorded as absent is not listed for again. Nothing else
is waited on: every model the settings name is used as they name it, and the
background refresh checks it.

``plan`` is pure. The catalog gives it what it knows now, lists what comes
back, concurrently, and asks again. A provider this start listed is never
planned again, so a second round lists only the built-in providers the first
round's listings made needed (they found absent the entries that could run,
or those of a computed role's tier), and there is no third.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from agentic_cli.models.providers import Provider
from agentic_cli.models.refs import ModelRef


@dataclass(frozen=True)
class NamedModel:
    """A model the settings name, as planning sees it.

    Attributes:
        ref: The model.
        entry: It is a ``models`` entry.
        reference: A setting, a role, an agent or the manager names it (it
            may be an entry too).
        tier: Its tier, from the table's family rules (a built-in provider's
            models only), else None.
        described: Something describes it without a new listing: its
            ``models`` entry sets a fact of it, a table rule or entry
            matches it, or the listing in hand serves it.
        not_served: Its provider does not serve it, as far as is known: the
            catalog's one rule for what the list leaves out.
    """

    ref: ModelRef
    entry: bool
    reference: bool
    tier: str | None
    described: bool
    not_served: bool


@dataclass(frozen=True)
class Known:
    """What is known of one provider's models when planning.

    Attributes:
        listing: A listing is in hand (fresh or stale).
        failed: Its last listing failed.
        listed: This start has listed it already, whatever it found.
    """

    listing: bool = False
    failed: bool = False
    listed: bool = False


@dataclass(frozen=True)
class StartupPlan:
    """The providers in use, and those to list now, each in catalog order."""

    in_use: tuple[str, ...]
    to_list: tuple[str, ...]


def plan(
    named: Sequence[NamedModel],
    computed_roles: Iterable[str],
    providers: Mapping[str, Provider],
    known: Mapping[str, Known],
) -> StartupPlan:
    """The providers in use, and the listings a start waits on.

    In use: the providers of the entries that can run, of the references
    (defined, with a key or not), and every built-in provider with a key when
    no entry can run or a computed role has no entry of its tier. To list:
    the gaps the module docstring names.

    Args:
        named: The models the settings name: the entries, then the
            references, each once.
        computed_roles: The built-in roles nothing maps.
        providers: Every provider, built in and configured, in catalog order.
        known: What is known of each provider's models; one left out is one
            nothing is known of.
    """

    def facts(name: str) -> Known:
        return known.get(name) or Known()

    def runs(model: NamedModel) -> bool:
        provider = providers.get(model.ref.provider)
        return provider is not None and provider.has_credentials and not model.not_served

    def listable(name: str) -> bool:
        provider = providers.get(name)
        return provider is not None and provider.has_credentials and provider.list_models

    running = [model for model in named if model.entry and runs(model)]
    tiers = {model.tier for model in running}
    builtins = not running or any(role not in tiers for role in computed_roles)
    keyed = [name for name, p in providers.items() if p.builtin and p.has_credentials]

    used = {model.ref.provider for model in running}
    used.update(m.ref.provider for m in named if m.reference and m.ref.provider in providers)
    wait: set[str] = set()
    if builtins:
        used.update(keyed)
        for name in keyed:
            known_now = facts(name)
            if listable(name) and not (known_now.listing or known_now.failed or known_now.listed):
                wait.add(name)
    for model in named:
        name, known_now = model.ref.provider, facts(model.ref.provider)
        if model.described or model.not_served or not listable(name) or known_now.listed:
            continue
        # A provider whose last listing failed, with none in hand, is left to
        # the background refresh: one that is down would cost every start.
        if known_now.failed and not known_now.listing:
            continue
        wait.add(name)
    return StartupPlan(
        in_use=tuple(name for name in providers if name in used),
        to_list=tuple(name for name in providers if name in wait),
    )
