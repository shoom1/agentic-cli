"""Listing records: what a provider's listing, the shipped table and the
``models`` entries make of its models.

``records_for`` is the catalog's one mapping from what is known of a provider
to the ``ModelRecord`` of each of its models. Pure functions, no state: the
catalog calls it whenever a provider's knowledge changes (a listing, a cache
entry) and swaps in the result.

Two maps come out. The provider's listing holds every model it **serves**, and
the catalog accepts each of their ids; the provider **shows** fewer, the
models its list offers (``ModelCatalog.records``). Where "shown" is decided
depends on the provider: Google's lister marks the text models; Anthropic's
marks a dated id with its date-stripped id, which this module takes as the
dated id's alias, serves and shows (it takes the first dated id listed's
facts) only for a family the table names (``ShippedTable.is_claude_alias``),
since Anthropic serves no ``claude-3-5-haiku`` for ``claude-3-5-haiku-20241022``,
and otherwise shows the dated id itself; OpenAI's curated list and its
reasoning models are picked here. Every value in a record keeps its source
(``ModelRecord.sources``), highest precedence first: the ``models`` entry
(``config``, ``with_entry``), the listing (``listing``), the shipped table
(``table``).

Claude's thinking is the listing's word where it gives one (its thinking types
and effort levels, whether thinking can be turned off), else the table's rule.
When the listing names a type the table's rule for the model does not expect,
the ladder or the budgets are the table's generic ones
(``ShippedTable.generic_claude_thinking``), so the table is their only source.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from agentic_cli.models.discovery import ListedModel, Listing
from agentic_cli.models.entries import DEFAULT_EFFORT_LEVELS, ModelEntry
from agentic_cli.models.providers import Provider, ProviderKind
from agentic_cli.models.records import ModelRecord, ThinkingMechanism, ThinkingRule
from agentic_cli.models.refs import ModelRef
from agentic_cli.models.table import ShippedTable

# OpenRouter's ``reasoning.effort`` values, lowest first.
_OPENROUTER_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")
_ENTRY_THINKING = {
    "none": ThinkingMechanism.NONE,
    "chat_template": ThinkingMechanism.CHAT_TEMPLATE,
    "effort": ThinkingMechanism.EFFORT,
}


def records_for(
    provider: Provider,
    listing: Listing | None,
    table: ShippedTable,
    entries: Mapping[ModelRef, ModelEntry] | None = None,
) -> tuple[Mapping[str, ModelRecord], tuple[ModelRecord, ...]]:
    """The records of one provider's models, each with its ``models`` entry.

    Without a listing the table's fallback models stand in (a configured
    provider has none).

    Args:
        provider: The provider.
        listing: Its last successful listing, if there is one.
        table: The shipped table.
        entries: The ``models`` entries, by reference; only this provider's
            apply, each to its model's record.

    Returns:
        The records of every model the provider serves, by id (the ids
        ``ModelCatalog.record`` accepts), and those it shows, in the order its
        list offers them. A shown record is the served one.
    """
    served, shown = _records_for(provider, listing, table)
    given = entries or {}
    served = {
        model: with_entry(record, given.get(ModelRef(provider.name, model)))
        for model, record in served.items()
    }
    return MappingProxyType(served), tuple(served[record.model] for record in shown)


def with_entry(record: ModelRecord, entry: ModelEntry | None) -> ModelRecord:
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


def _records_for(
    provider: Provider, listing: Listing | None, table: ShippedTable
) -> tuple[dict[str, ModelRecord], list[ModelRecord]]:
    name, kind = provider.name, provider.kind
    if kind is ProviderKind.OPENAI_COMPATIBLE:
        return _compatible_records(provider, listing, table)
    if listing is None:
        fallback = [table.record_for(name, kind, model) for model in table.fallback_models(kind)]
        return {record.model: record for record in fallback}, fallback
    # Every id the provider serves is accepted; only the shown ones are offered.
    accepted: dict[str, ModelRecord] = {}
    shown: dict[str, ModelRecord] = {}
    for listed in listing.models:
        if listed.id not in accepted:
            accepted[listed.id] = _listed_record(name, kind, listed, table)
        if listed.alias is None or not table.is_claude_alias(listed.alias):
            # A dated id whose alias Anthropic does not serve (the 3.x line's)
            # is shown as it is, and its date-stripped id is no id at all.
            if listed.shown or listed.alias is not None:
                shown.setdefault(listed.id, accepted[listed.id])
            continue
        # Anthropic: a dated id is shown as its alias, which takes the
        # first one listed's facts, as in 0.6.
        if listed.alias not in accepted:
            alias = dataclasses.replace(listed, id=listed.alias)
            accepted[listed.alias] = _listed_record(name, kind, alias, table)
        shown.setdefault(listed.alias, accepted[listed.alias])
    if kind is not ProviderKind.OPENAI:
        return accepted, list(shown.values())
    # R15: the curated ids the account has, then any other reasoning
    # model it lists. Other ids (embeddings, TTS, gpt-4.1) are accepted by
    # record() but never offered.
    curated = table.curated_openai()
    offered = [accepted[model] for model in curated if model in accepted]
    offered += [
        record
        for model, record in accepted.items()
        if model not in curated and record.thinking.mechanism is ThinkingMechanism.EFFORT
    ]
    return accepted, offered


def _listed_record(
    name: str, kind: ProviderKind, listed: ListedModel, table: ShippedTable
) -> ModelRecord:
    """One listed model's record: the table's, with what the listing says."""
    record = table.record_for(name, kind, listed.id)
    if kind is ProviderKind.GOOGLE:
        return _google(record, listed)
    if kind is ProviderKind.ANTHROPIC:
        return _anthropic(record, listed, table)
    if kind is ProviderKind.OPENROUTER:
        return _openrouter(record, listed)
    # OpenAI: ids only. The listing confirms the model exists; the table says what it is.
    return record


def _google(record: ModelRecord, listed: ListedModel) -> ModelRecord:
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


def _anthropic(record: ModelRecord, listed: ListedModel, table: ShippedTable) -> ModelRecord:
    return _merge(
        record,
        "listing",
        display_name=listed.display_name or None,
        context_window=listed.context_window,
        max_output=listed.max_output,
        image_input=listed.image_input,
        thinking=_claude_thinking(record.thinking, listed, table),
    )


def _openrouter(record: ModelRecord, listed: ListedModel) -> ModelRecord:
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


def _compatible_records(
    provider: Provider, listing: Listing | None, table: ShippedTable
) -> tuple[dict[str, ModelRecord], list[ModelRecord]]:
    """An OpenAI-compatible provider's served models: each the conservative
    record (tools assumed), with its served name and context length."""
    records: dict[str, ModelRecord] = {}
    for listed in listing.models if listing is not None else ():
        if listed.id not in records:
            record = table.record_for(provider.name, provider.kind, listed.id)
            records[listed.id] = _merge(
                record,
                "listing",
                display_name=listed.display_name or None,
                context_window=listed.context_window,
            )
    return records, list(records.values())


def _claude_thinking(
    rule: ThinkingRule, listed: ListedModel, table: ShippedTable
) -> ThinkingRule | None:
    """Claude's thinking from the listing; None leaves the table's rule.

    The listing's types decide the mechanism (adaptive wins when both are
    listed, with the listing's effort ladder, else the table's); without
    either, the table's (version) rule stands. Whether thinking can be turned
    off is the listing's answer when it gives one (its ``disabled`` type),
    else the table's.
    """
    if listed.reasoning is False:
        return ThinkingRule() if rule.mechanism is not ThinkingMechanism.NONE else None
    named = _claude_mechanism(rule, listed, table)
    if listed.can_disable is None:
        return named
    decided = named if named is not None else rule
    if decided.mechanism is ThinkingMechanism.NONE or decided.can_disable is listed.can_disable:
        return named
    return dataclasses.replace(decided, can_disable=listed.can_disable)


def _claude_mechanism(
    rule: ThinkingRule, listed: ListedModel, table: ShippedTable
) -> ThinkingRule | None:
    """The rule the listing's thinking types name; None when they name neither."""
    if "adaptive" in listed.thinking_types:
        table_adaptive = rule.mechanism is ThinkingMechanism.ANTHROPIC_ADAPTIVE
        default_levels = (
            rule.levels
            if table_adaptive
            else table.generic_claude_thinking(ThinkingMechanism.ANTHROPIC_ADAPTIVE).levels
        )
        levels = listed.effort_levels or default_levels
        return ThinkingRule(
            ThinkingMechanism.ANTHROPIC_ADAPTIVE,
            levels=tuple(levels),
            can_disable=rule.can_disable if table_adaptive else True,
        )
    if "enabled" in listed.thinking_types:
        if rule.mechanism is ThinkingMechanism.ANTHROPIC_BUDGET:
            return rule
        generic = table.generic_claude_thinking(ThinkingMechanism.ANTHROPIC_BUDGET)
        return ThinkingRule(
            ThinkingMechanism.ANTHROPIC_BUDGET,
            budgets=generic.budgets,
            budget_range=generic.budget_range,
        )
    return None


def _merge(record: ModelRecord, source: str, **values: Any) -> ModelRecord:
    """``record`` with the values that are not None, each marked as from ``source``."""
    changes = {name: value for name, value in values.items() if value is not None}
    if not changes:
        return record
    sources = {**record.sources, **dict.fromkeys(changes, source)}
    return dataclasses.replace(record, sources=MappingProxyType(sources), **changes)
