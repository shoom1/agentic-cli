"""The shipped model table (``data/table.json``) and the records it gives.

The table is the lowest-precedence source of model knowledge, below live
listings and the user's endpoint declarations. Per provider kind it holds:

- **rules**, first match wins: a regular expression over the model id (and
  optionally a minimum version), giving the tier, the thinking knob, the cache
  mechanism, the sampling rule and whether the model returns reasoning text;
- **models**: per-model numbers (context window, max output, tools, image
  input) and effort ladders, copied from the LangChain partner packages'
  bundled profiles (models.dev data);
- the **fallback** model lists used before or without a listing, and OpenAI's
  curated list.

A model no rule matches gets the conservative record: no thinking knob, tools
assumed, numbers unknown. Every rule and entry carries a ``source`` note.
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agentic_cli.models.endpoints import ProviderKind
from agentic_cli.models.records import (
    CacheMechanism,
    ModelRecord,
    ThinkingMechanism,
    ThinkingRule,
)

#: The record fields whose origin ``ModelRecord.sources`` reports.
TRACKED_FIELDS: tuple[str, ...] = (
    "display_name",
    "tier",
    "context_window",
    "max_output",
    "tools",
    "image_input",
    "thinking",
    "reasoning_text",
    "cache",
    "sampling",
)

_BUDGET_MECHANISMS = (ThinkingMechanism.GEMINI_BUDGET, ThinkingMechanism.ANTHROPIC_BUDGET)
_LADDER_MECHANISMS = (
    ThinkingMechanism.GEMINI_LEVEL,
    ThinkingMechanism.ANTHROPIC_ADAPTIVE,
    ThinkingMechanism.EFFORT,
)
_LEVEL = re.compile(r"[a-z][a-z0-9_]*")


# --- version numbers ------------------------------------------------------

# A token that carries the model's version: "2.5", "4", "4o", "o3".
_VERSION_TOKEN = re.compile(r"(?:[a-z]{1,2})?(\d{1,2}(?:\.\d{1,2})*)(?:[a-z]{1,2})?")
_MINOR_TOKEN = re.compile(r"\d{1,2}")


def version_key(model_id: str) -> tuple[int, ...]:
    """The model's version as a tuple, for "newest" comparisons.

    The version is the first hyphen-separated token that carries one (``2.5``,
    ``4``, ``4o``, ``o3``) plus the one- or two-digit tokens right after it
    (Claude's ``4-5``). Numeric parts are one or two digits only, so a date
    stamp is never read as a version: ``claude-opus-4-20250514`` is ``(4,)``,
    ``gpt-5-2025-08-07`` is ``(5,)``, ``gemini-2.5-flash-preview-09-2025`` is
    ``(2, 5)``. An id with no version token (``qwen/qwen3-235b-a22b``) falls
    back to its one- and two-digit numbers, and one with none is ``()``.
    """
    numbers: list[int] = []
    for token in model_id.split("-"):
        if not numbers:
            match = _VERSION_TOKEN.fullmatch(token)
            if match:
                numbers.extend(int(part) for part in match.group(1).split("."))
        elif _MINOR_TOKEN.fullmatch(token):
            numbers.append(int(token))
        else:
            break
    if numbers:
        return tuple(numbers)
    return tuple(int(n) for n in re.findall(r"(?<!\d)\d{1,2}(?!\d)", model_id))


# --- the table file's schema ----------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_levels(levels: list[str]) -> list[str]:
    bad = [level for level in levels if not _LEVEL.fullmatch(level)]
    if bad:
        raise ValueError(f"levels {bad} must be lowercase names")
    if len(set(levels)) != len(levels):
        raise ValueError(f"levels {levels} repeat a level")
    return levels


class _Thinking(_Strict):
    mechanism: ThinkingMechanism
    levels: list[str] = Field(default_factory=list)
    budgets: tuple[int, int, int] | None = None
    budget_range: tuple[int, int] | None = None
    can_disable: bool = True
    generic_map: dict[str, str] = Field(default_factory=dict)

    _levels = field_validator("levels")(_check_levels)

    @model_validator(mode="after")
    def _consistent(self) -> "_Thinking":
        if self.mechanism in _BUDGET_MECHANISMS:
            if self.budgets is None or self.budget_range is None:
                raise ValueError(f"{self.mechanism.value} needs budgets and budget_range")
            low, high = self.budget_range
            if not 0 < low <= high:
                raise ValueError(f"budget_range {self.budget_range} is not a range")
        if self.mechanism in _LADDER_MECHANISMS and not self.levels:
            raise ValueError(f"{self.mechanism.value} needs a levels ladder")
        off_ladder = {k: v for k, v in self.generic_map.items() if v not in self.levels}
        if off_ladder:
            raise ValueError(f"generic_map {off_ladder} maps to levels not on the ladder")
        return self


class _Rule(_Strict):
    match: str
    min_version: tuple[int, ...] | None = None
    tier: Literal["fast", "default", "strong"] | None = None
    thinking: _Thinking | None = None
    cache: CacheMechanism = CacheMechanism.NONE
    reasoning_text: bool | None = None
    sampling: Literal["always", "without_thinking", "never"] | None = None
    source: str = Field(min_length=1)

    @field_validator("match")
    @classmethod
    def _compiles(cls, pattern: str) -> str:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"match {pattern!r} is not a regular expression: {exc}") from None
        return pattern


class _ThinkingOverride(_Strict):
    levels: list[str] | None = None
    can_disable: bool | None = None

    @field_validator("levels")
    @classmethod
    def _levels(cls, levels: list[str] | None) -> list[str] | None:
        if levels is not None:
            if not levels:
                raise ValueError("levels must name at least one level")
            _check_levels(levels)
        return levels


class _Model(_Strict):
    display_name: str | None = None
    context_window: int | None = Field(default=None, ge=1)
    max_output: int | None = Field(default=None, ge=1)
    tools: bool | None = None
    image_input: bool | None = None
    reasoning_text: bool | None = None
    sampling: Literal["always", "without_thinking", "never"] | None = None
    thinking: _ThinkingOverride | None = None
    source: str = Field(min_length=1)


class _Provider(_Strict):
    fallback_models: list[str] = Field(default_factory=list)
    fallback_source: str | None = None
    curated: list[str] = Field(default_factory=list)
    curated_source: str | None = None
    rules: list[_Rule] = Field(default_factory=list)
    models: dict[str, _Model] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _lists_have_sources(self) -> "_Provider":
        if self.fallback_models and not self.fallback_source:
            raise ValueError("fallback_models needs a fallback_source note")
        if self.curated and not self.curated_source:
            raise ValueError("curated needs a curated_source note")
        return self


class _Document(_Strict):
    version: Literal[1]
    about: str
    providers: dict[ProviderKind, _Provider]


# --- the table ------------------------------------------------------------


@dataclass(frozen=True)
class _CompiledRule:
    pattern: re.Pattern[str]
    rule: _Rule

    def matches(self, model: str) -> bool:
        if not self.pattern.search(model):
            return False
        return self.rule.min_version is None or version_key(model) >= self.rule.min_version


class ShippedTable:
    """The shipped table, validated, with its rules compiled.

    Args:
        data: The parsed ``table.json`` document.

    Raises:
        ValueError: If the document does not match the table's schema (a
            rule or entry without a ``source`` note included).
    """

    def __init__(self, data: Mapping[str, Any]) -> None:
        document = _Document.model_validate(data)
        self._providers: dict[ProviderKind, _Provider] = dict(document.providers)
        self._rules: dict[ProviderKind, tuple[_CompiledRule, ...]] = {
            kind: tuple(_CompiledRule(re.compile(rule.match), rule) for rule in provider.rules)
            for kind, provider in self._providers.items()
        }

    def record_for(self, provider: str, kind: ProviderKind, model: str) -> ModelRecord:
        """The table's record for one model.

        The first matching family rule gives the tier, thinking knob, cache,
        sampling rule and reasoning text; the model's own entry adds its
        numbers and refines its ladder. With neither, the conservative record:
        no thinking knob, tools assumed, every source ``default``.
        """
        table = self._providers.get(kind)
        rule = next((r.rule for r in self._rules.get(kind, ()) if r.matches(model)), None)
        entry = table.models.get(model) if table is not None else None

        values: dict[str, Any] = {}
        if rule is not None:
            if rule.tier is not None:
                values["tier"] = rule.tier
            values["cache"] = rule.cache
            if rule.reasoning_text is not None:
                values["reasoning_text"] = rule.reasoning_text
            if rule.sampling is not None:
                values["sampling"] = rule.sampling
            values["thinking"] = _thinking_rule(rule.thinking, entry)
        if entry is not None:
            for name in ("display_name", "context_window", "max_output", "tools", "image_input",
                         "reasoning_text", "sampling"):
                value = getattr(entry, name)
                if value is not None:
                    values[name] = value

        sources = MappingProxyType(
            {name: "table" if name in values else "default" for name in TRACKED_FIELDS}
        )
        return ModelRecord(provider=provider, kind=kind, model=model, sources=sources, **values)

    def fallback_models(self, kind: ProviderKind) -> tuple[str, ...]:
        """The models assumed to exist before (or without) a listing.

        OpenAI's is its curated list; OpenRouter and endpoints have none.
        """
        table = self._providers.get(kind)
        if table is None:
            return ()
        return tuple(table.curated or table.fallback_models)

    def curated_openai(self) -> tuple[str, ...]:
        """OpenAI's chat and tool-calling models (its listing gives ids only)."""
        table = self._providers.get(ProviderKind.OPENAI)
        return tuple(table.curated) if table is not None else ()


def _thinking_rule(thinking: _Thinking | None, entry: _Model | None) -> ThinkingRule:
    """A rule's thinking knob, refined by the model's own entry.

    A generic level the model's ladder has needs no mapping, so the entries
    of ``generic_map`` naming one are dropped (Gemini 3.1 Pro has ``medium``;
    Gemini 3 Pro does not).
    """
    if thinking is None:
        return ThinkingRule()
    override = entry.thinking if entry is not None and entry.thinking is not None else None
    levels = tuple(override.levels) if override and override.levels else tuple(thinking.levels)
    can_disable = thinking.can_disable
    if override and override.can_disable is not None:
        can_disable = override.can_disable
    generic_map = {k: v for k, v in thinking.generic_map.items() if k not in levels and v in levels}
    return ThinkingRule(
        mechanism=thinking.mechanism,
        levels=levels,
        budgets=thinking.budgets,
        budget_range=thinking.budget_range,
        can_disable=can_disable,
        generic_map=MappingProxyType(generic_map),
    )


@functools.cache
def load_table() -> ShippedTable:
    """The shipped table, read once from the package's ``data/table.json``."""
    text = resources.files("agentic_cli.models").joinpath("data", "table.json").read_text("utf-8")
    return ShippedTable(json.loads(text))
