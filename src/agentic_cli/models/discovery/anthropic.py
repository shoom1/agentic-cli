"""Anthropic: ``GET https://api.anthropic.com/v1/models``.

The key goes in ``x-api-key``, with ``anthropic-version: 2023-06-01``. Pages
follow ``has_more`` and ``last_id`` (as ``after_id``).

Each model's ``capabilities`` say whether it thinks, with which thinking
types (``adaptive``, ``enabled``) and effort levels, whether thinking can be
turned off (``types.disabled``: false exactly when a request with thinking
disabled is a 400 from that model), and whether it takes image input.

Every model is listed under its own id, once (the first one listed wins), so
each id is accepted. A dated id (``claude-sonnet-4-5-20250929``, or the
``@2025-09-29`` form) is marked with its date-stripped id
(``claude-sonnet-4-5``) and not shown under its own id; whether that id is an
alias Anthropic serves (it is for the families the table names, not for
``claude-3-5-haiku``) is for ``listing_records`` to say, with the table.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from agentic_cli.models.discovery import (
    ListedModel,
    ListingError,
    child,
    count,
    entries,
    follow_pages,
    get_json,
    text,
)
from agentic_cli.models.providers import Provider
from agentic_cli.models.table import claude_alias

URL = "https://api.anthropic.com/v1/models"
API_VERSION = "2023-06-01"
_THINKING_TYPES = ("adaptive", "enabled")
_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


async def fetch(client: httpx.AsyncClient, provider: Provider) -> list[ListedModel]:
    headers = {"x-api-key": provider.api_key or "", "anthropic-version": API_VERSION}
    models: dict[str, ListedModel] = {}

    async def page(after: str | None) -> str | None:
        params: dict[str, object] = {"limit": 1000}
        if after:
            params["after_id"] = after
        document = await get_json(client, URL, headers=headers, params=params)
        for entry in entries(document, "data"):
            model = _model(entry)
            if model is not None:
                models.setdefault(model.id, model)
        if document.get("has_more") is not True:
            return None
        last = text(document.get("last_id"))
        if last is None:
            raise ListingError("unexpected response: has_more without a last_id")
        return last

    await follow_pages(page)
    return list(models.values())


def _model(entry: Mapping[str, Any]) -> ListedModel | None:
    model_id = text(entry.get("id"))
    if model_id is None:
        return None
    capabilities = child(entry, "capabilities")
    thinking = child(capabilities, "thinking")
    types = child(thinking, "types")
    effort = child(capabilities, "effort")
    alias = claude_alias(model_id)
    dated = bool(alias) and alias != model_id
    return ListedModel(
        id=model_id,
        display_name=text(entry.get("display_name")) or "",
        context_window=count(entry.get("max_input_tokens")),
        max_output=count(entry.get("max_tokens")),
        image_input=_flag(child(capabilities, "image_input")),
        reasoning=_flag(thinking),
        thinking_types=_supported(thinking, types, _THINKING_TYPES),
        effort_levels=_supported(effort, effort, _EFFORT_LEVELS),
        can_disable=_flag(child(types, "disabled")),
        shown=not dated,
        alias=alias if dated else None,
    )


def _flag(capability: Mapping[str, Any]) -> bool | None:
    supported = capability.get("supported")
    return supported if isinstance(supported, bool) else None


def _supported(
    capability: Mapping[str, Any], options: Mapping[str, Any], names: tuple[str, ...]
) -> tuple[str, ...]:
    """The ``names`` whose option says ``supported``, if the capability itself is."""
    if capability.get("supported") is False:
        return ()
    return tuple(name for name in names if _flag(child(options, name)) is True)
