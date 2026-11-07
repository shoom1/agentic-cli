"""Google: ``GET https://generativelanguage.googleapis.com/v1beta/models``.

The key goes in ``x-goog-api-key``. Pages follow ``nextPageToken``.

A model that supports ``generateContent`` is served for chat, so its id is
accepted; the others (embedding, Imagen and Veo models) are left out. Only
text models are shown, by 0.6's rule: a ``gemini-`` model that supports both
``generateContent`` and ``createCachedContent`` (which leaves out image
generation, TTS, native audio, robotics, computer use and deep research), and
not a point release (``-001``), a ``-latest`` alias, a dated snapshot
(``-09-2025``), a ``-customtools`` or an ``-exp-`` variant.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import httpx

from agentic_cli.models.discovery import (
    ListedModel,
    count,
    entries,
    follow_pages,
    get_json,
    text,
)
from agentic_cli.models.providers import Provider

URL = "https://generativelanguage.googleapis.com/v1beta/models"
# A chat request is a generateContent call.
_SERVED_METHOD = "generateContent"
_REQUIRED_METHODS = frozenset({_SERVED_METHOD, "createCachedContent"})
_EXCLUDED = re.compile(
    r"-\d{3}$"  # point releases: -001, -002
    r"|-latest$"  # aliases: gemini-flash-latest
    r"|-\d{2}-\d{4}$"  # dated snapshots: -09-2025
    r"|-customtools"  # variant: -customtools
    r"|-exp-"  # experimental: -exp-image-generation
)


async def fetch(client: httpx.AsyncClient, provider: Provider) -> list[ListedModel]:
    headers = {"x-goog-api-key": provider.api_key or ""}
    models: list[ListedModel] = []

    async def page(token: str | None) -> str | None:
        params: dict[str, object] = {"pageSize": 1000}
        if token:
            params["pageToken"] = token
        document = await get_json(client, URL, headers=headers, params=params)
        for entry in entries(document, "models"):
            model = _model(entry)
            if model is not None:
                models.append(model)
        return text(document.get("nextPageToken"))

    await follow_pages(page)
    return models


def _model(entry: Mapping[str, Any]) -> ListedModel | None:
    name = text(entry.get("name"))
    model_id = name.removeprefix("models/") if name is not None else ""
    methods = entry.get("supportedGenerationMethods")
    if not isinstance(methods, list):
        return None
    supported = {method for method in methods if isinstance(method, str)}
    if not model_id or _SERVED_METHOD not in supported:
        return None
    thinking = entry.get("thinking")
    return ListedModel(
        id=model_id,
        display_name=text(entry.get("displayName")) or "",
        context_window=count(entry.get("inputTokenLimit")),
        max_output=count(entry.get("outputTokenLimit")),
        reasoning=thinking if isinstance(thinking, bool) else None,
        shown=_shown(model_id, supported),
    )


def _shown(model_id: str, methods: set[str]) -> bool:
    """Whether 0.6's rule keeps the model: a ``gemini-`` text model, not an
    alias, snapshot or variant."""
    if not model_id.startswith("gemini-") or _EXCLUDED.search(model_id):
        return False
    return _REQUIRED_METHODS <= methods
