"""Google: ``GET https://generativelanguage.googleapis.com/v1beta/models``.

The key goes in ``x-goog-api-key``. Pages follow ``nextPageToken``.

Only text models are kept, by 0.6's rule: a ``gemini-`` model that supports
both ``generateContent`` and ``createCachedContent`` (which drops image
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
from agentic_cli.models.endpoints import Endpoint

URL = "https://generativelanguage.googleapis.com/v1beta/models"
_REQUIRED_METHODS = frozenset({"generateContent", "createCachedContent"})
_EXCLUDED = re.compile(
    r"-\d{3}$"  # point releases: -001, -002
    r"|-latest$"  # aliases: gemini-flash-latest
    r"|-\d{2}-\d{4}$"  # dated snapshots: -09-2025
    r"|-customtools"  # variant: -customtools
    r"|-exp-"  # experimental: -exp-image-generation
)


async def fetch(client: httpx.AsyncClient, endpoint: Endpoint) -> list[ListedModel]:
    headers = {"x-goog-api-key": endpoint.api_key or ""}
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
    if name is None:
        return None
    model_id = name.removeprefix("models/")
    if not model_id.startswith("gemini-") or _EXCLUDED.search(model_id):
        return None
    methods = entry.get("supportedGenerationMethods")
    if not isinstance(methods, list):
        return None
    if not _REQUIRED_METHODS <= {method for method in methods if isinstance(method, str)}:
        return None
    thinking = entry.get("thinking")
    return ListedModel(
        id=model_id,
        display_name=text(entry.get("displayName")) or "",
        context_window=count(entry.get("inputTokenLimit")),
        max_output=count(entry.get("outputTokenLimit")),
        reasoning=thinking if isinstance(thinking, bool) else None,
    )
