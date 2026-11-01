"""An OpenAI-compatible provider (vLLM, Together and the like): ``GET {base_url}/models``.

The bearer key is the one the provider's ``api_key_env`` names, or the
placeholder ``EMPTY`` that servers without authentication accept. The body is
OpenAI's ``{"data": [...]}`` or, from some servers, a bare array of models.
Each gives its id, and some its context length, under one of the names
servers use for it; nothing else is read (a model's thinking comes only from
its ``models`` entry). A list with entries but no model object in it (bare
id strings, say) is a shape this lister does not know: a failed listing, not
an empty one that would reject every id.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from agentic_cli.models.discovery import (
    ListedModel,
    ListingError,
    count,
    get_document,
    objects,
    text,
)
from agentic_cli.models.providers import Provider

#: The key sent to a provider that names none.
PLACEHOLDER_KEY = "EMPTY"
#: Where a model's context length may be, the first that is a count winning:
#: Together's and OpenRouter-style ``context_length``, ``context_window``, and
#: vLLM's ``max_model_len``.
_WINDOW_FIELDS = ("context_length", "context_window", "max_model_len")


async def fetch(client: httpx.AsyncClient, provider: Provider) -> list[ListedModel]:
    headers = {"authorization": f"Bearer {provider.api_key or PLACEHOLDER_KEY}"}
    document = await get_document(client, f"{provider.base_url}/models", headers=headers)
    if isinstance(document, Mapping):
        items = document.get("data")
        if not isinstance(items, list):
            raise ListingError("unexpected response: 'data' is not a list")
    elif isinstance(document, list):
        items = document
    else:
        raise ListingError("unexpected response: neither a JSON object nor an array")
    models = list(objects(items))
    if items and not models:
        raise ListingError("unexpected response: no model object in the list")
    return [
        ListedModel(id=model_id, context_window=_window(entry))
        for entry in models
        if (model_id := text(entry.get("id"))) is not None
    ]


def _window(entry: Mapping[str, Any]) -> int | None:
    for field in _WINDOW_FIELDS:
        window = count(entry.get(field))
        if window is not None:
            return window
    return None
