"""OpenRouter: the public ``GET https://openrouter.ai/api/v1/models``.

No key is needed; the bearer key is sent when there is one. Each model gives
its context length, the top provider's max completion tokens, its input
modalities and its supported parameters, which say whether it takes ``tools``
and ``reasoning``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from agentic_cli.models.discovery import ListedModel, child, count, entries, get_json, text
from agentic_cli.models.endpoints import OPENROUTER_BASE_URL, Endpoint


async def fetch(client: httpx.AsyncClient, endpoint: Endpoint) -> list[ListedModel]:
    headers = {"authorization": f"Bearer {endpoint.api_key}"} if endpoint.api_key else {}
    url = f"{endpoint.base_url or OPENROUTER_BASE_URL}/models"
    document = await get_json(client, url, headers=headers)
    models = (_model(entry) for entry in entries(document, "data"))
    return [model for model in models if model is not None]


def _model(entry: Mapping[str, Any]) -> ListedModel | None:
    model_id = text(entry.get("id"))
    if model_id is None:
        return None
    parameters = entry.get("supported_parameters")
    known = isinstance(parameters, list)
    modalities = child(entry, "architecture").get("input_modalities")
    return ListedModel(
        id=model_id,
        display_name=text(entry.get("name")) or "",
        context_window=count(entry.get("context_length")),
        max_output=count(child(entry, "top_provider").get("max_completion_tokens")),
        image_input="image" in modalities if isinstance(modalities, list) else None,
        tools="tools" in parameters if known else None,
        reasoning="reasoning" in parameters if known else None,
    )
