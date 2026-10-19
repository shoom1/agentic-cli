"""A user endpoint (vLLM and other OpenAI-compatible servers): ``GET {base_url}/models``.

The bearer key is the one the endpoint's ``api_key_env`` names, or the
placeholder ``EMPTY`` that servers without authentication accept. vLLM adds
``max_model_len``, the served context length; other servers send ids only.
"""

from __future__ import annotations

import httpx

from agentic_cli.models.discovery import ListedModel, count, entries, get_json, text
from agentic_cli.models.endpoints import Endpoint

#: The key sent to an endpoint that names none.
PLACEHOLDER_KEY = "EMPTY"


async def fetch(client: httpx.AsyncClient, endpoint: Endpoint) -> list[ListedModel]:
    headers = {"authorization": f"Bearer {endpoint.api_key or PLACEHOLDER_KEY}"}
    document = await get_json(client, f"{endpoint.base_url}/models", headers=headers)
    return [
        ListedModel(id=model_id, context_window=count(entry.get("max_model_len")))
        for entry in entries(document, "data")
        if (model_id := text(entry.get("id"))) is not None
    ]
