"""OpenAI: ``GET https://api.openai.com/v1/models`` with a bearer key.

The listing gives ids only (no context windows, no capabilities), so it
confirms which models the account can use; the shipped table says what they
are.
"""

from __future__ import annotations

import httpx

from agentic_cli.models.discovery import ListedModel, entries, get_json, text
from agentic_cli.models.providers import Provider

URL = "https://api.openai.com/v1/models"


async def fetch(client: httpx.AsyncClient, provider: Provider) -> list[ListedModel]:
    headers = {"authorization": f"Bearer {provider.api_key or ''}"}
    document = await get_json(client, URL, headers=headers)
    return [
        ListedModel(id=model_id)
        for entry in entries(document, "data")
        if (model_id := text(entry.get("id"))) is not None
    ]
