"""Fake provider APIs for LangChain chat models.

Each ``route`` sends the requests of the clients a LangChain chat model class
builds to a ``FakeAPI`` (``tests/workflow/adk/fakes.py``), keeping every
other argument the class gives its client: the key, base URL, headers,
timeout and retries the adapter set. What a test reads is the request the
class itself sends.
"""

from __future__ import annotations

from typing import Any

import httpx
import langchain_google_genai.chat_models as google_chat_models
import openrouter
from google.genai import types

from tests.workflow.adk.fakes import FakeAPI, chat_completion, http_library


def route(monkeypatch, module, client_class: str, api: FakeAPI) -> None:
    """Send the requests of every ``module.<client_class>`` built from now on to
    ``api``, keeping every other argument the chat model gives it."""
    real = getattr(module, client_class)
    library = http_library(real)

    def client(*args, **kwargs):
        return real(*args, **{**kwargs, "http_client": api.http_client(library)})

    monkeypatch.setattr(module, client_class, client)


def route_google(monkeypatch, api: FakeAPI) -> None:
    """Send the requests of every google-genai client ``ChatGoogleGenerativeAI``
    builds to ``api``, keeping its key and HTTP options."""
    real = google_chat_models.Client

    def client(**kwargs):
        options = kwargs.get("http_options") or types.HttpOptions()
        kwargs["http_options"] = options.model_copy(
            update={"httpx_async_client": httpx.AsyncClient(transport=api.transport)}
        )
        return real(**kwargs)

    monkeypatch.setattr(google_chat_models, "Client", client)


def route_openrouter(monkeypatch, api: FakeAPI) -> None:
    """Send the requests of every OpenRouter SDK client ``ChatOpenRouter``
    builds to ``api``, keeping its headers, server URL, timeout and retries."""
    real = openrouter.OpenRouter

    def client(**kwargs):
        headers = kwargs["async_client"].headers
        kwargs["async_client"] = httpx.AsyncClient(
            headers=headers, transport=api.transport, follow_redirects=True
        )
        return real(**kwargs)

    monkeypatch.setattr(openrouter, "OpenRouter", client)


def gemini_response(*parts: dict[str, Any]) -> dict[str, Any]:
    """A Gemini API ``generateContent`` response; one text part by default."""
    return {
        "candidates": [{
            "content": {"role": "model", "parts": list(parts) or [{"text": "ok"}]},
            "finishReason": "STOP",
        }],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1, "totalTokenCount": 4},
    }


def openrouter_completion(message: dict[str, Any], finish_reason: str = "stop") -> dict[str, Any]:
    """An OpenRouter chat completion (the SDK requires ``system_fingerprint``)."""
    return {
        **chat_completion(),
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "system_fingerprint": None,
    }
