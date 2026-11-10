"""An offline test's built-in provider clients point at a dead local port.

A provider SDK given no host reads one from the environment, and a test that
routes one client to a fake can miss another (ADK's ``AnthropicLlm`` module
binds ``AsyncAnthropic`` at import, so routing ``anthropic.AsyncAnthropic``
alone misses a native ADK YAML app's bare ``claude-*``). With the variables
below pointed at a local port nothing listens on, and no proxy for it, a
built-in provider's client that a test did not route fails at once, locally.
The root conftest's ``_dead_provider_urls`` does this for every test but the
live ones (marked ``llm``), which keep the real environment their keys come
from.

What it does not cover: a configured provider's host is its settings entry's
(``model_providers``), which no variable overrides, so routing alone protects
it; a child process given its own environment (``child_env``) runs without
the guard; so does a module- or session-scoped fixture, which is set up
before any test's fixtures.
"""

from __future__ import annotations

import os
import sys

import pytest

#: A local port nothing listens on.
DEAD_URL = "http://127.0.0.1:9"

#: The variables the provider SDKs read their host from, pointed at ``DEAD_URL``.
DEAD_BASE_URL_VARIABLES = (
    # anthropic: the framework's ``AsyncAnthropic``, ADK's own (a native app's
    # bare claude-*) and ChatAnthropic's.
    "ANTHROPIC_BASE_URL",
    # openai: ADK's OpenAIResponsesLlm, ChatOpenAI.
    "OPENAI_BASE_URL",
    # google-genai on the Gemini API (ADK's Gemini, ChatGoogleGenerativeAI)
    # and on Vertex AI.
    "GOOGLE_GEMINI_BASE_URL",
    "GOOGLE_VERTEX_BASE_URL",
    # langchain-openrouter's ChatOpenRouter, built without a base URL.
    "OPENROUTER_API_BASE",
    # No installed SDK reads it (the OpenRouter SDK reads no host variable);
    # kept for one that may.
    "OPENROUTER_BASE_URL",
)

#: No proxy for the dead URL. httpx, aiohttp and urllib take a proxy from
#: HTTP(S)_PROXY or ALL_PROXY, and on macOS from the system settings (read only
#: when no *_proxy variable is set), and would send it the request in plain
#: text, key header and all. Both spellings: a lowercase one wins.
NO_PROXY = "127.0.0.1,localhost"
NO_PROXY_VARIABLES = ("NO_PROXY", "no_proxy")

#: The variables that would send a client elsewhere whatever its host variable
#: says, removed.
CLEARED_VARIABLES = (
    # ChatAnthropic reads it before ANTHROPIC_BASE_URL.
    "ANTHROPIC_API_URL",
    # ChatOpenAI reads it before the SDK reads OPENAI_BASE_URL.
    "OPENAI_API_BASE",
    # The LangChain classes' LangSmith gateway, which ChatGoogleGenerativeAI
    # takes over GOOGLE_GEMINI_BASE_URL.
    "LANGSMITH_GATEWAY",
    # google-genai, ADK's Gemini and ChatGoogleGenerativeAI then run on Vertex AI.
    "GOOGLE_GENAI_USE_VERTEXAI",
    "GOOGLE_GENAI_USE_ENTERPRISE",
    # ChatAnthropic's and ChatOpenAI's own proxy, which NO_PROXY does not stop.
    "ANTHROPIC_PROXY",
    "OPENAI_PROXY",
)

#: LangSmith's tracing switches: ``langsmith.utils.tracing_is_enabled`` reads
#: TRACING_V2, then TRACING, each under LANGSMITH_ and LANGCHAIN_. Any one of
#: them set to "true" uploads every LangChain run.
TRACING_VARIABLES = (
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING",
)


def dead_base_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point every provider SDK's base URL at ``DEAD_URL``, with no proxy for
    it: a client a test did not route fails at once instead of reaching a
    provider."""
    for name in DEAD_BASE_URL_VARIABLES:
        monkeypatch.setenv(name, DEAD_URL)
    for name in NO_PROXY_VARIABLES:
        monkeypatch.setenv(name, NO_PROXY)
    for name in CLEARED_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def no_tracing() -> None:
    """Turn LangSmith tracing off for the rest of the process.

    langsmith caches what it reads of the environment (``get_env_var`` is an
    ``lru_cache``), so the switches are removed once, before any run, and its
    cache emptied in case it read them already.
    """
    for name in TRACING_VARIABLES:
        os.environ.pop(name, None)
    utils = sys.modules.get("langsmith.utils")
    if utils is not None:
        utils.get_env_var.cache_clear()
