"""No offline test reaches a model provider: every provider SDK's host, dead.

A provider SDK given no host reads one from the environment, and a test that
routes one client to a fake can miss another (ADK's ``AnthropicLlm`` module
binds ``AsyncAnthropic`` at import, so routing ``anthropic.AsyncAnthropic``
alone misses a native ADK YAML app's bare ``claude-*``). With the variables
below pointed at a local port nothing listens on, such a client fails at once,
locally, instead of reaching the provider with whatever key the environment
holds. The root conftest's ``_dead_provider_urls`` does this for every test
but the live ones (marked ``llm``), which keep the real environment their
keys come from.
"""

from __future__ import annotations

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
)


def dead_base_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point every provider SDK's base URL at ``DEAD_URL``: a client a test did
    not route fails at once instead of reaching a provider."""
    for name in DEAD_BASE_URL_VARIABLES:
        monkeypatch.setenv(name, DEAD_URL)
    for name in CLEARED_VARIABLES:
        monkeypatch.delenv(name, raising=False)
