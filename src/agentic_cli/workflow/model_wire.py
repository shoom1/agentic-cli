"""What a model request carries on the wire, the same on every backend.

The ADK adapter (``workflow/adk/model_factory.py``) and the LangChain adapter
(``workflow/langchain_models.py``) both read these, so an agent sends the same
thinking parameters, key, attribution headers and timeouts whichever backend
runs it. This module imports no provider SDK.
"""

from __future__ import annotations

from typing import Any

from agentic_cli.models import ProviderKind, ResolvedModel, ThinkingMechanism

#: A request to a provider without a timeout setting ends after this long: the
#: OpenAI SDK's own default, set explicitly. google-genai has no default, so a
#: hung Gemini connection would otherwise hang the turn.
REQUEST_TIMEOUT = 600.0
CONNECT_TIMEOUT = 5.0
#: The key an endpoint without one is sent: a local server without
#: authentication expects some key, and the client must never fall back to
#: OPENAI_API_KEY and send the user's OpenAI key to it.
PLACEHOLDER_KEY = "EMPTY"


def openrouter_headers(app_name: str) -> dict[str, str]:
    """OpenRouter's attribution headers, naming the application."""
    return {"HTTP-Referer": app_name, "X-Title": app_name}


def reasoning_body(resolved: ResolvedModel) -> dict[str, Any]:
    """The request fields that set an OpenAI-compatible model's thinking.

    OpenRouter takes ``reasoning.effort``; a user's endpoint takes what it
    declares: ``chat_template_kwargs.enable_thinking`` or ``reasoning_effort``.
    Empty for a model without a thinking knob.
    """
    thinking = resolved.thinking
    if thinking is None:
        return {}
    if thinking.mechanism is ThinkingMechanism.CHAT_TEMPLATE:
        return {"chat_template_kwargs": {"enable_thinking": thinking.enabled}}
    if resolved.endpoint.kind is ProviderKind.OPENROUTER:
        return {"reasoning": {"effort": thinking.level}}
    return {"reasoning_effort": thinking.level}  # an endpoint that declares effort
