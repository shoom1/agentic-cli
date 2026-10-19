"""ADK models from resolved models.

``build_adk_model`` turns one ``ResolvedModel`` into what an ADK ``LlmAgent``
takes: a model instance, the agent's ``generate_content_config`` and, for a
model with a thinking knob, a ``BuiltInPlanner``. Every agent gets an instance
built here; nothing is resolved from a model string through ADK's registry,
which would map a bare ``claude-*`` to Claude on Vertex AI and give every
model its class defaults.

``resolve_model`` has already decided the thinking value, the one
``max_tokens`` and the sampling settings to keep; this module only puts each
where its ADK class reads it:

- Google: ``Gemini``, its key in ``client_kwargs``. Thinking through the
  planner: ``thinking_budget`` (Gemini 2.5) or ``thinking_level`` (Gemini 3+).
  Retries are google-genai's ``HttpRetryOptions`` in the config (429 is not
  retried), with a timeout.
- Anthropic: ADK's ``AnthropicLlm`` with an ``AsyncAnthropic`` client (key,
  ``anthropic_request_timeout``, ``retry_max_attempts`` retries). The planner
  carries ``thinking_budget`` 0 (disabled), -1 (adaptive) or the budget, and
  ``AnthropicGenerateContentConfig`` the effort. ``max_tokens`` goes on the
  instance and, as ``max_output_tokens``, in the config: ADK sends the
  config's value when it is set.
- OpenAI: ``OpenAIResponsesLlm`` (``store=False``) with an ``AsyncOpenAI``
  client; the effort in ``OpenAIGenerateContentConfig``, for which ADK
  requests a concise reasoning summary.
- OpenRouter and the user's endpoints: ``OpenAICompatibleLlm`` (Chat
  Completions) with an ``AsyncOpenAI`` client on their base URL (OpenRouter's
  carries attribution headers naming the app). The reasoning parameter goes
  in ``extra_body``: OpenRouter's ``reasoning.effort``, or what the endpoint
  declares, ``chat_template_kwargs.enable_thinking`` or ``reasoning_effort``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
from google.adk.models.base_llm import BaseLlm
from google.adk.planners import BuiltInPlanner
from google.genai import types

from agentic_cli.logging import Loggers
from agentic_cli.models import NativeThinking, ProviderKind, ResolvedModel, ThinkingMechanism

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings

logger = Loggers.workflow()

# A request to a provider without a timeout setting ends after this long: the
# OpenAI SDK's own default, set explicitly. google-genai has no default, so a
# hung Gemini connection would otherwise hang the turn.
_REQUEST_TIMEOUT = 600.0
_CONNECT_TIMEOUT = 5.0
# google-genai retries these itself. A 429 is not retried: the turn fails with
# a rate-limit error the CLI reports.
_GEMINI_RETRY_STATUS_CODES = [500, 502, 503, 504]
_GEMINI_LEVELS = {
    "minimal": types.ThinkingLevel.MINIMAL,
    "low": types.ThinkingLevel.LOW,
    "medium": types.ThinkingLevel.MEDIUM,
    "high": types.ThinkingLevel.HIGH,
}
# ADK's AnthropicLlm reads a negative thinking budget as adaptive thinking.
_ADAPTIVE = -1
# The key an endpoint without one is sent: a local server without
# authentication expects some key, and the client must never fall back to
# OPENAI_API_KEY and send the user's OpenAI key to it.
_PLACEHOLDER_KEY = "EMPTY"


@dataclass(frozen=True)
class AdkModel:
    """What an ADK ``LlmAgent`` takes for one resolved model.

    Attributes:
        model: The model instance, with its own client.
        generate_content_config: The agent's generation settings, effort and
            (Gemini) retries.
        planner: The thinking config; None for a model without a thinking
            knob, and for GPT, whose effort is in the config.
    """

    model: BaseLlm
    generate_content_config: types.GenerateContentConfig
    planner: BuiltInPlanner | None


def build_adk_model(
    resolved: ResolvedModel, *, settings: "BaseSettings", app_name: str
) -> AdkModel:
    """Build an agent's ADK model from its resolved model.

    Args:
        resolved: The agent's model, as ``resolve_model`` decided it.
        settings: Supplies the retry and timeout settings.
        app_name: The application's name, which some providers are told.

    Raises:
        ValueError: For a provider kind the ADK backend cannot build.
    """
    kind = resolved.record.kind
    if kind is ProviderKind.GOOGLE:
        return _gemini(resolved, settings)
    if kind is ProviderKind.ANTHROPIC:
        return _claude(resolved, settings)
    if kind is ProviderKind.OPENAI:
        return _gpt(resolved, settings)
    if kind in (ProviderKind.OPENROUTER, ProviderKind.OPENAI_COMPATIBLE):
        return _openai_compatible(resolved, settings, app_name)
    raise ValueError(f"The ADK backend cannot build {resolved.ref} ({kind.value} models).")


def _gemini(resolved: ResolvedModel, settings: "BaseSettings") -> AdkModel:
    from google.adk.models import Gemini

    http_options = types.HttpOptions(
        timeout=int(_REQUEST_TIMEOUT * 1000),  # milliseconds
        retry_options=types.HttpRetryOptions(
            initial_delay=settings.retry_initial_delay,
            attempts=settings.retry_max_attempts,
            exp_base=settings.retry_backoff_factor,
            http_status_codes=_GEMINI_RETRY_STATUS_CODES,
        ),
    )
    thinking = resolved.thinking
    planner = None
    if thinking is not None:
        if thinking.mechanism is ThinkingMechanism.GEMINI_LEVEL:
            thinking_config = types.ThinkingConfig(
                include_thoughts=True, thinking_level=_GEMINI_LEVELS[thinking.level]
            )
        else:  # a budget; 0 turns thinking off
            thinking_config = types.ThinkingConfig(
                include_thoughts=thinking.enabled, thinking_budget=thinking.budget
            )
        planner = BuiltInPlanner(thinking_config=thinking_config)
    return AdkModel(
        model=Gemini(
            model=resolved.record.model,
            client_kwargs={
                "api_key": resolved.endpoint.api_key,
                # A bound for the calls ADK makes without a config of their
                # own; a request's own options (generation's above, a cache
                # create's) replace it.
                "http_options": types.HttpOptions(timeout=_GEMINI_CLIENT_TIMEOUT_MS),
            },
        ),
        generate_content_config=_config(
            resolved, types.GenerateContentConfig, http_options=http_options
        ),
        planner=planner,
    )


# How long a Gemini call without options of its own may take, in milliseconds.
# ADK's context cache deletes an agent's expired cache with no options, and
# google-genai then sets no timeout at all, so a delete that is never answered
# would hold the turn. With this, it costs at most this long, and the turn
# goes on.
_GEMINI_CLIENT_TIMEOUT_MS = 60_000


def _claude(resolved: ResolvedModel, settings: "BaseSettings") -> AdkModel:
    from anthropic import AsyncAnthropic
    from google.adk.models.anthropic_llm import (
        AnthropicGenerateContentConfig,
        AnthropicLlm,
    )

    thinking = resolved.thinking
    planner = None
    if thinking is not None:
        planner = BuiltInPlanner(
            thinking_config=types.ThinkingConfig(
                include_thoughts=thinking.enabled, thinking_budget=_claude_budget(thinking)
            )
        )
    client = AsyncAnthropic(
        api_key=resolved.endpoint.api_key,
        # Also what lets ADK's unstreamed requests carry a max_tokens the SDK
        # would otherwise insist on streaming.
        timeout=settings.anthropic_request_timeout,
        max_retries=settings.retry_max_attempts,
    )
    return AdkModel(
        model=AnthropicLlm(
            model=resolved.record.model, max_tokens=resolved.max_tokens, client=client
        ),
        generate_content_config=_config(
            resolved,
            AnthropicGenerateContentConfig,
            # Adaptive Claude's effort; None when thinking is off or on a budget.
            effort=thinking.level if thinking is not None else None,
        ),
        planner=planner,
    )


def _claude_budget(thinking: NativeThinking) -> int:
    """ADK's ``thinking_budget`` for Claude: 0 disabled, -1 adaptive, else the budget."""
    if not thinking.enabled:
        return 0
    if thinking.mechanism is ThinkingMechanism.ANTHROPIC_ADAPTIVE:
        return _ADAPTIVE
    return thinking.budget


def _gpt(resolved: ResolvedModel, settings: "BaseSettings") -> AdkModel:
    from google.adk.integrations.openai import (
        OpenAIGenerateContentConfig,
        OpenAIResponsesLlm,
    )

    level = resolved.thinking.level if resolved.thinking is not None else None
    # ADK's config takes the efforts it knows, which lack "none"; the model's
    # own reasoning default can say it.
    reasoning = {"effort": "none"} if level == "none" else None
    return AdkModel(
        model=OpenAIResponsesLlm(
            model=resolved.record.model,
            client=_openai_client(resolved, settings),
            store=False,
            reasoning=reasoning,
        ),
        generate_content_config=_config(
            resolved,
            OpenAIGenerateContentConfig,
            effort=level if reasoning is None else None,
        ),
        planner=None,
    )


def _openai_compatible(
    resolved: ResolvedModel, settings: "BaseSettings", app_name: str
) -> AdkModel:
    from agentic_cli.workflow.adk.openai_compatible import OpenAICompatibleLlm

    endpoint = resolved.endpoint
    headers = None
    if endpoint.kind is ProviderKind.OPENROUTER:
        headers = {"HTTP-Referer": app_name, "X-Title": app_name}
    client = _openai_client(
        resolved, settings, base_url=endpoint.base_url, default_headers=headers
    )
    return AdkModel(
        model=OpenAICompatibleLlm(
            model=resolved.record.model,
            max_tokens=resolved.max_tokens,
            client=client,
            extra_body=_reasoning_body(resolved),
        ),
        generate_content_config=_config(resolved, types.GenerateContentConfig),
        planner=None,
    )


def _reasoning_body(resolved: ResolvedModel) -> dict[str, Any]:
    """The request fields that set an OpenAI-compatible model's thinking."""
    thinking = resolved.thinking
    if thinking is None:
        return {}
    if thinking.mechanism is ThinkingMechanism.CHAT_TEMPLATE:
        return {"chat_template_kwargs": {"enable_thinking": thinking.enabled}}
    if resolved.endpoint.kind is ProviderKind.OPENROUTER:
        return {"reasoning": {"effort": thinking.level}}
    return {"reasoning_effort": thinking.level}  # an endpoint that declares effort


def _openai_client(resolved: ResolvedModel, settings: "BaseSettings", **kwargs: Any):
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        api_key=resolved.endpoint.api_key or _PLACEHOLDER_KEY,
        timeout=httpx.Timeout(_REQUEST_TIMEOUT, connect=_CONNECT_TIMEOUT),
        max_retries=settings.retry_max_attempts,
        **kwargs,
    )


def _config(
    resolved: ResolvedModel,
    config_class: type[types.GenerateContentConfig],
    **fields: Any,
) -> types.GenerateContentConfig:
    """The agent's ``generate_content_config``.

    The resolved generation settings, then ``fields``, then the keys of
    ``extra`` the config class takes; the other keys are logged and dropped.
    Unset values are left out.
    """
    values: dict[str, Any] = {
        "temperature": resolved.temperature,
        "top_p": resolved.top_p,
        "top_k": resolved.top_k,
        "max_output_tokens": resolved.max_tokens,
        "stop_sequences": (
            list(resolved.stop_sequences) if resolved.stop_sequences is not None else None
        ),
        **fields,
    }
    for key, value in resolved.extra.items():
        if key in config_class.model_fields:
            values[key] = value
        else:
            logger.warning("model_settings_extra_ignored", key=key, model=str(resolved.ref))
    return config_class(**{key: value for key, value in values.items() if value is not None})
