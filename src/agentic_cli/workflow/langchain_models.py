"""LangChain chat models from resolved models.

``build_chat_model`` turns one ``ResolvedModel`` into a LangChain chat model.
The LangGraph backend builds one per agent when it builds its graph; Deep
Agents will take the same instances. Each model is an instance of its
provider's class, built with its key, timeout and retries passed explicitly.
Nothing is built from a model string (``init_chat_model``), whose provider
guessing and environment lookups would decide the host and the key.

``resolve_model`` has already decided the thinking value, the one
``max_tokens`` and the sampling settings to keep; this module only puts each
where its class reads it:

- Google: ``ChatGoogleGenerativeAI``. Thinking as ``thinking_budget``
  (Gemini 2.5) or ``reasoning_effort`` (Gemini 3+), with ``include_thoughts``
  while the model thinks.
- Anthropic: ``ChatAnthropic``, with ``anthropic_request_timeout`` as its
  timeout (the class has none of its own). ``thinking`` is adaptive (with a
  summarized display) plus a ``reasoning_effort``, enabled with a budget, or
  disabled.
- OpenAI: ``ChatOpenAI`` on the Responses API with ``store=False``. A
  reasoning model gets ``reasoning={"effort", "summary": "auto"}`` and its
  encrypted reasoning back (``include``), so the reasoning carries across
  tool calls without being stored.
- OpenRouter: ``ChatOpenRouter`` on OpenRouter's URL, naming the app;
  ``reasoning={"effort"}``. The class keeps OpenRouter's ``reasoning_details``
  and sends them back with the next request.
- The user's endpoints: ``OpenAICompatibleChat`` (Chat Completions) on their
  base URL, with the endpoint's thinking parameter in ``extra_body``.

Every instance gets ``profile=`` from the model's record (``profile_for``), so
whatever reads a model's profile (Deep Agents' summarization, for one) sees
the catalog's numbers, never LangChain's bundled ones.

The timeouts, the placeholder key, OpenRouter's headers and the reasoning
parameter come from ``workflow/model_wire.py``, which the ADK adapter reads
too.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI

from agentic_cli.logging import Loggers
from agentic_cli.models import (
    ModelRecord,
    NativeThinking,
    ProviderKind,
    ResolvedModel,
    ThinkingMechanism,
)
from agentic_cli.workflow.model_wire import (
    CONNECT_TIMEOUT,
    PLACEHOLDER_KEY,
    REQUEST_TIMEOUT,
    openrouter_headers,
    reasoning_body,
)

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings

logger = Loggers.workflow()

#: The mechanisms whose levels have names (the record's ladder).
_LADDERS = frozenset({
    ThinkingMechanism.GEMINI_LEVEL,
    ThinkingMechanism.ANTHROPIC_ADAPTIVE,
    ThinkingMechanism.EFFORT,
})
#: Fields ``ModelSettings.extra`` may not set: which model, where the request
#: goes and with which key.
_RESERVED = re.compile(
    r"^model(_name)?$|api_key|api_base|api_url|base_url|client|credentials|proxy"
    r"|headers|vertexai|project|location|app_url|app_title"
)


def build_chat_model(
    resolved: ResolvedModel, *, settings: "BaseSettings", app_name: str
) -> BaseChatModel:
    """Build an agent's LangChain chat model from its resolved model.

    Args:
        resolved: The agent's model, as ``resolve_model`` decided it.
        settings: Supplies the retry and timeout settings.
        app_name: The application's name, which OpenRouter is told.

    Raises:
        ValueError: For a provider kind there is no chat model for.
    """
    kind = resolved.record.kind
    if kind is ProviderKind.GOOGLE:
        cls, fields = _gemini(resolved, settings)
    elif kind is ProviderKind.ANTHROPIC:
        cls, fields = _claude(resolved, settings)
    elif kind is ProviderKind.OPENAI:
        cls, fields = _gpt(resolved, settings)
    elif kind is ProviderKind.OPENROUTER:
        cls, fields = _openrouter(resolved, settings, app_name)
    elif kind is ProviderKind.OPENAI_COMPATIBLE:
        cls, fields = OpenAICompatibleChat, _endpoint(resolved, settings)
    else:
        raise ValueError(f"No LangChain chat model for {resolved.ref} ({kind.value} models).")
    fields.update(_extra(resolved, cls))
    return cls(profile=profile_for(resolved.record), **fields)


def profile_for(record: ModelRecord) -> dict[str, Any]:
    """A LangChain model profile from a model's record.

    The numbers the record does not know are left out, so a reader falls back
    to its own default rather than a guess.
    """
    rule = record.thinking
    profile: dict[str, Any] = {
        "max_input_tokens": record.context_window,
        "max_output_tokens": record.max_output,
        "tool_calling": record.tools,
        "image_inputs": record.image_input,
        "reasoning_output": rule.mechanism is not ThinkingMechanism.NONE,
    }
    if rule.mechanism in _LADDERS and rule.levels:
        profile["reasoning_effort_levels"] = list(rule.levels)
    return {key: value for key, value in profile.items() if value is not None}


# --- per provider --------------------------------------------------------------------


def _gemini(resolved: ResolvedModel, settings: "BaseSettings") -> tuple[type, dict[str, Any]]:
    from langchain_google_genai import ChatGoogleGenerativeAI

    fields = {
        "model": resolved.record.model,
        "api_key": resolved.endpoint.api_key,
        "timeout": REQUEST_TIMEOUT,
        "max_retries": settings.retry_max_attempts,
        # Passed even when unset: the class's own temperature is 0.7.
        "temperature": resolved.temperature,
        "top_p": resolved.top_p,
        "top_k": resolved.top_k,
        "max_output_tokens": resolved.max_tokens,
        "stop": _stop(resolved),
    }
    thinking = resolved.thinking
    if thinking is not None:
        if thinking.mechanism is ThinkingMechanism.GEMINI_LEVEL:
            fields.update(reasoning_effort=thinking.level, include_thoughts=True)
        else:  # a budget; 0 turns thinking off
            fields.update(thinking_budget=thinking.budget, include_thoughts=thinking.enabled)
    return ChatGoogleGenerativeAI, fields


def _claude(resolved: ResolvedModel, settings: "BaseSettings") -> tuple[type, dict[str, Any]]:
    from langchain_anthropic import ChatAnthropic

    fields = {
        "model": resolved.record.model,
        "api_key": resolved.endpoint.api_key,
        "max_tokens": resolved.max_tokens,
        "default_request_timeout": settings.anthropic_request_timeout,
        "max_retries": settings.retry_max_attempts,
        "temperature": resolved.temperature,
        "top_p": resolved.top_p,
        "top_k": resolved.top_k,
        "stop_sequences": _stop(resolved),
    }
    thinking = resolved.thinking
    if thinking is not None:
        fields["thinking"] = _claude_thinking(thinking)
        if thinking.enabled and thinking.mechanism is ThinkingMechanism.ANTHROPIC_ADAPTIVE:
            fields["reasoning_effort"] = thinking.level
    return ChatAnthropic, fields


def _claude_thinking(thinking: NativeThinking) -> dict[str, Any]:
    if not thinking.enabled:
        return {"type": "disabled"}
    if thinking.mechanism is ThinkingMechanism.ANTHROPIC_ADAPTIVE:
        # Opus 4.7 and later omit the thinking text unless asked for a summary.
        return {"type": "adaptive", "display": "summarized"}
    return {"type": "enabled", "budget_tokens": thinking.budget}


def _gpt(resolved: ResolvedModel, settings: "BaseSettings") -> tuple[type, dict[str, Any]]:
    fields = {
        **_openai_fields(resolved, settings),
        "api_key": resolved.endpoint.api_key,
        "use_responses_api": True,
        "store": False,
    }
    thinking = resolved.thinking
    if thinking is not None:
        reasoning = {"effort": thinking.level}
        if thinking.enabled:
            reasoning["summary"] = "auto"
        # Not stored, the reasoning comes back encrypted, to be sent with
        # the next request of the turn.
        fields.update(reasoning=reasoning, include=["reasoning.encrypted_content"])
    return ChatOpenAI, fields


def _openrouter(
    resolved: ResolvedModel, settings: "BaseSettings", app_name: str
) -> tuple[type, dict[str, Any]]:
    from langchain_openrouter import ChatOpenRouter

    _not_sent(resolved, "top_k")
    headers = openrouter_headers(app_name)
    fields = {
        "model": resolved.record.model,
        "api_key": resolved.endpoint.api_key,
        # Explicit, or the class reads its host and attribution from the
        # environment.
        "base_url": resolved.endpoint.base_url,
        "app_url": headers["HTTP-Referer"],
        "app_title": headers["X-Title"],
        "timeout": int(REQUEST_TIMEOUT * 1000),  # milliseconds
        # The SDK retries on a backoff whose window grows with this number.
        "max_retries": settings.retry_max_attempts,
        "max_tokens": resolved.max_tokens,
        "temperature": resolved.temperature,
        "top_p": resolved.top_p,
        "stop": _stop(resolved),
        **reasoning_body(resolved),  # {"reasoning": {"effort": ...}}
    }
    return ChatOpenRouter, fields


def _endpoint(resolved: ResolvedModel, settings: "BaseSettings") -> dict[str, Any]:
    return {
        **_openai_fields(resolved, settings),
        "api_key": resolved.endpoint.api_key or PLACEHOLDER_KEY,
        "base_url": resolved.endpoint.base_url,
        "use_responses_api": False,
        # ChatOpenAI streams usage only from OpenAI's own host unless told.
        "stream_usage": True,
        "extra_body": reasoning_body(resolved) or None,
    }


def _openai_fields(resolved: ResolvedModel, settings: "BaseSettings") -> dict[str, Any]:
    """What ``ChatOpenAI`` takes for GPT and the user's endpoints alike."""
    _not_sent(resolved, "top_k")
    return {
        "model": resolved.record.model,
        "timeout": httpx.Timeout(REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT),
        "max_retries": settings.retry_max_attempts,
        "max_tokens": resolved.max_tokens,
        "temperature": resolved.temperature,
        "top_p": resolved.top_p,
        "stop": _stop(resolved),
    }


def _stop(resolved: ResolvedModel) -> list[str] | None:
    return list(resolved.stop_sequences) if resolved.stop_sequences is not None else None


def _not_sent(resolved: ResolvedModel, setting: str) -> None:
    """Log a generation setting the model's API has no field for (as on ADK)."""
    if getattr(resolved, setting) is not None:
        logger.debug("model_setting_not_sent", setting=setting, model=str(resolved.ref))


def _extra(resolved: ResolvedModel, cls: type) -> dict[str, Any]:
    """The keys of ``extra`` the chat model class takes; the rest are logged."""
    accepted = set()
    for name, field in cls.model_fields.items():
        accepted.add(name)
        if field.alias:
            accepted.add(field.alias)
    kept = {}
    for key, value in resolved.extra.items():
        if key in accepted and not _RESERVED.search(key):
            kept[key] = value
        else:
            logger.warning("model_settings_extra_ignored", key=key, model=str(resolved.ref))
    return kept


# --- the user's endpoints --------------------------------------------------------------

#: What an endpoint's messages are tagged with: no provider LangChain has a
#: translator for, so ``content_blocks`` reads their reasoning.
_COMPATIBLE_PROVIDER = "openai_compatible"


class OpenAICompatibleChat(ChatOpenAI):
    """``ChatOpenAI`` for an OpenAI-compatible server, keeping its reasoning.

    vLLM and similar servers send the model's reasoning beside its answer, as
    ``reasoning`` (``reasoning_content`` on older servers), and ``ChatOpenAI``
    drops it. This class keeps it, streamed and final, in the message's
    ``additional_kwargs["reasoning_content"]``, where ``content_blocks`` reads
    it as a standard reasoning block. Its messages are tagged as an
    OpenAI-compatible server's, since OpenAI's own translator would skip the
    reasoning. The token limit goes as ``max_tokens``, which every compatible
    server knows, rather than OpenAI's newer ``max_completion_tokens``.
    """

    def _create_chat_result(self, response: Any, generation_info: dict | None = None) -> Any:
        result = super()._create_chat_result(response, generation_info)
        data = response if isinstance(response, Mapping) else response.model_dump()
        for generation, choice in zip(result.generations, data.get("choices") or ()):
            _keep_reasoning(generation.message, choice.get("message") or {})
        return result

    def _convert_chunk_to_generation_chunk(
        self, chunk: dict, default_chunk_class: type, base_generation_info: dict | None
    ) -> Any:
        generation = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices") or ()
        if generation is not None and choices:
            _keep_reasoning(generation.message, choices[0].get("delta") or {})
        return generation

    def _get_request_payload(self, input_: Any, *, stop: list[str] | None = None, **kwargs: Any):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if "max_completion_tokens" in payload:
            payload["max_tokens"] = payload.pop("max_completion_tokens")
        return payload


def _keep_reasoning(message: BaseMessage, fields: Mapping[str, Any]) -> None:
    """Keep a message's (or delta's) reasoning text, and tag it as ours."""
    reasoning = fields.get("reasoning") or fields.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        message.additional_kwargs["reasoning_content"] = reasoning
    message.response_metadata["model_provider"] = _COMPATIBLE_PROVIDER


# --- reasoning across providers --------------------------------------------------------

#: The tag each provider's class gives its AI messages
#: (``response_metadata["model_provider"]``).
_PROVIDER_TAGS = {
    ProviderKind.GOOGLE: "google_genai",
    ProviderKind.ANTHROPIC: "anthropic",
    ProviderKind.OPENAI: "openai",
    ProviderKind.OPENROUTER: "openrouter",
    ProviderKind.OPENAI_COMPATIBLE: _COMPATIBLE_PROVIDER,
}
_REASONING_BLOCKS = frozenset({"thinking", "redacted_thinking", "reasoning"})
_REASONING_FIELDS = frozenset({"reasoning_content", "reasoning", "reasoning_details"})


def without_foreign_reasoning(
    messages: Sequence[BaseMessage], kind: ProviderKind
) -> list[BaseMessage]:
    """``messages`` for a model of ``kind``, without other providers' reasoning.

    A model switch that keeps the session, or a tree of agents on several
    providers, sends a model AI messages another provider wrote. Their
    reasoning belongs to that provider: Claude rejects a thinking block it did
    not sign, Gemini drops a foreign one with a warning on every call, and
    OpenRouter would take an endpoint's reasoning as its own. Each such
    message is copied without its reasoning blocks and fields; the answer,
    tool calls and the given messages themselves are left as they are, and a
    message of the model's own provider keeps its reasoning (Claude's signed
    thinking must go back within a tool loop).
    """
    own = _PROVIDER_TAGS[kind]
    kept: list[BaseMessage] = []
    for message in messages:
        if not isinstance(message, AIMessage) or message.response_metadata.get(
            "model_provider"
        ) == own:
            kept.append(message)
            continue
        content = message.content
        if isinstance(content, list):
            content = [
                block
                for block in content
                if not (isinstance(block, dict) and block.get("type") in _REASONING_BLOCKS)
            ]
        fields = {
            key: value
            for key, value in message.additional_kwargs.items()
            if key not in _REASONING_FIELDS
        }
        if content == message.content and len(fields) == len(message.additional_kwargs):
            kept.append(message)
        else:
            kept.append(message.model_copy(update={"content": content, "additional_kwargs": fields}))
    return kept
