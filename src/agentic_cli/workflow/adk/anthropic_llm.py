"""Direct-API Anthropic LLM for ADK, with support for extra request params.

ADK's stock ``AnthropicLlm`` only forwards ``thinking`` to the Anthropic Messages
API — it never sends ``output_config`` (so the discrete effort ladder
``low|medium|high|xhigh|max`` is unreachable), and a bare ``claude-*`` model
string resolves through ``LLMRegistry`` to the *Vertex* ``Claude`` class (which
needs GOOGLE_CLOUD_PROJECT/LOCATION).

This module provides two things:

- ``DirectAnthropicLlm`` — a subclass that (a) uses the direct API
  (``ANTHROPIC_API_KEY``, inherited from ``AnthropicLlm``) and (b) injects extra
  ``messages.create`` kwargs — notably ``output_config={"effort": ...}`` — by
  wrapping ADK's ``_anthropic_client`` property. That property is ADK's own
  extension seam (the Vertex ``Claude`` class overrides the same property), so
  we get ADK's full request-build + streaming aggregation for free.
- ``register_direct_anthropic()`` — points the ADK ``LLMRegistry`` at this class
  for ``claude-*`` strings, so plain model strings in agent configs resolve to
  the direct API instead of Vertex.

Note: when a model is resolved *from a string* the registry constructs
``DirectAnthropicLlm(model=...)`` with class defaults (``max_tokens=8192``,
``effort=None``). Per-agent ``max_tokens``/``effort`` must be supplied by
constructing an instance directly (see ``manager._build_model_arg``); the
registry path is the convenience/safety net that fixes the Vertex gotcha.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any

from google.adk.models.anthropic_llm import AnthropicLlm
from google.adk.models.registry import LLMRegistry

from agentic_cli.logging import Loggers

logger = Loggers.workflow()

# Patterns to claim in the registry. The first two mirror ADK's ``Claude``
# (Vertex) patterns so registering overwrites those entries; the catch-all
# covers newer families (5.x, fable) that ADK's patterns miss.
_CLAUDE_PATTERNS = [r"claude-3-.*", r"claude-.*-4.*", r"claude-.*"]


class _ExtraParamMessages:
    """Wraps an Anthropic ``messages`` resource, merging fixed extra kwargs into
    every ``create`` call. Per-call kwargs win over the fixed extras."""

    def __init__(self, messages: Any, extra: dict[str, Any]) -> None:
        self._messages = messages
        self._extra = extra

    def __getattr__(self, name: str) -> Any:
        return getattr(self._messages, name)

    async def create(self, **kwargs: Any) -> Any:
        merged: dict[str, Any] = {**self._extra, **kwargs}
        return await self._messages.create(**merged)


class _ExtraParamClient:
    """Wraps an AsyncAnthropic client, swapping in an extra-param ``messages``."""

    def __init__(self, client: Any, extra: dict[str, Any]) -> None:
        self._client = client
        self._extra = extra

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    @cached_property
    def messages(self) -> _ExtraParamMessages:
        return _ExtraParamMessages(self._client.messages, self._extra)


class DirectAnthropicLlm(AnthropicLlm):
    """Anthropic via the direct API, with support for extra request params.

    Attributes:
        effort: Anthropic ``output_config.effort`` (``low|medium|high|xhigh|max``).
            Only valid on models that support it (Opus 4.5+/4.6/4.7/4.8,
            Sonnet 4.6); Sonnet 4.5 / Haiku 4.5 reject it. Leave ``None`` to omit.
        extra_params: Escape hatch merged into every ``messages.create`` call
            (e.g. ``{"service_tier": "..."}``). ADK's own per-call kwargs
            (model, messages, thinking, ...) always win over these.
        request_timeout: Overall client timeout (seconds). Set to a non-default
            value so ADK's non-streaming ``messages.create`` skips the SDK's
            "streaming required" guard for ``max_tokens`` above ~21k (the guard
            only fires when the client uses the default timeout).
        max_retries: Client-level retry count (defaults to the anthropic SDK's
            when ``None``); wire it from ``settings.retry_max_attempts``.
    """

    effort: str | None = None
    extra_params: dict[str, Any] = {}
    request_timeout: float | None = None
    max_retries: int | None = None

    @staticmethod
    def supported_models() -> list[str]:
        return list(_CLAUDE_PATTERNS)

    def _extra_create_params(self) -> dict[str, Any]:
        """The fixed kwargs to merge into ``messages.create`` for this instance."""
        extra: dict[str, Any] = dict(self.extra_params)
        if self.effort:
            output_config = dict(extra.get("output_config") or {})
            output_config.setdefault("effort", self.effort)
            extra["output_config"] = output_config
        return extra

    @cached_property
    def _anthropic_client(self):  # type: ignore[override]
        from anthropic import AsyncAnthropic

        client_kwargs: dict[str, Any] = {}
        if self.request_timeout is not None:
            client_kwargs["timeout"] = self.request_timeout
        if self.max_retries is not None:
            client_kwargs["max_retries"] = self.max_retries
        client = AsyncAnthropic(**client_kwargs)
        extra = self._extra_create_params()
        if not extra:
            return client
        return _ExtraParamClient(client, extra)


def register_direct_anthropic() -> None:
    """Resolve ``claude-*`` strings to ``DirectAnthropicLlm`` via ``LLMRegistry``.

    Overrides ADK's default (the Vertex ``Claude`` class) for the existing
    ``claude-3-.*`` / ``claude-.*-4.*`` patterns and adds a ``claude-.*``
    catch-all for newer families. Idempotent; clears the ``resolve()`` lru_cache
    so already-resolved names pick up the change.
    """
    LLMRegistry.register(DirectAnthropicLlm)
    cache_clear = getattr(LLMRegistry.resolve, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()
    logger.debug("direct_anthropic_registered", patterns=_CLAUDE_PATTERNS)
