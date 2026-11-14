"""Token usage: what each model call used, kept by conversation.

Every model call the framework makes is one ``LLM_USAGE`` event, on both
backends: an agent's own step, a summary of a conversation, and a side call
(``generate_simple``: the summaries ``web_fetch`` and the knowledge base ask
for, the memory proposals when a conversation ends). Its counts are
normalized where it is made (``normalized_tokens``): ``prompt_tokens`` is
every input token, the cache's reads and writes included, and
``completion_tokens`` every output token, the thinking included where the
provider reports it; ``cached_tokens``, ``cache_creation_tokens`` and
``thinking_tokens`` are parts of them. Providers report them otherwise
(Anthropic's API leaves the cache out of its input; Gemini, and ADK's Claude
adapter, keep the thinking out of the output), and the adapters in between
(ADK's, LangChain's partner packages) differ again: ``adk_tokens`` and
``langchain_tokens`` read each shape.

``metadata["purpose"]`` says why a call was made: none for the root's own
step, ``subagent`` for a sub-agent's, and a side call's purpose otherwise
(``PURPOSE_*``). The caller of a side call names it (``side_call``).

Each manager keeps a ledger per conversation (``UsageLedgers``), in memory
for the process: a row per call, recorded where the event is made, before an
app's ``on_event`` hook could drop it, under the turn it is made in. The
ledger answers totals by run, agent, model, purpose and turn
(``UsageLedger.totals``); ``BaseWorkflowManager.usage`` is the open
conversation's. Its rows carry a ``cost`` that is None while no price is
known.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agentic_cli.workflow.events import WorkflowEvent

#: Why a model call was made (``LLM_USAGE``'s ``metadata["purpose"]``). The
#: root's own step has none.
PURPOSE_SUBAGENT = "subagent"  # a sub-agent's own step
PURPOSE_SUMMARIZATION = "summarization"  # a summary of a conversation
PURPOSE_TOOL = "tool"  # a summary a tool asked for (web_fetch, the knowledge base)
PURPOSE_MEMORY_EXTRACTION = "memory_extraction"  # memory proposals as a conversation ends
PURPOSE_SIDE_CALL = "side_call"  # any other generate_simple call
#: The purposes of an agent's own steps; every other is a side call's.
AGENT_STEP_PURPOSES = frozenset({None, PURPOSE_SUBAGENT})

#: Provider kinds whose ADK adapter counts a reply's thinking apart from its
#: candidates (``thoughts_token_count`` beside ``candidates_token_count``):
#: Gemini's own counts, and ADK's Claude adapter (its candidates are Claude's
#: output less the thinking). OpenAI's adapters, and the OpenAI-compatible
#: one, count the reasoning inside the completion.
_THOUGHTS_APART = frozenset({"google", "anthropic"})

#: Anthropic's cache writes by time to live, which langchain-anthropic
#: reports instead of ``cache_creation`` when the API breaks them down.
_WRITES_BY_TTL = ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")


def _count(value: Any) -> int | None:
    """``value`` when it is a token count, else None (a fake's or a missing field)."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


@dataclass(frozen=True)
class UsageTotals:
    """Tokens summed over model calls (one call's: ``calls`` 1).

    Attributes:
        input: Every input token, cached and cache-write included.
        output: Every output token, thinking included where the provider
            reports it.
        cached: The input tokens read from a cache.
        cache_write: The input tokens written to a cache.
        thinking: The output tokens spent thinking.
        calls: How many model calls.
    """

    input: int = 0
    output: int = 0
    cached: int = 0
    cache_write: int = 0
    thinking: int = 0
    calls: int = 0

    def __add__(self, other: "UsageTotals") -> "UsageTotals":
        if not isinstance(other, UsageTotals):
            return NotImplemented
        return UsageTotals(
            input=self.input + other.input,
            output=self.output + other.output,
            cached=self.cached + other.cached,
            cache_write=self.cache_write + other.cache_write,
            thinking=self.thinking + other.thinking,
            calls=self.calls + other.calls,
        )


def normalized_tokens(
    *,
    input: int | None,
    output: int | None,
    cached: int | None = None,
    cache_write: int | None = None,
    thinking: int | None = None,
    input_counts_cache: bool = True,
    output_counts_thinking: bool = True,
) -> UsageTotals:
    """One call's tokens as the ledger counts them, from what was reported.

    The one place a provider's report becomes the ledger's counts: ``input``
    every input token, the cache's reads and writes included (Anthropic's
    own API leaves them out: ``input_counts_cache`` False), ``output`` every
    output token, the thinking included (Gemini's own counts, and ADK's
    Claude adapter, keep it apart: ``output_counts_thinking`` False). A count
    that was not reported is 0.
    """
    cached_n, write_n, thinking_n = cached or 0, cache_write or 0, thinking or 0
    return UsageTotals(
        input=(input or 0) + (0 if input_counts_cache else cached_n + write_n),
        output=(output or 0) + (0 if output_counts_thinking else thinking_n),
        cached=cached_n,
        cache_write=write_n,
        thinking=thinking_n,
        calls=1,
    )


def adk_tokens(usage: Any, kind: Any) -> UsageTotals:
    """A call's tokens from ADK's usage (``LlmResponse.usage_metadata``).

    Every ADK adapter counts the cache in ``prompt_token_count`` (ADK's
    Claude adapter adds Anthropic's reads and writes back), and Gemini's
    server-side tool prompt (``tool_use_prompt_token_count``) is input too.
    Gemini and ADK's Claude adapter count the thinking apart from the
    candidates, OpenAI's adapters inside them (``_THOUGHTS_APART``; an
    unknown kind is taken as apart, as a native ADK YAML app's bare model
    names are Gemini's and Claude's). ADK's Claude adapter keeps the cache
    writes under ``cache_creation_input_tokens``.

    Args:
        usage: The response's usage metadata.
        kind: The provider kind of the model that answered (a
            ``ProviderKind`` or its value); None when unknown.
    """
    prompt = _count(getattr(usage, "prompt_token_count", None))
    tool_prompt = _count(getattr(usage, "tool_use_prompt_token_count", None))
    kind = getattr(kind, "value", kind)
    return normalized_tokens(
        input=None if prompt is None and tool_prompt is None else (prompt or 0) + (tool_prompt or 0),
        output=_count(getattr(usage, "candidates_token_count", None)),
        cached=_count(getattr(usage, "cached_content_token_count", None)),
        cache_write=_count(getattr(usage, "cache_creation_input_tokens", None)),
        thinking=_count(getattr(usage, "thoughts_token_count", None)),
        output_counts_thinking=kind is not None and kind not in _THOUGHTS_APART,
    )


def langchain_tokens(usage: Mapping[str, Any] | None) -> UsageTotals:
    """A call's tokens from LangChain's ``usage_metadata``.

    Every partner package the framework builds on (Anthropic, OpenAI,
    Google, OpenRouter) keeps LangChain's convention: ``input_tokens``
    counts the cache (langchain-anthropic adds Anthropic's reads and writes
    back) and ``output_tokens`` the reasoning (langchain-google-genai adds
    Gemini's thoughts), their parts in ``input_token_details`` (``cache_read``,
    ``cache_creation``, or the writes by time to live, when Anthropic breaks
    them down and the package zeroes ``cache_creation``) and
    ``output_token_details.reasoning``.
    """
    usage = usage or {}
    inputs = usage.get("input_token_details") or {}
    outputs = usage.get("output_token_details") or {}
    by_ttl = sum(_count(inputs.get(key)) or 0 for key in _WRITES_BY_TTL)
    return normalized_tokens(
        input=_count(usage.get("input_tokens")),
        output=_count(usage.get("output_tokens")),
        cached=_count(inputs.get("cache_read")),
        cache_write=by_ttl or _count(inputs.get("cache_creation")),
        thinking=_count(outputs.get("reasoning")),
    )


def llm_usage_event(
    model: str,
    tokens: UsageTotals,
    *,
    total_tokens: int | None = None,
    invocation_id: str | None = None,
    latency_ms: float | None = None,
) -> WorkflowEvent:
    """``LLM_USAGE`` for one call on ``model``, its counts normalized.

    ``prompt_tokens`` is every input token and ``completion_tokens`` every
    output token (``normalized_tokens``); the parts (``cached_tokens``,
    ``cache_creation_tokens``, ``thinking_tokens``) are given when there are
    any. ``total_tokens`` is the provider's own, when it gave one.
    """
    return WorkflowEvent.llm_usage(
        model=model,
        prompt_tokens=tokens.input,
        completion_tokens=tokens.output,
        total_tokens=_count(total_tokens),
        thinking_tokens=tokens.thinking or None,
        cached_tokens=tokens.cached or None,
        cache_creation_tokens=tokens.cache_write or None,
        invocation_id=invocation_id,
        latency_ms=latency_ms,
    )
