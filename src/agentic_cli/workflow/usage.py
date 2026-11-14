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
for as long as that manager lives: a backend switch builds a new manager,
and the rows recorded before it do not follow (as with its ``agent_runs``).
A row per call, recorded where the event is made, before an app's
``on_event`` hook could drop it, under the turn it is made in, with how long
the call took (``latency_ms``: every call is timed, on both backends). The
ledger answers totals by run, agent, model, purpose and turn
(``UsageLedger.totals``); ``BaseWorkflowManager.usage`` is the open
conversation's. Its rows carry a ``cost`` that is None while no price is
known.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from agentic_cli.workflow.events import EventType, WorkflowEvent
from agentic_cli.workflow.sessions import (
    SessionRef,
    get_active_run,
    get_active_turn,
    turn_has_ended,
)

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

#: The service tiers langchain-openai names a part after when a call ran on
#: one (``priority_cache_read``, ``flex_reasoning``, ...).
_SERVICE_TIERS = ("", "priority_", "flex_")


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
    unreported = prompt is None and tool_prompt is None
    return normalized_tokens(
        input=None if unreported else (prompt or 0) + (tool_prompt or 0),
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
    ``output_token_details.reasoning``. langchain-openai names each part
    after the service tier a call ran on, priority or flex
    (``priority_cache_read``, ``flex_reasoning``): those count too.
    """
    usage = usage or {}
    inputs = usage.get("input_token_details") or {}
    outputs = usage.get("output_token_details") or {}
    by_ttl = sum(_count(inputs.get(key)) or 0 for key in _WRITES_BY_TTL)
    return normalized_tokens(
        input=_count(usage.get("input_tokens")),
        output=_count(usage.get("output_tokens")),
        cached=_part(inputs, "cache_read"),
        cache_write=by_ttl or _part(inputs, "cache_creation"),
        thinking=_part(outputs, "reasoning"),
    )


def _part(details: Mapping[str, Any], name: str) -> int | None:
    """A part of LangChain's usage details, on any service tier; None when none is reported."""
    counts = [_count(details.get(f"{tier}{name}")) for tier in _SERVICE_TIERS]
    reported = [count for count in counts if count is not None]
    return sum(reported) if reported else None


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


def adk_usage_event(
    model: str,
    usage: Any,
    kind: Any,
    *,
    invocation_id: str | None = None,
    latency_ms: float | None = None,
) -> WorkflowEvent:
    """``LLM_USAGE`` for one ADK call on ``model`` that reported ``usage``.

    Every place ADK's usage becomes an event makes it here: the event
    processor, the summarizer, ``generate_simple`` and the raw log.

    Args:
        model: The model's reference, as the event names it.
        usage: The response's usage metadata (``LlmResponse.usage_metadata``).
        kind: The provider kind of the model that answered; None when
            unknown (``adk_tokens``).
        invocation_id: ADK's invocation, for correlation.
        latency_ms: How long the call took, when it was measured.
    """
    return llm_usage_event(
        model,
        adk_tokens(usage, kind),
        total_tokens=getattr(usage, "total_token_count", None),
        invocation_id=invocation_id,
        latency_ms=latency_ms,
    )


# -- what a side call is for ----------------------------------------------------


@dataclass(frozen=True)
class SideCall:
    """What a side call (``generate_simple``) is made for.

    Attributes:
        purpose: Why it is made (``PURPOSE_*``).
        conversation: The conversation it is made for, outside a turn (the
            memory proposals of a conversation that ends); None inside one.
    """

    purpose: str
    conversation: SessionRef | None = None


_side_call: ContextVar[SideCall | None] = ContextVar("agentic_cli_side_call", default=None)


@contextlib.contextmanager
def side_call(purpose: str, conversation: SessionRef | None = None) -> Iterator[SideCall]:
    """The side calls made in the block are for ``purpose`` (and ``conversation``)."""
    made = SideCall(purpose, conversation)
    token = _side_call.set(made)
    try:
        yield made
    finally:
        _side_call.reset(token)


def current_side_call() -> SideCall | None:
    """What the side calls made now are for, as the caller said (``side_call``); None unsaid."""
    return _side_call.get()


# -- the ledger -------------------------------------------------------------------


@dataclass(frozen=True)
class UsageRow:
    """One model call in a conversation's ledger.

    Attributes:
        run: The run that made it (``RunRef.run``); None for the root's own
            work, and a call made in no run (outside a turn).
        agent: The agent whose work it was.
        purpose: Why it was made (``PURPOSE_*``); None for the root's own step.
        model: The model that answered.
        input: Every input token, cached and cache-write included.
        output: Every output token, thinking included.
        cached: The input tokens read from a cache.
        cache_write: The input tokens written to a cache.
        thinking: The output tokens spent thinking.
        latency_ms: How long the call took, when it was measured.
        turn: The turn of the conversation it was made in, as this manager
            counts them (``begin_turn``, from 1: ``/compact`` and a job's
            notice each run as a turn, and a conversation resumed in another
            manager starts again at 1), not ``SessionInfo.turns``, the
            user's own; None outside one of the conversation's turns.
        cost: What it cost; None while no price is known.
    """

    run: str | None
    agent: str | None
    purpose: str | None
    model: str
    input: int
    output: int
    cached: int
    cache_write: int
    thinking: int
    latency_ms: float | None
    turn: int | None
    cost: float | None = None

    @property
    def tokens(self) -> UsageTotals:
        """Its counts, as one call's totals."""
        return UsageTotals(
            input=self.input,
            output=self.output,
            cached=self.cached,
            cache_write=self.cache_write,
            thinking=self.thinking,
            calls=1,
        )


class UsageLedger:
    """One conversation's model calls, a row each, in memory for its manager's life.

    ``record`` makes a row from an ``LLM_USAGE`` event; the totals are read
    by run, agent, model, purpose and turn (``totals``), or grouped
    (``by_agent``, ``by_purpose``, ``by_model``).

    Args:
        conversation: The conversation it keeps (``UsageLedgers`` names
            it): a call carries a turn's number only while a turn of this
            conversation runs.
    """

    def __init__(self, conversation: SessionRef | None = None) -> None:
        self.conversation = conversation
        self._rows: list[UsageRow] = []
        self._total = UsageTotals()
        #: The turns begun in the conversation (``begin_turn``).
        self.turns = 0

    @property
    def rows(self) -> tuple[UsageRow, ...]:
        """Every call recorded, in order."""
        return tuple(self._rows)

    def begin_turn(self) -> int:
        """Count a turn of the conversation starting; the calls made in it carry its number."""
        self.turns += 1
        return self.turns

    def record(self, event: WorkflowEvent) -> UsageRow | None:
        """A row for ``event``, an ``LLM_USAGE`` whose counts are normalized; else None.

        The call is the run's the event names, else the one in progress
        (``get_active_run``), which the event is then given, so the stream
        and the ledger agree; its agent is the event's (``metadata["agent"]``),
        else that run's; its turn the number of the turn of this conversation
        running, None outside one (``_in_a_turn``).
        """
        if event.type is not EventType.LLM_USAGE:
            return None
        if event.run is None:
            event.run = get_active_run()
        ref = event.run
        metadata = event.metadata
        latency = metadata.get("latency_ms")
        row = UsageRow(
            run=ref.run if ref is not None else None,
            agent=metadata.get("agent") or (ref.agent if ref is not None else None),
            purpose=metadata.get("purpose"),
            model=str(metadata.get("model") or "unknown"),
            input=_count(metadata.get("prompt_tokens")) or 0,
            output=_count(metadata.get("completion_tokens")) or 0,
            cached=_count(metadata.get("cached_tokens")) or 0,
            cache_write=_count(metadata.get("cache_creation_tokens")) or 0,
            thinking=_count(metadata.get("thinking_tokens")) or 0,
            latency_ms=float(latency) if isinstance(latency, (int, float)) else None,
            turn=self.turns if self._in_a_turn() else None,
        )
        self._rows.append(row)
        self._total = self._total + row.tokens
        return row

    def _in_a_turn(self) -> bool:
        """Whether a turn of this conversation runs in this context.

        Not another conversation's (a conversation left proposes its
        memories outside its own turns), and not a turn that has ended (a
        task it started that outlived it: ``turn_has_ended``).
        """
        turn = get_active_turn()
        return (
            bool(self.turns)
            and turn is not None
            and turn == self.conversation
            and not turn_has_ended()
        )

    def totals(
        self,
        *,
        run: str | None = ...,  # type: ignore[assignment]
        agent: str | None = None,
        model: str | None = None,
        purpose: str | None = None,
        turn: int | None = None,
    ) -> UsageTotals:
        """The tokens of the calls that match, all of them by default.

        Args:
            run: One run's calls (``researcher-1#1``); None for the root's own
                work (and calls made in no run). Any run when not given.
            agent: One agent's calls (its steps and its tools' side calls).
            model: One model's calls.
            purpose: The calls made for one purpose.
            turn: The calls of one turn (``turns`` is the last one begun).
        """
        if run is ... and agent is None and model is None and purpose is None and turn is None:
            return self._total
        total = UsageTotals()
        for row in self._rows:
            if (
                (run is ... or row.run == run)
                and (agent is None or row.agent == agent)
                and (model is None or row.model == model)
                and (purpose is None or row.purpose == purpose)
                and (turn is None or row.turn == turn)
            ):
                total = total + row.tokens
        return total

    def by_agent(self) -> dict[str, UsageTotals]:
        """The agents' own steps, by agent name, in the order each first ran."""
        return self._grouped(
            lambda row: row.agent, lambda row: row.purpose in AGENT_STEP_PURPOSES
        )

    def by_purpose(self) -> dict[str, UsageTotals]:
        """The side calls, by purpose, in the order each was first made."""
        return self._grouped(
            lambda row: row.purpose, lambda row: row.purpose not in AGENT_STEP_PURPOSES
        )

    def by_model(
        self, *, run: str | None = ...  # type: ignore[assignment]
    ) -> dict[str, UsageTotals]:
        """The calls by model; one run's when ``run`` is given (None: the root's own)."""
        return self._grouped(lambda row: row.model, lambda row: run is ... or row.run == run)

    def run_usage(self, run: str) -> dict[str, dict[str, int]]:
        """What ``run`` used, by model, as its ``AGENT_RUN`` end carries it.

        ``{model: {"input": ..., "output": ...}}``: the run's own calls, its
        steps and its tools' side calls; a run it started has its own.
        """
        return {
            model: {"input": totals.input, "output": totals.output}
            for model, totals in self.by_model(run=run).items()
        }

    def _grouped(
        self,
        key: Callable[[UsageRow], str | None],
        keep: Callable[[UsageRow], bool],
    ) -> dict[str, UsageTotals]:
        grouped: dict[str, UsageTotals] = {}
        for row in self._rows:
            name = key(row)
            if name is None or not keep(row):
                continue
            grouped[name] = grouped.get(name, UsageTotals()) + row.tokens
        return grouped


class UsageLedgers:
    """A manager's ledgers, one per conversation (a ``SessionRef``, or None
    for a processor driven outside a manager, as unit tests do)."""

    def __init__(self) -> None:
        self._ledgers: dict[Any, UsageLedger] = {}

    def ledger(self, conversation: SessionRef | None) -> UsageLedger:
        """``conversation``'s ledger, made on first use."""
        ledger = self._ledgers.get(conversation)
        if ledger is None:
            ledger = self._ledgers[conversation] = UsageLedger(conversation)
        return ledger

    def record(self, event: WorkflowEvent) -> UsageRow | None:
        """Record an ``LLM_USAGE`` in the ledger of the turn it is made in
        (``get_active_turn``)."""
        return self.ledger(get_active_turn()).record(event)
