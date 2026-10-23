"""Token usage tracking for LLM invocations.

Accumulates token usage from LLM_USAGE events and provides
compact formatting for status bar display and /status breakdown.

The status bar's ``ctx`` segment is the main agent's last prompt against the
size at which its conversation is summarized (the ``context_trigger`` its
calls' usage carries): ``ctx 45k/170k (26%)``. A sub-agent's call and a
summary's add to the totals only, so they never move it.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Calls whose prompt is not the main agent's context: a summary's prompt is
#: the history it condenses, a sub-agent's its brief (or, inheriting, the
#: conversation, which is still not the main agent's context).
_NOT_THE_CONTEXT = ("summarization", "subagent")


def format_tokens(count: int) -> str:
    """Format token count for compact display.

    Args:
        count: Token count to format

    Returns:
        Compact string: "42", "1.5k", "12.5k", "146k", "1.2M". The value is
        rounded to one decimal first, so a whole result shows none (146,036
        is "146k"), and one that rounds up to 1,000k is "1M".
    """
    if count < 1_000:
        return str(count)
    value, unit = round(count / 1_000, 1), "k"
    if value >= 1_000:
        value, unit = round(count / 1_000_000, 1), "M"
    return f"{value:.1f}".removesuffix(".0") + unit


@dataclass
class UsageTracker:
    """Accumulates token usage from LLM_USAGE events.

    Fields are cumulative across all LLM invocations in a session, but for
    the main agent's context, which is its last call's.

    Attributes:
        last_prompt_tokens: The main agent's last prompt, in tokens.
        context_trigger: The prompt size its conversation is summarized at,
            from that call's usage; None when the backend gave none.
        window_known: Whether that trigger comes from the model's own window
            rather than the 128k floor; None when the backend gave none.
        context_trimmed_count: The ``CONTEXT_TRIMMED`` events seen: summaries
            and requests trimmed to fit (counted by the message processor).
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    cached_tokens: int = 0
    cache_creation_tokens: int = 0
    invocation_count: int = 0
    total_latency_ms: float = 0.0

    # The main agent's context
    last_prompt_tokens: int = 0
    context_trigger: int | None = None
    window_known: bool | None = None
    context_trimmed_count: int = 0

    def record(self, metadata: dict) -> None:
        """Accumulate values from an LLM_USAGE event's metadata dict.

        Handles None values and missing keys gracefully.

        A summarization call (``purpose="summarization"``) or a sub-agent's call
        (``purpose="subagent"``) adds to the totals only: its prompt is the
        history it condenses or the sub-agent's own, not the main agent's
        context. Any other call sets the context: its prompt, and the
        ``context_trigger`` and ``window_known`` its usage carries.

        Args:
            metadata: Metadata dict from a WorkflowEvent (LLM_USAGE type).
        """
        if metadata.get("purpose") not in _NOT_THE_CONTEXT:
            self.last_prompt_tokens = metadata.get("prompt_tokens") or 0
            self.context_trigger = metadata.get("context_trigger")
            self.window_known = metadata.get("window_known")

        self.prompt_tokens += metadata.get("prompt_tokens") or 0
        self.completion_tokens += metadata.get("completion_tokens") or 0
        self.thinking_tokens += metadata.get("thinking_tokens") or 0
        self.cached_tokens += metadata.get("cached_tokens") or 0
        self.cache_creation_tokens += metadata.get("cache_creation_tokens") or 0
        self.total_latency_ms += metadata.get("latency_ms") or 0.0
        self.invocation_count += 1

    @property
    def total_tokens(self) -> int:
        """Total tokens (prompt + completion)."""
        return self.prompt_tokens + self.completion_tokens

    def format_context(self) -> str:
        """The ``ctx`` segment: the main agent's last prompt against its trigger.

        Returns:
            ``"ctx 45k/170k (26%)"`` when the trigger is known and comes from
            the model's own window; ``"ctx 45k"`` when the window is unknown
            (the trigger then rests on the 128k floor) or the backend gives no
            trigger; ``""`` before the main agent's first call.
        """
        if self.last_prompt_tokens <= 0:
            return ""
        context = f"ctx {format_tokens(self.last_prompt_tokens)}"
        if self.context_trigger and self.window_known:
            share = round(100 * self.last_prompt_tokens / self.context_trigger)
            context += f"/{format_tokens(self.context_trigger)} ({share}%)"
        return context

    def format_status_bar(self) -> str:
        """Format token counts for status bar display.

        The context comes first: the bar is one line, clipped at the
        terminal's width, and late in a long session the totals are long.
        What the edge cuts is then the totals, which ``/status`` shows in
        full, never how close the next summary is.

        Returns:
            ``"ctx 45k/170k (26%) | Tokens: 12.5k in / 3.2k out"``, or ``""``
            if no usage yet (see ``format_context``).
        """
        if self.invocation_count == 0:
            return ""
        parts = [
            f"Tokens: {format_tokens(self.prompt_tokens)} in"
            f" / {format_tokens(self.completion_tokens)} out"
        ]
        context = self.format_context()
        if context:
            parts.insert(0, context)
        return " | ".join(parts)

    def format_detail_rows(self) -> list[tuple[str, str]]:
        """Format token usage as (label, value) rows for detailed display.

        The main agent's context (its last prompt) is ``/status``'s own row,
        beside the model's window and trigger.

        Returns:
            List of (label, formatted_value) tuples suitable for a table.
            Empty list if no invocations have been recorded.
        """
        if self.invocation_count == 0:
            return []
        rows: list[tuple[str, str]] = [
            ("", ""),  # Spacer
            ("LLM Invocations", str(self.invocation_count)),
            ("Input Tokens", format_tokens(self.prompt_tokens)),
            ("Output Tokens", format_tokens(self.completion_tokens)),
            ("Total Tokens", format_tokens(self.total_tokens)),
        ]
        if self.cached_tokens > 0:
            rows.append(("Cached Tokens", format_tokens(self.cached_tokens)))
        if self.cache_creation_tokens > 0:
            rows.append(("Cache Creation", format_tokens(self.cache_creation_tokens)))
        if self.thinking_tokens > 0:
            rows.append(("Thinking Tokens", format_tokens(self.thinking_tokens)))
        if self.total_latency_ms > 0:
            avg_ms = self.total_latency_ms / self.invocation_count
            rows.append(("Avg Latency", f"{avg_ms:.0f}ms"))
        if self.context_trimmed_count > 0:
            rows.append(("Context Trimmed", f"{self.context_trimmed_count} time(s)"))
        return rows

    def reset_context(self) -> None:
        """Forget the main agent's context; the totals stay.

        For a conversation the tracker has not seen (another backend's, after
        an orchestrator swap): the segment shows nothing until its first call.
        """
        self.last_prompt_tokens = 0
        self.context_trigger = None
        self.window_known = None

    def reset(self) -> None:
        """Zero all counters."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.thinking_tokens = 0
        self.cached_tokens = 0
        self.cache_creation_tokens = 0
        self.invocation_count = 0
        self.total_latency_ms = 0.0
        self.reset_context()
        self.context_trimmed_count = 0
