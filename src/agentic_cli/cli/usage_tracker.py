"""Token usage tracking for LLM invocations.

Accumulates token usage from LLM_USAGE events and provides
compact formatting for status bar display and /status breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def format_tokens(count: int) -> str:
    """Format token count for compact display.

    Args:
        count: Token count to format

    Returns:
        Compact string: "42", "1.5k", "12.5k", "1.2M"
    """
    if count >= 1_000_000:
        value = count / 1_000_000
        return f"{value:.1f}M" if value != int(value) else f"{int(value)}M"
    if count >= 1_000:
        value = count / 1_000
        return f"{value:.1f}k" if value != int(value) else f"{int(value)}k"
    return str(count)


@dataclass
class UsageTracker:
    """Accumulates token usage from LLM_USAGE events.

    Fields are cumulative across all LLM invocations in a session.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    cached_tokens: int = 0
    cache_creation_tokens: int = 0
    invocation_count: int = 0
    total_latency_ms: float = 0.0

    # Context window tracking
    last_prompt_tokens: int = 0
    context_trimmed_count: int = 0

    def record(self, metadata: dict, context_trimmed_already: bool = False) -> bool:
        """Accumulate values from an LLM_USAGE event's metadata dict.

        Handles None values and missing keys gracefully.
        Detects context trimming via token-drop heuristic when the
        caller has not already reported a CONTEXT_TRIMMED event.

        Args:
            metadata: Metadata dict from a WorkflowEvent (LLM_USAGE type).
            context_trimmed_already: Set True if a CONTEXT_TRIMMED event
                was already received for this invocation (suppresses the
                token-drop heuristic).

        Returns:
            True if a token-drop heuristic trim was detected, False otherwise.
        """
        prev_prompt = self.last_prompt_tokens
        current_prompt = metadata.get("prompt_tokens") or 0
        self.last_prompt_tokens = current_prompt

        self.prompt_tokens += current_prompt
        self.completion_tokens += metadata.get("completion_tokens") or 0
        self.thinking_tokens += metadata.get("thinking_tokens") or 0
        self.cached_tokens += metadata.get("cached_tokens") or 0
        self.cache_creation_tokens += metadata.get("cache_creation_tokens") or 0
        self.total_latency_ms += metadata.get("latency_ms") or 0.0
        self.invocation_count += 1

        # Token-drop heuristic: if prompt_tokens dropped and no explicit
        # trim event was already received, infer context was trimmed.
        if (
            not context_trimmed_already
            and prev_prompt > 0
            and current_prompt < prev_prompt
        ):
            self.context_trimmed_count += 1
            return True
        return False

    @property
    def total_tokens(self) -> int:
        """Total tokens (prompt + completion)."""
        return self.prompt_tokens + self.completion_tokens

    def format_status_bar(self) -> str:
        """Format token counts for status bar display.

        Returns:
            "Tokens: 12.5k in / 3.2k out | ctx: 5k" or "" if no usage yet.
            Appends "(trimmed)" if context trimming was detected.
        """
        if self.invocation_count == 0:
            return ""
        parts = [
            f"Tokens: {format_tokens(self.prompt_tokens)} in"
            f" / {format_tokens(self.completion_tokens)} out"
        ]
        if self.last_prompt_tokens > 0:
            ctx = f"ctx: {format_tokens(self.last_prompt_tokens)}"
            if self.context_trimmed_count > 0:
                ctx += " (trimmed)"
            parts.append(ctx)
        return " | ".join(parts)

    def format_detail_rows(self) -> list[tuple[str, str]]:
        """Format token usage as (label, value) rows for detailed display.

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
        if self.last_prompt_tokens > 0:
            rows.append(("Context Window", format_tokens(self.last_prompt_tokens)))
        if self.context_trimmed_count > 0:
            rows.append(("Context Trimmed", f"{self.context_trimmed_count} time(s)"))
        return rows

    def reset(self) -> None:
        """Zero all counters."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.thinking_tokens = 0
        self.cached_tokens = 0
        self.cache_creation_tokens = 0
        self.invocation_count = 0
        self.total_latency_ms = 0.0
        self.last_prompt_tokens = 0
        self.context_trimmed_count = 0
