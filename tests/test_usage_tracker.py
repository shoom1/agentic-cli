"""Tests for UsageTracker and format_tokens."""

import pytest

from agentic_cli.cli.usage_tracker import UsageTracker, format_tokens


# === format_tokens tests ===


class TestFormatTokens:
    def test_small_numbers(self):
        assert format_tokens(0) == "0"
        assert format_tokens(1) == "1"
        assert format_tokens(42) == "42"
        assert format_tokens(999) == "999"

    def test_thousands_exact(self):
        assert format_tokens(1000) == "1k"
        assert format_tokens(2000) == "2k"
        assert format_tokens(10000) == "10k"

    def test_thousands_fractional(self):
        assert format_tokens(1500) == "1.5k"
        assert format_tokens(12500) == "12.5k"
        assert format_tokens(100500) == "100.5k"

    def test_millions_exact(self):
        assert format_tokens(1000000) == "1M"
        assert format_tokens(2000000) == "2M"

    def test_millions_fractional(self):
        assert format_tokens(1200000) == "1.2M"
        assert format_tokens(1500000) == "1.5M"


# === UsageTracker tests ===


class TestUsageTracker:
    def test_initial_state(self):
        tracker = UsageTracker()
        assert tracker.prompt_tokens == 0
        assert tracker.completion_tokens == 0
        assert tracker.thinking_tokens == 0
        assert tracker.cached_tokens == 0
        assert tracker.cache_creation_tokens == 0
        assert tracker.invocation_count == 0
        assert tracker.total_latency_ms == 0.0

    def test_record_single_event(self):
        tracker = UsageTracker()
        tracker.record({
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "thinking_tokens": 20,
            "cached_tokens": 30,
            "cache_creation_tokens": 10,
            "latency_ms": 500.0,
        })
        assert tracker.prompt_tokens == 100
        assert tracker.completion_tokens == 50
        assert tracker.thinking_tokens == 20
        assert tracker.cached_tokens == 30
        assert tracker.cache_creation_tokens == 10
        assert tracker.invocation_count == 1
        assert tracker.total_latency_ms == 500.0

    def test_record_accumulates(self):
        tracker = UsageTracker()
        tracker.record({"prompt_tokens": 100, "completion_tokens": 50})
        tracker.record({"prompt_tokens": 200, "completion_tokens": 80})
        assert tracker.prompt_tokens == 300
        assert tracker.completion_tokens == 130
        assert tracker.invocation_count == 2

    def test_record_none_values(self):
        tracker = UsageTracker()
        tracker.record({
            "prompt_tokens": None,
            "completion_tokens": None,
            "thinking_tokens": None,
        })
        assert tracker.prompt_tokens == 0
        assert tracker.completion_tokens == 0
        assert tracker.thinking_tokens == 0
        assert tracker.invocation_count == 1

    def test_record_missing_keys(self):
        tracker = UsageTracker()
        tracker.record({})
        assert tracker.prompt_tokens == 0
        assert tracker.completion_tokens == 0
        assert tracker.invocation_count == 1

    def test_record_extra_keys_ignored(self):
        tracker = UsageTracker()
        tracker.record({
            "prompt_tokens": 100,
            "model": "claude-sonnet-4-5-20250929",
            "invocation_id": "abc123",
        })
        assert tracker.prompt_tokens == 100
        assert tracker.invocation_count == 1

    def test_total_tokens(self):
        tracker = UsageTracker()
        tracker.record({"prompt_tokens": 100, "completion_tokens": 50})
        assert tracker.total_tokens == 150

    def test_total_tokens_empty(self):
        tracker = UsageTracker()
        assert tracker.total_tokens == 0

    def test_format_status_bar_empty(self):
        tracker = UsageTracker()
        assert tracker.format_status_bar() == ""

    def test_format_status_bar_populated(self):
        tracker = UsageTracker()
        tracker.record({"prompt_tokens": 12500, "completion_tokens": 3200})
        assert tracker.format_status_bar() == "Tokens: 12.5k in / 3.2k out | ctx: 12.5k"

    def test_format_status_bar_small_values(self):
        tracker = UsageTracker()
        tracker.record({"prompt_tokens": 42, "completion_tokens": 10})
        assert tracker.format_status_bar() == "Tokens: 42 in / 10 out | ctx: 42"

    def test_reset(self):
        tracker = UsageTracker()
        tracker.record({
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "thinking_tokens": 20,
            "cached_tokens": 30,
            "cache_creation_tokens": 10,
            "latency_ms": 500.0,
        })
        tracker.reset()
        assert tracker.prompt_tokens == 0
        assert tracker.completion_tokens == 0
        assert tracker.thinking_tokens == 0
        assert tracker.cached_tokens == 0
        assert tracker.cache_creation_tokens == 0
        assert tracker.invocation_count == 0
        assert tracker.total_latency_ms == 0.0
        assert tracker.last_prompt_tokens == 0
        assert tracker.prev_prompt_tokens == 0
        assert tracker.context_trimmed_count == 0
        assert tracker.format_status_bar() == ""


# === Context window tracking tests ===


class TestContextWindowTracking:
    def test_last_prompt_tokens_tracked(self):
        tracker = UsageTracker()
        tracker.record({"prompt_tokens": 5000, "completion_tokens": 200})
        assert tracker.last_prompt_tokens == 5000

    def test_last_prompt_tokens_updates(self):
        tracker = UsageTracker()
        tracker.record({"prompt_tokens": 5000, "completion_tokens": 200})
        tracker.record({"prompt_tokens": 6000, "completion_tokens": 300})
        assert tracker.last_prompt_tokens == 6000
        assert tracker.prev_prompt_tokens == 5000

    def test_record_does_not_increment_trimmed_count(self):
        """record() no longer auto-detects trims; count is externally managed."""
        tracker = UsageTracker()
        tracker.record({"prompt_tokens": 10000})
        tracker.record({"prompt_tokens": 5000})  # Drop — but no auto-increment
        assert tracker.context_trimmed_count == 0

    def test_no_trim_on_increase(self):
        tracker = UsageTracker()
        tracker.record({"prompt_tokens": 5000})
        tracker.record({"prompt_tokens": 8000})
        assert tracker.context_trimmed_count == 0

    def test_no_trim_on_first_invocation(self):
        tracker = UsageTracker()
        tracker.record({"prompt_tokens": 5000})
        assert tracker.context_trimmed_count == 0

    def test_external_increment_works(self):
        """context_trimmed_count can be incremented by external event handlers."""
        tracker = UsageTracker()
        tracker.context_trimmed_count += 1
        tracker.context_trimmed_count += 1
        assert tracker.context_trimmed_count == 2

    def test_status_bar_includes_context(self):
        tracker = UsageTracker()
        tracker.record({"prompt_tokens": 5000, "completion_tokens": 200})
        bar = tracker.format_status_bar()
        assert "ctx: 5k" in bar

    def test_status_bar_shows_trimmed_when_set_externally(self):
        """Status bar shows (trimmed) regardless of how count was set."""
        tracker = UsageTracker()
        tracker.record({"prompt_tokens": 5000, "completion_tokens": 200})
        tracker.context_trimmed_count = 1
        bar = tracker.format_status_bar()
        assert "ctx: 5k" in bar
        assert "(trimmed)" in bar

    def test_status_bar_no_trimmed_without_trim(self):
        tracker = UsageTracker()
        tracker.record({"prompt_tokens": 5000, "completion_tokens": 200})
        tracker.record({"prompt_tokens": 8000, "completion_tokens": 300})
        bar = tracker.format_status_bar()
        assert "(trimmed)" not in bar

    def test_reset_clears_context_fields(self):
        tracker = UsageTracker()
        tracker.record({"prompt_tokens": 10000})
        tracker.context_trimmed_count = 3
        tracker.reset()
        assert tracker.last_prompt_tokens == 0
        assert tracker.prev_prompt_tokens == 0
        assert tracker.context_trimmed_count == 0
