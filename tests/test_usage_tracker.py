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

    @pytest.mark.parametrize("count, shown", [
        (146_036, "146k"),  # Claude 4.5's trigger
        (45_020, "45k"),
        (103_360, "103.4k"),
        (1_000_400, "1M"),
        (999_999, "1M"),  # rounds up into the next unit
        (999, "999"),
    ])
    def test_rounded_to_a_whole_number_it_shows_no_decimal(self, count, shown):
        assert format_tokens(count) == shown


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
        assert tracker.format_status_bar() == "ctx 12.5k | Tokens: 12.5k in / 3.2k out"

    def test_format_status_bar_small_values(self):
        tracker = UsageTracker()
        tracker.record({"prompt_tokens": 42, "completion_tokens": 10})
        assert tracker.format_status_bar() == "ctx 42 | Tokens: 42 in / 10 out"

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
        assert tracker.context_trimmed_count == 0
        assert tracker.format_status_bar() == ""


# === The main agent's context ===

#: The usage of a main agent's call whose trigger is 170k on a known window.
MAIN = {"prompt_tokens": 45_000, "completion_tokens": 100,
        "context_trigger": 170_000, "window_known": True}


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

    def test_external_increment_works(self):
        """context_trimmed_count can be incremented by external event handlers."""
        tracker = UsageTracker()
        tracker.context_trimmed_count += 1
        tracker.context_trimmed_count += 1
        assert tracker.context_trimmed_count == 2

    def test_the_main_agents_prompt_against_its_trigger(self):
        tracker = UsageTracker()
        tracker.record(MAIN)

        assert tracker.format_status_bar() == "ctx 45k/170k (26%) | Tokens: 45k in / 100 out"
        assert (tracker.last_prompt_tokens, tracker.context_trigger, tracker.window_known) == (
            45_000, 170_000, True
        )

    @pytest.mark.parametrize("prompt, shown", [
        (85_000, "ctx 85k/170k (50%)"),
        (1_000, "ctx 1k/170k (1%)"),
        (180_000, "ctx 180k/170k (106%)"),  # the trigger lags a call
    ])
    def test_the_share_is_a_whole_percent(self, prompt, shown):
        tracker = UsageTracker()
        tracker.record({**MAIN, "prompt_tokens": prompt})

        assert tracker.format_status_bar().startswith(f"{shown} | ")

    def test_on_an_unknown_window_the_prompt_alone(self):
        # The trigger then rests on the 128k floor, not on the model.
        tracker = UsageTracker()
        tracker.record({**MAIN, "context_trigger": 92_000, "window_known": False})

        assert tracker.format_status_bar() == "ctx 45k | Tokens: 45k in / 100 out"

    def test_without_a_trigger_the_prompt_alone(self):
        # A backend that manages no context (LangGraph) reports none.
        tracker = UsageTracker()
        tracker.record({"prompt_tokens": 45_000, "completion_tokens": 100})

        assert tracker.format_status_bar() == "ctx 45k | Tokens: 45k in / 100 out"

    @pytest.mark.parametrize("purpose", ["subagent", "summarization"])
    def test_a_sub_agent_or_summary_call_never_moves_it(self, purpose):
        """A summary's prompt is the history it condenses; a sub-agent's is its
        brief, or the whole conversation when it inherits it, and still not
        the main agent's context. Each adds to the totals only."""
        tracker = UsageTracker()
        tracker.record(MAIN)

        result = tracker.record({
            "prompt_tokens": 150_000, "completion_tokens": 30, "purpose": purpose,
            "context_trigger": 100_000, "window_known": False,
        })

        assert result is None
        assert tracker.format_status_bar() == (
            "ctx 45k/170k (26%) | Tokens: 195k in / 130 out"
        )
        assert tracker.invocation_count == 2

    def test_a_smaller_prompt_is_no_trim(self):
        # The token-drop heuristic is gone: a summary or a trim says so itself.
        tracker = UsageTracker()
        tracker.record({**MAIN, "prompt_tokens": 100_000})

        assert tracker.record({**MAIN, "prompt_tokens": 9_000}) is None
        assert tracker.context_trimmed_count == 0
        assert tracker.format_status_bar().startswith("ctx 9k/170k (5%) | ")

    def test_counted_trims_do_not_mark_the_segment(self):
        tracker = UsageTracker()
        tracker.record(MAIN)
        tracker.context_trimmed_count = 2

        assert "(trimmed)" not in tracker.format_status_bar()

    def test_reset_clears_context_fields(self):
        tracker = UsageTracker()
        tracker.record({**MAIN, "prompt_tokens": 10000})
        tracker.context_trimmed_count = 3
        tracker.reset()
        assert tracker.last_prompt_tokens == 0
        assert (tracker.context_trigger, tracker.window_known) == (None, None)
        assert tracker.context_trimmed_count == 0

    def test_the_detail_rows_leave_the_context_to_status(self):
        # /status shows the last prompt beside the model's window and trigger.
        tracker = UsageTracker()
        tracker.record(MAIN)

        assert "Context Window" not in dict(tracker.format_detail_rows())
