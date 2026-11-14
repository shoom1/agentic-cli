"""Token usage: each provider's report, normalized.

Every ``LLM_USAGE`` counts the same way on both backends: ``prompt_tokens``
every input token, the cache's reads and writes included, and
``completion_tokens`` every output token, the thinking included. Each
provider reports otherwise, and each adapter in between differs again; these
pin how each shape is read (``adk_tokens``, ``langchain_tokens``, both through
``normalized_tokens``).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agentic_cli.workflow.events import EventType
from agentic_cli.workflow.usage import (
    UsageTotals,
    adk_tokens,
    adk_usage_event,
    langchain_tokens,
    llm_usage_event,
    normalized_tokens,
)


def adk_usage(**counts) -> SimpleNamespace:
    """ADK's usage shape (``GenerateContentResponseUsageMetadata``), its unset fields None."""
    fields = ("prompt_token_count", "candidates_token_count", "total_token_count",
              "cached_content_token_count", "thoughts_token_count", "tool_use_prompt_token_count")
    return SimpleNamespace(**{name: counts.pop(name, None) for name in fields}, **counts)


class TestNormalizedTokens:
    def test_counts_as_reported(self):
        assert normalized_tokens(input=100, output=20, cached=60, cache_write=10, thinking=5) == (
            UsageTotals(input=100, output=20, cached=60, cache_write=10, thinking=5, calls=1)
        )

    def test_an_input_that_leaves_the_cache_out_gets_it_back(self):
        tokens = normalized_tokens(input=50, output=7, cached=1_200, cache_write=300,
                                   input_counts_cache=False)

        assert (tokens.input, tokens.cached, tokens.cache_write) == (1_550, 1_200, 300)

    def test_an_output_that_leaves_the_thinking_out_gets_it_back(self):
        tokens = normalized_tokens(input=900, output=5, thinking=25, output_counts_thinking=False)

        assert (tokens.output, tokens.thinking) == (30, 25)

    def test_what_was_not_reported_is_none(self):
        assert normalized_tokens(input=None, output=None) == UsageTotals(calls=1)


class TestADKShapes:
    """What ADK 2.12's adapters put in ``usage_metadata``, by provider kind."""

    def test_gemini_counts_its_cache_in_the_prompt_and_its_thinking_apart(self):
        usage = adk_usage(prompt_token_count=900, candidates_token_count=5,
                          cached_content_token_count=800, thoughts_token_count=25)

        assert adk_tokens(usage, "google") == UsageTotals(
            input=900, output=30, cached=800, thinking=25, calls=1
        )

    def test_gemini_s_server_side_tool_prompt_is_input(self):
        usage = adk_usage(prompt_token_count=900, candidates_token_count=5,
                          tool_use_prompt_token_count=40)

        assert adk_tokens(usage, "google").input == 940

    def test_claude_s_adapter_folds_the_cache_in_and_keeps_its_writes_apart(self):
        # ADK's Claude adapter: prompt = input + cache reads + cache writes;
        # candidates = output - thinking; the writes on an attribute of its own.
        usage = adk_usage(prompt_token_count=1_550, candidates_token_count=4,
                          cached_content_token_count=1_200, thoughts_token_count=3,
                          cache_creation_input_tokens=300)

        assert adk_tokens(usage, "anthropic") == UsageTotals(
            input=1_550, output=7, cached=1_200, cache_write=300, thinking=3, calls=1
        )

    @pytest.mark.parametrize("kind", ["openai", "openrouter", "openai_compatible"])
    def test_openai_s_adapters_count_the_reasoning_in_the_completion(self, kind):
        usage = adk_usage(prompt_token_count=1_550, candidates_token_count=70,
                          cached_content_token_count=1_200, thoughts_token_count=64)

        assert adk_tokens(usage, kind) == UsageTotals(
            input=1_550, output=70, cached=1_200, thinking=64, calls=1
        )

    def test_a_provider_kind_counts_as_its_value(self):
        from agentic_cli.models.providers import ProviderKind

        usage = adk_usage(prompt_token_count=10, candidates_token_count=5, thoughts_token_count=2)

        assert adk_tokens(usage, ProviderKind.GOOGLE).output == 7
        assert adk_tokens(usage, ProviderKind.OPENAI).output == 5

    def test_an_unknown_kind_counts_the_thinking_apart(self):
        # A native ADK YAML app's bare model names are Gemini's and Claude's.
        usage = adk_usage(prompt_token_count=10, candidates_token_count=5, thoughts_token_count=2)

        assert adk_tokens(usage, None).output == 7

    def test_a_fake_s_fields_are_no_counts(self):
        assert adk_tokens(MagicMock(), "google") == UsageTotals(calls=1)


class TestLangChainShapes:
    """What each partner package puts in ``usage_metadata``: LangChain's convention."""

    def test_anthropic_s_package_adds_the_cache_back_itself(self):
        usage = {"input_tokens": 1_550, "output_tokens": 7, "total_tokens": 1_557,
                 "input_token_details": {"cache_read": 1_200, "cache_creation": 300},
                 "output_token_details": {"reasoning": 3}}

        assert langchain_tokens(usage) == UsageTotals(
            input=1_550, output=7, cached=1_200, cache_write=300, thinking=3, calls=1
        )

    def test_anthropic_s_writes_by_time_to_live(self):
        usage = {"input_tokens": 1_550, "output_tokens": 7,
                 "input_token_details": {"cache_read": 1_200, "cache_creation": 0,
                                         "ephemeral_5m_input_tokens": 300,
                                         "ephemeral_1h_input_tokens": 0}}

        assert langchain_tokens(usage).cache_write == 300

    def test_google_s_package_adds_the_thoughts_to_the_output(self):
        usage = {"input_tokens": 900, "output_tokens": 30, "total_tokens": 930,
                 "input_token_details": {"cache_read": 800},
                 "output_token_details": {"reasoning": 25}}

        assert langchain_tokens(usage) == UsageTotals(
            input=900, output=30, cached=800, thinking=25, calls=1
        )

    @pytest.mark.parametrize("package", ["openai", "openrouter"])
    def test_openai_s_and_openrouter_s_packages(self, package):
        usage = {"input_tokens": 1_550, "output_tokens": 70, "total_tokens": 1_620,
                 "input_token_details": {"cache_read": 1_200, "cache_creation": 100},
                 "output_token_details": {"reasoning": 64}}

        assert langchain_tokens(usage) == UsageTotals(
            input=1_550, output=70, cached=1_200, cache_write=100, thinking=64, calls=1
        )

    @pytest.mark.parametrize("tier", ["priority", "flex"])
    def test_openai_s_package_on_a_service_tier(self, tier):
        # langchain-openai prefixes the parts with the tier, and counts the
        # tier's own tokens beside them (the input or output less the parts).
        usage = {"input_tokens": 1_550, "output_tokens": 70, "total_tokens": 1_620,
                 "input_token_details": {f"{tier}_cache_read": 1_200,
                                         f"{tier}_cache_creation": 100, tier: 350},
                 "output_token_details": {f"{tier}_reasoning": 64, tier: 6}}

        assert langchain_tokens(usage) == UsageTotals(
            input=1_550, output=70, cached=1_200, cache_write=100, thinking=64, calls=1
        )

    def test_none_is_nothing(self):
        assert langchain_tokens(None) == UsageTotals(calls=1)


class TestTheEvent:
    def test_it_carries_the_normalized_counts_and_their_parts(self):
        event = llm_usage_event(
            "anthropic:claude-sonnet-4-6",
            UsageTotals(input=1_550, output=7, cached=1_200, cache_write=300, thinking=3, calls=1),
            total_tokens=1_557,
        )

        assert event.type is EventType.LLM_USAGE
        assert event.metadata == {
            "model": "anthropic:claude-sonnet-4-6", "prompt_tokens": 1_550,
            "completion_tokens": 7, "total_tokens": 1_557, "thinking_tokens": 3,
            "cached_tokens": 1_200, "cache_creation_tokens": 300,
        }

    def test_parts_that_are_none_are_left_out(self):
        event = llm_usage_event("m", UsageTotals(input=10, output=5, calls=1))

        assert event.metadata == {"model": "m", "prompt_tokens": 10, "completion_tokens": 5}


class TestTheADKEvent:
    def test_it_reads_adk_s_usage_by_kind(self):
        usage = adk_usage(prompt_token_count=900, candidates_token_count=5,
                          total_token_count=930, cached_content_token_count=800,
                          thoughts_token_count=25)

        event = adk_usage_event("google:gemini-2.5-flash", usage, "google",
                                invocation_id="i1", latency_ms=12.5)

        assert event.metadata == {
            "model": "google:gemini-2.5-flash", "prompt_tokens": 900,
            "completion_tokens": 30, "total_tokens": 930, "thinking_tokens": 25,
            "cached_tokens": 800, "invocation_id": "i1", "latency_ms": 12.5,
        }
