"""When a conversation is summarized, and how a context overflow is recognized.

``workflow/context.py`` computes one model's numbers for both backends and
the CLI. For a window W, an output limit M (the resolved ``max_tokens``),
``context_summarize_at`` f and ``context_max_tokens`` C:

- the input budget B = floor(0.95 W) - M;
- the trigger T = min(floor(f B), C), or floor(f B) when C is 0;
- the tail kept word for word K = floor(0.12 T).

A model whose window is unknown counts as 128,000 tokens, and a budget under
16,000 tokens is a build error. ``is_context_overflow`` recognizes each
provider's "too long" error from its attributes, without importing a
provider SDK; the tests build the errors from the real SDK classes.
"""

from __future__ import annotations

import dataclasses

import anthropic
import httpx
import openai
import pytest
from google.genai import errors as genai_errors
from pydantic import ValidationError

from agentic_cli.config import BaseSettings
from agentic_cli.models import ModelCatalog, ProviderKind, RoleMap, resolve_model
from agentic_cli.settings_persistence import PROJECT_SETTABLE_KEYS
from agentic_cli.workflow.config import AgentGraphError
from agentic_cli.workflow.context import (
    KEEP_FRACTION,
    MIN_BUDGET,
    OVERFLOW_ERROR_CODE,
    UNKNOWN_WINDOW,
    ContextBudget,
    ContextBudgetError,
    chars_per_token,
    context_budget,
    is_context_overflow,
    overflow_message,
)
from tests.models.fakes import endpoints as fake_endpoints

#: Claude Sonnet 4.5: a 200,000-token window, max_tokens 8,192 (Anthropic's default).
SONNET = "anthropic:claude-sonnet-4-5"
#: Claude Opus 4.7: a 1,000,000-token window, max_tokens 8,192.
OPUS = "anthropic:claude-opus-4-7"


def resolved(spec: str, agent: str = "writer"):
    """``spec`` resolved for ``agent`` as a manager would, without listings."""
    endpoints = fake_endpoints(anthropic="a-test-key", google="g-test-key", openai="o-test-key")
    return resolve_model(
        agent=agent, spec=spec, generation=None, thinking_effort="none",
        roles=RoleMap(default_model=None, model_roles={}),
        catalog=ModelCatalog(endpoints=endpoints), endpoints=endpoints,
    )


def windowed(model, window: int | None):
    """``model`` with its record's window replaced."""
    record = dataclasses.replace(model.record, context_window=window)
    return dataclasses.replace(model, record=record)


def settings(**values) -> BaseSettings:
    return BaseSettings(**values)


class TestTheConstants:
    def test_the_numbers_the_spec_names(self):
        assert (UNKNOWN_WINDOW, MIN_BUDGET, KEEP_FRACTION) == (128_000, 16_000, 0.12)
        assert OVERFLOW_ERROR_CODE == "CONTEXT_OVERFLOW"

    def test_a_budget_error_is_a_build_error(self):
        assert issubclass(ContextBudgetError, AgentGraphError)


class TestContextBudget:
    def test_a_200k_claude(self):
        sonnet = resolved(SONNET)
        assert (sonnet.record.context_window, sonnet.max_tokens) == (200_000, 8_192)

        # B = 190,000 - 8,192; T = floor(0.85 B), under the ceiling; K = floor(0.12 T).
        assert context_budget(sonnet, settings()) == ContextBudget(
            window=200_000, window_known=True, budget=181_808, trigger=154_536, keep=18_544
        )

    def test_on_a_1m_model_the_ceiling_wins(self):
        opus = resolved(OPUS)
        assert (opus.record.context_window, opus.max_tokens) == (1_000_000, 8_192)

        # 0.85 B is about 800k tokens: the 200,000 ceiling is the trigger.
        assert context_budget(opus, settings()) == ContextBudget(
            window=1_000_000, window_known=True, budget=941_808, trigger=200_000, keep=24_000
        )

    def test_a_zero_ceiling_means_none(self):
        budget = context_budget(resolved(OPUS), settings(context_max_tokens=0))

        assert (budget.trigger, budget.keep) == (800_536, 96_064)

    def test_summarize_at_scales_the_trigger(self):
        budget = context_budget(resolved(SONNET), settings(context_summarize_at=0.5))

        assert (budget.budget, budget.trigger, budget.keep) == (181_808, 90_904, 10_908)

    def test_without_an_output_limit_the_budget_is_95_percent(self):
        sonnet = dataclasses.replace(resolved(SONNET), max_tokens=None)

        budget = context_budget(sonnet, settings())

        assert (budget.budget, budget.trigger) == (190_000, 161_500)

    def test_an_unknown_window_counts_as_128k(self):
        unknown = windowed(resolved(SONNET), None)

        assert context_budget(unknown, settings()) == ContextBudget(
            window=128_000, window_known=False, budget=113_408, trigger=96_396, keep=11_567
        )

    def test_it_is_frozen(self):
        budget = context_budget(resolved(SONNET), settings())

        with pytest.raises(dataclasses.FrozenInstanceError):
            budget.trigger = 1  # type: ignore[misc]

    def test_a_budget_too_small_to_work_names_the_agent_the_model_and_max_tokens(self):
        # A 20,000-token window: B = 19,000 - 8,192 = 10,808, under 16,000.
        small = windowed(resolved(SONNET, agent="summarizer"), 20_000)

        with pytest.raises(ContextBudgetError) as caught:
            context_budget(small, settings())

        message = str(caught.value)
        assert "'summarizer'" in message
        assert SONNET in message
        assert "20,000" in message and "10,808" in message
        assert "max_tokens" in message and "8,192" in message
        assert "lower" in message.lower()

    def test_a_budget_of_exactly_16k_is_enough(self):
        # floor(0.95 * 25,456) = 24,183; minus 8,183 is 16,000.
        model = dataclasses.replace(windowed(resolved(SONNET), 25_456), max_tokens=8_183)

        assert context_budget(model, settings()).budget == MIN_BUDGET

    def test_an_unknown_window_with_a_huge_output_limit_fails_too(self):
        model = dataclasses.replace(windowed(resolved(SONNET), None), max_tokens=120_000)

        with pytest.raises(ContextBudgetError, match="unknown") as caught:
            context_budget(model, settings())

        # An endpoint model's usual fix: say how large its window is.
        assert "context_window" in str(caught.value)
        assert "model_endpoints" in str(caught.value)

    def test_a_known_window_is_not_told_to_declare_one(self):
        small = windowed(resolved(SONNET), 20_000)

        with pytest.raises(ContextBudgetError) as caught:
            context_budget(small, settings())

        assert "model_endpoints" not in str(caught.value)


class TestSettings:
    def test_the_defaults(self):
        defaults = BaseSettings()

        assert defaults.context_summarize_at == 0.85
        assert defaults.context_max_tokens == 200_000

    @pytest.mark.parametrize("value", [0.49, 0.96, 0.0, 1.0])
    def test_summarize_at_outside_half_to_95_percent_is_rejected(self, value):
        with pytest.raises(ValidationError, match="context_summarize_at"):
            BaseSettings(context_summarize_at=value)

    @pytest.mark.parametrize("value", [0.5, 0.95])
    def test_summarize_at_accepts_its_bounds(self, value):
        assert BaseSettings(context_summarize_at=value).context_summarize_at == value

    @pytest.mark.parametrize("value", [15_999, 1, -1])
    def test_a_ceiling_under_16k_is_rejected(self, value):
        with pytest.raises(ValidationError, match="context_max_tokens"):
            BaseSettings(context_max_tokens=value)

    @pytest.mark.parametrize("value", [0, 16_000, 1_000_000])
    def test_a_ceiling_of_zero_or_at_least_16k_is_accepted(self, value):
        assert BaseSettings(context_max_tokens=value).context_max_tokens == value

    def test_a_project_may_set_both(self):
        assert {"context_summarize_at", "context_max_tokens"} <= PROJECT_SETTABLE_KEYS

    def test_both_read_as_settings_in_the_dialog(self):
        fields = BaseSettings.model_fields

        assert fields["context_summarize_at"].title == "Summarize At"
        assert fields["context_max_tokens"].title == "Context Ceiling"
        assert fields["context_summarize_at"].description
        assert "0" in fields["context_max_tokens"].description


class TestCharsPerToken:
    def test_claude_is_denser_than_the_others(self):
        assert chars_per_token(ProviderKind.ANTHROPIC) == 3.3
        for kind in ProviderKind:
            if kind is not ProviderKind.ANTHROPIC:
                assert chars_per_token(kind) == 4.0


# --- overflow errors, from the real SDK classes ----------------------------------------------


def _anthropic(status: int, kind: str, message: str):
    error_class = {
        400: anthropic.BadRequestError,
        413: anthropic.RequestTooLargeError,
        429: anthropic.RateLimitError,
    }[status]
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return error_class(
        message,
        response=httpx.Response(status, request=request),
        body={"type": "error", "error": {"type": kind, "message": message}},
    )


def _openai(message: str, code):
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return openai.BadRequestError(
        message,
        response=httpx.Response(400, request=request),
        body={"message": message, "type": "invalid_request_error", "code": code},
    )


def _gemini(message: str, code: int = 400):
    return genai_errors.ClientError(
        code, {"error": {"code": code, "message": message, "status": "INVALID_ARGUMENT"}}
    )


class _StatusError(Exception):
    """A proxy's error: only a status code and a message."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class TestIsContextOverflow:
    def test_claude_prompt_is_too_long(self):
        error = _anthropic(
            400, "invalid_request_error", "prompt is too long: 210000 tokens > 200000 maximum"
        )

        assert is_context_overflow(error)

    def test_claude_request_too_large(self):
        error = _anthropic(413, "request_too_large", "Request exceeds the maximum size")

        assert is_context_overflow(error)

    def test_openai_context_length_exceeded_is_read_from_the_code(self):
        # The text alone names nothing; the code does.
        error = _openai("Invalid request.", "context_length_exceeded")
        assert error.code == "context_length_exceeded"

        assert is_context_overflow(error)

    def test_openrouter_int_code_and_maximum_context_length(self):
        error = _openai(
            "This endpoint's maximum context length is 32768 tokens. However, you "
            "requested about 40000 tokens.",
            400,
        )
        assert error.code == 400

        assert is_context_overflow(error)

    def test_gemini_too_many_tokens(self):
        error = _gemini(
            "The input token count (1200000) exceeds the maximum number of tokens "
            "allowed (1048576)."
        )
        # The int code wins over the string status.
        assert (error.code, error.status) == (400, "INVALID_ARGUMENT")

        assert is_context_overflow(error)

    def test_gemini_input_token_count_needs_exceeds(self):
        assert is_context_overflow(_gemini("Input token count 1200000 exceeds the limit."))
        assert not is_context_overflow(_gemini("Input token count must be positive."))

    def test_langchains_own_overflow(self):
        exceptions = pytest.importorskip("langchain_core.exceptions")

        assert is_context_overflow(exceptions.ContextOverflowError("anything at all"))

    def test_a_proxy_422(self):
        assert is_context_overflow(_StatusError("Context window exceeded for model x", 422))

    def test_claude_input_plus_max_tokens_over_the_window(self):
        # Claude before Sonnet 4.5: the prompt and max_tokens together.
        error = _anthropic(
            400, "invalid_request_error",
            "input length and `max_tokens` exceed context limit: 188240 + 21333 > 200000, "
            "decrease input length or `max_tokens` and try again",
        )

        assert is_context_overflow(error)

    def test_a_429_is_never_an_overflow(self):
        assert not is_context_overflow(
            _anthropic(429, "rate_limit_error", "rate limit: token limit reached")
        )
        assert not is_context_overflow(
            _anthropic(429, "rate_limit_error", "prompt is too long")
        )

    def test_a_429_is_never_an_overflow_even_as_langchains(self):
        # langchain-google-genai raises its ContextOverflowError for any
        # ClientError whose text holds "token limit", a rate limit included.
        chat_models = pytest.importorskip("langchain_google_genai.chat_models")
        error = chat_models.GoogleContextOverflowError(429, {"error": {
            "code": 429, "message": "Token limit per minute exceeded.",
            "status": "RESOURCE_EXHAUSTED",
        }})
        assert error.code == 429

        assert not is_context_overflow(error)

    def test_a_plain_400_is_not(self):
        assert not is_context_overflow(
            _anthropic(400, "invalid_request_error", "temperature: must be at most 1")
        )
        assert not is_context_overflow(_openai("Unknown parameter: 'foo'.", "unknown_parameter"))
        assert not is_context_overflow(_gemini("Request contains an invalid argument."))

    def test_an_error_without_a_status_is_not(self):
        assert not is_context_overflow(ValueError("prompt is too long"))
        assert not is_context_overflow(_StatusError("prompt is too long", 500))


def test_the_overflow_message():
    assert overflow_message("anthropic:claude-sonnet-4-5") == (
        "This conversation no longer fits anthropic:claude-sonnet-4-5's context window. "
        "Use /compact or /new."
    )
