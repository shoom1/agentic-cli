"""How much of a model's window a conversation may fill, and its overflow errors.

One module computes these numbers for both backends and the CLI, so a
summary starts at the same point on ADK and on Deep Agents, and the status
bar shows the point it starts at.

**When to summarize.** For a model with window W and output limit M (its
resolved ``max_tokens``), with ``context_summarize_at`` f and
``context_max_tokens`` C:

- the input budget ``B = floor(0.95 W) - M`` (Deep Agents' own formula);
- the trigger ``T = min(floor(f B), C)``, or ``floor(f B)`` when C is 0;
- the tail kept word for word ``K = floor(0.12 T)``.

A model whose window is unknown (an endpoint model without a declared
window, an OpenRouter id before a listing) counts as ``UNKNOWN_WINDOW``. A
budget under ``MIN_BUDGET`` leaves no room to work in, so it fails where the
agents are built, naming the model and its ``max_tokens``.

**Overflow errors.** Each provider says "too long" its own way:
``is_context_overflow`` reads the error's attributes (status, code, text) and
never imports a provider SDK, and ``is_overflow_text`` the text alone, for an
overflow reported in an answer. LangChain's own ``ContextOverflowError`` is
recognized when LangChain is installed.

Imports nothing from ``cli/``, no provider SDK and no LangChain at import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agentic_cli.models import ProviderKind
from agentic_cli.workflow.config import AgentGraphError

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings
    from agentic_cli.models import ResolvedModel

#: The window assumed for a model whose window is unknown.
UNKNOWN_WINDOW = 128_000
#: The smallest input budget an agent can work in.
MIN_BUDGET = 16_000
#: The share of the trigger kept word for word after a summary.
KEEP_FRACTION = 0.12
#: The ``error_code`` of the error a conversation too long for its model ends in.
OVERFLOW_ERROR_CODE = "CONTEXT_OVERFLOW"

#: The share of the window the input budget starts from (Deep Agents' margin).
_WINDOW_SHARE = 95  # percent

#: Characters per token when estimating a request's size: Claude's tokenizer
#: gives fewer characters per token than the others.
_CHARS_PER_TOKEN = {ProviderKind.ANTHROPIC: 3.3}
_DEFAULT_CHARS_PER_TOKEN = 4.0
#: What an estimate from characters adds for what it misses (10%).
ESTIMATE_MARGIN = 1.1

#: The code OpenAI (and servers copying it) give an overflow.
_OVERFLOW_CODE = "context_length_exceeded"
#: The statuses an overflow comes with.
_OVERFLOW_STATUSES = frozenset({400, 413, 422})
#: Rate limits: never an overflow, whatever the text or the type says.
_RATE_LIMITED = 429
#: What an overflow's text says, lowercased: Deep Agents' list (OpenAI,
#: Claude, OpenRouter, vLLM and other OpenAI-compatible servers), Gemini's,
#: and Claude's for a prompt that fits but leaves no room for max_tokens.
_OVERFLOW_MARKERS = (
    "context_length_exceeded",
    "contextwindowexceedederror",
    "maximum context length",
    "exceeds the context window",
    "exceeds the available context size",
    "context window exceeded",
    "context limit exceeded",
    "input tokens exceed the configured limit",
    "prompt is too long",
    "exceeds the maximum number of tokens allowed",  # Gemini
    "exceed context limit",  # Claude before Sonnet 4.5: input length and max_tokens
)
#: Gemini's other wording: both parts must be there.
_OVERFLOW_MARKER_PAIRS = (("input token count", "exceeds"),)


class ContextBudgetError(AgentGraphError):
    """An agent's model leaves too small an input budget to work in.

    An ``AgentGraphError``: it fails where the agents are built, as the other
    build errors do.
    """


@dataclass(frozen=True)
class ContextBudget:
    """One agent's context numbers, in tokens.

    Attributes:
        window: The model's window, or ``UNKNOWN_WINDOW``.
        window_known: Whether the window is the model's own.
        budget: The input budget B: what a request may hold.
        trigger: The prompt size T at which the conversation is summarized.
        keep: The newest K tokens a summary keeps word for word.
    """

    window: int
    window_known: bool
    budget: int
    trigger: int
    keep: int


@dataclass(frozen=True)
class CompactResult:
    """What one summary did to a conversation.

    Attributes:
        messages_summarized: How many messages the summary stands for (a
            summary it replaced counts as one).
        tokens_before: The conversation's estimated size before, in tokens.
        tokens_after: Its estimated size after: the summary and what it kept.
    """

    messages_summarized: int
    tokens_before: int
    tokens_after: int


def context_budget(resolved: "ResolvedModel", settings: "BaseSettings") -> ContextBudget:
    """The context numbers of an agent on ``resolved``.

    Args:
        resolved: The agent's model: its window and its ``max_tokens``.
        settings: ``context_summarize_at`` and ``context_max_tokens``.

    Raises:
        ContextBudgetError: The budget is under ``MIN_BUDGET``. The message
            names the agent, its model, the window and ``max_tokens``; for a
            window too small whatever ``max_tokens`` is, the smallest window
            that works.
    """
    known = resolved.record.context_window
    window = known if known is not None else UNKNOWN_WINDOW
    max_tokens = resolved.max_tokens or 0
    budget = window * _WINDOW_SHARE // 100 - max_tokens
    if budget < MIN_BUDGET:
        who = f"Agent {resolved.agent!r}" if resolved.agent else "The default model"
        if window * _WINDOW_SHARE // 100 < MIN_BUDGET:
            # No max_tokens can help: the window itself is too small.
            smallest = -(-MIN_BUDGET * 100 // _WINDOW_SHARE)
            raise ContextBudgetError(
                f"{who} cannot work on {resolved.ref}: its window ({window:,} tokens) "
                f"is under the {smallest:,} an agent needs for an input budget of "
                f"{MIN_BUDGET:,} tokens. Choose a model with a larger window."
            )
        if known is not None:
            size = f"{window:,} tokens"
            advice = "Lower its max_tokens, or choose a model with a larger window."
        else:
            size = f"unknown, {window:,} assumed"
            advice = (
                "Declare the model's context_window (in model_endpoints for a model "
                "your server serves), lower its max_tokens, or choose a model with a "
                "larger window."
            )
        raise ContextBudgetError(
            f"{who} cannot work on {resolved.ref}: its window ({size}) less max_tokens "
            f"{max_tokens:,} leaves an input budget of {budget:,} tokens, under the "
            f"{MIN_BUDGET:,} needed. {advice}"
        )
    trigger = int(settings.context_summarize_at * budget)
    if settings.context_max_tokens:
        trigger = min(trigger, settings.context_max_tokens)
    return ContextBudget(
        window=window,
        window_known=known is not None,
        budget=budget,
        trigger=trigger,
        keep=int(KEEP_FRACTION * trigger),
    )


def usage_fields(budget: ContextBudget | None) -> dict[str, int | bool]:
    """What an agent's ``LLM_USAGE`` says of its context, for its metadata.

    ``context_trigger`` is the prompt size at which the agent's conversation
    is summarized, and ``window_known`` whether its window is the model's own
    or the ``UNKNOWN_WINDOW`` floor: the CLI shows the main agent's last
    prompt against that trigger. Nothing for an agent with no budget, one
    whose context nobody manages (a native ADK YAML app's, LangGraph's).
    """
    if budget is None:
        return {}
    return {"context_trigger": budget.trigger, "window_known": budget.window_known}


def chars_per_token(kind: ProviderKind) -> float:
    """Characters per token for estimating a request to a ``kind`` provider."""
    return _CHARS_PER_TOKEN.get(kind, _DEFAULT_CHARS_PER_TOKEN)


def is_context_overflow(error: BaseException) -> bool:
    """Whether ``error`` is a provider saying the request is too long for the model.

    True for LangChain's ``ContextOverflowError``; for the code
    ``context_length_exceeded`` (OpenAI); for a 413 ``RequestTooLargeError``
    (Claude's request-size limit); and for a 400, 413 or 422 whose text holds
    one of the providers' overflow phrases. A 429 is never an overflow, even
    as LangChain's type: langchain-google-genai raises that for any error
    whose text holds "token limit", a rate limit included.

    The status is the first int among ``status_code``, ``code`` and
    ``status``: Gemini's ``ClientError`` has an int ``code`` and a string
    ``status``, OpenRouter's errors an int ``code``. A string ``code`` is
    OpenAI's error code.
    """
    status = next(
        (
            value
            for value in (
                getattr(error, name, None) for name in ("status_code", "code", "status")
            )
            if isinstance(value, int) and not isinstance(value, bool)
        ),
        None,
    )
    if status == _RATE_LIMITED:
        return False
    if _is_langchain_overflow(error):
        return True
    if getattr(error, "code", None) == _OVERFLOW_CODE:
        return True
    if status == 413 and type(error).__name__ == "RequestTooLargeError":
        return True
    if status not in _OVERFLOW_STATUSES:
        return False
    return is_overflow_text(str(error))


def is_overflow_text(text: str) -> bool:
    """Whether ``text`` says what a provider says of a request too long for its model.

    The text half of ``is_context_overflow``: an overflow code
    (``context_length_exceeded``) or one of the providers' phrases, in any
    case. For an error a provider reports in its answer instead of raising
    (OpenAI Responses' ``status: failed``), where no status is there to read.
    """
    lowered = text.lower()
    return any(marker in lowered for marker in _OVERFLOW_MARKERS) or any(
        all(part in lowered for part in pair) for pair in _OVERFLOW_MARKER_PAIRS
    )


def overflow_message(model: str) -> str:
    """What the user is told when a conversation no longer fits ``model``."""
    return f"This conversation no longer fits {model}'s context window. Use /compact or /new."


def _is_langchain_overflow(error: BaseException) -> bool:
    """Whether ``error`` is LangChain's ``ContextOverflowError`` (False without LangChain)."""
    try:
        from langchain_core.exceptions import ContextOverflowError
    except ImportError:
        return False
    return isinstance(error, ContextOverflowError)
