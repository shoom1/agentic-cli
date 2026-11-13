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

A model whose window is unknown (an OpenAI-compatible provider's model whose
``models`` entry gives none, an OpenRouter id before a listing) counts as
``UNKNOWN_WINDOW``. A budget under ``MIN_BUDGET`` leaves no room to work in,
so it fails where the agents are built, naming the model and its
``max_tokens``.

**A summary that failed** is checked again before every model call of its
turn, over the same range, and would fail again. ``FailedSummaries`` keeps
it for the rest of that turn, until its range has grown by ``MIN_SAVING`` of
the trigger.

**Overflow errors.** Each provider says "too long" its own way:
``is_context_overflow`` reads the error's attributes (status, code, text,
the SDK it came from) and never imports a provider SDK, and
``is_overflow_text`` the text alone, for an overflow reported in an answer.
LangChain's own ``ContextOverflowError`` is recognized when LangChain is
installed.

**A side call's prompt** (a summary of a fetched page, fact extraction) goes
to the default model whole: ``prompt_room`` is how many characters it may
hold, and ``cut_middle`` leaves out the middle of a longer one (never to
less than ``MIN_PROMPT_ROOM``).

Imports nothing from ``cli/``, no provider SDK and no LangChain at import.
"""

from __future__ import annotations

import weakref
from collections.abc import Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agentic_cli.models import ProviderKind
from agentic_cli.workflow.config import AgentGraphError

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings
    from agentic_cli.models import ResolvedModel
    from agentic_cli.workflow.live_events import LiveEvents

#: The window assumed for a model whose window is unknown.
UNKNOWN_WINDOW = 128_000
#: The smallest input budget an agent can work in.
MIN_BUDGET = 16_000
#: The share of the trigger kept word for word after a summary.
KEEP_FRACTION = 0.12
#: The smallest saving worth a summary, as a share of the trigger; also how
#: much a range must grow before a summary that failed in a turn is tried again.
MIN_SAVING = 0.2
#: The ``error_code`` of the error a conversation too long for its model ends in.
OVERFLOW_ERROR_CODE = "CONTEXT_OVERFLOW"
#: What the user is told, once a turn, when a summary could not be written, on
#: both backends. True whatever comes next: Deep Agents sends the conversation
#: whole; ADK does too, unless it is over its budget, when the fit guard or the
#: overflow handler leaves its oldest turns out and says so itself.
SUMMARY_FAILED = (
    "Could not summarize the earlier conversation; the agent goes on without a new summary."
)
#: The ``error_code`` of that warning.
SUMMARY_FAILED_CODE = "summary_failed"

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
#: Claude's request-size limit: a 413 from Anthropic's SDK, whose class is
#: ``RequestTooLargeError`` before anthropic 2 and ``ContentTooLargeError``
#: (or a subclass naming the cause) since. Its text names no context.
_REQUEST_TOO_LARGE = 413
_ANTHROPIC_SDK = "anthropic"
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


class FailedSummaries:
    """The automatic summaries that failed in the current turn, by range.

    A conversation past its trigger is checked for a summary before every
    model call, and a summary that failed (no text written, a provider's
    error) fails again over the same range, each attempt paying for all of
    it: a turn of 20 tool steps sent the range 20 times. So a failure is
    kept for its turn, whose later calls skip its range (``skips``) until it
    has grown by ``MIN_SAVING`` of the trigger, the saving a summary must be
    worth. A range is known by where it starts: after a summary written
    since (``/compact``, an overflow's), the range starts past it, a range of
    its own. The next turn tries again.

    The turn is its ``LiveEvents`` (``workflow/live_events.py``), held
    weakly, so a finished turn's is not kept alive; without one, nothing is
    kept.

    Args:
        trigger: The trigger the growth is a share of, in tokens.
    """

    def __init__(self, trigger: int) -> None:
        self._growth = MIN_SAVING * trigger
        self._turn: weakref.ref[LiveEvents] | None = None
        # The size of each range that failed, in tokens, by where it starts.
        self._failed: dict[Hashable, int] = {}

    def skips(self, turn: "LiveEvents | None", start: Hashable, tokens: int) -> bool:
        """Whether the summary of ``tokens`` from ``start`` failed in ``turn``,
        and the range has not grown enough since."""
        failed = self._failed.get(start) if self._is_current(turn) else None
        return failed is not None and tokens < failed + self._growth

    def failed(self, turn: "LiveEvents | None", start: Hashable, tokens: int) -> None:
        """Keep, for ``turn``, that the summary of ``tokens`` from ``start`` failed."""
        if turn is None:
            return
        if not self._is_current(turn):
            self._turn, self._failed = weakref.ref(turn), {}
        self._failed[start] = tokens

    def _is_current(self, turn: "LiveEvents | None") -> bool:
        return turn is not None and self._turn is not None and self._turn() is turn


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
                "Set the model's context_window in models, lower its max_tokens, or "
                "choose a model with a larger window."
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
    prompt against that trigger. Nothing for an agent with no budget: one
    whose model did not resolve (a native ADK YAML app's), or whose context
    nobody manages (LangGraph's).
    """
    if budget is None:
        return {}
    return {"context_trigger": budget.trigger, "window_known": budget.window_known}


def chars_per_token(kind: ProviderKind) -> float:
    """Characters per token for estimating a request to a ``kind`` provider."""
    return _CHARS_PER_TOKEN.get(kind, _DEFAULT_CHARS_PER_TOKEN)


#: Where the middle of a side call's prompt too long for its model is left out.
PROMPT_MIDDLE_OMITTED = "[... middle of the text omitted: too long for the model ...]"
#: The smallest room, in characters, a side call's prompt is cut down to; with
#: less, the prompt goes as it is and the provider answers.
MIN_PROMPT_ROOM = 4_000


def prompt_room(resolved: "ResolvedModel | None", max_tokens: int) -> int:
    """How many characters a one-off prompt to ``resolved`` may hold.

    95% of its window (``UNKNOWN_WINDOW`` when it is unknown, or when there
    is no model yet), less what the call may write, at the provider's
    characters per token with the estimates' margin: as the conversation
    summarizer sizes its rendering. What the call may write is ``resolved``'s
    ``max_tokens`` (the resolver clamps a request to the model's max output),
    else ``max_tokens``. Never negative.
    """
    window, ratio, output = UNKNOWN_WINDOW, _DEFAULT_CHARS_PER_TOKEN, max_tokens
    if resolved is not None:
        window = resolved.record.context_window or UNKNOWN_WINDOW
        ratio = chars_per_token(resolved.record.kind)
        output = resolved.max_tokens or max_tokens
    tokens = window * _WINDOW_SHARE // 100 - output
    return max(int(tokens * ratio / ESTIMATE_MARGIN), 0)


def cut_middle(text: str, room: int) -> str:
    """``text`` in ``room`` characters: its start and its end, half each, and
    ``PROMPT_MIDDLE_OMITTED`` between them, so the instructions a prompt
    starts or ends with stay."""
    if len(text) <= room:
        return text
    omitted = f"\n\n{PROMPT_MIDDLE_OMITTED}\n\n"
    space = max(room - len(omitted), 0)
    start = space // 2
    end = space - start
    return f"{text[:start]}{omitted}{text[len(text) - end:]}"


def is_context_overflow(error: BaseException) -> bool:
    """Whether ``error`` is a provider saying the request is too long for the model.

    True for LangChain's ``ContextOverflowError``; for the code
    ``context_length_exceeded`` (OpenAI); for a 413 Anthropic's SDK raised
    (Claude's request-size limit, whatever the SDK names its class and
    whatever the body says); and for a 400, 413 or 422 whose text holds one
    of the providers' overflow phrases. A 429 is never an overflow, even as
    LangChain's type: langchain-google-genai raises that for any error whose
    text holds "token limit", a rate limit included.

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
    if status == _REQUEST_TOO_LARGE and _raised_by(error, _ANTHROPIC_SDK):
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


def subagent_overflow_message(agent: str, model: str) -> str:
    """What the user is warned of when a sub-agent's request no longer fits ``model``.

    A recoverable warning, ``error_code`` ``OVERFLOW_ERROR_CODE``, on both
    backends: the turn goes on, and the sub-agent's parent is answered.
    """
    return f"{agent}: the request no longer fits {model}'s context window."


def _raised_by(error: BaseException, package: str) -> bool:
    """Whether ``error``'s class, or a class it derives from, is ``package``'s."""
    return any(cls.__module__.partition(".")[0] == package for cls in type(error).__mro__)


def _is_langchain_overflow(error: BaseException) -> bool:
    """Whether ``error`` is LangChain's ``ContextOverflowError`` (False without LangChain)."""
    try:
        from langchain_core.exceptions import ContextOverflowError
    except ImportError:
        return False
    return isinstance(error, ContextOverflowError)
