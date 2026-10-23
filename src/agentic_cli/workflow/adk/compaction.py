"""Summarizing the main agent's conversation on ADK.

ADK compacts a session through the app's ``EventsCompactionConfig``. Before
each model call it reads the calling agent's last prompt size, and past the
config's ``token_threshold`` it hands the summarizer the events since the
last summary (that summary first, as a "seed" event) and appends the event
the summarizer returns. Its request builder then sends the summary in place
of the events the summary's range covers; the session keeps them all, for
``/sessions`` and fact extraction.

``AgentCliSummarizer`` is that summarizer:

- **The main agent only.** ``MainAgentPlugin`` records the agent that is
  running (``running_agent()``), and nothing is written for another one. A
  delegated sub-agent's check would compact the shared session under it: a
  ``fresh`` one then lost its brief and looped (100 requests in one turn,
  offline). A sub-agent too big for its own model is the fit guard's.
- **The range** (``summary_range``) keeps the newest ``keep`` tokens word for
  word and cuts where a user turn starts, never between a call and its
  result, never inside the current turn. It is chosen before anything is
  paid for: with no clean cut, or a saving under a fifth of the trigger,
  nothing is summarized.
- **The call** runs on the main agent's own model, thinking off, at most
  ``SUMMARY_MAX_TOKENS`` written, with the shared summary prompt
  (``workflow/summary_prompt.py``) over the events as text
  (``render_events``: thoughts left out, each tool call and result cut to
  ``TOOL_PAYLOAD_CHARS``). It is sized to the model that writes it, not to
  the conversation: after a switch to a smaller model the range may be
  larger than its window. Then the summary so far stays whole and the middle
  of the rest is left out (``MIDDLE_OMITTED``), as LangChain's summary helper
  does on an overflow, but before the call, so none is refused.
- **The summary** is user-role text starting with ``SUMMARY_PREFIX``. ADK
  sends it as the first message from then on, and a model-role first message
  is one Claude refuses.
- **Reporting** goes through the turn's live events
  (``workflow/live_events.py``), while the turn's stream waits on the call:
  ``CONTEXT_SUMMARIZING`` as it starts, its ``LLM_USAGE``
  (``purpose="summarization"``, the main agent's: its name and context
  trigger), then ``CONTEXT_TRIMMED``.
- **A failure** of any kind is one recoverable ``ERROR`` a turn; nothing is
  written, ADK goes on with the whole history, and its next model call tries
  again. A summary asked for (``manual``) reports none of its own: its
  caller reads ``last_error``. A cancel is no failure: it ends the turn, and
  nothing is written.

ADK's own pass after each turn would make the user wait for a summary after
the answer; ``NoAfterTurnCompactionRunner`` switches it off (it would also
run with no agent, and write nothing).

``compact_session`` runs ADK's compaction once, now (``/compact``, an
overflow), through ADK's private ``_run_compaction_for_token_threshold_config``.
Contract tests pin that function, ``Runner._run_post_invocation_compaction``
and the summarizer interface.
"""

from __future__ import annotations

import json
import re
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from google.adk import Runner
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.base_events_summarizer import BaseEventsSummarizer
from google.adk.events import Event, EventActions
from google.adk.events.event_actions import EventCompaction
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from agentic_cli.logging import Loggers
from agentic_cli.workflow import live_events
from agentic_cli.workflow.adk.model_factory import generate_text
from agentic_cli.workflow.context import (
    ESTIMATE_MARGIN,
    CompactResult,
    chars_per_token,
    usage_fields,
)
from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.summary_prompt import SUMMARY_PREFIX, summary_prompt

if TYPE_CHECKING:
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.sessions import BaseSessionService, Session

    from agentic_cli.config import BaseSettings
    from agentic_cli.models import ProviderKind, ResolvedModel
    from agentic_cli.workflow.context import ContextBudget

logger = Loggers.workflow()

#: The most a summary may be, in tokens.
SUMMARY_MAX_TOKENS = 8_000
#: The most of one tool call or result the summary call reads, in characters.
TOOL_PAYLOAD_CHARS = 2_000
#: The smallest saving worth a summary, as a share of the trigger.
MIN_SAVING = 0.2

#: What the user is told, once a turn, when a summary could not be written.
SUMMARY_FAILED = (
    "Could not summarize the earlier conversation; older turns are left out instead."
)
SUMMARY_FAILED_CODE = "summary_failed"

#: Where the middle of a conversation too long for the summarizing model is left out.
MIDDLE_OMITTED = "[... middle of the conversation omitted ...]"
_OMITTED = f"\n\n{MIDDLE_OMITTED}\n\n"

#: How the summary so far is shown to the call that writes the next one.
_SEED_HEADING = "Summary of the conversation before this point:"

_running_agent: ContextVar[str | None] = ContextVar(
    "agentic_cli_adk_running_agent", default=None
)


def running_agent() -> str | None:
    """The agent whose turn this is, as ``MainAgentPlugin`` recorded it.

    ADK runs each agent in a task of its own, which copies the context: a
    sub-agent's name never reaches its parent's model calls, and ADK's own
    after-turn pass, run outside every agent, sees None.
    """
    return _running_agent.get()


class MainAgentPlugin(BasePlugin):
    """Records the running agent before ADK runs it, for ``running_agent()``."""

    def __init__(self) -> None:
        super().__init__(name="main_agent")

    async def before_agent_callback(
        self, *, agent: "BaseAgent", callback_context: "CallbackContext"
    ) -> None:
        _running_agent.set(agent.name)
        return None


class NoAfterTurnCompactionRunner(Runner):
    """ADK's runner without its compaction pass after each turn.

    ADK compacts before an agent's model call (in the flow) and, again, after
    the turn, before ``run_async`` ends: the user would wait for a summary
    after the answer. The pass before the next model call does the same work
    while the turn waits anyway. The method is private ADK; a contract test
    pins it, and if ADK renamed it, only that wait would come back.
    """

    async def _run_post_invocation_compaction(self, *, session: "Session") -> None:
        return None


@dataclass(frozen=True)
class WrittenSummary:
    """A summary just written, for the call it was asked for (an overflow).

    Attributes:
        text: The summary as the model reads it (its prefix included).
        kept_turns: How many of the main agent's user turns it kept word for
            word: the turns a request after it holds besides the summary.
    """

    text: str
    kept_turns: int


class AgentCliSummarizer(BaseEventsSummarizer):
    """Summarizes the main agent's earlier conversation (see the module docstring).

    Args:
        root: The main agent's name.
        model: Its model, resolved for summaries (thinking off,
            ``SUMMARY_MAX_TOKENS``).
        budget: Its context budget: what is kept (``keep``), and the trigger
            the saving is measured against.
        settings: The manager's settings (retries, timeouts).
        app_name: The application's name, which some providers are told.
        manual: Asked for now (``/compact``, an overflow): written whoever
            runs, whatever the saving. A failure is the caller's to report
            (``last_error``): no warning of its own.
        focus: What the summary should keep above all (``/compact <focus>``).
        source: What ``CONTEXT_TRIMMED`` names as the summary's cause:
            ``"compact"`` when ``manual``, else ``"summarization"``, unless given
            (a summary an overflow asked for is ``"summarization"``).

    Attributes:
        last_result: What the last summary written did; None until one is.
        last_kept_turns: How many of the main agent's user turns the last
            summary written kept word for word.
        last_error: Why the last call wrote no summary, if it failed; None
            otherwise (cleared at each call).
        last_summary: The last summary written, as the model reads it (its
            prefix included); None until one is.
    """

    #: The window a summary call is sized to, when not the model's: the limit
    #: a refusal stated (see ``_summarize``).
    _sized_to: int | None = None

    def __init__(
        self,
        *,
        root: str,
        model: "ResolvedModel",
        budget: "ContextBudget",
        settings: "BaseSettings",
        app_name: str,
        manual: bool = False,
        focus: str | None = None,
        source: str | None = None,
    ) -> None:
        self.root = root
        self.model = model
        self.budget = budget
        self.manual = manual
        self.focus = focus
        self.source = source or ("compact" if manual else "summarization")
        self.last_result: CompactResult | None = None
        self.last_kept_turns = 0
        self.last_error: Exception | None = None
        self.last_summary: str | None = None
        self._settings = settings
        self._app_name = app_name

    async def maybe_summarize_events(self, *, events: list[Event]) -> Event | None:
        self.last_error = None
        if not self.manual and running_agent() != self.root:
            return None
        try:
            return await self._summarize(events)
        except Exception as error:  # noqa: BLE001 - the turn goes on uncompacted
            self._failed(error)
            return None

    async def _summarize(self, events: list[Event]) -> Event | None:
        kind = self.model.record.kind
        cut = summary_range(events, keep_tokens=self.budget.keep, kind=kind)
        if cut is None:
            return None
        summarized, kept = events[:cut], events[cut:]
        # The summary so far is replaced, not removed: it is no saving.
        measured = summarized[1:] if _is_seed(summarized[0]) else summarized
        if not self.manual and estimate_event_tokens(measured, kind) < (
            MIN_SAVING * self.budget.trigger
        ):
            return None

        live_events.report(WorkflowEvent.context_summarizing(self.root))
        try:
            summary, usage = await generate_text(
                self.model, summary_prompt(self._rendered(summarized), self.focus),
                settings=self._settings, app_name=self._app_name,
            )
        except Exception as error:
            # Refused as too long: the window was assumed (an endpoint model
            # declared without one counts as 128,000) or is wrong. Once more,
            # sized to the limit the provider stated, else to half the window.
            if not _is_overflow(error):
                raise
            self._sized_to = _summary_window(str(error), self.budget.window)
            try:
                summary, usage = await generate_text(
                    self.model, summary_prompt(self._rendered(summarized), self.focus),
                    settings=self._settings, app_name=self._app_name,
                )
            finally:
                self._sized_to = None
        if not summary.strip():
            raise ValueError(f"{self.model.ref} wrote an empty summary")

        text = f"{SUMMARY_PREFIX}\n\n{summary.strip()}"
        event = Event(
            author="user",
            invocation_id=Event.new_id(),
            actions=EventActions(compaction=EventCompaction(
                start_timestamp=min(e.timestamp for e in summarized),
                end_timestamp=max(e.timestamp for e in summarized),
                compacted_content=types.Content(
                    role="user", parts=[types.Part.from_text(text=text)]
                ),
            )),
        )
        result = CompactResult(
            messages_summarized=cut,
            tokens_before=estimate_event_tokens(events, kind),
            tokens_after=estimate_event_tokens([event, *kept], kind),
        )
        live_events.report(self._usage(usage))
        live_events.report(WorkflowEvent.context_trimmed(
            len(events), len(kept) + 1, self.source,
            self.root, messages_summarized=result.messages_summarized,
            tokens_before=result.tokens_before, tokens_after=result.tokens_after,
        ))
        self.last_result, self.last_summary = result, text
        self.last_kept_turns = sum(
            1 for e in kept if _is_user_text(e) and _in_conversation(e)
        )
        logger.debug(
            "adk_conversation_summarized", agent=self.root,
            messages=result.messages_summarized, tokens_before=result.tokens_before,
            tokens_after=result.tokens_after,
        )
        return event

    def _rendered(self, summarized: list[Event]) -> str:
        """``summarized`` as text, as much of it as the summarizing model takes.

        The room is its window's 95% less the summary it may write
        (``SUMMARY_MAX_TOKENS``) and the prompt around the messages, at the
        provider's characters per token with the estimates' 10% margin. A
        longer rendering keeps the summary so far whole and leaves out the
        middle of the rest: a quarter of what room is left for its start, the
        rest for its end. A summary so far too long for the room alone is cut
        with the rest, the same way.
        """
        ratio = chars_per_token(self.model.record.kind)
        window = self._sized_to or self.budget.window
        room = int(
            (int(0.95 * window) - SUMMARY_MAX_TOKENS) * ratio / ESTIMATE_MARGIN
        ) - len(summary_prompt("", self.focus))
        seed = summarized[:1] if _is_seed(summarized[0]) else []
        head, rest = render_events(seed), render_events(summarized[len(seed):])
        whole = "\n".join(text for text in (head, rest) if text)
        if len(whole) <= room:
            return whole
        logger.debug("adk_summary_range_cut", agent=self.root, chars=len(whole), room=room)
        left = room - len(head) - 1
        if head and left > 2 * len(_OMITTED):
            return f"{head}\n{_cut_middle(rest, left)}"
        return _cut_middle(whole, room)

    def _usage(self, usage: types.GenerateContentResponseUsageMetadata | None) -> WorkflowEvent:
        """The summary call's usage, credited to the agent it summarized.

        It carries that agent's context numbers (an overflow's summary too),
        as its own calls' usage does.
        """
        event = WorkflowEvent.llm_usage(
            model=str(self.model.ref),
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            completion_tokens=getattr(usage, "candidates_token_count", None),
            total_tokens=getattr(usage, "total_token_count", None),
            thinking_tokens=getattr(usage, "thoughts_token_count", None),
            cached_tokens=getattr(usage, "cached_content_token_count", None),
        )
        event.metadata.update(
            purpose="summarization", agent=self.root, **usage_fields(self.budget)
        )
        return event

    def _failed(self, error: Exception) -> None:
        """One warning a turn: ADK goes on with the whole history and tries again.

        A summary asked for reports nothing itself: its caller reads
        ``last_error`` (``/compact`` raises it).
        """
        self.last_error = error
        logger.debug(
            "adk_summary_failed", agent=self.root, error_type=type(error).__name__,
            error=str(error),
        )
        if self.manual:
            return
        live = live_events.current()
        if live is not None and live.first(SUMMARY_FAILED_CODE):
            live.emit(WorkflowEvent.error(
                SUMMARY_FAILED, error_code=SUMMARY_FAILED_CODE, recoverable=True
            ))


async def compact_session(
    *,
    session: "Session",
    session_service: "BaseSessionService",
    root_agent: "BaseAgent",
    summarizer: AgentCliSummarizer,
) -> bool:
    """Run ADK's compaction of ``session`` once, now, with ``summarizer``.

    ADK picks the events since the last summary (that summary first), as
    before a model call, and appends what ``summarizer`` returns; the
    summarizer still chooses the range. A throwaway config makes it run
    whatever the last prompt size (a threshold of one token). Private ADK; a
    contract test pins it.

    Returns:
        Whether a summary was appended.
    """
    from google.adk.apps.compaction import _run_compaction_for_token_threshold_config

    return await _run_compaction_for_token_threshold_config(
        config=EventsCompactionConfig(
            summarizer=summarizer, token_threshold=1, event_retention_size=0
        ),
        session=session,
        session_service=session_service,
        agent=root_agent,
        agent_name="",
        current_branch=None,
    )


# --- the range -------------------------------------------------------------------------------


def summary_range(
    events: list[Event], *, keep_tokens: int, kind: "ProviderKind"
) -> int | None:
    """Where to cut ``events``: ``events[:i]`` is summarized, the rest kept.

    Walking back from the newest event until about ``keep_tokens`` are kept,
    the cut goes back to where that user turn started (a user's text, not a
    tool result), and further back while it would part a call from its
    result or leave a kept event stamped no later than a summarized one
    (ADK's range is by timestamp). A job's call (``long_running_tool_ids``)
    may be parted from its final result: ADK puts the call back in front of
    it. The newest user turn, the current one, is never summarized.

    ADK hands over only events whose calls are answered, but for calls it
    proved can never be: those may be summarized.

    Returns:
        The cut, or None when nothing would be summarized but the summary so
        far (the seed) or nothing at all.
    """
    starts = [i for i, event in enumerate(events) if _is_user_text(event)]
    ratio = chars_per_token(kind)
    cut, kept = len(events), 0.0
    while cut > 0 and kept < keep_tokens:
        cut -= 1
        kept += _chars(events[cut]) / ratio
    pending = set().union(*(event.long_running_tool_ids or () for event in events))
    for start in reversed(starts):
        if start <= cut and _clean_cut(events, start, pending):
            break
    else:
        return None
    if start == 0 or (start == 1 and _is_seed(events[0])):
        return None
    return start


def _clean_cut(events: list[Event], cut: int, long_running: set[str]) -> bool:
    """Whether cutting at ``cut`` keeps every call with its result, and the times apart."""
    if cut == 0:
        return True
    called = {
        call.id for event in events[:cut] for call in event.get_function_calls() if call.id
    } - long_running
    answered_later = {
        response.id
        for event in events[cut:]
        for response in event.get_function_responses()
        if response.id
    }
    if called & answered_later:
        return False
    last = max(event.timestamp for event in events[:cut])
    return all(event.timestamp > last for event in events[cut:])


def _is_user_text(event: Event) -> bool:
    """A user's message: where a turn starts (a tool result is no start)."""
    content = event.content
    if event.author != "user" or content is None or not content.parts:
        return False
    if event.actions and event.actions.compaction:
        return False
    return not event.get_function_responses() and any(
        part.text and not part.thought for part in content.parts
    )


def _is_seed(event: Event) -> bool:
    """The summary so far, which ADK hands back first (unmarked but for its words)."""
    return event.author != "user" and _text(event).startswith(SUMMARY_PREFIX)


# --- what the summary is written from --------------------------------------------------------


def render_events(events: list[Event]) -> str:
    """The conversation in ``events`` as the summary call reads it.

    One line per part: ``<author>: <text>``; a call ``<author> called
    <name>(<args JSON>)`` and a result ``<name> returned <response JSON>``,
    each cut to ``TOOL_PAYLOAD_CHARS``. Thoughts are left out, and so is a
    sub-agent's own work (events on its branch): its parent read its answer
    as the call's result. The summary so far comes first, as itself.
    """
    lines: list[str] = []
    for event in events:
        if not _in_conversation(event):
            continue
        if _is_seed(event):
            summary = _text(event).removeprefix(SUMMARY_PREFIX).strip()
            lines.append(f"{_SEED_HEADING}\n{summary}")
            continue
        for part in (event.content.parts if event.content else None) or []:
            if part.function_call:
                call = part.function_call
                lines.append(_cut(f"{event.author} called {call.name}({_json(call.args or {})})"))
            elif part.function_response:
                response = part.function_response
                lines.append(_cut(f"{response.name} returned {_json(response.response)}"))
            elif part.text and not part.thought:
                lines.append(f"{event.author}: {part.text}")
    return "\n".join(lines)


def estimate_event_tokens(events: list[Event], kind: "ProviderKind") -> int:
    """About how many tokens ``events`` add to a request of the main agent's.

    The characters of each event's content (a summary's, for a compaction
    event) as JSON, at the provider's characters per token. A sub-agent's
    own work (events on its branch) is not in the main agent's requests.
    """
    return int(sum(_chars(event) for event in events) / chars_per_token(kind))


def _chars(event: Event) -> int:
    if not _in_conversation(event):
        return 0
    compaction = event.actions.compaction if event.actions else None
    content = compaction.compacted_content if compaction is not None else event.content
    return len(content.model_dump_json(exclude_none=True)) if content is not None else 0


def _in_conversation(event: Event) -> bool:
    """Whether ``event`` is the main agent's conversation, not a sub-agent's branch."""
    return not event.branch


def _text(event: Event) -> str:
    parts = event.content.parts if event.content else None
    return "".join(part.text or "" for part in parts or [] if not part.thought)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _cut_middle(text: str, room: int) -> str:
    """``text`` in ``room`` characters: its start (a quarter) and its end, the
    middle left out, as LangChain's summary helper cuts an overflowing history."""
    if len(text) <= room:
        return text
    space = max(room - len(_OMITTED), 0)
    start = space // 4
    end = space - start
    return f"{text[:start]}{_OMITTED}{text[len(text) - end:]}"


def _cut(line: str) -> str:
    if len(line) <= TOOL_PAYLOAD_CHARS:
        return line
    return f"{line[:TOOL_PAYLOAD_CHARS]}…[{len(line) - TOOL_PAYLOAD_CHARS:,} characters cut]"


#: How providers state the limit a request went over: OpenAI-compatible
#: servers (vLLM, OpenRouter) "maximum context length is N", Claude "> N
#: maximum", Gemini "allowed (N)".
_STATED_LIMIT = re.compile(
    r"maximum context length is (\d[\d,]*)|> (\d[\d,]*) maximum|allowed \((\d[\d,]*)\)",
    re.IGNORECASE,
)


def _is_overflow(error: Exception) -> bool:
    from agentic_cli.workflow.context import is_context_overflow

    return is_context_overflow(error)


def _summary_window(message: str, window: int) -> int:
    """The window a summary refused as too long is sized to next.

    The limit the provider's ``message`` states, when it is under the
    ``window`` the refused call was sized to; else half that window.
    """
    match = _STATED_LIMIT.search(message)
    if match:
        stated = int(next(group for group in match.groups() if group).replace(",", ""))
        if stated < window:
            return stated
    return window // 2
