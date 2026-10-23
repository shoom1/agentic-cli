"""The main agent's conversation summarized on ADK: the summarizer and its pieces.

ADK's ``EventsCompactionConfig`` hands a summarizer the events since the
last summary (that summary first, as a "seed" event) before a model call
whose agent's last prompt passed the config's threshold, and appends the
event it returns. ``AgentCliSummarizer`` (``workflow/adk/compaction.py``)
writes for the main agent only, picks its own range (``summary_range``: the
newest ``keep`` tokens kept, cut where a user turn starts, never inside a
call and its result, never in the current turn) before paying for anything,
and returns a user-role summary. These tests drive it on hand-built events
and, for the summary call, on ``FakeAPI`` (Claude). The manager's wiring is
in ``tests/workflow/test_adk_compaction.py``.

The private ADK pieces it rests on have contract tests at the end.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx
import pytest

pytest.importorskip("google.adk")

from google.adk import Runner  # noqa: E402
from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.apps import App  # noqa: E402
from google.adk.apps import compaction as adk_compaction  # noqa: E402
from google.adk.apps.app import EventsCompactionConfig  # noqa: E402
from google.adk.apps.base_events_summarizer import BaseEventsSummarizer  # noqa: E402
from google.adk.events import Event  # noqa: E402
from google.adk.models.base_llm import BaseLlm  # noqa: E402
from google.adk.models.llm_response import LlmResponse  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.models import ProviderKind  # noqa: E402
from agentic_cli.workflow.adk import compaction  # noqa: E402
from agentic_cli.workflow.adk.compaction import (  # noqa: E402
    MIN_SAVING,
    SUMMARY_MAX_TOKENS,
    TOOL_PAYLOAD_CHARS,
    AgentCliSummarizer,
    MainAgentPlugin,
    NoAfterTurnCompactionRunner,
    compact_session,
    estimate_event_tokens,
    render_events,
    running_agent,
    summary_range,
)
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.context import CompactResult, ContextBudget  # noqa: E402
from agentic_cli.workflow.events import EventType, WorkflowEvent  # noqa: E402
from agentic_cli.workflow.live_events import LiveEvents, reporting  # noqa: E402
from agentic_cli.workflow.summary_prompt import SUMMARY_PREFIX  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI, anthropic_message  # noqa: E402
from tests.workflow.test_manager_models import route  # noqa: E402

CLAUDE = ProviderKind.ANTHROPIC
#: The trigger and the kept tail of a model whose summaries start at 16,000 tokens.
BUDGET = ContextBudget(window=200_000, window_known=True, budget=171_808, trigger=16_000,
                       keep=1_920)
LONG = "word " * 2_400  # 12,000 characters, about 3,600 tokens on Claude


# --- events as ADK stores them ---------------------------------------------------------------


def user(text: str, t: float) -> Event:
    return Event(author="user", timestamp=t, invocation_id=f"i{t}",
                 content=types.Content(role="user", parts=[types.Part.from_text(text=text)]))


def said(text: str, t: float, author: str = "solo", *, thought: str | None = None,
         **fields) -> Event:
    parts = [types.Part(text=thought, thought=True)] if thought else []
    parts.append(types.Part.from_text(text=text))
    return Event(author=author, timestamp=t,
                 content=types.Content(role="model", parts=parts), **fields)


def called(name: str, call_id: str, t: float, author: str = "solo", *,
           long_running: bool = False, **args) -> Event:
    event = Event(author=author, timestamp=t, content=types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(id=call_id, name=name, args=args))
    ]))
    if long_running:
        event.long_running_tool_ids = {call_id}
    return event


def answered(name: str, call_id: str, t: float, author: str = "solo", **response) -> Event:
    return Event(author=author, timestamp=t, content=types.Content(role="user", parts=[
        types.Part(function_response=types.FunctionResponse(
            id=call_id, name=name, response=response
        ))
    ]))


def seed(text: str, t: float) -> Event:
    """A previous summary, as ADK hands it back first."""
    return Event(author="model", timestamp=t, content=types.Content(
        role="user", parts=[types.Part.from_text(text=f"{SUMMARY_PREFIX}\n\n{text}")]
    ))


def tokens(*events: Event) -> int:
    return estimate_event_tokens(list(events), CLAUDE)


# --- the range -------------------------------------------------------------------------------


class TestSummaryRange:
    def test_the_current_turn_is_never_summarized(self):
        events = [user("One.", 0), said("A1.", 1), user("Two.", 2),
                  called("read_file", "c1", 3, path="a"), answered("read_file", "c1", 4, text="x")]

        assert summary_range(events, keep_tokens=0, kind=CLAUDE) == 2

    def test_about_keep_tokens_are_kept(self):
        events = [user(LONG, 0), said("A1.", 1), user(LONG, 2), said("A2.", 3), user("Three.", 4)]

        # Keeping the last two events is not enough: the cut goes back to "LONG" at 2.
        i = summary_range(events, keep_tokens=tokens(*events[3:]) + 1, kind=CLAUDE)

        assert i == 2
        assert tokens(*events[i:]) >= tokens(*events[3:]) + 1

    def test_the_cut_lands_where_a_user_turn_starts(self):
        events = [user("One.", 0), said("A1.", 1), user("Two.", 2), said("A2.", 3),
                  user("Three.", 4), said("A3.", 5), user("Four.", 6)]

        # The walk back stops at "A3." (5); the turn it is in starts at "Three." (4).
        assert summary_range(events, keep_tokens=tokens(events[6]) + 1, kind=CLAUDE) == 4

    def test_never_between_a_call_and_its_response(self):
        events = [user("Zero.", 0), said("A0.", 1), user("One.", 2),
                  called("lookup", "c1", 3), user("Two.", 4), answered("lookup", "c1", 5, n=1),
                  said("A2.", 6), user("Three.", 7)]

        # A cut at "Two." (4) would part c1 from its result: the turn before it is taken.
        assert summary_range(events, keep_tokens=tokens(*events[6:]) + 1, kind=CLAUDE) == 2

    def test_a_call_never_answered_does_not_hold_the_cut_back(self):
        # A turn cancelled mid-tool leaves a call no result will ever answer;
        # ADK hands it over as dead. Kept out, it would stand before every
        # later cut, and the conversation would never be summarized again.
        events = [user("Zero.", 0), said("A0.", 1), user("One.", 2), called("lookup", "c1", 3),
                  user("Two.", 4), said("A2.", 5), user("Three.", 6)]

        assert summary_range(events, keep_tokens=tokens(*events[5:]) + 1, kind=CLAUDE) == 4

    @pytest.mark.parametrize("long_running, cut", [(True, 5), (False, 2)])
    def test_a_long_running_call_does_not_hold_the_cut_back(self, long_running, cut):
        # A job's call is answered at once with a placeholder, and again when it
        # finishes; ADK puts the call back in front of that later answer.
        events = [user("Zero.", 0), said("A0.", 1), user("Start the job.", 2),
                  called("start_job", "j1", 3, long_running=long_running),
                  answered("start_job", "j1", 3.5, state="running"),
                  user("Two.", 4), answered("start_job", "j1", 5, state="done"),
                  said("A2.", 6), user("Three.", 7)]

        assert summary_range(events, keep_tokens=tokens(*events[6:]) + 1, kind=CLAUDE) == cut

    def test_equal_timestamps_move_the_cut_back(self):
        # ADK's range is inclusive: "Two." stamped like "A1." would be summarized with it.
        events = [user("Zero.", 0), said("A0.", 1), user("One.", 2), said("A1.", 3),
                  user("Two.", 3), said("A2.", 4), user("Three.", 5)]

        assert summary_range(events, keep_tokens=tokens(*events[5:]) + 1, kind=CLAUDE) == 2

    def test_a_kept_event_stamped_inside_the_range_moves_the_cut_back(self):
        # Parallel work can be appended after an event stamped later: "A2." is
        # kept, but stamped inside the range a cut at "Two." would summarize.
        events = [user("Zero.", 0), said("A0.", 1), user("One.", 2), said("A1.", 4),
                  user("Two.", 5), said("A2.", 3.5), user("Three.", 6)]

        assert summary_range(events, keep_tokens=tokens(*events[5:]) + 1, kind=CLAUDE) == 2

    def test_the_seed_and_the_turns_after_it_are_summarized_together(self):
        events = [seed("Earlier.", 0), user("One.", 1), said("A1.", 2), user("Two.", 3)]

        assert summary_range(events, keep_tokens=0, kind=CLAUDE) == 3

    def test_the_seed_alone_is_nothing_to_summarize(self):
        events = [seed("Earlier.", 0), user("One.", 1), said("A1.", 2)]

        assert summary_range(events, keep_tokens=0, kind=CLAUDE) is None

    def test_the_first_turn_alone_is_nothing_to_summarize(self):
        events = [user("One.", 0), said("A1.", 1), called("lookup", "c1", 2),
                  answered("lookup", "c1", 3, n=1)]

        assert summary_range(events, keep_tokens=0, kind=CLAUDE) is None

    def test_no_user_turn_no_cut(self):
        assert summary_range([said("A.", 0), said("B.", 1)], keep_tokens=0, kind=CLAUDE) is None


# --- what the summary is written from --------------------------------------------------------


class TestRenderEvents:
    def test_one_line_per_part(self):
        text = render_events([user("Hi.", 0), said("Hello.", 1), said("Bye.", 2, author="solo")])

        assert text == "user: Hi.\nsolo: Hello.\nsolo: Bye."

    def test_thoughts_are_left_out(self):
        text = render_events([said("Hello.", 1, thought="Let me think about it.")])

        assert text == "solo: Hello."

    def test_calls_and_their_results(self):
        text = render_events([
            called("read_file", "c1", 0, path="a.txt"),
            answered("read_file", "c1", 1, success=True, content="text"),
        ])

        assert text.splitlines() == [
            'solo called read_file({"path": "a.txt"})',
            'read_file returned {"success": true, "content": "text"}',
        ]

    def test_a_tool_payload_is_cut_to_2000_characters(self):
        text = render_events([
            called("write_file", "c1", 0, content="x" * 10_000),
            answered("write_file", "c1", 1, content="y" * 10_000),
        ])

        call, result = text.splitlines()
        assert TOOL_PAYLOAD_CHARS == 2_000
        assert call.startswith('solo called write_file({"content": "xxx')
        assert result.startswith('write_file returned {"content": "yyy')
        for line in (call, result):
            head, marker, _ = line.partition("…[")
            assert len(head) == TOOL_PAYLOAD_CHARS
            assert marker
            assert line.endswith(" characters cut]")

    def test_the_seed_is_the_summary_so_far(self):
        text = render_events([seed("The user is Ada.", 0), user("Hi.", 1)])

        assert text == (
            "Summary of the conversation before this point:\nThe user is Ada.\nuser: Hi."
        )

    def test_a_sub_agents_own_work_is_left_out(self):
        # Its parent read its answer as the tool's result: that is what the
        # conversation holds.
        text = render_events([
            called("researcher", "c1", 0, request="Find X."),
            said("Looking.", 1, author="researcher", branch="researcher@c1"),
            answered("researcher", "c1", 2, result="X is 42."),
        ])

        assert text.splitlines() == [
            'solo called researcher({"request": "Find X."})',
            'researcher returned {"result": "X is 42."}',
        ]


class TestEstimateEventTokens:
    def test_claude_counts_more_tokens_for_the_same_text(self):
        events = [user(LONG, 0)]

        assert estimate_event_tokens(events, CLAUDE) > estimate_event_tokens(
            events, ProviderKind.OPENAI
        )

    def test_tool_payloads_count(self):
        assert tokens(answered("read_file", "c1", 0, content=LONG)) > 3_000

    def test_a_sub_agents_own_work_does_not(self):
        assert tokens(said(LONG, 0, author="researcher", branch="researcher@c1")) == 0


# --- the summarizer --------------------------------------------------------------------------


class Recorded(LiveEvents):
    """A turn's live events, kept in a list."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[WorkflowEvent] = []

    def emit(self, event: WorkflowEvent) -> None:
        self.events.append(event)


async def as_agent(agent: str | None, work, live: LiveEvents | None = None):
    """Run ``work()`` as ADK runs a model call of ``agent``: in a task of its own,
    after the agent's callbacks."""

    async def run():
        if agent is not None:
            await MainAgentPlugin().before_agent_callback(
                agent=SimpleNamespace(name=agent), callback_context=None
            )
        if live is None:
            return await work()
        with reporting(live):
            return await work()

    return await asyncio.create_task(run())


def conversation() -> list[Event]:
    """Two long turns and a short current one: the first turn is worth a summary.

    Stamped from 1: ADK hands over only events after the last summary's end,
    0.0 before the first."""
    return [user(LONG, 1), said("A1.", 2), user(LONG, 3), said("A2.", 4), user("Three?", 5)]


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    (tmp_path / "home").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for variable in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-test-key")
    return tmp_path


class Claude(FakeAPI):
    """Claude answering from ``script``, then with "ok"."""

    def __init__(self) -> None:
        self.script: list = []
        super().__init__(lambda request: self.script.pop(0) if self.script else anthropic_message())


@pytest.fixture
def api(project, monkeypatch) -> Claude:
    api = Claude()
    route(monkeypatch, anthropic, "AsyncAnthropic", api)
    return api


@pytest.fixture
async def manager(api):
    """A manager whose main agent runs on Haiku, its default model on Sonnet,
    thinking high."""
    manager = GoogleADKWorkflowManager(
        agent_configs=[AgentConfig(name="solo", prompt="You help.",
                                   model="anthropic:claude-haiku-4-5")],
        settings=BaseSettings(session_store="memory", default_model="claude-sonnet-4-5",
                              thinking_effort="high"),
    )
    await manager.initialize_services()
    try:
        yield manager
    finally:
        await manager.cleanup()


def summarizer(manager, budget: ContextBudget = BUDGET, **options) -> AgentCliSummarizer:
    return AgentCliSummarizer(
        root="solo", model=manager._simple_model(SUMMARY_MAX_TOKENS, agent="solo"),
        budget=budget, settings=manager._settings, app_name=manager.app_name, **options,
    )


def written(text: str = "The summary.") -> dict:
    message = anthropic_message({"type": "text", "text": text})
    message["usage"] = {"input_tokens": 7_300, "output_tokens": 40}
    return message


class TestTheSummary:
    async def test_it_is_a_user_role_summary_of_the_range(self, manager, api):
        api.script = [written()]
        events = conversation()

        event = await as_agent("solo", lambda: summarizer(manager).maybe_summarize_events(
            events=events
        ))

        assert event.author == "user"
        span = event.actions.compaction
        assert (span.start_timestamp, span.end_timestamp) == (1, 2)  # "LONG" and "A1."
        assert span.compacted_content.role == "user"
        assert [part.text for part in span.compacted_content.parts] == [
            f"{SUMMARY_PREFIX}\n\nThe summary."
        ]

    async def test_it_runs_on_the_main_agents_model_without_thinking(self, manager, api):
        api.script = [written()]

        await as_agent("solo", lambda: summarizer(manager).maybe_summarize_events(
            events=conversation()
        ))

        (body,) = api.bodies
        assert body["model"] == "claude-haiku-4-5"
        assert body["max_tokens"] == SUMMARY_MAX_TOKENS == 8_000
        assert body["thinking"] == {"type": "disabled"}
        assert "system" not in body
        (message,) = body["messages"]
        prompt = message["content"][0]["text"]
        assert prompt.startswith("<role>\nContext Extraction Assistant\n</role>")
        assert prompt.endswith(f"Messages to summarize:\nuser: {LONG}\nsolo: A1.\n</messages>")

    async def test_a_focus_reaches_the_prompt(self, manager, api):
        api.script = [written()]

        await as_agent("solo", lambda: summarizer(
            manager, focus="the API decisions"
        ).maybe_summarize_events(events=conversation()))

        prompt = api.bodies[0]["messages"][0]["content"][0]["text"]
        assert "<focus>\nFocus the summary on: the API decisions\n</focus>\n\n<messages>" in prompt

    async def test_it_reports_as_it_goes(self, manager, api):
        api.script = [written()]
        live = Recorded()
        events = conversation()
        made = summarizer(manager)

        event = await as_agent("solo", lambda: made.maybe_summarize_events(events=events), live)

        starting, usage, trimmed = live.events
        assert (starting.type, starting.metadata) == (
            EventType.CONTEXT_SUMMARIZING, {"agent": "solo"}
        )
        assert usage.type is EventType.LLM_USAGE
        assert usage.metadata["model"] == "anthropic:claude-haiku-4-5"
        assert (usage.metadata["purpose"], usage.metadata["agent"]) == ("summarization", "solo")
        assert (usage.metadata["prompt_tokens"], usage.metadata["completion_tokens"]) == (
            7_300, 40
        )
        assert trimmed.type is EventType.CONTEXT_TRIMMED
        metadata = trimmed.metadata
        assert (metadata["source"], metadata["agent"]) == ("summarization", "solo")
        assert (metadata["messages_before"], metadata["messages_after"]) == (5, 4)
        assert metadata["messages_summarized"] == 2
        assert metadata["tokens_before"] == estimate_event_tokens(events, CLAUDE)
        assert metadata["tokens_after"] == estimate_event_tokens([event, *events[2:]], CLAUDE)
        assert metadata["tokens_after"] < metadata["tokens_before"]
        assert made.last_result == CompactResult(
            messages_summarized=2, tokens_before=metadata["tokens_before"],
            tokens_after=metadata["tokens_after"],
        )
        # The user turns it kept word for word: "LONG" (the second) and "Three?".
        assert made.last_kept_turns == 2

    async def test_a_small_saving_costs_no_call(self, manager, api):
        api.script = [written()]
        events = [user("One.", 0), said("A1.", 1), user(LONG, 2), said("A2.", 3),
                  user("Three?", 4)]
        assert tokens(*events[:2]) < MIN_SAVING * BUDGET.trigger

        live = Recorded()
        event = await as_agent("solo", lambda: summarizer(manager).maybe_summarize_events(
            events=events
        ), live)

        assert (event, api.requests, live.events) == (None, [], [])

    async def test_the_saving_leaves_the_summary_so_far_out(self, manager, api):
        """A new summary replaces the one before it rather than removing it: a
        large summary so far and a small turn are no saving worth a call."""
        api.script = [written()]
        events = [seed(LONG, 1), user("One.", 2), said("A1.", 3), user(LONG, 4),
                  said("A2.", 5), user("Three?", 6)]
        assert tokens(*events[:3]) >= MIN_SAVING * BUDGET.trigger  # with the seed
        assert tokens(*events[1:3]) < MIN_SAVING * BUDGET.trigger  # without it

        event = await as_agent("solo", lambda: summarizer(manager).maybe_summarize_events(
            events=events
        ))

        assert (event, api.requests) == (None, [])

    @pytest.mark.parametrize("agent", ["researcher", None], ids=["sub-agent", "after-turn"])
    async def test_only_the_main_agent_is_summarized(self, manager, api, agent):
        api.script = [written()]

        event = await as_agent(agent, lambda: summarizer(manager).maybe_summarize_events(
            events=conversation()
        ))

        assert (event, api.requests) == (None, [])

    async def test_asked_for_it_skips_both_checks(self, manager, api):
        """``/compact``: whoever runs, and whatever the saving."""
        api.script = [written()]
        live = Recorded()
        events = [user("One.", 0), said("A1.", 1), user(LONG, 2), said("A2.", 3),
                  user("Three?", 4)]

        event = await as_agent(None, lambda: summarizer(manager, manual=True)
                               .maybe_summarize_events(events=events), live)

        assert event is not None
        assert len(api.requests) == 1
        assert live.events[-1].metadata["source"] == "compact"


#: A 20,000-token model: 19,000 tokens for a request, its summaries at 16,150.
SMALL = ContextBudget(window=20_000, window_known=True, budget=19_000, trigger=16_150,
                      keep=1_938)


def summary_request_tokens(body: dict) -> float:
    """The summary request as the summarizing model counts it (Claude's
    characters per token, the fit guard's 10% margin), and the answer it may write."""
    prompt = body["messages"][0]["content"][0]["text"]
    return len(prompt) / 3.3 * 1.1 + SUMMARY_MAX_TOKENS


def long_range(n: int = 10) -> list[Event]:
    """``n`` turns of 20,000 characters each (200,000 for ten), then the current one."""
    events: list[Event] = []
    for i in range(n):
        events += [user(f"Turn {i}: " + "x" * 20_000, 2 * i + 1), said(f"A{i}.", 2 * i + 2)]
    return [*events, user("Now?", 2 * n + 1)]


class TestASummaryTheModelCanWrite:
    """The summary call is sized to the model that writes it, not to the
    conversation: after a switch to a smaller model, the whole range to
    summarize may be larger than its window. The summary so far stays whole;
    the middle of the rest is cut."""

    async def test_a_range_larger_than_the_window_is_cut_in_the_middle(self, manager, api):
        api.script = [written()]
        events = long_range()

        event = await as_agent("solo", lambda: summarizer(manager, budget=SMALL)
                               .maybe_summarize_events(events=events))

        (body,) = api.bodies
        prompt = body["messages"][0]["content"][0]["text"]
        assert event is not None
        assert summary_request_tokens(body) <= 0.95 * SMALL.window
        assert compaction.MIDDLE_OMITTED == "[... middle of the conversation omitted ...]"
        assert prompt.count(compaction.MIDDLE_OMITTED) == 1
        # Its start and its end: the first turn, and the last one summarized.
        start, _, end = prompt.partition(compaction.MIDDLE_OMITTED)
        assert "user: Turn 0: xxx" in start
        assert end.rstrip().endswith("solo: A8.\n</messages>")
        # A quarter of the room for the start, the rest for the end.
        assert len(start) < len(end)

    async def test_the_summary_so_far_stays_whole(self, manager, api):
        api.script = [written()]
        so_far = "The user is Ada. " * 300  # 5,100 characters
        events = [seed(so_far, 0), *long_range()]

        await as_agent("solo", lambda: summarizer(manager, budget=SMALL)
                       .maybe_summarize_events(events=events))

        (body,) = api.bodies
        prompt = body["messages"][0]["content"][0]["text"]
        assert summary_request_tokens(body) <= 0.95 * SMALL.window
        assert f"Summary of the conversation before this point:\n{so_far.strip()}" in prompt
        assert compaction.MIDDLE_OMITTED in prompt.partition(so_far.strip())[2]

    async def test_a_summary_so_far_too_big_alone_is_cut_too(self, manager, api):
        api.script = [written()]
        events = [seed("y" * 60_000, 0), *long_range(2)]

        await as_agent("solo", lambda: summarizer(manager, budget=SMALL)
                       .maybe_summarize_events(events=events))

        (body,) = api.bodies
        assert summary_request_tokens(body) <= 0.95 * SMALL.window
        assert compaction.MIDDLE_OMITTED in body["messages"][0]["content"][0]["text"]

    async def test_a_range_that_fits_is_sent_whole(self, manager, api):
        api.script = [written()]

        await as_agent("solo", lambda: summarizer(manager).maybe_summarize_events(
            events=conversation()
        ))

        assert compaction.MIDDLE_OMITTED not in api.bodies[0]["messages"][0]["content"][0]["text"]


class _Logs:
    """Stands in for a module's structlog logger."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def __getattr__(self, level: str):
        def log(event: str, **fields) -> None:
            self.calls.append((level, event, fields))

        return log


class TestAFailedSummary:
    @staticmethod
    def refused() -> httpx.Response:
        return httpx.Response(400, json={"type": "error", "error": {
            "type": "invalid_request_error", "message": "max_tokens: 100000 > 64000"
        }})

    async def test_it_is_one_warning_a_turn_and_no_summary(self, manager, api, monkeypatch):
        logs = _Logs()
        monkeypatch.setattr(compaction, "logger", logs)
        api.script = [self.refused(), self.refused(), self.refused()]
        made = summarizer(manager)
        turn, next_turn = Recorded(), Recorded()

        first = await as_agent("solo", lambda: made.maybe_summarize_events(
            events=conversation()
        ), turn)
        again = await as_agent("solo", lambda: made.maybe_summarize_events(
            events=conversation()
        ), turn)
        later = await as_agent("solo", lambda: made.maybe_summarize_events(
            events=conversation()
        ), next_turn)

        assert (first, again, later) == (None, None, None)
        assert made.last_result is None
        errors = [e for e in turn.events if e.type is EventType.ERROR]
        (error,) = errors
        assert error.content == (
            "Could not summarize the earlier conversation; older turns are left out instead."
        )
        assert error.metadata == {"recoverable": True, "error_code": "summary_failed"}
        assert [e.type for e in next_turn.events] == [
            EventType.CONTEXT_SUMMARIZING, EventType.ERROR
        ]
        assert [(level, event) for level, event, _ in logs.calls] == [
            ("debug", "adk_summary_failed")
        ] * 3
        assert logs.calls[0][2]["error_type"] == "BadRequestError"

    async def test_the_failure_is_kept_until_the_next_call(self, manager, api):
        api.script = [self.refused(), written()]
        made = summarizer(manager)

        await as_agent("solo", lambda: made.maybe_summarize_events(events=conversation()))
        failed = made.last_error
        await as_agent("solo", lambda: made.maybe_summarize_events(events=conversation()))

        assert isinstance(failed, anthropic.BadRequestError)
        assert made.last_error is None

    async def test_asked_for_one_reports_no_warning_of_its_own(self, manager, api):
        """``/compact`` reports its own failure (``compact()`` raises it)."""
        api.script = [self.refused()]
        made = summarizer(manager, manual=True)
        live = Recorded()

        event = await as_agent(None, lambda: made.maybe_summarize_events(
            events=conversation()
        ), live)

        assert event is None
        assert isinstance(made.last_error, anthropic.BadRequestError)
        assert [e.type for e in live.events] == [EventType.CONTEXT_SUMMARIZING]

    async def test_an_empty_summary_is_a_failure(self, manager, api):
        api.script = [written("")]
        live = Recorded()

        event = await as_agent("solo", lambda: summarizer(manager).maybe_summarize_events(
            events=conversation()
        ), live)

        assert event is None
        assert [e.type for e in live.events] == [EventType.CONTEXT_SUMMARIZING, EventType.ERROR]

    async def test_a_cancel_is_no_failure(self, manager, monkeypatch):
        async def cancelled(*args, **kwargs):
            raise asyncio.CancelledError

        monkeypatch.setattr(compaction, "generate_text", cancelled)
        live = Recorded()

        with pytest.raises(asyncio.CancelledError):
            await as_agent("solo", lambda: summarizer(manager).maybe_summarize_events(
                events=conversation()
            ), live)

        assert [e.type for e in live.events] == [EventType.CONTEXT_SUMMARIZING]


# --- compacting now --------------------------------------------------------------------------


class TestCompactSession:
    @staticmethod
    async def session_of(service, *events: Event):
        session = await service.create_session(app_name="app", user_id="u")
        for event in events:
            await service.append_event(session, event)
        return await service.get_session(app_name="app", user_id="u", session_id=session.id)

    async def test_it_appends_the_summary(self, manager, api):
        api.script = [written()]
        service = InMemorySessionService()
        session = await self.session_of(service, *conversation())
        made = summarizer(manager, manual=True)

        compacted = await compact_session(
            session=session, session_service=service, root_agent=LlmAgent(name="solo"),
            summarizer=made,
        )

        stored = await service.get_session(app_name="app", user_id="u", session_id=session.id)
        assert compacted is True
        (summary,) = [e for e in stored.events if e.actions.compaction]
        assert summary.actions.compaction.compacted_content.parts[0].text.endswith(
            "The summary."
        )
        assert made.last_result.messages_summarized == 2

    async def test_nothing_to_summarize_appends_nothing(self, manager, api):
        api.script = [written()]
        service = InMemorySessionService()
        session = await self.session_of(service, user("One.", 1), said("A1.", 2))

        compacted = await compact_session(
            session=session, session_service=service, root_agent=LlmAgent(name="solo"),
            summarizer=summarizer(manager, manual=True),
        )

        stored = await service.get_session(app_name="app", user_id="u", session_id=session.id)
        assert compacted is False
        assert api.requests == []
        assert not [e for e in stored.events if e.actions.compaction]


# --- the ADK it rests on ---------------------------------------------------------------------


class Canned(BaseLlm):
    """A model answering every call with "ok", reporting ``prompt`` prompt tokens."""

    model: str = "canned"
    prompt: int = 10

    async def generate_content_async(
        self, llm_request, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part.from_text(text="ok")]),
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=self.prompt, candidates_token_count=1,
                total_token_count=self.prompt + 1,
            ),
        )


class Counting(BaseEventsSummarizer):
    def __init__(self) -> None:
        self.calls: list[str | None] = []

    async def maybe_summarize_events(self, *, events):
        self.calls.append(running_agent())
        return None


async def one_turn(runner_class: type[Runner], summarizer: BaseEventsSummarizer) -> None:
    app = App(
        name="contract", root_agent=LlmAgent(name="solo", model=Canned(prompt=5_000)),
        plugins=[MainAgentPlugin()],
        events_compaction_config=EventsCompactionConfig(
            summarizer=summarizer, token_threshold=1_000, event_retention_size=0
        ),
    )
    runner = runner_class(app=app, session_service=InMemorySessionService())
    session = await runner.session_service.create_session(app_name="contract", user_id="u")
    for text in ("one", "two"):
        async for _ in runner.run_async(
            user_id="u", session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part.from_text(text=text)]),
        ):
            pass


class TestTheADKItRestsOn:
    def test_a_summarizer_is_handed_the_events_by_keyword(self):
        parameters = inspect.signature(BaseEventsSummarizer.maybe_summarize_events).parameters

        assert list(parameters) == ["self", "events"]
        assert parameters["events"].kind is inspect.Parameter.KEYWORD_ONLY

    def test_the_after_turn_pass_is_one_private_method(self):
        method = Runner._run_post_invocation_compaction
        parameters = inspect.signature(method).parameters

        assert inspect.iscoroutinefunction(method)
        assert list(parameters) == ["self", "session"]
        assert parameters["session"].kind is inspect.Parameter.KEYWORD_ONLY

    async def test_the_runner_runs_that_pass_after_every_turn(self):
        """Overriding it is how the after-turn pass goes: ADK must still call it."""
        after: list[str] = []

        class Recording(Runner):
            async def _run_post_invocation_compaction(self, *, session) -> None:
                after.append(session.id)
                return await super()._run_post_invocation_compaction(session=session)

        summarizer = Counting()
        await one_turn(Recording, summarizer)

        assert len(after) == 2
        # After the first turn, with no agent; before the second turn's model
        # call, in the agent's task; after the second turn.
        assert summarizer.calls == [None, "solo", None]

    async def test_without_it_only_the_checks_before_model_calls_run(self):
        summarizer = Counting()

        await one_turn(NoAfterTurnCompactionRunner, summarizer)

        assert summarizer.calls == ["solo"]

    def test_compacting_now_is_one_private_function(self):
        function = adk_compaction._run_compaction_for_token_threshold_config
        parameters = inspect.signature(function).parameters

        assert inspect.iscoroutinefunction(function)
        assert list(parameters) == [
            "config", "session", "session_service", "agent", "agent_name", "current_branch",
        ]
        assert {p.kind for p in parameters.values()} == {inspect.Parameter.KEYWORD_ONLY}

    def test_a_compaction_event_carries_its_range_and_content(self):
        from google.adk.events.event_actions import EventCompaction

        assert {"start_timestamp", "end_timestamp", "compacted_content"} <= set(
            EventCompaction.model_fields
        )
