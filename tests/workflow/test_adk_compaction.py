"""ADK summarizes the main agent's conversation, as Deep Agents does.

A manager built from agent configs gives its App an ``EventsCompactionConfig``
with ``AgentCliSummarizer`` (``workflow/adk/compaction.py``), at the main
agent's trigger. Before each of the main agent's model calls, ADK reads its
last prompt size, and past the trigger the earlier conversation is
summarized: a user-role summary stored as a compaction event, sent from then
on in place of the turns it covers. The current turn is never summarized. The
summary is reported live while the turn waits on it (``CONTEXT_SUMMARIZING``,
its usage, ``CONTEXT_TRIMMED``). A sub-agent past its trigger never compacts
the shared session; ADK's pass after each turn does not run; a model whose
budget is too small fails initialization. A native ADK YAML app gets none of
this. ``compact()`` (``/compact``) summarizes the same way now, whatever the
last prompt's size.

Turns run the real manager on ``FakeAPI`` (Claude, ``context_max_tokens`` at
16,000 so the trigger is small); usage is scripted per call, since ADK reads
the trigger from the provider's count.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call

import anthropic
import httpx
import openai
import pytest

pytest.importorskip("google.adk")

from google.adk import Runner  # noqa: E402
from google.adk.apps import compaction as adk_compaction  # noqa: E402

from agentic_cli.cli.message_processor import CANCELLED_LINE, MessageProcessor  # noqa: E402
from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.tools.jobs import JobState  # noqa: E402
from agentic_cli.cli.usage_tracker import UsageTracker  # noqa: E402
from agentic_cli.tools.registry import ToolCategory, register_tool  # noqa: E402
from agentic_cli.workflow import live_events  # noqa: E402
from agentic_cli.workflow.adk import compaction  # noqa: E402
from agentic_cli.workflow.adk.compaction import (  # noqa: E402
    SUMMARY_MAX_TOKENS,
    AgentCliSummarizer,
    MainAgentPlugin,
    NoAfterTurnCompactionRunner,
    running_agent,
)
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.context import (  # noqa: E402
    CompactResult,
    ContextBudgetError,
    context_budget,
)
from agentic_cli.workflow.events import EventType  # noqa: E402
from agentic_cli.workflow.permissions import EXEMPT  # noqa: E402
from agentic_cli.workflow.summary_prompt import SUMMARY_PREFIX  # noqa: E402
from tests.cli.dialog_fakes import autospec_session, opening_boxes  # noqa: E402
from tests.event_replay import ReplayController  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI, anthropic_message, openai_response  # noqa: E402
from tests.workflow.test_adk_context_cache import _native_yaml  # noqa: E402
from tests.workflow.test_manager_models import route  # noqa: E402

LONG = "word " * 2_400  # 12,000 characters, about 3,600 tokens on Claude
#: A prompt past the 16,000-token trigger, and one well under it.
OVER, UNDER = 17_000, 4_000
TURNS = [f"One: {LONG}", f"Two: {LONG}", "Three?", "Four?"]
SUMMARY = "The user wrote twice at length."
#: Each agent's system prompt begins with its prompt.
PROMPTS = {"solo": "You help.", "coordinator": "You coordinate.", "researcher": "You research."}


def says(text: str, prompt_tokens: int = UNDER) -> dict:
    message = anthropic_message({"type": "text", "text": text})
    message["usage"] = {"input_tokens": prompt_tokens, "output_tokens": 5}
    return message


def calls(name: str, call_id: str, prompt_tokens: int = UNDER, **args) -> dict:
    message = says("", prompt_tokens)
    message["content"] = [{"type": "tool_use", "id": call_id, "name": name, "input": args}]
    message["stop_reason"] = "tool_use"
    return message


def is_summary(body: dict) -> bool:
    first = body["messages"][0]["content"]
    text = first if isinstance(first, str) else first[0].get("text", "")
    return "system" not in body and text.startswith("<role>\nContext Extraction Assistant")


def system_text(body: dict) -> str:
    system = body.get("system") or ""
    if isinstance(system, list):
        return "\n".join(block.get("text", "") for block in system)
    return system


def lines(body: dict) -> list[str]:
    """Each message of a Claude request as ``"<role>: <block>"``, block by block."""
    out: list[str] = []
    for message in body["messages"]:
        content = message["content"]
        if isinstance(content, str):
            out.append(f"{message['role']}: {content}")
            continue
        for block in content:
            if block["type"] == "text":
                out.append(f"{message['role']}: {block['text']}")
            elif block["type"] == "tool_use":
                out.append(f"{message['role']}: called {block['name']}#{block['id']}")
            elif block["type"] == "tool_result":
                out.append(f"{message['role']}: result#{block['tool_use_id']}")
    return out


class Claude:
    """Claude for each agent (told apart by its system prompt) and for the
    summary calls; every request is recorded under who it was for."""

    def __init__(self, summaries: list | None = None, **scripts: list) -> None:
        self.scripts = {name: list(script) for name, script in scripts.items()}
        self.summaries = list(summaries or [])
        self.sent: list[tuple[str, dict]] = []
        #: Held summary calls: set ``hold`` to hold the next one until ``release``.
        self.hold = False
        self.held = asyncio.Event()
        self.release = asyncio.Event()

    def __call__(self, request):
        body = json.loads(request.content)
        if is_summary(body):
            self.sent.append(("summary", body))
            answer = self.summaries.pop(0) if self.summaries else says(SUMMARY, 7_000)
            if self.hold:
                self.hold = False
                return self._held(answer)
            return answer
        system = system_text(body)
        agent = next(name for name, prompt in PROMPTS.items() if system.startswith(prompt))
        self.sent.append((agent, body))
        script = self.scripts.get(agent) or []
        return script.pop(0) if script else says(f"[{agent} has no scripted answer]")

    async def _held(self, answer):
        self.held.set()
        await self.release.wait()
        return answer

    @property
    def asked(self) -> list[str]:
        return [who for who, _ in self.sent]

    def bodies(self, who: str) -> list[dict]:
        return [body for name, body in self.sent if name == who]


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    """A hermetic project (the cwd) and HOME, with a fake Claude key."""
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path / "project")
    for variable in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-test-key")
    return tmp_path / "project"


def settings(**values) -> BaseSettings:
    return BaseSettings(**{
        "session_store": "memory", "default_model": "claude-sonnet-4-6",
        "context_max_tokens": 16_000, **values,
    })


def solo(**values) -> GoogleADKWorkflowManager:
    return GoogleADKWorkflowManager(
        agent_configs=[AgentConfig(name="solo", prompt=PROMPTS["solo"],
                                   include_state_tools=False)],
        settings=settings(**values),
    )


def serve(monkeypatch, answers: Claude) -> Claude:
    route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(answers))
    return answers


async def turn(manager, text: str, session_id: str = "s") -> list:
    return [event async for event in manager.process(text, "u", session_id=session_id)]


def texts(events) -> list[str]:
    return [event.content for event in events if event.type is EventType.TEXT]


async def stored(manager, session_id: str = "s"):
    session = await manager._session_service.get_session(
        app_name=manager.app_name, user_id="u", session_id=session_id
    )
    return session.events


def summaries_in(events) -> list:
    return [event for event in events if event.actions and event.actions.compaction]


def four_turns() -> Claude:
    """The second turn's prompt passes the trigger: the third is summarized first."""
    return Claude(solo=[says("A1."), says("A2.", OVER), says("A3."), says("A4.")])


def pumps() -> list[asyncio.Task]:
    """The live-events pump tasks still running."""
    return [t for t in asyncio.all_tasks() if t.get_name() == live_events.PUMP_TASK]


class TestSummarizing:
    async def test_past_the_trigger_the_earlier_turns_are_summarized(self, project, monkeypatch):
        answers = serve(monkeypatch, four_turns())
        manager = solo()
        try:
            for text in TURNS:
                await turn(manager, text)
            events = await stored(manager)
        finally:
            await manager.cleanup()

        assert answers.asked == ["solo", "solo", "summary", "solo", "solo"]
        # Turn one is summarized; turn two, kept word for word, is the tail.
        (summary,) = summaries_in(events)
        span = summary.actions.compaction
        assert (span.start_timestamp, span.end_timestamp) == (
            events[0].timestamp, events[1].timestamp
        )
        assert summary.author == "user"
        assert span.compacted_content.role == "user"
        prompt = lines(answers.bodies("summary")[0])[0]
        assert f"user: {TURNS[0]}\nsolo: A1." in prompt and "Two:" not in prompt
        # The summary takes their place, from the third turn on.
        summarized = f"user: {SUMMARY_PREFIX}\n\n{SUMMARY}"
        third, fourth = answers.bodies("solo")[2:]
        assert lines(third) == [summarized, f"user: {TURNS[1]}", "assistant: A2.",
                                "user: Three?"]
        assert lines(fourth) == [*lines(third), "assistant: A3.", "user: Four?"]

    async def test_the_summary_is_reported_while_the_turn_waits_on_it(self, project, monkeypatch):
        serve(monkeypatch, four_turns())
        manager = solo()
        try:
            for text in TURNS[:2]:
                await turn(manager, text)
            third = await turn(manager, TURNS[2])
        finally:
            await manager.cleanup()

        kinds = [event.type for event in third]
        assert kinds[:3] == [
            EventType.CONTEXT_SUMMARIZING, EventType.LLM_USAGE, EventType.CONTEXT_TRIMMED
        ]
        assert kinds.index(EventType.CONTEXT_TRIMMED) < kinds.index(EventType.TEXT)
        usage, trimmed = third[1], third[2]
        assert usage.metadata["purpose"] == "summarization"
        assert usage.metadata["model"] == "anthropic:claude-sonnet-4-6"
        assert (trimmed.metadata["source"], trimmed.metadata["messages_summarized"]) == (
            "summarization", 2
        )
        assert texts(third) == ["A3."]
        # The summary's usage, like the answer's, is the main agent's: its
        # trigger (the 16,000-token ceiling) and a known window.
        answer_usage = [e for e in third if e.type is EventType.LLM_USAGE][-1]
        for event in (usage, answer_usage):
            assert (event.metadata["agent"], event.metadata["context_trigger"],
                    event.metadata["window_known"]) == ("solo", 16_000, True)
        assert "purpose" not in answer_usage.metadata

    async def test_the_status_bar_never_shows_the_summarys_prompt(self, project, monkeypatch):
        """The summary call's prompt (7,000 tokens here) is the history it
        condenses: the main agent's context goes from the second turn's prompt
        to the third's, past the trigger and back under it."""
        serve(monkeypatch, four_turns())
        manager = solo()
        tracker, shown = UsageTracker(), []
        try:
            for text in TURNS[:3]:
                for event in await turn(manager, text):
                    if event.type is EventType.LLM_USAGE:
                        tracker.record(event.metadata)
                        shown.append(tracker.format_context())
        finally:
            await manager.cleanup()

        assert shown == [
            "ctx 4k/16k (25%)",
            "ctx 17k/16k (106%)",
            "ctx 17k/16k (106%)",  # the summary's call
            "ctx 4k/16k (25%)",
        ]

    async def test_a_second_summary_replaces_the_first(self, project, monkeypatch):
        """The main path of a long conversation: the second summary starts where
        the first did (ADK then drops the first), reads it as the summary so
        far, and only it is sent."""
        answers = serve(monkeypatch, Claude(
            solo=[says("A1."), says("A2.", OVER), says("A3."), says("A4.", OVER), says("A5."),
                  says("A6.")],
            summaries=[says("First summary.", 100), says("Second summary.", 100)],
        ))
        manager = solo()
        try:
            for text in [f"One: {LONG}", f"Two: {LONG}", f"Three: {LONG}", f"Four: {LONG}",
                         "Five?", "Six?"]:
                await turn(manager, text)
            events = await stored(manager)
        finally:
            await manager.cleanup()

        first, second = summaries_in(events)
        a, b = first.actions.compaction, second.actions.compaction
        assert b.start_timestamp == a.start_timestamp and b.end_timestamp > a.end_timestamp
        prompt = lines(answers.bodies("summary")[1])[0]
        assert "Summary of the conversation before this point:\nFirst summary." in prompt
        last = lines(answers.bodies("solo")[-1])
        assert last[0] == f"user: {SUMMARY_PREFIX}\n\nSecond summary."
        assert not any("First summary." in line for line in last)

    async def test_under_the_trigger_nothing_is_summarized(self, project, monkeypatch):
        answers = serve(monkeypatch, Claude(solo=[says(f"A{n}.") for n in range(1, 5)]))
        manager = solo()
        try:
            for text in TURNS:
                await turn(manager, text)
            events = await stored(manager)
        finally:
            await manager.cleanup()

        assert answers.asked == ["solo"] * 4
        assert summaries_in(events) == []

    async def test_the_pass_after_each_turn_does_not_run(self, project, monkeypatch):
        passes = []
        real = adk_compaction._run_compaction_for_sliding_window

        def recorded(*args, **kwargs):
            passes.append(kwargs.get("session") or args[1])
            return real(*args, **kwargs)

        monkeypatch.setattr(adk_compaction, "_run_compaction_for_sliding_window", recorded)
        answers = serve(monkeypatch, four_turns())
        manager = solo()
        try:
            for text in TURNS:
                await turn(manager, text)
            runner = manager._runner
        finally:
            await manager.cleanup()

        assert type(runner) is NoAfterTurnCompactionRunner
        # The summary was written before the third turn's call, not after the second.
        assert passes == []
        assert answers.asked.index("summary") == 2

    async def test_a_model_too_small_to_work_on_fails_initialization(self, project, monkeypatch):
        answers = serve(monkeypatch, Claude())
        api = FakeAPI(openai_response())
        route(monkeypatch, openai, "AsyncOpenAI", api)
        manager = solo(
            model_endpoints={"local": {"base_url": "http://localhost:8000/v1",
                                       "models": {"tiny": {"context_window": 10_000}}}},
            default_model="local:tiny",
        )

        with pytest.raises(ContextBudgetError, match=r"'solo'.*local:tiny"):
            await manager.initialize_services()

        assert not manager.is_initialized
        assert (answers.sent, api.requests) == ([], [])


class TestTheApp:
    async def test_the_main_agents_trigger_and_model_are_the_summarys(self, project, monkeypatch):
        serve(monkeypatch, Claude())
        configs = [
            AgentConfig(name="coordinator", prompt=PROMPTS["coordinator"],
                        sub_agents=["researcher"], description="Leads.",
                        include_state_tools=False),
            AgentConfig(name="researcher", prompt=PROMPTS["researcher"], description="Finds.",
                        model="anthropic:claude-haiku-4-5", include_state_tools=False),
        ]
        manager = GoogleADKWorkflowManager(agent_configs=configs, settings=settings())
        try:
            await manager.initialize_services()
            app, budgets = manager._runner.app, dict(manager._budgets)
            models = dict(manager.agent_models)
        finally:
            await manager.cleanup()

        assert budgets == {name: context_budget(model, manager._settings)
                           for name, model in models.items()}
        config = app.events_compaction_config
        assert (config.token_threshold, config.event_retention_size) == (
            budgets["coordinator"].trigger, 0
        )
        summarizer = config.summarizer
        assert isinstance(summarizer, AgentCliSummarizer)
        assert (summarizer.root, summarizer.budget) == ("coordinator", budgets["coordinator"])
        assert summarizer.model.ref == models["coordinator"].ref
        assert summarizer.model.max_tokens == SUMMARY_MAX_TOKENS
        assert isinstance(app.plugins[0], MainAgentPlugin)

    async def test_a_native_adk_app_gets_none_of_it(self, project, monkeypatch, tmp_path):
        serve(monkeypatch, Claude())
        manager = GoogleADKWorkflowManager(
            agent_configs=[], settings=settings(),
            adk_config_path=_native_yaml(tmp_path, "claude-sonnet-4-6"),
        )
        try:
            await manager.initialize_services()
            runner, budgets = manager._runner, manager._budgets
        finally:
            await manager.cleanup()

        assert type(runner) is Runner
        assert runner.app.events_compaction_config is None
        assert not any(isinstance(p, MainAgentPlugin) for p in runner.app.plugins)
        assert budgets == {}


def team(context: str | None) -> list[AgentConfig]:
    return [
        AgentConfig(name="coordinator", prompt=PROMPTS["coordinator"],
                    sub_agents=["researcher"], description="Leads.", include_state_tools=False),
        AgentConfig(name="researcher", prompt=PROMPTS["researcher"], description="Finds facts.",
                    context=context),
    ]


class TestSubAgents:
    @pytest.mark.parametrize("context", ["fresh", "inherit"])
    async def test_a_sub_agent_past_the_trigger_never_compacts(
        self, project, monkeypatch, context
    ):
        consulted: list[str | None] = []
        real = AgentCliSummarizer.maybe_summarize_events

        async def spy(self, *, events):
            consulted.append(running_agent())
            return await real(self, events=events)

        monkeypatch.setattr(AgentCliSummarizer, "maybe_summarize_events", spy)
        answers = serve(monkeypatch, Claude(
            coordinator=[says("Noted."), says("Noted again."),
                         calls("researcher", "toolu_c1", request="Find X."), says("X is 42.")],
            # Both of the researcher's prompts pass its trigger.
            researcher=[calls("get_plan", "toolu_r1", OVER), says("X is 42.", OVER)],
        ))
        manager = GoogleADKWorkflowManager(agent_configs=team(context), settings=settings())
        try:
            # Turn one would be worth summarizing, were the coordinator past its trigger.
            for text in TURNS[:2]:
                await turn(manager, text)
            events = await turn(manager, "Find X, please.")
            session = await stored(manager)
        finally:
            await manager.cleanup()

        # ADK asked: for the researcher, whose last prompt passed the trigger.
        assert "researcher" in consulted
        assert "summary" not in answers.asked
        assert answers.asked == ["coordinator"] * 3 + ["researcher", "researcher",
                                                        "coordinator"]  # no loop
        assert summaries_in(session) == []
        first, second = answers.bodies("researcher")
        assert lines(first)[-1] == "user: Find X."  # its brief
        assert lines(second) == [*lines(first), "assistant: called get_plan#toolu_r1",
                                 "user: result#toolu_r1"]
        assert texts(events)[-1] == "X is 42."

    @pytest.mark.parametrize("context", ["fresh", "inherit"])
    async def test_a_delegation_after_a_summary(self, project, monkeypatch, context):
        """A fresh sub-agent is sent its brief alone; an inheriting one the
        conversation as it now is, the summary first."""
        answers = serve(monkeypatch, Claude(
            coordinator=[says("Noted."), says("Noted again.", OVER),
                         calls("researcher", "toolu_c1", request="Find X."), says("X is 42.")],
            researcher=[says("X is 42.")],
        ))
        manager = GoogleADKWorkflowManager(agent_configs=team(context), settings=settings())
        try:
            for text in TURNS[:2]:
                await turn(manager, text)
            events = await turn(manager, "Find X, please.")
            session = await stored(manager)
        finally:
            await manager.cleanup()

        assert answers.asked == ["coordinator"] * 2 + ["summary", "coordinator", "researcher",
                                                        "coordinator"]
        assert len(summaries_in(session)) == 1
        (research,) = answers.bodies("researcher")
        sent = lines(research)
        if context == "fresh":
            assert sent == ["user: Find X."]
        else:
            assert sent[0] == f"user: {SUMMARY_PREFIX}\n\n{SUMMARY}"
            assert sent[-1] == "user: Find X."
        assert texts(events)[-1] == "X is 42."

    async def test_the_main_agent_still_summarizes_after_a_delegation(self, project, monkeypatch):
        """A sub-agent runs in a task of its own: what it records as the running
        agent never reaches its coordinator's next model call."""
        consulted: list[str | None] = []
        real = AgentCliSummarizer.maybe_summarize_events

        async def spy(self, *, events):
            consulted.append(running_agent())
            return await real(self, events=events)

        monkeypatch.setattr(AgentCliSummarizer, "maybe_summarize_events", spy)
        # The coordinator's delegating call passes its trigger.
        answers = serve(monkeypatch, Claude(
            coordinator=[says("Noted."), says("Noted again."),
                         calls("researcher", "toolu_c1", OVER, request="Find X."),
                         says("X is 42.")],
            researcher=[says("X is 42.")],
        ))
        manager = GoogleADKWorkflowManager(agent_configs=team("fresh"), settings=settings())
        try:
            for text in TURNS[:2]:
                await turn(manager, text)
            await turn(manager, "Find X, please.")
            session = await stored(manager)
        finally:
            await manager.cleanup()

        assert answers.asked == ["coordinator"] * 3 + ["researcher", "summary", "coordinator"]
        assert consulted[-1] == "coordinator"
        assert len(summaries_in(session)) == 1
        final = lines(answers.bodies("coordinator")[-1])
        assert final[0] == f"user: {SUMMARY_PREFIX}\n\n{SUMMARY}"
        assert final[-2:] == ["assistant: called researcher#toolu_c1", "user: result#toolu_c1"]


class TestCancelledTurns:
    """Ctrl+C while a model call runs, the summary call included."""

    async def test_ctrl_c_during_a_model_call_leaves_no_pump_and_frees_the_turn(
        self, project, monkeypatch
    ):
        started, release = asyncio.Event(), asyncio.Event()

        async def slow(answer):
            started.set()
            await release.wait()
            return answer

        script = [slow(says("A1.")), says("Again.")]
        api = FakeAPI(lambda request: script.pop(0))
        route(monkeypatch, anthropic, "AsyncAnthropic", api)
        manager = solo()
        try:
            consumer = asyncio.create_task(turn(manager, "First."))
            await asyncio.wait_for(started.wait(), 5)
            consumer.cancel()
            with pytest.raises(asyncio.CancelledError):  # not a TimeoutError: no hang
                await asyncio.wait_for(consumer, 5)
            left = pumps()
            # The turn lock is free: the next turn runs at once.
            again = await asyncio.wait_for(turn(manager, "Again."), 5)
        finally:
            release.set()
            await manager.cleanup()

        assert left == []
        assert texts(again) == ["Again."]

    async def test_ctrl_c_during_a_summary_writes_nothing_and_the_next_call_summarizes_again(
        self, project, monkeypatch
    ):
        answers = serve(monkeypatch, Claude(
            solo=[says("A1."), says("A2.", OVER), says("A3.")]
        ))
        manager = solo()
        third: list = []

        async def consume() -> None:
            async for event in manager.process(TURNS[2], "u", session_id="s"):
                third.append(event)

        try:
            for text in TURNS[:2]:
                await turn(manager, text)
            answers.hold = True
            consumer = asyncio.create_task(consume())
            await asyncio.wait_for(answers.held.wait(), 5)
            await asyncio.sleep(0.05)
            # The line is out while the summary call still runs.
            while_summarizing = [event.type for event in third]
            consumer.cancel()
            with pytest.raises(asyncio.CancelledError):  # not a TimeoutError: no hang
                await asyncio.wait_for(consumer, 5)
            left = pumps()
            after_cancel = summaries_in(await stored(manager))
            answers.release.set()
            again = await asyncio.wait_for(turn(manager, TURNS[2]), 5)
            events = await stored(manager)
        finally:
            answers.release.set()
            await manager.cleanup()

        assert while_summarizing == [EventType.CONTEXT_SUMMARIZING]
        assert left == []
        assert after_cancel == []  # nothing written
        assert [event.type for event in again][:3] == [
            EventType.CONTEXT_SUMMARIZING, EventType.LLM_USAGE, EventType.CONTEXT_TRIMMED,
        ]
        assert len(summaries_in(events)) == 1
        assert texts(again) == ["A3."]

    async def test_ctrl_c_during_a_summary_says_so_once(self, project, monkeypatch):
        answers = serve(monkeypatch, Claude(solo=[says("A1."), says("A2.", OVER)]))
        manager = solo()
        ui = opening_boxes(autospec_session())
        ui_settings = SimpleNamespace(default_user="u", verbose_thinking=False,
                                      dialog_placement="box")
        try:
            for text in TURNS[:2]:
                await turn(manager, text)
            answers.hold = True
            request = asyncio.create_task(MessageProcessor().process(
                TURNS[2], ReplayController(manager), ui, ui_settings, session_id="s"
            ))
            await asyncio.wait_for(answers.held.wait(), 5)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(request, 5)
            after_cancel = summaries_in(await stored(manager))
        finally:
            answers.release.set()
            await manager.cleanup()

        said = [c for c in ui.add_message.call_args_list if c == call("system", CANCELLED_LINE)]
        assert said == [call("system", CANCELLED_LINE)]
        assert after_cancel == []


# --- a job resumed after a summary (Review Focus 2) ------------------------------------------

#: The call id each start of the probe job was given.
STARTED: list[str] = []


@register_tool(category=ToolCategory.EXECUTION, capabilities=EXEMPT, long_running=True,
               description="Start a background job (an exempt test tool).")
def compaction_probe_start_job(command: str = "", tool_context=None) -> dict:
    """Start a job and return its id at once, as a job starter does."""
    STARTED.append(getattr(tool_context, "function_call_id", None))
    return {"success": True, "job_id": "j1", "state": "running"}


def _record(call_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        job_id="j1", tool="compaction_probe_start_job", call_id=call_id,
        call_name="compaction_probe_start_job", session_id="s", user_id="u",
        state=JobState.SUCCEEDED, exit_code=0, error=None,
    )


class TestAJobResumedAfterASummary:
    async def test_its_result_finds_its_call_and_the_turn_answers(self, project, monkeypatch):
        answers = serve(monkeypatch, Claude(solo=[
            calls("compaction_probe_start_job", "toolu_j1", command="make"), says("Started."),
            says("A2."), says("A3.", OVER), says("A4."), says("The build finished."),
        ]))
        # The durable store, which refuses a stale session (StaleSessionError).
        manager = GoogleADKWorkflowManager(
            agent_configs=[AgentConfig(name="solo", prompt=PROMPTS["solo"],
                                       tools=[compaction_probe_start_job],
                                       include_state_tools=False)],
            settings=settings(session_store="sqlite"),
        )
        try:
            for text in ["Start the job.", f"Two: {LONG}", f"Three: {LONG}", "Four?"]:
                await turn(manager, text)
            summarized = summaries_in(await stored(manager))
            resumed = [event async for event in manager.resume_with_job_result(
                _record(STARTED[-1]), result="built ok"
            )]
        finally:
            await manager.cleanup()

        # The job's call and its first answer were summarized with the turns around them.
        assert answers.asked == ["solo"] * 4 + ["summary", "solo", "solo"]
        assert len(summarized) == 1
        assert "solo called compaction_probe_start_job" in lines(answers.bodies("summary")[0])[0]
        # ADK puts the call back in front of the job's result.
        last = lines(answers.bodies("solo")[-1])
        assert last[0].startswith(f"user: {SUMMARY_PREFIX}")
        assert last[-2:] == ["assistant: called compaction_probe_start_job#toolu_j1",
                             "user: result#toolu_j1"]
        assert "built ok" in json.dumps(answers.bodies("solo")[-1]["messages"][-1])
        assert texts(resumed) == ["The build finished."]
        assert [e for e in resumed if e.type is EventType.ERROR] == []


    async def test_a_resume_turn_summarizes_too(self, project, monkeypatch):
        """A job's result resumes the main agent past its trigger: the resume
        runs the root, so its check summarizes as a user's turn would."""
        answers = serve(monkeypatch, Claude(solo=[
            calls("compaction_probe_start_job", "toolu_j1", command="make"), says("Started."),
            says("A2."), says("A3.", OVER), says("The build finished."),
        ]))
        manager = GoogleADKWorkflowManager(
            agent_configs=[AgentConfig(name="solo", prompt=PROMPTS["solo"],
                                       tools=[compaction_probe_start_job],
                                       include_state_tools=False)],
            settings=settings(),
        )
        try:
            for text in ["Start the job.", f"Two: {LONG}", f"Three: {LONG}"]:
                await turn(manager, text)
            resumed = [event async for event in manager.resume_with_job_result(
                _record(STARTED[-1]), result="built ok"
            )]
            events = await stored(manager)
        finally:
            await manager.cleanup()

        assert answers.asked == ["solo"] * 4 + ["summary", "solo"]
        assert len(summaries_in(events)) == 1
        assert texts(resumed) == ["The build finished."]


# --- a call left open by Ctrl+C mid-tool (deviation 2 on a real session) ---------------------

#: Set while ``compaction_probe_wait`` waits.
WAITING: list[asyncio.Event] = []


@register_tool(category=ToolCategory.EXECUTION, capabilities=EXEMPT,
               description="Wait until cancelled (an exempt test tool).")
async def compaction_probe_wait(seconds: int = 1) -> dict:
    """Wait, as a long tool does, until the turn is cancelled."""
    WAITING[-1].set()
    await asyncio.Event().wait()
    return {"success": True}


async def test_a_call_left_open_by_ctrl_c_does_not_stop_summaries(project, monkeypatch):
    """Ctrl+C while a tool runs leaves its call unanswered for good. ADK hands it
    over as dead, and it is summarized with the turns around it: later turns
    past the trigger are still summarized."""
    WAITING.append(asyncio.Event())
    answers = serve(monkeypatch, Claude(solo=[
        says("A0."), calls("compaction_probe_wait", "toolu_w1", seconds=1),
        says("A1."), says("A2.", OVER), says("A3."), says("A4."),
    ]))
    manager = GoogleADKWorkflowManager(
        agent_configs=[AgentConfig(name="solo", prompt=PROMPTS["solo"],
                                   tools=[compaction_probe_wait], include_state_tools=False)],
        settings=settings(),
    )
    try:
        await turn(manager, "Zero.")
        consumer = asyncio.create_task(turn(manager, "Wait, please."))
        await asyncio.wait_for(WAITING[-1].wait(), 5)
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(consumer, 5)
        answered = [texts(await turn(manager, text))
                    for text in [f"One: {LONG}", f"Two: {LONG}", "Three?", "Four?"]]
        session = await stored(manager)
    finally:
        await manager.cleanup()

    assert answered == [["A1."], ["A2."], ["A3."], ["A4."]]
    assert len(summaries_in(session)) == 1
    assert "solo called compaction_probe_wait" in lines(answers.bodies("summary")[0])[0]


# --- a model switch after a summary (Review Focus 3) -----------------------------------------


class TestAModelSwitchAfterASummary:
    async def test_the_new_model_gets_the_summary_and_its_own_budget(self, project, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "o-test-key")
        serve(monkeypatch, four_turns())
        gpt = FakeAPI(openai_response("From GPT."))
        route(monkeypatch, openai, "AsyncOpenAI", gpt)
        manager = solo()
        try:
            for text in TURNS[:3]:
                await turn(manager, text)
            before = manager._budgets["solo"]
            await manager.reinitialize(model="openai:gpt-5.5")
            after = manager._budgets["solo"]
            model = manager.agent_models["solo"]
            threshold = manager._runner.app.events_compaction_config.token_threshold
            answer = await turn(manager, "Four?")
        finally:
            await manager.cleanup()

        assert str(model.ref) == "openai:gpt-5.5"
        assert after == context_budget(model, manager._settings) != before
        assert threshold == after.trigger
        (body,) = gpt.bodies
        first = body["input"][0]
        assert first["role"] == "user"
        assert first["content"][0]["text"] == f"{SUMMARY_PREFIX}\n\n{SUMMARY}"
        assert texts(answer) == ["From GPT."]


# --- summarizing now: compact() (/compact) --------------------------------------------------

#: Three turns, each under the trigger: nothing is summarized on its own.
THREE = [f"One: {LONG}", f"Two: {LONG}", "Three?"]


def three_turns(**more) -> Claude:
    return Claude(solo=[says("A1."), says("A2."), says("A3."), says("A4.")], **more)


async def compact_now(manager, focus=None, *, events: list | None = None,
                      session_id: str = "s"):
    """``manager.compact`` as ``/compact`` runs it: what it reports merged in.

    Every event reported goes to ``events`` (a new list unless given), so a
    test still reads them when ``compact`` raises.
    """
    events = [] if events is None else events
    outcome: dict = {}

    async def source():
        outcome["result"] = await manager.compact(
            manager.session_ref(session_id, "u"), focus
        )
        if False:  # pragma: no cover - an async generator that yields nothing
            yield

    live = live_events.LiveEvents()
    with live_events.reporting(live):
        async with contextlib.aclosing(live_events.merged(source(), live)) as items:
            async for event in items:
                events.append(event)
    return outcome["result"], events


class TestCompactNow:
    async def test_the_earlier_turns_are_summarized_now_and_sent_from_then_on(
        self, project, monkeypatch
    ):
        """ADK's own compaction runs once, now: one summary appended, and the
        next request starts with it in place of the turns it covers."""
        answers = serve(monkeypatch, three_turns())
        manager = solo()
        try:
            for text in THREE:
                await turn(manager, text)
            result, _ = await compact_now(manager)
            events = await stored(manager)
            await turn(manager, "Four?")
        finally:
            await manager.cleanup()

        assert answers.asked == ["solo"] * 3 + ["summary", "solo"]
        (summary,) = summaries_in(events)
        span = summary.actions.compaction
        assert (span.start_timestamp, span.end_timestamp) == (
            events[0].timestamp, events[1].timestamp
        )
        assert (summary.author, span.compacted_content.role) == ("user", "user")
        prompt = lines(answers.bodies("summary")[0])[0]
        assert f"user: {THREE[0]}\nsolo: A1." in prompt and "Two:" not in prompt
        assert isinstance(result, CompactResult)
        assert result.messages_summarized == 2
        assert 0 < result.tokens_after < result.tokens_before
        fourth = answers.bodies("solo")[3]
        assert lines(fourth) == [
            f"user: {SUMMARY_PREFIX}\n\n{SUMMARY}", f"user: {THREE[1]}", "assistant: A2.",
            "user: Three?", "assistant: A3.", "user: Four?",
        ]
        # The summary once, folded into the first kept message (the fit
        # guard's last step): no two user messages in a row.
        assert json.dumps(fourth).count(SUMMARY_PREFIX) == 1
        assert [m["role"] for m in fourth["messages"]] == [
            "user", "assistant", "user", "assistant", "user"
        ]

    async def test_the_focus_reaches_the_summary_prompt(self, project, monkeypatch):
        answers = serve(monkeypatch, three_turns())
        manager = solo()
        try:
            for text in THREE:
                await turn(manager, text)
            await compact_now(manager, "keep the API decisions")
        finally:
            await manager.cleanup()

        prompt = lines(answers.bodies("summary")[0])[0]
        assert "Focus the summary on: keep the API decisions" in prompt
        assert prompt.index("Focus the summary on:") < prompt.index(f"user: {THREE[0]}")

    async def test_it_reports_as_a_summary_in_a_turn_does(self, project, monkeypatch):
        serve(monkeypatch, three_turns())
        manager = solo()
        try:
            for text in THREE:
                await turn(manager, text)
            result, events = await compact_now(manager)
        finally:
            await manager.cleanup()

        assert [event.type for event in events] == [
            EventType.CONTEXT_SUMMARIZING, EventType.LLM_USAGE, EventType.CONTEXT_TRIMMED
        ]
        summarizing, usage, trimmed = events
        assert summarizing.metadata["agent"] == "solo"
        assert (usage.metadata["purpose"], usage.metadata["agent"],
                usage.metadata["context_trigger"], usage.metadata["window_known"]) == (
            "summarization", "solo", 16_000, True
        )
        assert (trimmed.metadata["source"], trimmed.metadata["agent"]) == ("compact", "solo")
        assert (trimmed.metadata["messages_summarized"], trimmed.metadata["tokens_before"],
                trimmed.metadata["tokens_after"]) == (
            result.messages_summarized, result.tokens_before, result.tokens_after
        )

    async def test_a_short_conversation_has_nothing_to_summarize(self, project, monkeypatch):
        """Everything fits in the kept tail: no call, nothing appended."""
        answers = serve(monkeypatch, three_turns())
        manager = solo()
        try:
            for text in ["Hello.", "And again?"]:
                await turn(manager, text)
            result, events = await compact_now(manager)
            session = await stored(manager)
        finally:
            await manager.cleanup()

        assert (result, events) == (None, [])
        assert answers.asked == ["solo", "solo"]
        assert summaries_in(session) == []

    async def test_before_the_first_turn_there_is_nothing_to_summarize(
        self, project, monkeypatch
    ):
        answers = serve(monkeypatch, three_turns())
        manager = solo()
        try:
            result, events = await compact_now(manager)
            session = await manager._session_service.get_session(
                app_name=manager.app_name, user_id="u", session_id="s"
            )
        finally:
            await manager.cleanup()

        assert (result, events) == (None, [])
        assert session is None  # none is created
        assert answers.sent == []

    async def test_ctrl_c_during_the_summary_writes_nothing(self, project, monkeypatch):
        answers = serve(monkeypatch, three_turns())
        manager = solo()
        try:
            for text in THREE:
                await turn(manager, text)
            answers.hold = True
            seen: list = []
            compacting = asyncio.create_task(compact_now(manager, events=seen))
            await asyncio.wait_for(answers.held.wait(), 5)
            await asyncio.sleep(0.05)
            while_summarizing = [event.type for event in seen]
            compacting.cancel()
            with pytest.raises(asyncio.CancelledError):  # not a TimeoutError: no hang
                await asyncio.wait_for(compacting, 5)
            left = pumps()
            after_cancel = summaries_in(await stored(manager))
            answers.release.set()
            # The turn is free: a /compact again summarizes.
            again, _ = await asyncio.wait_for(compact_now(manager), 5)
        finally:
            answers.release.set()
            await manager.cleanup()

        assert while_summarizing == [EventType.CONTEXT_SUMMARIZING]
        assert left == []
        assert after_cancel == []
        assert again.messages_summarized == 2

    async def test_a_cancel_while_the_summary_is_stored_says_none_was_made(
        self, project, monkeypatch
    ):
        """What the summary did is reported once ADK has stored it: a cancel
        during the append (a durable store's write) shows the call and its
        usage, and no "Context summarized" for a summary never stored."""
        serve(monkeypatch, three_turns())
        manager = solo()
        appending, seen = asyncio.Event(), []
        try:
            for text in THREE:
                await turn(manager, text)
            service = manager._session_service
            append = service.append_event

            async def held(*, session, event):
                if event.actions and event.actions.compaction:
                    appending.set()
                    await asyncio.Event().wait()
                return await append(session=session, event=event)

            monkeypatch.setattr(service, "append_event", held)
            compacting = asyncio.create_task(compact_now(manager, events=seen))
            await asyncio.wait_for(appending.wait(), 5)
            await asyncio.sleep(0.05)
            compacting.cancel()
            with pytest.raises(asyncio.CancelledError):  # not a TimeoutError: no hang
                await asyncio.wait_for(compacting, 5)
            monkeypatch.setattr(service, "append_event", append)
            after = summaries_in(await stored(manager))
        finally:
            await manager.cleanup()

        assert [event.type for event in seen] == [
            EventType.CONTEXT_SUMMARIZING, EventType.LLM_USAGE
        ]
        assert after == []

    async def test_right_after_ctrl_c_mid_tool_the_cancelled_turn_is_kept(
        self, project, monkeypatch
    ):
        """A /compact right after a turn cancelled while its tool ran: ADK
        hands over the events before the open call (the newest invocation's),
        the cancelled turn is the kept tail, and the next turn answers."""
        WAITING.append(asyncio.Event())
        answers = serve(monkeypatch, Claude(solo=[
            says("A1."), says("A2."), calls("compaction_probe_wait", "toolu_w1", seconds=1),
            says("A4."),
        ]))
        manager = GoogleADKWorkflowManager(
            agent_configs=[AgentConfig(name="solo", prompt=PROMPTS["solo"],
                                       tools=[compaction_probe_wait], include_state_tools=False)],
            settings=settings(),
        )
        try:
            for text in THREE[:2]:
                await turn(manager, text)
            waiting = asyncio.create_task(turn(manager, "Wait, please."))
            await asyncio.wait_for(WAITING[-1].wait(), 5)
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(waiting, 5)
            result, _ = await compact_now(manager)
            answer = await turn(manager, "Four?")
        finally:
            await manager.cleanup()

        assert result.messages_summarized == 2  # One and A1; the cancelled turn is kept
        assert "Wait, please." not in lines(answers.bodies("summary")[0])[0]
        last = answers.bodies("solo")[-1]
        assert json.dumps(last).count(SUMMARY_PREFIX) == 1
        assert texts(answer) == ["A4."]

    async def test_it_waits_for_a_running_turn(self, project, monkeypatch):
        """Turn admission: a /compact never interleaves with a turn (or a job's
        resume), so it summarizes the turn that ran before it, all of it."""
        started, release = asyncio.Event(), asyncio.Event()

        async def slow(answer):
            started.set()
            await release.wait()
            return answer

        answers = serve(monkeypatch, Claude(solo=[says("A1."), says("A2."), slow(says("A3."))]))
        manager = solo()
        try:
            for text in THREE[:2]:
                await turn(manager, text)
            third = asyncio.create_task(turn(manager, THREE[2]))
            await asyncio.wait_for(started.wait(), 5)
            compacting = asyncio.create_task(compact_now(manager))
            await asyncio.sleep(0.1)
            waited = not compacting.done() and "summary" not in answers.asked
            release.set()
            answered = texts(await asyncio.wait_for(third, 5))
            result, _ = await asyncio.wait_for(compacting, 5)
        finally:
            release.set()
            await manager.cleanup()

        assert waited
        assert answered == ["A3."]
        assert answers.asked == ["solo"] * 3 + ["summary"]
        assert result.messages_summarized == 2

    async def test_a_summary_that_fails_is_raised_and_nothing_is_written(
        self, project, monkeypatch
    ):
        """Asked for, a summary that fails is the caller's to report: it is
        raised, with no warning of its own."""
        refused = httpx.Response(400, json={"type": "error", "error": {
            "type": "invalid_request_error", "message": "max_tokens: 8000 > 4096",
        }})
        answers = serve(monkeypatch, three_turns(summaries=[refused]))
        manager = solo()
        seen: list = []
        try:
            for text in THREE:
                await turn(manager, text)
            with pytest.raises(anthropic.BadRequestError, match="max_tokens: 8000 > 4096"):
                await compact_now(manager, events=seen)
            session = await stored(manager)
        finally:
            await manager.cleanup()

        assert answers.asked == ["solo"] * 3 + ["summary"]
        assert [event.type for event in seen] == [EventType.CONTEXT_SUMMARIZING]
        assert summaries_in(session) == []

    async def test_a_second_compact_replaces_the_first(self, project, monkeypatch):
        answers = serve(monkeypatch, Claude(
            solo=[says(f"A{n}.") for n in range(1, 7)],
            summaries=[says("First summary.", 100), says("Second summary.", 100)],
        ))
        manager = solo()
        try:
            for text in THREE:
                await turn(manager, text)
            await compact_now(manager)
            for text in [f"Four: {LONG}", "Five?"]:
                await turn(manager, text)
            second, _ = await compact_now(manager)
            events = await stored(manager)
            await turn(manager, "Six?")
        finally:
            await manager.cleanup()

        first_span, second_span = (e.actions.compaction for e in summaries_in(events))
        assert second_span.start_timestamp == first_span.start_timestamp
        assert second_span.end_timestamp > first_span.end_timestamp
        prompt = lines(answers.bodies("summary")[1])[0]
        assert "Summary of the conversation before this point:\nFirst summary." in prompt
        # The first summary and the four messages it was followed by.
        assert second.messages_summarized == 5
        last = answers.bodies("solo")[-1]
        assert lines(last)[0] == f"user: {SUMMARY_PREFIX}\n\nSecond summary."
        assert "First summary." not in json.dumps(last)
        assert json.dumps(last).count(SUMMARY_PREFIX) == 1

    async def test_a_native_adk_app_has_none(self, project, monkeypatch, tmp_path):
        serve(monkeypatch, Claude())
        manager = GoogleADKWorkflowManager(
            agent_configs=[], settings=settings(),
            adk_config_path=_native_yaml(tmp_path, "claude-sonnet-4-6"),
        )
        try:
            supports = manager.supports_compact
            with pytest.raises(NotImplementedError, match="native ADK"):
                await manager.compact(manager.session_ref("s", "u"))
        finally:
            await manager.cleanup()

        assert supports is False
