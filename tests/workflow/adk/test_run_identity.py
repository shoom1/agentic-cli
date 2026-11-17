"""Each ADK event is its run's: the conversation's own, or the sub-agent run whose runner wrote it.

The conversation's processor names each event of its runner the root's
(``root_run``), an agent the root transfers to at depth 0 (a native ADK YAML
app's), and the user's message no one's. A sub-agent run runs on its type's
own runner, on its child session, read by a processor made for that run
(``ADKEventProcessor(run=)``): every event is the run's, its text is
activity and its usage the sub-agent's, it reports no ``ERROR`` of its own,
and it keeps what ends the run (its last answer with text, that answer's
outcome, its failure) for the manager. The ``AGENT_RUN`` events are the
framework's (``agents/delegate.py``). A ``task`` call's result is read back
from its JSON text and shown from its envelope.

A delegation's call is its parent's: its after-tool callbacks (every
plugin's) see the parent's run, whether the sub-agent answered or failed;
the run's own calls ran in a task of their own.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx
import pytest

pytest.importorskip("google.adk")

from google.adk.events import Event  # noqa: E402
from google.genai import types  # noqa: E402

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.workflow.adk.compaction import MainAgentPlugin, running_agent  # noqa: E402
from agentic_cli.workflow.adk.event_processor import ADKEventProcessor  # noqa: E402
from agentic_cli.workflow.adk.interrupted_calls_plugin import InterruptedCallsPlugin  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.adk.subagent_outcome_plugin import OUTCOME_KEY  # noqa: E402
from agentic_cli.workflow.agents.envelope import envelope  # noqa: E402
from agentic_cli.workflow.agents.outcome import CUT  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.context import OVERFLOW_ERROR_CODE  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from agentic_cli.workflow.sessions import RunRef, get_active_run, root_run  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI, anthropic_message  # noqa: E402
from tests.workflow.test_manager_models import route  # noqa: E402

RESEARCHER = RunRef(
    agent="researcher", instance="researcher-1", run="researcher-1#1", parent_run=None, depth=1
)
MIDDLE = RunRef(agent="middle", instance="middle-1", run="middle-1#1", parent_run=None, depth=1)


def _text(author: str, text: str = "X is 42.", **fields) -> Event:
    return Event(
        author=author, invocation_id="i1",
        content=types.Content(role="model", parts=[types.Part.from_text(text=text)]),
        **fields,
    )


def _calls(author: str, *made: tuple[str, str, dict]) -> Event:
    return Event(
        author=author, invocation_id="i1",
        content=types.Content(role="model", parts=[
            types.Part(function_call=types.FunctionCall(id=call_id, name=name, args=args))
            for name, call_id, args in made
        ]),
    )


def _results(author: str, *made: tuple[str, str, dict], error_code: str | None = None) -> Event:
    return Event(
        author=author, invocation_id="i1", error_code=error_code,
        content=types.Content(role="user", parts=[
            types.Part(function_response=types.FunctionResponse(id=call_id, name=name, response=response))
            for name, call_id, response in made
        ]),
    )


def _processor(**kwargs) -> ADKEventProcessor:
    return ADKEventProcessor(model="m", root="coordinator", **kwargs)


async def _process(processor: ADKEventProcessor, event: Event) -> list:
    return [e async for e in processor.process_event(event, "s1")]


class TestEveryEventIsItsRuns:
    async def test_the_roots_events_are_its_own(self):
        (text,) = await _process(_processor(), _text("coordinator"))

        assert text.run == root_run("coordinator")

    async def test_an_agent_the_root_transfers_to_runs_at_depth_0(self):
        processor = ADKEventProcessor(
            model="m", root="coordinator", reply_agents={"coordinator", "helper"}
        )

        (text,) = await _process(processor, _text("helper"))

        assert text.run == RunRef(
            agent="helper", instance="helper", run=None, parent_run=None, depth=0
        )

    async def test_the_users_own_message_is_no_ones(self):
        event = Event(
            author="user", invocation_id="i1",
            content=types.Content(role="user", parts=[types.Part.from_text(text="hi")]),
        )

        (text,) = await _process(_processor(), event)

        assert text.run is None


class TestCallIds:
    async def test_a_call_and_its_result_name_the_call(self):
        processor = _processor()

        (call,) = await _process(processor, _calls("coordinator", ("lookup", "toolu_1", {"q": "x"})))
        (result,) = await _process(
            processor, _results("coordinator", ("lookup", "toolu_1", {"success": True}))
        )

        assert call.metadata["tool_call_id"] == "toolu_1"
        assert result.metadata["tool_call_id"] == "toolu_1"


class TestATaskCallsResult:
    async def test_an_envelope_is_shown_from_its_result(self):
        found = envelope(RESEARCHER, "completed", result="X is 42.\nMore.")

        (result,) = await _process(_processor(), _results(
            "coordinator", ("task", "toolu_c1", {"result": json.dumps(found)})
        ))

        assert (result.content, result.metadata["success"]) == ("X is 42.", True)
        assert result.metadata["result"] == found

    async def test_a_failed_one_is_a_failed_result(self):
        found = envelope(RESEARCHER, "failed", error="overloaded")

        (result,) = await _process(_processor(), _results(
            "coordinator", ("task", "toolu_c1", {"result": json.dumps(found)})
        ))

        assert result.content == "Failed: overloaded"
        assert (result.metadata["success"], result.metadata["error"]) == (False, "overloaded")

    async def test_a_brief_that_started_no_run_is_an_error_result(self):
        refused = {"success": False, "error": "task needs agent_type: one of researcher."}

        (result,) = await _process(_processor(), _results(
            "coordinator", ("task", "toolu_c1", {"result": json.dumps(refused)})
        ))

        assert result.metadata["result"] == refused
        assert (result.metadata["success"], result.metadata["error"]) == (
            False, refused["error"]
        )

    async def test_another_tools_result_is_read_as_it_is(self):
        (result,) = await _process(_processor(), _results(
            "coordinator", ("lookup", "toolu_c1", {"result": json.dumps({"success": False})})
        ))

        assert result.metadata["result"] == {"result": json.dumps({"success": False})}
        assert result.metadata["success"] is True


class TestARunsProcessor:
    """``run=``: every event is that run's, and what ends the run is kept."""

    async def test_every_event_is_the_runs(self):
        processor = _processor(run=RESEARCHER)

        events = [
            *await _process(processor, _calls("researcher", ("lookup", "toolu_1", {"q": "x"}))),
            *await _process(processor, _results("researcher", ("lookup", "toolu_1", {"v": 1}))),
            *await _process(processor, _text("researcher")),
        ]

        assert [(e.type, e.run) for e in events] == [
            (EventType.TOOL_CALL, RESEARCHER), (EventType.TOOL_RESULT, RESEARCHER),
            (EventType.TEXT, RESEARCHER),
        ]
        assert (events[-1].metadata["agent"], events[-1].metadata["subagent"]) == (
            "researcher", True
        )

    async def test_its_usage_is_the_sub_agents(self):
        event = _text("researcher", usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=50, candidates_token_count=5,
        ))

        (usage, _) = await _process(_processor(run=RESEARCHER), event)

        assert (usage.run, usage.metadata["agent"], usage.metadata["purpose"]) == (
            RESEARCHER, "researcher", "subagent"
        )

    async def test_its_last_answer_with_text_is_kept(self):
        processor = _processor(run=RESEARCHER)

        await _process(processor, _text("researcher", "X is 42."))
        # Anthropic may end with an empty end_turn after its final tool call.
        await _process(processor, Event(
            author="researcher", invocation_id="i1", error_code="MODEL_RETURNED_NO_CONTENT",
        ))

        assert (processor.answer, processor.outcome, processor.failure) == ("X is 42.", None, None)

    async def test_an_answers_text_counts_beside_its_tool_call(self):
        """As Deep Agents' own ``task`` reads a run: its last text, whatever
        else the reply held."""
        processor = _processor(run=RESEARCHER)
        event = _calls("researcher", ("lookup", "toolu_1", {}))
        event.content.parts.insert(0, types.Part.from_text(text="X is 42. "))

        await _process(processor, event)

        assert processor.answer == "X is 42."

    async def test_its_outcome_mark_is_kept(self):
        processor = _processor(run=RESEARCHER)

        await _process(processor, _text("researcher", "X is", custom_metadata={OUTCOME_KEY: CUT}))

        assert (processor.answer, processor.outcome) == ("X is", CUT)

    async def test_an_overflow_is_its_failure_and_no_error_is_reported(self):
        processor = _processor(run=RESEARCHER)
        event = Event(
            author="researcher", invocation_id="i1", error_code=OVERFLOW_ERROR_CODE,
            error_message="researcher: the request no longer fits m's context window.",
        )

        assert await _process(processor, event) == []
        assert processor.failure == (
            "researcher: the request no longer fits m's context window.", OVERFLOW_ERROR_CODE
        )

    async def test_a_reply_that_is_no_answer_is_its_failure(self):
        processor = _processor(run=RESEARCHER, models={"researcher": "anthropic:claude"})
        event = Event(
            author="researcher", invocation_id="i1", error_code="MALFORMED_FUNCTION_CALL",
            error_message="bad call",
        )

        assert await _process(processor, event) == []
        assert processor.failure == (
            "anthropic:claude returned no answer: bad call", "MALFORMED_FUNCTION_CALL"
        )

    async def test_it_reports_no_task_progress(self):
        event = _results("researcher", ("save_tasks", "toolu_1", {"success": True}))
        event.actions.state_delta["tasks"] = [
            {"id": "1", "description": "Look", "status": "in_progress"}
        ]

        events = await _process(_processor(run=RESEARCHER), event)

        assert [e.type for e in events] == [EventType.TOOL_RESULT]


class TestTheRunningAgent:
    async def test_main_agent_plugin_records_the_agent_alone(self):
        async def start() -> tuple[str | None, RunRef | None]:
            await MainAgentPlugin().before_agent_callback(
                agent=SimpleNamespace(name="researcher"), callback_context=None
            )
            return running_agent(), get_active_run()

        assert await asyncio.create_task(start()) == ("researcher", None)


# -- after a delegation, its parent's run: two turns on Claude, through FakeAPI --

PROMPTS = {"coordinator": "You coordinate.", "middle": "You plan.", "leaf": "You look up."}


def _says(text: str) -> dict:
    return anthropic_message({"type": "text", "text": text})


def _delegates(agent: str, call_id: str, objective: str) -> dict:
    message = anthropic_message({
        "type": "tool_use", "id": call_id, "name": "task",
        "input": {"agent_type": agent, "title": objective, "objective": objective},
    })
    message["stop_reason"] = "tool_use"
    return message


class _Team:
    """Claude for the team: each agent answers from its own list, told apart by its prompt."""

    def __init__(self, **answers: list) -> None:
        self.answers = {name: list(said) for name, said in answers.items()}

    def __call__(self, request) -> object:
        body = json.loads(request.content)
        system = body.get("system") or ""
        if isinstance(system, list):
            system = "\n".join(block.get("text", "") for block in system)
        name = next(name for name, prompt in PROMPTS.items() if system.startswith(prompt))
        return self.answers[name].pop(0)


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


class TestAfterADelegation:
    """A delegation's call is its parent's: its after-tool callbacks (every
    plugin's) see the parent's run, whether the sub-agent answered or failed."""

    async def _after_tool_runs(self, monkeypatch, team: _Team) -> list[tuple[str, RunRef | None]]:
        """Each tool call's (name, the active run) as its after-tool callbacks see them."""
        seen: list[tuple[str, RunRef | None]] = []
        original = InterruptedCallsPlugin.after_tool_callback

        async def recording(plugin, **kwargs):
            seen.append((kwargs["tool"].name, get_active_run()))
            return await original(plugin, **kwargs)

        monkeypatch.setattr(InterruptedCallsPlugin, "after_tool_callback", recording)
        route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(team))
        manager = GoogleADKWorkflowManager(
            agent_configs=[
                AgentConfig(name="coordinator", prompt=PROMPTS["coordinator"],
                            sub_agents=["middle"], include_state_tools=False),
                AgentConfig(name="middle", prompt=PROMPTS["middle"], sub_agents=["leaf"],
                            description="Plans.", include_state_tools=False),
                AgentConfig(name="leaf", prompt=PROMPTS["leaf"], description="Looks up.",
                            include_state_tools=False),
            ],
            settings=BaseSettings(session_store="memory", default_model="claude-sonnet-4-6"),
        )
        try:
            [event async for event in manager.process("What is X?", "u", session_id="s")]
        finally:
            await manager.cleanup()
        return seen

    async def test_after_a_sub_agent_answered(self, project, monkeypatch):
        seen = await self._after_tool_runs(monkeypatch, _Team(
            coordinator=[_delegates("middle", "toolu_c1", "Plan X."), _says("Planned.")],
            middle=[_delegates("leaf", "toolu_m1", "Look X up."), _says("X is 42.")],
            leaf=[_says("42.")],
        ))

        assert seen == [("task", MIDDLE), ("task", None)]

    async def test_after_a_sub_agent_failed(self, project, monkeypatch):
        refused = httpx.Response(400, json={
            "type": "error", "error": {"type": "invalid_request_error", "message": "bad request"},
        })

        seen = await self._after_tool_runs(monkeypatch, _Team(
            coordinator=[_delegates("middle", "toolu_c1", "Plan X."), _says("Planned.")],
            middle=[_delegates("leaf", "toolu_m1", "Look X up."), _says("The leaf failed.")],
            leaf=[refused],
        ))

        assert seen == [("task", MIDDLE), ("task", None)]
