"""Each ADK event is its run's: the branch a sub-agent's work runs on names it.

ADK runs a sub-agent its parent delegated to on the branch
``<agent>@<call id>``, under its parent's (``middle@c1.leaf@m1`` two levels
down); the root, and an agent the root transfers to, on none. The event
processor names each event's run from it, gives a delegation's call and
result an ``AGENT_RUN`` (started; completed, failed or interrupted), and
records it; ``MainAgentPlugin`` makes the run the active one as the agent
starts, and its parent's the active one again as it ends. ADK runs a
delegated sub-agent inside its parent's call (``run_node``, in that call's
own task), so what the agent sets as it starts is still set when it returns:
the call's after-tool callbacks are the parent's, and see the parent's run.
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
from agentic_cli.workflow.adk.branches import (  # noqa: E402
    branch_run,
    delegations,
    parent_branch_run,
)
from agentic_cli.workflow.adk.compaction import MainAgentPlugin, running_agent  # noqa: E402
from agentic_cli.workflow.adk.event_processor import ADKEventProcessor  # noqa: E402
from agentic_cli.workflow.adk.interrupted_calls_plugin import (  # noqa: E402
    ABORTED,
    InterruptedCallsPlugin,
)
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.agents import (  # noqa: E402
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_INTERRUPTED,
    RUN_STARTED,
    AgentRuns,
)
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from agentic_cli.workflow.interrupts import INTERRUPTED_BY_USER  # noqa: E402
from agentic_cli.workflow.sessions import RunRef, get_active_run, root_run  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI, anthropic_message  # noqa: E402
from tests.workflow.test_manager_models import route  # noqa: E402

RESEARCHER = RunRef(
    agent="researcher", instance="researcher-1", run="researcher-1#1", parent_run=None, depth=1
)
MIDDLE = RunRef(agent="middle", instance="middle-1", run="middle-1#1", parent_run=None, depth=1)
LEAF = RunRef(agent="leaf", instance="leaf-1", run="leaf-1#1", parent_run="middle-1#1", depth=2)


def _text(author: str, text: str = "X is 42.", *, branch: str | None = None) -> Event:
    return Event(
        author=author, invocation_id="i1", branch=branch,
        content=types.Content(role="model", parts=[types.Part.from_text(text=text)]),
    )


def _calls(author: str, *made: tuple[str, str, dict], branch: str | None = None) -> Event:
    return Event(
        author=author, invocation_id="i1", branch=branch,
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
    return ADKEventProcessor(
        model="m", root="coordinator", delegates={"researcher"}, **kwargs
    )


async def _process(processor: ADKEventProcessor, event: Event) -> list:
    return [e async for e in processor.process_event(event, "s1")]


async def _delegated(processor: ADKEventProcessor, request: str = "Find X.") -> None:
    await _process(processor, _calls("coordinator", ("researcher", "toolu_c1", {"request": request})))


class TestBranches:
    def test_a_branch_is_its_delegations_outermost_first(self):
        assert delegations("middle@toolu_c1.leaf@toolu_m1") == [
            ("middle", "toolu_c1"), ("leaf", "toolu_m1"),
        ]

    def test_no_branch_is_no_delegation(self):
        assert delegations(None) == []
        assert delegations("") == []

    def test_a_segment_without_a_call_is_no_delegation(self):
        assert delegations("node.researcher@toolu_c1") == [("researcher", "toolu_c1")]

    def test_a_branch_names_its_run_under_its_parents(self):
        assert branch_run(AgentRuns(), None, "middle@c1.leaf@m1") == LEAF
        assert branch_run(AgentRuns(), None, None) is None

    def test_a_branchs_parent_run_made_its_delegation(self):
        runs = AgentRuns()

        assert parent_branch_run(runs, None, "middle@c1.leaf@m1") == MIDDLE
        assert parent_branch_run(runs, None, "middle@c1") is None
        assert parent_branch_run(runs, None, None) is None


class TestEveryEventIsItsRuns:
    async def test_the_roots_events_are_its_own(self):
        (text,) = await _process(_processor(), _text("coordinator"))

        assert text.run == root_run("coordinator")

    async def test_a_sub_agents_events_are_its_runs(self):
        (text,) = await _process(_processor(), _text("researcher", branch="researcher@toolu_c1"))

        assert text.run == RESEARCHER

    async def test_a_nested_run_is_named_under_its_parents(self):
        (text,) = await _process(_processor(), _text("leaf", branch="middle@c1.leaf@m1"))

        assert text.run == LEAF

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


class TestDelegationRuns:
    async def test_a_delegation_starts_a_run_after_its_call(self):
        events = await _process(_processor(), _calls(
            "coordinator", ("researcher", "toolu_c1", {"request": "Find X.\nThen report."})
        ))

        assert [e.type for e in events] == [EventType.TOOL_CALL, EventType.AGENT_RUN]
        call, started = events
        assert call.run == root_run("coordinator")
        assert started.run == RESEARCHER
        assert started.content == "researcher-1 started"
        assert started.metadata == {
            "status": RUN_STARTED, "agent": "researcher", "partial": False,
            "call_id": "toolu_c1", "title": "Find X.",
        }

    async def test_its_result_ends_it(self):
        processor = _processor()
        await _delegated(processor)

        events = await _process(processor, _results(
            "coordinator", ("researcher", "toolu_c1", {"result": "X is 42."})
        ))

        assert [e.type for e in events] == [EventType.TOOL_RESULT, EventType.AGENT_RUN]
        ended = events[1]
        assert ended.run == RESEARCHER
        assert (ended.metadata["status"], ended.metadata["result"]) == (RUN_COMPLETED, "X is 42.")
        (record,) = processor.runs.records(None)
        assert (record.status, record.title, record.result) == (RUN_COMPLETED, "Find X.", "X is 42.")

    async def test_a_long_result_is_cut(self):
        processor = _processor()
        await _delegated(processor)

        (_, ended) = await _process(processor, _results(
            "coordinator", ("researcher", "toolu_c1", {"result": "z" * 500})
        ))

        assert ended.metadata["result"] == "z" * 200

    async def test_a_sub_agent_that_failed_ends_its_run_failed(self):
        processor = _processor()
        await _delegated(processor)

        (_, ended) = await _process(processor, _results(
            "coordinator",
            ("researcher", "toolu_c1", {"result": "Error running sub-agent: 529 overloaded"}),
        ))

        assert ended.metadata["status"] == RUN_FAILED
        assert (ended.metadata["error"], ended.metadata["result"]) == ("529 overloaded", "")

    async def test_a_stopped_turns_seal_ends_it_interrupted(self):
        processor = _processor()
        await _delegated(processor)

        (_, ended) = await _process(processor, _results(
            "coordinator",
            ("researcher", "toolu_c1", {"success": False, "error": INTERRUPTED_BY_USER}),
            error_code=ABORTED,
        ))

        assert ended.metadata["status"] == RUN_INTERRUPTED

    async def test_another_tool_starts_no_run(self):
        events = await _process(_processor(), _calls("coordinator", ("lookup", "toolu_1", {})))

        assert [e.type for e in events] == [EventType.TOOL_CALL]


class TestTheActiveRun:
    async def test_an_agent_on_a_delegations_branch_runs_as_that_run(self):
        plugin = MainAgentPlugin(AgentRuns())

        async def start() -> RunRef | None:
            await plugin.before_agent_callback(
                agent=SimpleNamespace(name="leaf"),
                callback_context=SimpleNamespace(branch="middle@c1.leaf@m1"),
            )
            return get_active_run()

        # ADK runs each agent in a task of its own: the run is that task's.
        assert await asyncio.create_task(start()) == LEAF
        assert get_active_run() is None

    async def test_the_root_runs_as_no_run(self):
        plugin = MainAgentPlugin(AgentRuns())

        async def start() -> RunRef | None:
            await plugin.before_agent_callback(
                agent=SimpleNamespace(name="coordinator"),
                callback_context=SimpleNamespace(branch=None),
            )
            return get_active_run()

        assert await asyncio.create_task(start()) is None

    async def test_without_the_runs_it_records_the_agent_alone(self):
        async def start() -> tuple[str | None, RunRef | None]:
            await MainAgentPlugin().before_agent_callback(
                agent=SimpleNamespace(name="researcher"), callback_context=None
            )
            return running_agent(), get_active_run()

        assert await asyncio.create_task(start()) == ("researcher", None)

    async def test_as_an_agent_ends_its_parents_run_is_active_again(self):
        plugin = MainAgentPlugin(AgentRuns())
        leaf, on_leaf = SimpleNamespace(name="leaf"), SimpleNamespace(branch="middle@c1.leaf@m1")

        async def run() -> tuple[RunRef | None, RunRef | None]:
            await plugin.before_agent_callback(agent=leaf, callback_context=on_leaf)
            during = get_active_run()
            await plugin.after_agent_callback(agent=leaf, callback_context=on_leaf)
            return during, get_active_run()

        assert await asyncio.create_task(run()) == (LEAF, MIDDLE)

    async def test_as_an_agent_fails_its_parents_run_is_active_again(self):
        plugin = MainAgentPlugin(AgentRuns())
        leaf, on_leaf = SimpleNamespace(name="leaf"), SimpleNamespace(branch="middle@c1.leaf@m1")

        async def run() -> RunRef | None:
            await plugin.before_agent_callback(agent=leaf, callback_context=on_leaf)
            await plugin.on_agent_error_callback(
                agent=leaf, callback_context=on_leaf, error=RuntimeError("overloaded")
            )
            return get_active_run()

        assert await asyncio.create_task(run()) == MIDDLE


# -- after a delegation, its parent's run: two turns on Claude, through FakeAPI --

PROMPTS = {"coordinator": "You coordinate.", "middle": "You plan.", "leaf": "You look up."}


def _says(text: str) -> dict:
    return anthropic_message({"type": "text", "text": text})


def _delegates(agent: str, call_id: str, request: str) -> dict:
    message = anthropic_message(
        {"type": "tool_use", "id": call_id, "name": agent, "input": {"request": request}}
    )
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

        assert seen == [("leaf", MIDDLE), ("middle", None)]

    async def test_after_a_sub_agent_failed(self, project, monkeypatch):
        refused = httpx.Response(400, json={
            "type": "error", "error": {"type": "invalid_request_error", "message": "bad request"},
        })

        seen = await self._after_tool_runs(monkeypatch, _Team(
            coordinator=[_delegates("middle", "toolu_c1", "Plan X."), _says("Planned.")],
            middle=[_delegates("leaf", "toolu_m1", "Look X up."), _says("The leaf failed.")],
            leaf=[refused],
        ))

        assert seen == [("leaf", MIDDLE), ("middle", None)]
