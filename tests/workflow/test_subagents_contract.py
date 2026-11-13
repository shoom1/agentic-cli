"""One sub-agent contract, on both backends.

Each scenario runs a scripted turn on ADK (Claude, through ``FakeAPI``) and on
Deep Agents (``ScriptedChat``) and expects the same events, run records and
results: the two backends behave alike. A scenario's model answers are
written once, in backend-neutral steps (``says``; ``calls`` with ``delegate``
and ``use``), which the harness turns into each backend's: on ADK a delegation is a call of the tool named after
the sub-agent (``request``), on Deep Agents a ``task`` call
(``description``, ``subagent_type``). Each agent's prompt is
``You are <name>.``, by which ADK's fake API tells the agents apart.

Offline: no provider is reached and no catalog lists (``tests/conftest.py``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
import pytest

from agentic_cli.config import BaseSettings
from agentic_cli.tools.registry import register_tool
from agentic_cli.workflow.agents import (
    RUN_COMPLETED,
    RUN_INTERRUPTED,
    RUN_STARTED,
    RunRecord,
)
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.events import EventType, UserInputRequest, WorkflowEvent
from agentic_cli.workflow.permissions import EXEMPT
from agentic_cli.workflow.permissions.prompt import DENY_CHOICE
from agentic_cli.workflow.sessions import (
    RunRef,
    SessionRef,
    get_active_run,
    get_active_tool_call,
    root_run,
)
from tests.workflow.adk.fakes import FakeAPI, anthropic_message
from tests.workflow.test_manager_models import route

BACKENDS = ("adk", "deepagents")
USER = "u"
SESSION = "s"


# -- backend-neutral model answers ---------------------------------------------


@dataclass(frozen=True)
class Call:
    """One call a model answer makes: a delegation to ``agent``, or a call of ``tool``."""

    call_id: str
    agent: str | None = None
    brief: str = ""
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)


def delegate(agent: str, brief: str, call_id: str) -> Call:
    """A delegation of ``brief`` to the sub-agent ``agent``."""
    return Call(call_id=call_id, agent=agent, brief=brief)


def use(tool: str, call_id: str, **args: Any) -> Call:
    """A call of ``tool`` with ``args``."""
    return Call(call_id=call_id, tool=tool, args=args)


@dataclass(frozen=True)
class Step:
    """One model answer: its text, its calls, and why it stopped.

    ``stop`` is a text answer's stop reason (Claude's ``end_turn``); an answer
    with calls stops for them. A scenario that needs another kind of answer
    adds a keyword here, with a default, and handles it in both
    ``_claude_answer`` and ``_scripted_answer``.
    """

    text: str = ""
    calls: tuple[Call, ...] = ()
    stop: str = "end_turn"


def says(text: str, *, stop: str = "end_turn") -> Step:
    """An answer of ``text`` alone."""
    return Step(text=text, stop=stop)


def calls(*made: Call, text: str = "") -> Step:
    """An answer that makes ``made`` (in one response: they run in parallel)."""
    return Step(text=text, calls=made)


def prompt_of(name: str) -> str:
    """The prompt of the agent ``name``: how ADK's fake API knows whose request it is."""
    return f"You are {name}."


def agent(
    name: str,
    *,
    sub_agents: tuple[str, ...] = (),
    tools: tuple[Any, ...] = (),
    **fields: Any,
) -> AgentConfig:
    """An agent of a scenario's team, without state tools (its requests hold only its own)."""
    return AgentConfig(
        name=name,
        prompt=prompt_of(name),
        sub_agents=list(sub_agents),
        tools=list(tools),
        description=fields.pop("description", f"The {name}."),
        include_state_tools=False,
        **fields,
    )


def _claude_answer(step: Step) -> dict[str, Any]:
    """``step`` as Claude's Messages API answers, for ADK."""
    blocks: list[dict[str, Any]] = []
    if step.text:
        blocks.append({"type": "text", "text": step.text})
    for call in step.calls:
        if call.agent is not None:
            blocks.append({"type": "tool_use", "id": call.call_id, "name": call.agent,
                           "input": {"request": call.brief}})
        else:
            blocks.append({"type": "tool_use", "id": call.call_id, "name": call.tool,
                           "input": dict(call.args)})
    answer = anthropic_message()
    answer["content"] = blocks
    answer["stop_reason"] = "tool_use" if step.calls else step.stop
    return answer


def _scripted_answer(step: Step) -> Any:
    """``step`` as a ``ScriptedChat`` answers, for Deep Agents."""
    from tests.workflow.deepagents.fakes import ai, task

    made = [
        task(call.agent, call.brief, call.call_id) if call.agent is not None
        else (call.tool, dict(call.args), call.call_id)
        for call in step.calls
    ]
    return ai(step.text, calls=made)


class _ClaudeTeam:
    """Claude for a team, behind ``FakeAPI``: each agent answers from its own
    script, told apart by its prompt; every request is kept under its agent."""

    def __init__(self, scripts: dict[str, list[Step]]) -> None:
        self.scripts = {name: list(steps) for name, steps in scripts.items()}
        self.sent: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, request: Any) -> dict[str, Any]:
        body = json.loads(request.content)
        system = body.get("system") or ""
        if isinstance(system, list):
            system = "\n".join(block.get("text", "") for block in system)
        name = next(name for name in self.scripts if system.startswith(prompt_of(name)))
        self.sent.append((name, body))
        script = self.scripts[name]
        return _claude_answer(script.pop(0) if script else says(f"[{name} has no scripted answer]"))


def _claude_results(body: dict[str, Any]) -> dict[str, str]:
    """The tool results a Claude request holds, by call id, as text."""
    results: dict[str, str] = {}
    for message in body["messages"]:
        content = message["content"]
        if isinstance(content, str):
            continue
        for block in content:
            if block.get("type") != "tool_result":
                continue
            held = block.get("content")
            if isinstance(held, list):
                held = "".join(part.get("text", "") for part in held if isinstance(part, dict))
            results[block["tool_use_id"]] = held if isinstance(held, str) else json.dumps(held)
    return results


# -- the crew: a team's manager on one backend -----------------------------------


@dataclass
class Crew:
    """A team's manager on one backend, and what its models were sent."""

    backend: str
    manager: Any
    conversation: SessionRef
    _results_read: Callable[[str, int], dict[str, str]]
    _requests: Callable[[str], int]

    async def turn(self, text: str = "Go.") -> list[WorkflowEvent]:
        """One turn of the conversation, run to its end; its events."""
        return [event async for event in self.manager.process(text, USER, session_id=SESSION)]

    def start(self, text: str = "Go.") -> "asyncio.Task[list[WorkflowEvent]]":
        """A turn in a task of its own, to stop it as Ctrl+C does."""
        return asyncio.create_task(self.turn(text))

    @staticmethod
    async def stop(turn: "asyncio.Task[Any]") -> None:
        """Stop ``turn`` as Ctrl+C does, and wait until it has unwound."""
        turn.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await turn

    def records(self) -> list[RunRecord]:
        """The conversation's run records, in the order the runs started."""
        return self.manager.agent_runs.records(self.conversation)

    def answer_with(self, answer: Callable[[UserInputRequest], Awaitable[str]]) -> None:
        """Answer every question and permission prompt with ``answer(request)``."""
        self.manager.set_input_callback(answer)

    def results_read(self, agent: str, call: int = -1) -> dict[str, str]:
        """The tool results ``agent``'s model request ``call`` held, by call id, as text."""
        return self._results_read(agent, call)

    def requests(self, agent: str) -> int:
        """How many model requests ``agent`` made."""
        return self._requests(agent)


def _settings() -> BaseSettings:
    return BaseSettings(session_store="memory", default_model="claude-sonnet-4-6")


def _adk_crew(team: list[AgentConfig], scripts: dict[str, list[Step]], monkeypatch) -> Crew:
    from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager

    claude = _ClaudeTeam(scripts)
    route(monkeypatch, anthropic, "AsyncAnthropic", FakeAPI(claude))
    manager = GoogleADKWorkflowManager(agent_configs=team, settings=_settings())

    def results_read(name: str, call: int) -> dict[str, str]:
        return _claude_results([body for sent, body in claude.sent if sent == name][call])

    def requests(name: str) -> int:
        return sum(1 for sent, _ in claude.sent if sent == name)

    return Crew("adk", manager, manager.session_ref(SESSION, USER), results_read, requests)


def _deepagents_crew(team: list[AgentConfig], scripts: dict[str, list[Step]], monkeypatch) -> Crew:
    from langchain_core.messages import ToolMessage

    from agentic_cli.workflow.deepagents import manager as deepagents_manager
    from tests.workflow.deepagents.fakes import ScriptedModels

    models = ScriptedModels(**{
        name: [_scripted_answer(step) for step in steps] for name, steps in scripts.items()
    })
    monkeypatch.setattr(deepagents_manager, "build_chat_model", models)
    manager = deepagents_manager.DeepAgentsWorkflowManager(agent_configs=team, settings=_settings())

    def results_read(name: str, call: int) -> dict[str, str]:
        return {
            message.tool_call_id: message.text
            for message in models.chats[name].conversation(call)
            if isinstance(message, ToolMessage)
        }

    def requests(name: str) -> int:
        return len(models.chats[name].requests)

    return Crew("deepagents", manager, manager.session_ref(SESSION, USER), results_read, requests)


async def until(condition: Callable[[], bool], *, within: float = 10.0) -> None:
    """Wait until ``condition()`` holds, looking every 10 ms."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + within
    while not condition():
        if loop.time() > deadline:
            raise AssertionError(f"not within {within} s")
        await asyncio.sleep(0.01)


def runs_of(events: list[WorkflowEvent], status: str | None = None) -> list[WorkflowEvent]:
    """The turn's ``AGENT_RUN`` events, those of ``status`` when given."""
    return [
        event for event in events
        if event.type is EventType.AGENT_RUN
        and (status is None or event.metadata["status"] == status)
    ]


# -- the tools the scenarios call ------------------------------------------------

#: What each ``contract_whoami`` call saw: the active run, and its own call id.
SEEN: list[tuple[RunRef | None, str | None]] = []
#: ``BLOCKING["started"]`` is set when a ``contract_block`` call starts (made per test).
BLOCKING: dict[str, asyncio.Event] = {}


@register_tool(capabilities=EXEMPT, description="Look a thing up (a contract-test tool).")
def contract_lookup(q: str = "") -> dict:
    """Look ``q`` up."""
    return {"success": True, "value": 42}


@register_tool(capabilities=EXEMPT, description="Say which run calls (a contract-test tool).")
async def contract_whoami() -> dict:
    """Record the run, and the call, this call runs as."""
    call = get_active_tool_call()
    SEEN.append((get_active_run(), call.call_id if call is not None else None))
    return {"success": True}


@register_tool(capabilities=EXEMPT, description="Wait until stopped (a contract-test tool).")
async def contract_block() -> dict:
    """Wait, as a slow tool does, until the turn is stopped."""
    BLOCKING["started"].set()
    await asyncio.Event().wait()
    return {"success": True}


# -- fixtures --------------------------------------------------------------------


@pytest.fixture(params=BACKENDS)
def backend(request) -> str:
    if request.param == "deepagents":
        pytest.importorskip("deepagents")
    return request.param


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


@pytest.fixture
async def make_crew(backend, project, monkeypatch):
    """``make_crew(team, **scripts)``: ``team``'s manager on ``backend``, each
    agent answering from its script (an agent without one has none); every
    manager made is cleaned up after the test."""
    SEEN.clear()
    BLOCKING["started"] = asyncio.Event()
    made: list[Crew] = []

    def make(team: list[AgentConfig], **scripts: list[Step]) -> Crew:
        complete = {config.name: list(scripts.get(config.name, ())) for config in team}
        build = _adk_crew if backend == "adk" else _deepagents_crew
        crew = build(team, complete, monkeypatch)
        made.append(crew)
        return crew

    yield make
    for crew in made:
        await crew.manager.cleanup()


# -- the team most scenarios use ----------------------------------------------------

RESEARCHER = RunRef(
    agent="researcher", instance="researcher-1", run="researcher-1#1", parent_run=None, depth=1
)
COORDINATOR = root_run("coordinator")


def team() -> list[AgentConfig]:
    """A coordinator, and the researcher it delegates to."""
    return [
        agent("coordinator", sub_agents=("researcher",), tools=(contract_whoami,)),
        agent("researcher", tools=(contract_lookup, contract_whoami, contract_block, "write_file")),
    ]


# -- run identity: each run named, recorded, and active in its own calls ------------------


class TestRunIdentity:
    async def test_a_delegation_is_a_run_from_its_call_to_its_result(self, make_crew):
        crew = make_crew(
            team(),
            coordinator=[calls(delegate("researcher", "Find X.\nUse what you know.", "c1")),
                         says("Done.")],
            researcher=[says("X is 42.")],
        )

        events = await crew.turn()

        started, ended = runs_of(events)
        assert started.run == ended.run == RESEARCHER
        assert started.metadata == {
            "status": RUN_STARTED, "agent": "researcher", "partial": False,
            "call_id": "c1", "title": "Find X.",
        }
        assert (ended.metadata["status"], ended.metadata["call_id"], ended.metadata["result"]) == (
            RUN_COMPLETED, "c1", "X is 42.",
        )
        call = next(i for i, e in enumerate(events)
                    if e.type is EventType.TOOL_CALL and e.metadata.get("tool_call_id") == "c1")
        result = next(i for i, e in enumerate(events)
                      if e.type is EventType.TOOL_RESULT and e.metadata.get("tool_call_id") == "c1")
        assert events.index(started) == call + 1
        assert events.index(ended) == result + 1
        (record,) = crew.records()
        assert (record.ref, record.call_id, record.title, record.status, record.result) == (
            RESEARCHER, "c1", "Find X.", RUN_COMPLETED, "X is 42.",
        )
        assert record.ended is not None and record.ended >= record.started

    async def test_every_event_an_agent_wrote_names_its_run(self, make_crew):
        crew = make_crew(
            team(),
            coordinator=[calls(delegate("researcher", "Find X.", "c1")), says("Done.")],
            researcher=[calls(use("contract_lookup", "r1", q="X")), says("X is 42.")],
        )

        events = await crew.turn()

        written = [e for e in events if e.type in (
            EventType.TEXT, EventType.TOOL_CALL, EventType.TOOL_RESULT, EventType.LLM_USAGE
        )]
        assert {e.run for e in written} == {COORDINATOR, RESEARCHER}
        assert [e.run for e in written if e.metadata.get("tool_name") == "contract_lookup"] == [
            RESEARCHER, RESEARCHER,
        ]
        assert [e.run for e in written if e.type is EventType.TEXT] == [RESEARCHER, COORDINATOR]
        assert all(
            e.metadata.get("tool_call_id")
            for e in written if e.type in (EventType.TOOL_CALL, EventType.TOOL_RESULT)
        )

    async def test_parallel_runs_of_one_agent_are_told_apart(self, make_crew):
        crew = make_crew(
            team(),
            coordinator=[calls(delegate("researcher", "Find A.", "c1"),
                               delegate("researcher", "Find B.", "c2")),
                         says("Both done.")],
            researcher=[says("One answer."), says("The other answer.")],
        )

        events = await crew.turn()

        started = runs_of(events, RUN_STARTED)
        assert sorted(e.run.run for e in started) == ["researcher-1#1", "researcher-2#1"]
        assert sorted(e.metadata["call_id"] for e in started) == ["c1", "c2"]
        said = {e.run.run: e.content for e in events
                if e.type is EventType.TEXT and e.run != COORDINATOR}
        assert sorted(said.values()) == ["One answer.", "The other answer."]
        # Each run ends with its own answer, not the other's.
        assert {e.run.run: e.metadata["result"] for e in runs_of(events, RUN_COMPLETED)} == said
        assert [r.status for r in crew.records()] == [RUN_COMPLETED, RUN_COMPLETED]

    async def test_a_nested_run_is_its_parents_child(self, make_crew):
        crew = make_crew(
            [
                agent("coordinator", sub_agents=("middle",)),
                agent("middle", sub_agents=("leaf",)),
                agent("leaf", tools=(contract_whoami,)),
            ],
            coordinator=[calls(delegate("middle", "Plan X.", "c1")), says("Planned.")],
            middle=[calls(delegate("leaf", "Look X up.", "m1")), says("X is 42, says the leaf.")],
            leaf=[calls(use("contract_whoami", "l1")), says("42.")],
        )

        events = await crew.turn()

        middle = RunRef(agent="middle", instance="middle-1", run="middle-1#1",
                        parent_run=None, depth=1)
        leaf = RunRef(agent="leaf", instance="leaf-1", run="leaf-1#1",
                      parent_run="middle-1#1", depth=2)
        assert [(e.run, e.metadata["status"]) for e in runs_of(events)] == [
            (middle, RUN_STARTED), (leaf, RUN_STARTED),
            (leaf, RUN_COMPLETED), (middle, RUN_COMPLETED),
        ]
        assert SEEN == [(leaf, "l1")]
        assert [e.run for e in events if e.type is EventType.TEXT] == [
            leaf, middle, root_run("coordinator"),
        ]

    async def test_a_call_to_another_agents_sub_agent_is_no_run(self, make_crew):
        """Only a delegation to one of the caller's own sub-agents is a run.

        The coordinator delegates to ``leaf``, the middle's sub-agent: no
        sub-agent runs (ADK answers an unknown tool, Deep Agents an unknown
        type), so the call is an ordinary tool call, with no ``AGENT_RUN``,
        no record and no active run, and ``leaf``'s numbering is untouched
        when the middle runs the real one.
        """
        crew = make_crew(
            [
                agent("coordinator", sub_agents=("middle",), tools=(contract_whoami,)),
                agent("middle", sub_agents=("leaf",)),
                agent("leaf", tools=(contract_whoami,)),
            ],
            coordinator=[calls(delegate("leaf", "Look X up.", "c1")),
                         calls(use("contract_whoami", "c2")),
                         calls(delegate("middle", "Plan X.", "c3")),
                         says("Planned.")],
            middle=[calls(delegate("leaf", "Look X up.", "m1")), says("X is 42, says the leaf.")],
            leaf=[calls(use("contract_whoami", "l1")), says("42.")],
        )

        events = await crew.turn()

        middle = RunRef(agent="middle", instance="middle-1", run="middle-1#1",
                        parent_run=None, depth=1)
        leaf = RunRef(agent="leaf", instance="leaf-1", run="leaf-1#1",
                      parent_run="middle-1#1", depth=2)
        assert [(e.run, e.metadata["status"]) for e in runs_of(events)] == [
            (middle, RUN_STARTED), (leaf, RUN_STARTED),
            (leaf, RUN_COMPLETED), (middle, RUN_COMPLETED),
        ]
        assert [r.ref for r in crew.records()] == [middle, leaf]
        assert SEEN == [(None, "c2"), (leaf, "l1")]
        answered = [e for e in events if e.type is EventType.TOOL_RESULT
                    and e.metadata.get("tool_call_id") == "c1"]
        assert [e.run for e in answered] == [COORDINATOR]

    async def test_a_sub_agents_tools_and_prompts_run_as_its_run(self, make_crew, project):
        crew = make_crew(
            team(),
            coordinator=[calls(use("contract_whoami", "c0")),
                         calls(delegate("researcher", "Write a note.", "c1")),
                         says("It could not.")],
            researcher=[calls(use("contract_whoami", "r1")),
                        calls(use("write_file", "r2", path="note.txt", content="hi")),
                        says("Writing was denied.")],
        )
        asked: list[tuple[str, RunRef | None]] = []

        async def deny(request: UserInputRequest) -> str:
            asked.append((request.tool_name, get_active_run()))
            return DENY_CHOICE

        crew.answer_with(deny)
        await crew.turn()

        assert SEEN == [(None, "c0"), (RESEARCHER, "r1")]
        assert asked == [("write_file", RESEARCHER)]
        assert not (project / "note.txt").exists()
        assert get_active_run() is None

    async def test_a_stopped_turn_leaves_its_open_run_interrupted(self, make_crew):
        crew = make_crew(
            team(),
            coordinator=[calls(delegate("researcher", "Wait.", "c1"))],
            researcher=[calls(use("contract_block", "r1"))],
        )

        turn = crew.start()
        try:
            await asyncio.wait_for(BLOCKING["started"].wait(), 10)
            await until(lambda: [r.status for r in crew.records()] == [RUN_STARTED])
        finally:
            # Stopped whatever happened: a turn left running holds the turn
            # lock, and the fixture's cleanup would wait for it forever.
            await crew.stop(turn)

        (record,) = crew.records()
        assert (record.ref, record.status) == (RESEARCHER, RUN_INTERRUPTED)
        assert record.ended is not None
