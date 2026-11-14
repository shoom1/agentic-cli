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
import functools
import json
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
import httpx
import pytest

from agentic_cli.config import BaseSettings
from agentic_cli.constants import TOOL_SUMMARY_MAX_LENGTH, truncate
from agentic_cli.tools.jobs.backends import TERMINAL_STATES
from agentic_cli.tools.registry import register_tool
from agentic_cli.workflow.agents import (
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_INTERRUPTED,
    RUN_REFUSED,
    RUN_STARTED,
    RunRecord,
)
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.events import (
    EventType,
    UserInputRequest,
    UserInputUnavailable,
    WorkflowEvent,
)
from agentic_cli.workflow.interrupts import INTERRUPTED_BY_USER, NOT_RUN
from agentic_cli.workflow.permissions import EXEMPT, Capability
from agentic_cli.workflow.permissions.prompt import DENY_CHOICE
from agentic_cli.workflow.sessions import (
    RunRef,
    SessionRef,
    get_active_run,
    get_active_tool_call,
    reset_active_turn,
    root_run,
    set_active_turn,
)
from agentic_cli.workflow.service_registry import (
    JOB_MANAGER,
    PERMISSION_ENGINE,
    WORKFLOW,
    get_service,
)
from agentic_cli.workflow.usage import (
    PURPOSE_MEMORY_EXTRACTION,
    PURPOSE_SUMMARIZATION,
    PURPOSE_TOOL,
    UsageLedger,
    UsageTotals,
    side_call,
)
from tests.workflow.adk.fakes import FakeAPI, anthropic_message
from tests.workflow.compacting import compact_now
from tests.workflow.test_manager_models import route

BACKENDS = ("adk", "deepagents")
USER = "u"
SESSION = "s"
#: The script that answers side calls (``generate_simple``): a model request
#: that is no agent's (ADK: no agent's prompt; Deep Agents: the default model's).
SIDE_CALLS = "side_calls"
#: The script that answers summary calls, whichever agent's conversation they
#: summarize: a request whose prompt holds the summary prompt's ``<messages>``
#: (ADK: a request that is no agent's; Deep Agents: one an agent's model gets).
SUMMARIES = "summaries"
#: What a summary prompt holds (``workflow/summary_prompt.py``, both backends).
SUMMARY_MARKER = "<messages>"


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
    with calls stops for them. ``usage`` is what the call reports,
    ``(input, output)`` tokens; each fake's own when None (they differ). A
    scenario that needs another kind of answer adds a keyword here, with a
    default, and handles it in both ``_claude_answer`` and ``_scripted_answer``.
    """

    text: str = ""
    calls: tuple[Call, ...] = ()
    stop: str = "end_turn"
    #: The provider refuses the request (a 400) with this message.
    fails: str | None = None
    usage: tuple[int, int] | None = None


def says(text: str, *, stop: str = "end_turn", usage: tuple[int, int] | None = None) -> Step:
    """An answer of ``text`` alone."""
    return Step(text=text, stop=stop, usage=usage)


def fails(message: str) -> Step:
    """The provider refusing the request with ``message``, as a 400 does."""
    return Step(fails=message)


def calls(*made: Call, text: str = "", usage: tuple[int, int] | None = None) -> Step:
    """An answer that makes ``made`` (in one response: they run in parallel)."""
    return Step(text=text, calls=made, usage=usage)


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


def _provider_error(message: str) -> dict[str, Any]:
    """The body of the 400 Claude's API answers a request it refuses with."""
    return {"type": "error", "error": {"type": "invalid_request_error", "message": message}}


def _claude_answer(step: Step) -> dict[str, Any] | httpx.Response:
    """``step`` as Claude's Messages API answers, for ADK."""
    if step.fails is not None:
        return httpx.Response(400, json=_provider_error(step.fails))
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
    if step.usage is not None:
        answer["usage"] = {"input_tokens": step.usage[0], "output_tokens": step.usage[1]}
    return answer


def _scripted_answer(step: Step) -> Any:
    """``step`` as a ``ScriptedChat`` answers, for Deep Agents: its stop
    reason in ``response_metadata``, where ChatAnthropic reports Claude's."""
    from tests.workflow.deepagents.fakes import ai, task

    if step.fails is not None:
        # What the Anthropic client raises for that 400.
        body = _provider_error(step.fails)
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        return anthropic.BadRequestError(
            f"Error code: 400 - {body}", response=httpx.Response(400, request=request), body=body
        )

    made = [
        task(call.agent, call.brief, call.call_id) if call.agent is not None
        else (call.tool, dict(call.args), call.call_id)
        for call in step.calls
    ]
    usage = None
    if step.usage is not None:
        tokens_in, tokens_out = step.usage
        usage = {"input_tokens": tokens_in, "output_tokens": tokens_out,
                 "total_tokens": tokens_in + tokens_out}
    answer = ai(step.text, calls=made, usage=usage)
    stop = "tool_use" if step.calls else step.stop
    return answer.model_copy(update={"response_metadata": {"stop_reason": stop}})


class _ClaudeTeam:
    """Claude for a team, behind ``FakeAPI``: each agent answers from its own
    script, told apart by its prompt, and a request that is no agent's (a
    side call) from ``SIDE_CALLS``'s; every request is kept under its agent."""

    def __init__(self, scripts: dict[str, list[Step]]) -> None:
        self.scripts = {name: list(steps) for name, steps in scripts.items()}
        self.sent: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, request: Any) -> dict[str, Any] | httpx.Response:
        body = json.loads(request.content)
        system = body.get("system") or ""
        if isinstance(system, list):
            system = "\n".join(block.get("text", "") for block in system)
        name = next(
            (name for name in self.scripts if system.startswith(prompt_of(name))), SIDE_CALLS
        )
        if name == SIDE_CALLS and SUMMARY_MARKER in _last_text(body):
            name = SUMMARIES
        self.sent.append((name, body))
        script = self.scripts[name]
        return _claude_answer(script.pop(0) if script else says(f"[{name} has no scripted answer]"))


def _last_text(body: dict[str, Any]) -> str:
    """The text of the last message a Claude request holds."""
    content = body["messages"][-1]["content"] if body.get("messages") else ""
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict))


@functools.cache
def _team_chat() -> type:
    """A ``ScriptedChat`` whose summary requests the team's summaries script answers.

    Deep Agents summarizes an agent's conversation on that agent's model; the
    scenarios script the summaries apart, as ADK sends them apart.
    """
    from tests.workflow.deepagents.fakes import ScriptedChat

    class TeamChat(ScriptedChat):
        #: The chat that answers summary requests (Any: not copied by pydantic).
        summaries: Any = None

        def _answer(self, messages):
            if self.summaries is not None and SUMMARY_MARKER in (messages[-1].text or ""):
                return self.summaries._answer(messages)
            return super()._answer(messages)

    return TeamChat


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


def _claude_result_ids(body: dict[str, Any]) -> list[str]:
    """The call ids of the tool results a Claude request holds, one per result, in order."""
    return [
        block["tool_use_id"]
        for message in body["messages"] if not isinstance(message["content"], str)
        for block in message["content"] if block.get("type") == "tool_result"
    ]


# -- the crew: a team's manager on one backend -----------------------------------


@dataclass
class Crew:
    """A team's manager on one backend, and what its models were sent."""

    backend: str
    manager: Any
    conversation: SessionRef
    _results_read: Callable[[str, int], dict[str, str]]
    _requests: Callable[[str], int]
    _result_ids: Callable[[str, int], list[str]] | None = None
    _answers: Callable[[str, str], Awaitable[list[str]]] | None = None
    _last_message: Callable[[str, int], str] | None = None

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

    def ledger(self) -> UsageLedger:
        """The conversation's token usage (``manager.usage_of``)."""
        return self.manager.usage_of(self.conversation)

    def last_message(self, agent: str, call: int = -1) -> str:
        """The text of the last message ``agent``'s model request ``call`` held."""
        assert self._last_message is not None
        return self._last_message(agent, call)

    def result_ids(self, agent: str, call: int = -1) -> list[str]:
        """The call ids of the tool results ``agent``'s model request ``call``
        held, one per result, in order: a call answered twice is there twice."""
        assert self._result_ids is not None
        return self._result_ids(agent, call)

    async def answers(self, agent: str, call_id: str) -> list[str]:
        """The answers ``agent``'s call ``call_id`` has after a stopped turn,
        one per answer, as text: a call answered twice is there twice.

        ADK keeps every agent's conversation, so these are the answers its
        session holds. Deep Agents keeps only the root's (a sub-agent is
        compiled without a checkpointer), so a call's answer there is the
        one its stack gives it from the record (``OpenCalls``, under the run
        of ``agent``); read it before the next turn ends, which clears it.
        """
        assert self._answers is not None
        return await self._answers(agent, call_id)

    def recorded(self, call_id: str, run: str | None = None) -> bool:
        """Whether the record holds ``call_id`` of ``run`` (None: the root's
        own call) as finished: its result is kept for a stop to answer with."""
        token = set_active_turn(self.conversation)
        try:
            return self.manager._open_calls.recorded(call_id, run=run) is not None
        finally:
            reset_active_turn(token)


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

    def result_ids(name: str, call: int) -> list[str]:
        return _claude_result_ids([body for sent, body in claude.sent if sent == name][call])

    async def answers(name: str, call_id: str) -> list[str]:
        session = await manager._session_service.get_session(
            app_name=manager.session_ref(SESSION, USER).app_name,
            user_id=USER, session_id=SESSION,
        )
        return [
            json.dumps(response.response)
            for event in session.events if event.author == name
            for response in event.get_function_responses() if response.id == call_id
        ]

    def last_message(name: str, call: int) -> str:
        body = [body for sent, body in claude.sent if sent == name][call]
        content = body["messages"][-1]["content"]
        return content if isinstance(content, str) else "".join(
            block.get("text", "") for block in content if block.get("type") == "text"
        )

    return Crew(
        "adk", manager, manager.session_ref(SESSION, USER), results_read, requests, result_ids,
        answers, last_message,
    )


def _deepagents_crew(team: list[AgentConfig], scripts: dict[str, list[Step]], monkeypatch) -> Crew:
    from langchain_core.messages import ToolMessage

    from agentic_cli.workflow.deepagents import manager as deepagents_manager
    from tests.workflow.deepagents.fakes import ScriptedModels

    models = ScriptedModels(**{
        name: [_scripted_answer(step) for step in steps] for name, steps in scripts.items()
    })
    # A side call runs on the default model, which serves no agent.
    models.chats[None] = models.chats.pop(SIDE_CALLS)
    # A summary runs on the agent's model: its own script answers it.
    summaries = models.chats.pop(SUMMARIES)
    for name, chat in list(models.chats.items()):
        if name is not None:
            models.chats[name] = _team_chat()(
                label=chat.label, script=chat.script, summaries=summaries
            )
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

    def result_ids(name: str, call: int) -> list[str]:
        return [
            message.tool_call_id
            for message in models.chats[name].conversation(call)
            if isinstance(message, ToolMessage)
        ]

    async def answers(name: str, call_id: str) -> list[str]:
        conversation = manager.session_ref(SESSION, USER)
        runs = [r.ref.run for r in manager.agent_runs.records(conversation) if r.ref.agent == name]
        token = set_active_turn(conversation)
        try:
            answer = manager._open_calls.answer(
                call_id, by_user=True, run=runs[-1] if runs else None
            )
        finally:
            reset_active_turn(token)
        message = getattr(answer, "message", None)  # a finished call's ToolMessage
        return [message.text if message is not None else json.dumps(answer)]

    def last_message(name: str, call: int) -> str:
        return models.chats[name].conversation(call)[-1].text

    return Crew(
        "deepagents", manager, manager.session_ref(SESSION, USER), results_read, requests,
        result_ids, answers, last_message,
    )


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


@register_tool(capabilities=EXEMPT, description="Summarize a text (a contract-test tool).")
async def contract_summarize(text: str = "") -> dict:
    """Summarize ``text`` with the workflow's model: a side call inside the run."""
    summary = await get_service(WORKFLOW).summarize(text, f"Summarize this: {text}")
    return {"success": True, "summary": summary}


@register_tool(capabilities=EXEMPT, description="Wait until stopped (a contract-test tool).")
async def contract_block() -> dict:
    """Wait, as a slow tool does, until the turn is stopped."""
    BLOCKING["started"].set()
    await asyncio.Event().wait()
    return {"success": True}


#: What ``contract_leave_behind`` did: ``LEFT["during"]`` is the answer to the
#: question it asked in its turn, ``LEFT["go"]`` lets its task go on,
#: ``LEFT["task"]`` is the task.
LEFT: dict[str, Any] = {}


@register_tool(capabilities=EXEMPT, description="Leave a task behind (a contract-test tool).")
async def contract_leave_behind() -> dict:
    """Ask the user once, then start a task that outlives the turn; once let go,
    that task asks the user and the gate."""
    workflow = get_service(WORKFLOW)
    engine = get_service(PERMISSION_ENGINE)
    LEFT["during"] = await workflow.request_user_input(UserInputRequest(
        request_id="during", tool_name="contract_leave_behind", prompt="Here?",
    ))
    go = LEFT["go"] = asyncio.Event()

    async def later() -> dict[str, Any]:
        await go.wait()
        try:
            await workflow.request_user_input(UserInputRequest(
                request_id="late", tool_name="contract_leave_behind", prompt="Still there?",
            ))
            question = "answered"
        except UserInputUnavailable:
            question = "unavailable"
        checked = await engine.check(
            "write_file", [Capability("filesystem.write", target_arg="path")],
            {"path": "late.txt"},
        )
        return {"question": question, "permission": (checked.allowed, checked.reason)}

    LEFT["task"] = asyncio.create_task(later())
    return {"success": True}


@register_tool(capabilities=EXEMPT, description="Ask the user (a contract-test tool).")
async def contract_ask() -> dict:
    """Ask the user a question, its request named after this call; return the answer."""
    call = get_active_tool_call()
    answer = await get_service(WORKFLOW).request_user_input(UserInputRequest(
        request_id=call.call_id if call is not None else "ask",
        tool_name="contract_ask", prompt="Which one?",
    ))
    return {"success": True, "answer": answer}


def _built() -> str:
    return "built ok"


@register_tool(
    capabilities=EXEMPT, requires=JOB_MANAGER,
    description="Start a background job (a contract-test tool).",
)
def contract_start_job() -> dict:
    """Start a job that ends at once, the agent told when it has."""
    record = get_service(JOB_MANAGER).submit(
        tool="contract_start_job", backend="inprocess", spec={"target": _built},
        resume_on_complete=True,
    )
    return {"success": True, "job_id": record.job_id, "state": record.state.value}


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
        complete[SIDE_CALLS] = list(scripts.get(SIDE_CALLS, ()))
        complete[SUMMARIES] = list(scripts.get(SUMMARIES, ()))
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
        agent("researcher", tools=(contract_lookup, contract_whoami, contract_block,
                                   contract_summarize, "write_file")),
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


# -- each call is its own run's: a call of the same id never stands for another's ---------


class TestACallIsItsOwnRuns:
    """A provider that numbers its call ids per agent (``call_0``, ...) gives a
    sub-agent's call the id of its parent's delegation: neither call ever
    stands for the other, while the turn runs or after Ctrl+C."""

    async def test_a_delegation_ends_at_its_own_result(self, make_crew):
        crew = make_crew(
            team(),
            coordinator=[calls(delegate("researcher", "Find X.", "c1")), says("Done.")],
            researcher=[calls(use("contract_lookup", "c1", q="X")), says("X is 42.")],
        )

        events = await crew.turn()

        results = [e for e in events
                   if e.type is EventType.TOOL_RESULT and e.metadata["tool_call_id"] == "c1"]
        assert [e.run for e in results] == [RESEARCHER, COORDINATOR]
        assert results[0].metadata["tool_name"] == "contract_lookup"
        (ended,) = runs_of(events, RUN_COMPLETED)
        assert (ended.run, ended.metadata["result"]) == (RESEARCHER, "X is 42.")
        assert events.index(ended) == events.index(results[1]) + 1
        assert crew.results_read("coordinator")["c1"] == "X is 42."

    async def test_after_ctrl_c_a_delegation_is_answered_as_interrupted(self, make_crew):
        crew = make_crew(
            team(),
            coordinator=[calls(delegate("researcher", "Find X.", "c1")), says("Starting over.")],
            researcher=[calls(use("contract_lookup", "c1", q="X")),
                        calls(use("contract_block", "c2"))],
        )

        turn = crew.start()
        try:
            await asyncio.wait_for(BLOCKING["started"].wait(), 10)
        finally:
            await crew.stop(turn)
        await crew.turn("Never mind.")

        answered = crew.results_read("coordinator")["c1"]
        assert INTERRUPTED_BY_USER in answered
        assert "42" not in answered
        (record,) = crew.records()
        assert (record.ref, record.status, record.result) == (RESEARCHER, RUN_INTERRUPTED, "")

    async def test_after_ctrl_c_at_a_sub_agents_prompt_its_parents_call_was_not_asking(
        self, make_crew
    ):
        crew = make_crew(
            team(),
            coordinator=[calls(delegate("researcher", "Write a note.", "c1")),
                         says("Starting over.")],
            researcher=[calls(use("write_file", "c1", path="note.txt", content="hi"))],
        )
        asking = asyncio.Event()

        async def never_answered(request: UserInputRequest) -> str:
            asking.set()
            await asyncio.Event().wait()
            return DENY_CHOICE

        crew.answer_with(never_answered)
        turn = crew.start()
        try:
            await asyncio.wait_for(asking.wait(), 10)
        finally:
            await crew.stop(turn)
        await crew.turn("Never mind.")

        answered = crew.results_read("coordinator")["c1"]
        assert INTERRUPTED_BY_USER in answered
        assert NOT_RUN not in answered

    async def test_after_ctrl_c_two_levels_down_every_open_run_is_interrupted_and_answered_once(
        self, make_crew
    ):
        crew = make_crew(
            [
                agent("coordinator", sub_agents=("middle",)),
                agent("middle", sub_agents=("leaf",)),
                agent("leaf", tools=(contract_lookup, contract_block)),
            ],
            coordinator=[calls(delegate("middle", "Plan X.", "c1")), says("Starting over.")],
            middle=[calls(delegate("leaf", "Look X up.", "c1"))],
            leaf=[calls(use("contract_lookup", "c1", q="X")), calls(use("contract_block", "c2"))],
        )

        turn = crew.start()
        try:
            await asyncio.wait_for(BLOCKING["started"].wait(), 10)
        finally:
            await crew.stop(turn)
        middles = await crew.answers("middle", "c1")
        await crew.turn("Never mind.")

        assert [(r.ref.run, r.status) for r in crew.records()] == [
            ("middle-1#1", RUN_INTERRUPTED), ("leaf-1#1", RUN_INTERRUPTED),
        ]
        assert crew.result_ids("coordinator") == ["c1"]
        answered = crew.results_read("coordinator")["c1"]
        assert INTERRUPTED_BY_USER in answered and "42" not in answered
        # The middle's delegation is answered once, as the user's, as the
        # turn stops, and the next turn answers it no more.
        (middle,) = middles
        assert INTERRUPTED_BY_USER in middle and "42" not in middle
        assert await crew.answers("middle", "c1") == middles
        for name in ("coordinator", "middle", "leaf"):
            for call in range(crew.requests(name)):
                ids = crew.result_ids(name, call)
                assert len(ids) == len(set(ids)), (name, call, ids)

    async def test_after_ctrl_c_a_finished_sibling_of_a_delegation_keeps_its_result(
        self, make_crew
    ):
        """The coordinator's own call ``c2`` finished beside its delegation
        ``c1``, whose sub-agent then used both ids: each is answered once,
        with its own answer."""
        crew = make_crew(
            team(),
            coordinator=[calls(delegate("researcher", "Find X.", "c1"),
                               use("contract_whoami", "c2")),
                         says("Starting over.")],
            researcher=[calls(use("contract_lookup", "c1", q="X")),
                        calls(use("contract_lookup", "c2", q="Y")),
                        calls(use("contract_block", "c3"))],
        )

        turn = crew.start()
        try:
            await asyncio.wait_for(BLOCKING["started"].wait(), 10)
            await until(lambda: crew.recorded("c2"))
        finally:
            await crew.stop(turn)
        await crew.turn("Never mind.")

        assert sorted(crew.result_ids("coordinator")) == ["c1", "c2"]
        answered = crew.results_read("coordinator")
        assert json.loads(answered["c2"]) == {"success": True}
        assert INTERRUPTED_BY_USER in answered["c1"] and "42" not in answered["c1"]

    async def test_after_ctrl_c_a_finished_delegation_keeps_its_result(self, make_crew):
        """Two delegations run side by side, each sub-agent using the other's
        call id: the one that finished keeps its sub-agent's answer, the
        other is answered as the user's, each once."""
        crew = make_crew(
            [
                agent("coordinator", sub_agents=("researcher", "waiter")),
                agent("researcher", tools=(contract_lookup,)),
                agent("waiter", tools=(contract_block,)),
            ],
            coordinator=[calls(delegate("researcher", "Find X.", "c1"),
                               delegate("waiter", "Wait.", "c2")),
                         says("Starting over.")],
            researcher=[calls(use("contract_lookup", "c2", q="X")), says("X is 42.")],
            waiter=[calls(use("contract_block", "c1"))],
        )

        turn = crew.start()
        try:
            await asyncio.wait_for(BLOCKING["started"].wait(), 10)
            await until(lambda: crew.recorded("c1"))
        finally:
            await crew.stop(turn)
        await crew.turn("Never mind.")

        assert sorted(crew.result_ids("coordinator")) == ["c1", "c2"]
        answered = crew.results_read("coordinator")
        assert answered["c1"] == "X is 42."
        assert INTERRUPTED_BY_USER in answered["c2"]


# -- a sub-agent's refusal and cut-off answer: said so to its parent, and in its run --------

#: What a parent reads for a sub-agent whose model refused: Deep Agents 0.7.34's notice.
REFUSAL = (
    "The subagent stopped because a model refused the request. "
    "Retrying the same request is unlikely to succeed; rephrase it or continue without it."
)
#: What a parent reads for a sub-agent whose answer "X is" was cut at the output limit.
CUT_OFF = "X is\n\n[cut off at the output limit]"


class TestRefusalAndCutOff:
    async def test_a_refusal_reaches_the_parent_as_the_notice_and_the_run_is_refused(
        self, make_crew
    ):
        crew = make_crew(
            team(),
            coordinator=[calls(delegate("researcher", "Find X.", "c1")), says("It refused.")],
            researcher=[says("", stop="refusal")],
        )

        events = await crew.turn()

        assert crew.results_read("coordinator")["c1"] == REFUSAL
        (ended,) = runs_of(events, RUN_REFUSED)
        assert ended.run == RESEARCHER
        assert (ended.metadata["result"], ended.metadata["error"], ended.metadata["partial"]) == (
            REFUSAL, "a model refused the request", False,
        )
        assert [e for e in events if e.type is EventType.ERROR] == []
        (record,) = crew.records()
        assert (record.status, record.result, record.error) == (
            RUN_REFUSED, REFUSAL, "a model refused the request",
        )

    async def test_a_cut_answer_is_kept_marked_partial_and_warned_once(self, make_crew):
        crew = make_crew(
            team(),
            coordinator=[calls(delegate("researcher", "Find X.", "c1")), says("Partly done.")],
            researcher=[says("X is", stop="max_tokens")],
        )

        events = await crew.turn()

        assert crew.results_read("coordinator")["c1"] == CUT_OFF
        (ended,) = runs_of(events, RUN_COMPLETED)
        assert (ended.run, ended.metadata["result"], ended.metadata["partial"]) == (
            RESEARCHER, CUT_OFF, True,
        )
        warnings = [e for e in events if e.type is EventType.ERROR]
        assert [
            (w.content, w.metadata["recoverable"], w.metadata["error_code"],
             w.metadata["agent"], w.run)
            for w in warnings
        ] == [(
            "researcher-1's answer was cut off at the output limit", True, "MAX_TOKENS",
            "researcher", RESEARCHER,
        )]
        assert events.index(warnings[0]) == events.index(ended) + 1
        (record,) = crew.records()
        assert (record.status, record.partial, record.result) == (RUN_COMPLETED, True, CUT_OFF)

    async def test_parallel_runs_each_end_by_their_own_last_answer(self, make_crew):
        crew = make_crew(
            team(),
            coordinator=[calls(delegate("researcher", "Find A.", "c1"),
                               delegate("researcher", "Find B.", "c2")),
                         says("One of them answered.")],
            researcher=[says("", stop="refusal"), says("B is 7.")],
        )

        events = await crew.turn()

        read = crew.results_read("coordinator")
        assert sorted(read[call] for call in ("c1", "c2")) == sorted([REFUSAL, "B is 7."])
        ended = {
            e.metadata["call_id"]: (e.metadata["status"], e.metadata["result"])
            for e in runs_of(events) if e.metadata["status"] != RUN_STARTED
        }
        assert ended == {
            call: (RUN_REFUSED if read[call] == REFUSAL else RUN_COMPLETED, read[call])
            for call in ("c1", "c2")
        }
        assert [e for e in events if e.type is EventType.ERROR] == []

    async def test_an_answer_that_ended_well_is_not_touched(self, make_crew):
        crew = make_crew(
            team(),
            coordinator=[calls(delegate("researcher", "Find X.", "c1")), says("Done.")],
            researcher=[says("X is 42.", stop="end_turn")],
        )

        events = await crew.turn()

        assert crew.results_read("coordinator")["c1"] == "X is 42."
        (ended,) = runs_of(events, RUN_COMPLETED)
        assert ended.metadata["partial"] is False
        assert [e for e in events if e.type is EventType.ERROR] == []

    async def test_the_roots_own_cut_answer_is_left_as_its_model_gave_it(self, make_crew):
        """Only a sub-agent's answer is said so: the root's, after its
        delegation returned, reaches the user as its model gave it."""
        crew = make_crew(
            team(),
            coordinator=[calls(delegate("researcher", "Find X.", "c1")),
                         says("X is 42, and", stop="max_tokens")],
            researcher=[says("X is 42.")],
        )

        events = await crew.turn()

        reply = [e.content for e in events if e.type is EventType.TEXT and e.run == COORDINATOR]
        assert reply == ["X is 42, and"]
        assert [e for e in events if e.type is EventType.ERROR] == []
        (ended,) = runs_of(events, RUN_COMPLETED)
        assert (ended.run, ended.metadata["partial"]) == (RESEARCHER, False)


# -- a sub-agent's failure: its parent's error result, and the turn goes on ----------------


class TestASubAgentsFailure:
    async def test_the_parent_reads_why_and_the_turn_goes_on(self, make_crew):
        crew = make_crew(
            team(),
            coordinator=[calls(delegate("researcher", "Find X.", "c1")),
                         says("The researcher could not do it.")],
            researcher=[fails("max_tokens: 100000 > 64000")],
        )

        events = await crew.turn()

        (warning,) = [e for e in events if e.type is EventType.ERROR]
        assert warning.content.startswith("researcher failed: ")
        assert "max_tokens: 100000 > 64000" in warning.content
        assert (warning.metadata["recoverable"], warning.metadata["agent"], warning.run) == (
            True, "researcher", RESEARCHER,
        )
        read = crew.results_read("coordinator")["c1"]
        assert read.startswith("Error running sub-agent: ")
        (ended,) = runs_of(events, RUN_FAILED)
        assert (ended.run, ended.metadata["result"]) == (RESEARCHER, "")
        assert ended.metadata["error"] == read.removeprefix("Error running sub-agent: ")
        if crew.backend == "deepagents":
            # A named difference: ADK's private delegation tool words its
            # reason "Dynamic node researcher failed" (the provider's reason
            # reaches the user in the warning above).
            assert "max_tokens: 100000 > 64000" in read
        assert [r.status for r in crew.records()] == [RUN_FAILED]
        assert [e.content for e in events
                if e.type is EventType.TEXT and e.run == COORDINATOR] == [
            "The researcher could not do it."
        ]
        assert crew.requests("coordinator") == 2

    async def test_the_next_turn_runs(self, make_crew):
        crew = make_crew(
            team(),
            coordinator=[calls(delegate("researcher", "Find X.", "c1")),
                         says("It failed."), says("Asked again: still no answer.")],
            researcher=[fails("overloaded")],
        )

        await crew.turn()
        events = await crew.turn("And now?")

        assert [e.type for e in events if e.type is EventType.ERROR] == []
        assert [e.content for e in events if e.type is EventType.TEXT] == [
            "Asked again: still no answer."
        ]

    async def test_an_overflow_is_warned_of_alike(self, make_crew):
        """A request too long for the sub-agent's model: one warning, in the
        same words, its code ``CONTEXT_OVERFLOW``, as the sub-agent's run."""
        too_long = "prompt is too long: 208310 tokens > 200000 maximum"
        crew = make_crew(
            team(),
            coordinator=[calls(delegate("researcher", "Find X.", "c1")),
                         says("The researcher could not do it.")],
            # Enough for either backend's retry, should it make one.
            researcher=[fails(too_long), fails(too_long)],
        )

        events = await crew.turn()

        (warning,) = [e for e in events if e.type is EventType.ERROR]
        assert warning.content == (
            "researcher: the request no longer fits anthropic:claude-sonnet-4-6's "
            "context window."
        )
        assert (warning.metadata["recoverable"], warning.metadata["error_code"],
                warning.metadata["agent"], warning.run) == (
            True, "CONTEXT_OVERFLOW", "researcher", RESEARCHER,
        )
        read = crew.results_read("coordinator")["c1"]
        (ended,) = [e for e in runs_of(events) if e.metadata["status"] != RUN_STARTED]
        if crew.backend == "adk":
            # A named difference: ADK's overflow handler answers the
            # sub-agent's model call with text, so its run completes; on
            # Deep Agents the parent reads the error, and the run fails.
            assert read.startswith("I could not do this task: the request no longer fits ")
            assert ended.metadata["status"] == RUN_COMPLETED
        else:
            assert read.startswith("Error running sub-agent: ")
            assert ended.metadata["status"] == RUN_FAILED
        assert [e.content for e in events
                if e.type is EventType.TEXT and e.run == COORDINATOR] == [
            "The researcher could not do it."
        ]

    async def test_a_nested_failure_is_the_leafs_and_its_parent_answers(self, make_crew):
        """coordinator → middle → leaf, the leaf's model failing: the warning
        names the leaf's run, the middle reads why and answers, and the turn
        completes."""
        crew = make_crew(
            [
                agent("coordinator", sub_agents=("middle",)),
                agent("middle", sub_agents=("leaf",)),
                agent("leaf"),
            ],
            coordinator=[calls(delegate("middle", "Plan X.", "c1")),
                         says("Planned without the leaf.")],
            middle=[calls(delegate("leaf", "Look X up.", "m1")),
                    says("The leaf could not look X up.")],
            leaf=[fails("overloaded")],
        )

        events = await crew.turn()

        middle = RunRef(agent="middle", instance="middle-1", run="middle-1#1",
                        parent_run=None, depth=1)
        leaf = RunRef(agent="leaf", instance="leaf-1", run="leaf-1#1",
                      parent_run="middle-1#1", depth=2)
        (warning,) = [e for e in events if e.type is EventType.ERROR]
        assert warning.content.startswith("leaf failed: ")
        assert "overloaded" in warning.content
        assert (warning.metadata["recoverable"], warning.metadata["agent"], warning.run) == (
            True, "leaf", leaf,
        )
        assert crew.results_read("middle")["m1"].startswith("Error running sub-agent: ")
        assert [(e.run, e.metadata["status"]) for e in runs_of(events)] == [
            (middle, RUN_STARTED), (leaf, RUN_STARTED), (leaf, RUN_FAILED), (middle, RUN_COMPLETED),
        ]
        assert crew.results_read("coordinator")["c1"] == "The leaf could not look X up."
        assert [e.content for e in events
                if e.type is EventType.TEXT and e.run == COORDINATOR] == [
            "Planned without the leaf."
        ]
        assert [(r.ref, r.status) for r in crew.records()] == [
            (middle, RUN_COMPLETED), (leaf, RUN_FAILED),
        ]


# -- a delegation's result shown alike, and the raw log per model call ---------------------


class TestDelegationResult:
    async def test_a_delegations_result_shows_what_the_sub_agent_answered(self, make_crew):
        crew = make_crew(
            team(),
            coordinator=[calls(delegate("researcher", "Find X.", "c1")), says("Done.")],
            researcher=[says("X is 42.\nThe handbook says so.")],
        )

        events = await crew.turn()

        (result,) = [
            e for e in events
            if e.type is EventType.TOOL_RESULT and e.metadata.get("tool_call_id") == "c1"
        ]
        assert (result.content, result.metadata["success"]) == ("X is 42.", True)

    async def test_a_refused_delegation_shows_the_notices_first_line(self, make_crew):
        crew = make_crew(
            team(),
            coordinator=[calls(delegate("researcher", "Find X.", "c1")), says("It refused.")],
            researcher=[says("", stop="refusal")],
        )

        events = await crew.turn()

        (result,) = [
            e for e in events
            if e.type is EventType.TOOL_RESULT and e.metadata.get("tool_call_id") == "c1"
        ]
        # The notice is one line, longer than a summary: it is cut.
        assert "\n" not in REFUSAL and len(REFUSAL) > TOOL_SUMMARY_MAX_LENGTH
        assert (result.content, result.metadata["success"]) == (
            REFUSAL[:TOOL_SUMMARY_MAX_LENGTH] + "...", True,
        )

    async def test_a_cut_delegation_shows_the_kept_texts_first_line(self, make_crew):
        crew = make_crew(
            team(),
            coordinator=[calls(delegate("researcher", "Find X.", "c1")), says("Partly done.")],
            researcher=[says("X is", stop="max_tokens")],
        )

        events = await crew.turn()

        (result,) = [
            e for e in events
            if e.type is EventType.TOOL_RESULT and e.metadata.get("tool_call_id") == "c1"
        ]
        assert crew.results_read("coordinator")["c1"] == CUT_OFF
        assert (result.content, result.metadata["success"]) == ("X is", True)

    async def test_a_failed_delegation_shows_why(self, make_crew):
        crew = make_crew(
            team(),
            coordinator=[calls(delegate("researcher", "Find X.", "c1")), says("It failed.")],
            researcher=[fails("overloaded")],
        )

        events = await crew.turn()

        (result,) = [
            e for e in events
            if e.type is EventType.TOOL_RESULT and e.metadata.get("tool_call_id") == "c1"
        ]
        read = crew.results_read("coordinator")["c1"]
        reason = read.removeprefix("Error running sub-agent: ")
        first = truncate(reason.strip().splitlines()[0], TOOL_SUMMARY_MAX_LENGTH)
        assert result.content == f"Failed: {first}"
        assert (result.metadata["success"], result.metadata["error"]) == (False, read)
        if crew.backend == "adk":
            # A named difference: ADK's private delegation tool words the
            # parent's reason itself.
            assert result.content == "Failed: Dynamic node researcher failed"
        else:
            assert "overloaded" in reason


class TestRawLog:
    async def test_each_record_names_its_agent_and_its_run(self, make_crew, monkeypatch):
        settings = BaseSettings(
            session_store="memory", default_model="claude-sonnet-4-6", raw_llm_logging=True
        )
        monkeypatch.setattr(sys.modules[__name__], "_settings", lambda: settings)
        crew = make_crew(
            team(),
            coordinator=[calls(delegate("researcher", "Find X.", "c1")), says("Done.")],
            researcher=[says("X is 42.")],
        )

        events = await crew.turn()

        coordinator = [("llm_request", "coordinator", None), ("llm_response", "coordinator", None)]
        researcher = [
            ("llm_request", "researcher", "researcher-1#1"),
            ("llm_response", "researcher", "researcher-1#1"),
        ]
        expected = sorted(coordinator * 2 + researcher)
        streamed = [e for e in events if e.type in (EventType.LLM_REQUEST, EventType.LLM_RESPONSE)]
        assert sorted(
            (e.type.value, e.metadata["agent"], e.run.run if e.run else None) for e in streamed
        ) == expected
        log = Path.home() / f".{settings.app_name}" / "logs" / "llm_events.jsonl"
        records = [json.loads(line) for line in log.read_text().splitlines()]
        logged = [r for r in records if r["type"] in ("llm_request", "llm_response")]
        assert sorted(
            (r["type"], r["metadata"]["agent"], (r.get("run") or {}).get("run")) for r in logged
        ) == expected
        assert {json.dumps(r["run"], sort_keys=True) for r in logged if "run" in r} == {
            json.dumps({"agent": "researcher", "instance": "researcher-1",
                        "run": "researcher-1#1", "parent_run": None, "depth": 1}, sort_keys=True)
        }


# -- the input callback after its turn, and a sub-agent's jobs ----------------------------


class TestAfterItsTurn:
    async def test_a_task_a_turn_left_behind_asks_no_one(self, make_crew):
        crew = make_crew(
            [agent("coordinator", sub_agents=("researcher",)),
             agent("researcher", tools=(contract_leave_behind,))],
            coordinator=[calls(delegate("researcher", "Leave a task behind.", "c1")),
                         says("Done.")],
            researcher=[calls(use("contract_leave_behind", "r1")), says("Left one.")],
        )
        asked: list[UserInputRequest] = []

        async def answer(request: UserInputRequest) -> str:
            asked.append(request)
            return "yes"

        crew.answer_with(answer)  # installed outside any turn, as a script does
        await crew.turn()
        LEFT["go"].set()
        late = await asyncio.wait_for(LEFT["task"], 10)

        assert late == {
            "question": "unavailable",
            "permission": (False, "no rule + no interactive approver to ask"),
        }
        # Inside its turn the callback answered; after it, nothing reached it.
        assert LEFT["during"] == "yes"
        assert [request.request_id for request in asked] == ["during"]

    async def test_a_stopped_turns_question_leaves_the_next_turn_asking(self, make_crew):
        # A pin: the span refuses only a request from a turn that has ended,
        # never the next turn's, even after a cancel at a dialog.
        crew = make_crew(
            [agent("coordinator", sub_agents=("researcher",)),
             agent("researcher", tools=(contract_ask,))],
            coordinator=[calls(delegate("researcher", "Ask.", "c1")),
                         calls(delegate("researcher", "Ask again.", "c2")),
                         says("Done.")],
            researcher=[calls(use("contract_ask", "r1")),
                        calls(use("contract_ask", "r2")), says("Got yes.")],
        )
        asked: list[str] = []

        async def answer(request: UserInputRequest) -> str:
            asked.append(request.request_id)
            if len(asked) == 1:
                BLOCKING["started"].set()
                await asyncio.Event().wait()  # at its dialog until the turn is stopped
            return "yes"

        crew.answer_with(answer)
        turn = crew.start()
        try:
            await asyncio.wait_for(BLOCKING["started"].wait(), 10)
        finally:
            await crew.stop(turn)
        events = await crew.turn("Again.")

        assert asked == ["r1", "r2"]
        assert json.loads(crew.results_read("researcher")["r2"])["answer"] == "yes"
        assert [e.content for e in events if e.type is EventType.TEXT][-1] == "Done."


class TestASubAgentsJob:
    async def test_its_record_and_its_notice_name_it(self, make_crew):
        crew = make_crew(
            [agent("coordinator", sub_agents=("researcher",)),
             agent("researcher", tools=(contract_start_job,))],
            coordinator=[calls(delegate("researcher", "Start the build.", "c1")),
                         says("The researcher started a build."),
                         says("The build finished.")],
            researcher=[calls(use("contract_start_job", "r1")), says("Started the job.")],
        )

        await crew.turn()
        jobs = crew.manager.job_manager
        (record,) = jobs.list()
        await until(lambda: jobs.get(record.job_id).state in TERMINAL_STATES)
        resumed = [event async for event in crew.manager.resume_with_job_result(record)]

        assert (record.started_by, record.call_id) == ("researcher-1", "r1")
        notice = crew.last_message("coordinator").splitlines()
        assert notice[1] == (
            "A background job researcher-1 started has finished. Tell the user what it means."
        )
        assert [e.content for e in resumed if e.type is EventType.TEXT] == ["The build finished."]

    async def test_after_ctrl_c_it_never_answers_its_coordinators_call(self, make_crew):
        """The researcher's call that started the job has the id of the
        coordinator's delegation (a provider that numbers call ids per
        agent): after Ctrl+C the delegation is answered as the user's, never
        with the researcher's job."""
        crew = make_crew(
            [agent("coordinator", sub_agents=("researcher",)),
             agent("researcher", tools=(contract_start_job, contract_block))],
            coordinator=[calls(delegate("researcher", "Start the build.", "c1")),
                         says("Starting over.")],
            researcher=[calls(use("contract_start_job", "c1")),
                        calls(use("contract_block", "c2"))],
        )

        turn = crew.start()
        try:
            await asyncio.wait_for(BLOCKING["started"].wait(), 10)
        finally:
            await crew.stop(turn)
        (record,) = crew.manager.job_manager.list()
        await crew.turn("Never mind.")

        assert (record.call_id, record.started_by) == ("c1", "researcher-1")
        answered = crew.results_read("coordinator")["c1"]
        assert INTERRUPTED_BY_USER in answered
        assert record.job_id not in answered


# -- token usage: each model call in its conversation's ledger, by run and agent ------------


class TestUsage:
    async def test_the_ledger_adds_up_to_what_the_providers_reported(self, make_crew):
        crew = make_crew(
            team(),
            coordinator=[calls(delegate("researcher", "Find X.", "c1"), usage=(100, 10)),
                         says("Done.", usage=(120, 12))],
            researcher=[calls(use("contract_summarize", "r1", text="A long page."), usage=(50, 5)),
                        says("X is 42.", usage=(60, 6))],
            side_calls=[says("A summary.", usage=(30, 3))],
        )

        events = await crew.turn()

        ledger = crew.ledger()
        model = str(crew.manager.agent_models["researcher"].ref)
        reported = [e for e in events if e.type is EventType.LLM_USAGE]
        # Every call's usage reached the stream too, the side call's among them.
        assert sum(e.metadata["prompt_tokens"] for e in reported) == 360
        (side,) = [e for e in reported if e.metadata.get("purpose") == PURPOSE_TOOL]
        assert (side.run, side.metadata["agent"]) == (RESEARCHER, "researcher")
        assert ledger.totals() == UsageTotals(input=360, output=36, calls=5)
        assert ledger.by_agent() == {
            "coordinator": UsageTotals(input=220, output=22, calls=2),
            "researcher": UsageTotals(input=110, output=11, calls=2),
        }
        assert ledger.by_purpose() == {PURPOSE_TOOL: UsageTotals(input=30, output=3, calls=1)}
        # The run's own: its steps and its tool's side call.
        assert ledger.totals(run="researcher-1#1") == UsageTotals(input=140, output=14, calls=3)
        assert ledger.totals(run=None) == UsageTotals(input=220, output=22, calls=2)
        assert {row.turn for row in ledger.rows} == {1}
        (ended,) = runs_of(events, RUN_COMPLETED)
        assert ended.metadata["usage"] == {model: {"input": 140, "output": 14}}
        (record,) = crew.records()
        assert record.usage == {model: {"input": 140, "output": 14}}

    async def test_a_side_call_outside_a_turn_is_its_conversations(self, make_crew):
        crew = make_crew(team(), side_calls=[says("[]", usage=(40, 4))])

        with side_call(PURPOSE_MEMORY_EXTRACTION, crew.conversation):
            await crew.manager.generate_simple("Propose memories.", max_tokens=200)

        (row,) = crew.ledger().rows
        assert (row.purpose, row.run, row.turn) == (PURPOSE_MEMORY_EXTRACTION, None, None)
        assert (row.input, row.output, row.agent) == (40, 4, "coordinator")

    async def test_a_stopped_runs_record_keeps_what_it_used(self, make_crew):
        crew = make_crew(
            team(),
            coordinator=[calls(delegate("researcher", "Wait.", "c1"), usage=(100, 10))],
            researcher=[calls(use("contract_block", "r1"), usage=(50, 5))],
        )

        turn = crew.start()
        try:
            await asyncio.wait_for(BLOCKING["started"].wait(), 10)
        finally:
            await crew.stop(turn)

        model = str(crew.manager.agent_models["researcher"].ref)
        (record,) = crew.records()
        assert (record.status, record.usage) == (
            RUN_INTERRUPTED, {model: {"input": 50, "output": 5}}
        )

    @pytest.mark.parametrize("summary", ["A summary.", ""], ids=["written", "empty"])
    async def test_a_summarys_usage_is_recorded_whether_it_is_written_or_not(
        self, make_crew, summary
    ):
        # Three long turns: /compact has the first to summarize. An empty
        # summary is no summary, but its call was paid for all the same.
        crew = make_crew(
            team(),
            coordinator=[says("One."), says("Two."), says("Three.")],
            summaries=[says(summary, usage=(700, 20))],
        )
        for n in (1, 2, 3):
            await crew.turn(f"Turn {n}: {LONG_TURN}")
        reported: list[WorkflowEvent] = []

        if summary:
            await compact_now(crew.manager, events=reported, session_id=SESSION, user_id=USER)
        else:
            with pytest.raises(ValueError, match="empty summary"):
                await compact_now(
                    crew.manager, events=reported, session_id=SESSION, user_id=USER
                )

        (row,) = [row for row in crew.ledger().rows if row.purpose == PURPOSE_SUMMARIZATION]
        assert (row.agent, row.run, row.input, row.output, row.turn) == (
            "coordinator", None, 700, 20, 4
        )
        (usage,) = [e for e in reported if e.type is EventType.LLM_USAGE]
        assert usage.metadata["purpose"] == PURPOSE_SUMMARIZATION


#: A long user turn: three of them hold more than a summary keeps word for word.
LONG_TURN = "word " * 8_000
