"""Whole Deep Agents turns through the real stack, offline.

Every agent here is a real ``create_agent`` graph on the chat model
``build_chat_model`` builds for it, with the whole middleware stack, a real
checkpointer and the real permission engine. Only the provider's HTTP is
fake: ``FakeAPI`` answers each agent from its own script and records what it
was sent, and, like the real APIs, refuses a request with a tool call left
without its result. Nothing reaches a provider.

Covered: a sub-agent's permission prompt, denial and raising tool; usage and
cache tokens; a turn cancelled mid-tool and the turn after it, on Claude and
on OpenAI's Responses API (Deep Agents issue #5781); a model switch that
keeps the session; the thread-sensitive sync tools on LangChain's executor
thread; and Review Focus 1, 2 and 5.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import anthropic
import httpx
import openai
import pytest

pytest.importorskip("deepagents")

from agentic_cli.cli.usage_tracker import UsageTracker  # noqa: E402
from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.tools import compile_document, execute_python, kb_search  # noqa: E402
from agentic_cli.tools.registry import Capability, register_tool  # noqa: E402
from agentic_cli.tools.sandbox import sandbox_execute  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager  # noqa: E402
from agentic_cli.workflow.deepagents.sessions import run_config  # noqa: E402
from agentic_cli.workflow.events import EventType, UserInputRequest  # noqa: E402
from agentic_cli.workflow.permissions import EXEMPT  # noqa: E402
from agentic_cli.workflow.permissions.prompt import (  # noqa: E402
    ALLOW_ONCE_CHOICE,
    ALLOW_SESSION_CHOICE,
    DENY_CHOICE,
)
from agentic_cli.workflow.service_registry import WORKFLOW, get_service  # noqa: E402
from agentic_cli.workflow.sessions import get_active_turn  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI, anthropic_message, openai_response  # noqa: E402
from tests.workflow.langchain_fakes import (  # noqa: E402
    anthropic_reply,
    gemini_reply,
    openai_function_call,
    route,
    route_google,
)

CLAUDE = "anthropic:claude-opus-4-7"
KEYS = {
    "GOOGLE_API_KEY": "g-test-key",
    "ANTHROPIC_API_KEY": "a-test-key",
    "OPENAI_API_KEY": "o-test-key",
}

# What the tools below wait on and report to, set by the test that runs them.
_HOLD: dict[str, asyncio.Event] = {}
_LOOP: dict[str, asyncio.AbstractEventLoop] = {}


@register_tool(capabilities=[Capability("filesystem.read", target_arg="path")])
def dt_read_notes(path: str) -> dict:
    """Reads a notes file (a test tool: outside the project, it asks first)."""
    return {"success": True, "notes": f"notes of {Path(path).name}"}


@register_tool(capabilities=EXEMPT)
def dt_explode() -> dict:
    """Always raises (an exempt test tool)."""
    raise RuntimeError("the tool broke")


@register_tool(capabilities=EXEMPT)
async def dt_wait() -> dict:
    """Waits until the test lets it finish (an exempt test tool)."""
    _HOLD["started"].set()
    await _HOLD["release"].wait()
    return {"success": True}


@register_tool(capabilities=EXEMPT)
def dt_where() -> dict:
    """Reports where it runs and what it sees of its turn (an exempt test tool).

    It asks the user a question from its own thread, as a sync tool would.
    """
    workflow = get_service(WORKFLOW)
    turn = get_active_turn()
    question = UserInputRequest(request_id="q1", tool_name="dt_where", prompt="Colour?")
    answer = asyncio.run_coroutine_threadsafe(
        workflow.request_user_input(question), _LOOP["loop"]
    ).result(timeout=5)
    return {
        "success": True,
        "worker_thread": threading.current_thread() is not threading.main_thread(),
        "workflow": type(workflow).__name__,
        "turn": [turn.user_id, turn.session_id] if turn else None,
        "answer": answer,
    }


@register_tool(capabilities=EXEMPT)
def dt_lookup(topic: str) -> dict:
    """Looks a topic up (an exempt test tool)."""
    return {"success": True, "topic": topic, "fact": f"{topic} is 42"}


# --- the fake APIs ------------------------------------------------------------------------


def _system(body: dict) -> str:
    """The text of an Anthropic request's system prompt (plain, or cache-marked blocks)."""
    system = body.get("system") or ""
    if isinstance(system, list):
        return "".join(block.get("text", "") for block in system)
    return system


def unanswered_tool_uses(body: dict) -> list[str]:
    """``tool_use`` ids no ``tool_result`` answers in the very next message."""
    messages = body["messages"]
    missing: list[str] = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant" or not isinstance(message["content"], list):
            continue
        calls = [b["id"] for b in message["content"] if b.get("type") == "tool_use"]
        following = messages[index + 1]["content"] if index + 1 < len(messages) else []
        answered = {
            b.get("tool_use_id")
            for b in (following if isinstance(following, list) else [])
            if isinstance(b, dict) and b.get("type") == "tool_result"
        }
        missing += [call for call in calls if call not in answered]
    return missing


def unanswered_function_calls(body: dict) -> list[str]:
    """Responses API ``function_call`` items without a ``function_call_output``."""
    items = [item for item in body.get("input", []) if isinstance(item, dict)]
    answered = {i["call_id"] for i in items if i.get("type") == "function_call_output"}
    return [i["call_id"] for i in items if i.get("type") == "function_call"
            and i["call_id"] not in answered]


class Claude:
    """Anthropic's API for a team: each agent answers from its own script.

    A request belongs to the agent whose system prompt it carries. Like the
    real API it refuses (400) a ``tool_use`` left without its result.
    """

    def __init__(self, **scripts: list) -> None:
        self.scripts = {system: list(script) for system, script in scripts.items()}
        self.refused: list[list[str]] = []
        self.api = FakeAPI(self._answer)

    def _answer(self, request):
        body = json.loads(request.content)
        missing = unanswered_tool_uses(body)
        if missing:
            self.refused.append(missing)
            return httpx.Response(400, json={"type": "error", "error": {
                "type": "invalid_request_error",
                "message": f"tool_use ids were found without tool_result blocks: {missing}",
            }})
        reply = self.scripts[_system(body)].pop(0)
        return reply(request) if callable(reply) else reply

    def bodies(self, system: str) -> list[dict]:
        return [b for b in self.api.bodies if _system(b) == system]


def say(text: str) -> object:
    return anthropic_reply({"type": "text", "text": text}, model="claude-opus-4-7")


def calls(*uses: tuple[str, dict, str]) -> object:
    """Claude calling tools: ``(name, input, id)`` each."""
    return anthropic_reply(
        *[{"type": "tool_use", "id": i, "name": n, "input": a} for n, a, i in uses],
        model="claude-opus-4-7",
    )


def delegate(subagent: str, brief: str, call_id: str) -> tuple[str, dict, str]:
    return ("task", {"description": brief, "subagent_type": subagent}, call_id)


def bad_request(message: str = "The provider refused the request.") -> httpx.Response:
    return httpx.Response(400, json={"type": "error", "error": {
        "type": "invalid_request_error", "message": message,
    }})


# --- managers -------------------------------------------------------------------------


@pytest.fixture
def hermetic(tmp_path, monkeypatch) -> Path:
    """A hermetic HOME, a project directory as cwd, and fake keys."""
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    monkeypatch.chdir(tmp_path / "project")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    for variable, key in KEYS.items():
        monkeypatch.setenv(variable, key)
    return tmp_path


def manager_for(configs, *, model: str = CLAUDE, **settings) -> DeepAgentsWorkflowManager:
    settings.setdefault("session_store", "memory")
    return DeepAgentsWorkflowManager(
        agent_configs=configs,
        settings=BaseSettings(default_model=model, **settings),
    )


async def turn(manager, text: str, *, user: str = "u", session: str = "s") -> list:
    return [event async for event in manager.process(text, user, session_id=session)]


class Answers:
    """The input callback: answers each prompt in turn, recording them."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.asked: list[UserInputRequest] = []

    async def __call__(self, request: UserInputRequest) -> str:
        self.asked.append(request)
        return self.answers.pop(0)


def of(events, kind: EventType, agent: str | None = None) -> list:
    return [e for e in events
            if e.type is kind and (agent is None or e.metadata.get("agent") == agent)]


def results(events, tool: str) -> list:
    return [e for e in of(events, EventType.TOOL_RESULT) if e.metadata["tool_name"] == tool]


LEAD = AgentConfig(name="lead", prompt="Lead.", sub_agents=["worker"])


# --- a delegating turn ------------------------------------------------------------------


class TestADelegatingTurn:
    async def test_a_sub_agent_asks_is_denied_survives_a_raise_and_answers(
        self, hermetic, monkeypatch
    ):
        notes = str(hermetic / "elsewhere" / "notes.txt")  # outside the project: asks
        claude = Claude(**{
            "Lead.": [calls(delegate("worker", "Read the notes.", "t1")), say("All done.")],
            "Work.": [
                calls(("dt_read_notes", {"path": notes}, "r1")),
                calls(("dt_read_notes", {"path": notes}, "r2")),
                calls(("dt_explode", {}, "e1")),
                say("Worker done."),
            ],
        })
        route(monkeypatch, anthropic, "AsyncClient", claude.api)
        manager = manager_for([LEAD, AgentConfig(
            name="worker", prompt="Work.", description="Works.",
            tools=[dt_read_notes, dt_explode],
        )])
        answers = Answers(ALLOW_ONCE_CHOICE, DENY_CHOICE)
        manager.set_input_callback(answers)
        try:
            events = await turn(manager, "Go.")
        finally:
            await manager.cleanup()

        assert of(events, EventType.ERROR) == []
        assert [a.tool_name for a in answers.asked] == ["dt_read_notes", "dt_read_notes"]
        allowed, denied = results(events, "dt_read_notes")
        assert allowed.metadata["success"] is True
        assert allowed.metadata["result"]["notes"] == "notes of notes.txt"
        assert denied.metadata["success"] is False
        assert denied.metadata["error"] == "Permission denied: no rule + user denied"
        (raised,) = results(events, "dt_explode")
        assert raised.metadata["success"] is False
        assert raised.metadata["error"] == "dt_explode failed: RuntimeError: the tool broke"
        assert {e.metadata["agent"] for e in (allowed, denied, raised)} == {"worker"}
        assert [e.content for e in of(events, EventType.TEXT)] == ["Worker done.", "All done."]
        (delegation,) = results(events, "task")
        assert (delegation.metadata["agent"], delegation.metadata["result"]) == (
            "lead", "Worker done."
        )
        # The worker read every outcome as a result, the error ones flagged.
        last = claude.bodies("Work.")[-1]["messages"]
        tool_results = [b for m in last if isinstance(m["content"], list)
                        for b in m["content"] if b.get("type") == "tool_result"]
        assert [b.get("is_error", False) for b in tool_results] == [False, True, True]


# --- usage and cache --------------------------------------------------------------------


def _record(events) -> UsageTracker:
    """What the status bar adds up from a turn's usage."""
    tracker = UsageTracker()
    for event in of(events, EventType.LLM_USAGE):
        tracker.record(event.metadata)
    return tracker


class TestUsageAndCache:
    async def test_a_delegation_is_not_taken_for_a_trimmed_context(
        self, hermetic, monkeypatch
    ):
        """The worker's prompt is its brief: the status bar keeps the
        coordinator's context and reports no trim."""

        def reply(input_tokens: int, block: dict) -> dict:
            stop = "tool_use" if block["type"] == "tool_use" else "end_turn"
            return {**anthropic_message(block, model="claude-opus-4-7"), "stop_reason": stop,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 10}}

        brief = {"description": "Look it up.", "subagent_type": "worker"}
        claude = Claude(**{
            "Lead.": [
                reply(5000, {"type": "tool_use", "id": "t1", "name": "task", "input": brief}),
                reply(5100, {"type": "text", "text": "All done."}),
            ],
            "Work.": [reply(400, {"type": "text", "text": "Worker done."})],
        })
        route(monkeypatch, anthropic, "AsyncClient", claude.api)
        manager = manager_for([LEAD, AgentConfig(name="worker", prompt="Work.", description="Works.")])
        try:
            events = await turn(manager, "Go.")
        finally:
            await manager.cleanup()

        tracker = UsageTracker()
        trims = [tracker.record(e.metadata) for e in of(events, EventType.LLM_USAGE)]
        assert trims == [False, False, False]
        assert tracker.context_trimmed_count == 0
        assert tracker.last_prompt_tokens == 5100
        assert tracker.prompt_tokens == 10500

    async def test_claude_cache_reads_and_writes_reach_the_status_bar(
        self, hermetic, monkeypatch
    ):
        reply = {**anthropic_message({"type": "text", "text": "Hi."}, model="claude-opus-4-7"),
                 "usage": {"input_tokens": 50, "output_tokens": 7,
                           "cache_read_input_tokens": 1200, "cache_creation_input_tokens": 300,
                           # The API also breaks the writes down by TTL, and
                           # langchain-anthropic then zeroes "cache_creation".
                           "cache_creation": {"ephemeral_5m_input_tokens": 300,
                                              "ephemeral_1h_input_tokens": 0}}}
        api = FakeAPI(reply)
        route(monkeypatch, anthropic, "AsyncClient", api)
        manager = manager_for([AgentConfig(name="solo", prompt="Answer.")])
        try:
            events = await turn(manager, "Hello.")
        finally:
            await manager.cleanup()

        (usage,) = of(events, EventType.LLM_USAGE)
        assert usage.metadata["model"] == CLAUDE
        tracker = _record(events)
        assert (tracker.cached_tokens, tracker.cache_creation_tokens) == (1200, 300)
        assert (tracker.prompt_tokens, tracker.completion_tokens) == (1550, 7)
        assert api.bodies[0]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}

    async def test_gemini_cache_reads_and_thinking_reach_the_status_bar(
        self, hermetic, monkeypatch
    ):
        def answer(request):
            thought = {"text": "Weighing it.", "thought": True}
            reply = gemini_reply(thought, {"text": "Hi."})(request)
            reply["usageMetadata"] = {
                "promptTokenCount": 900, "candidatesTokenCount": 5, "totalTokenCount": 930,
                "cachedContentTokenCount": 800, "thoughtsTokenCount": 25,
            }
            return reply

        route_google(monkeypatch, FakeAPI(answer))
        manager = manager_for([AgentConfig(name="solo", prompt="Answer.")],
                              model="google:gemini-2.5-flash")
        try:
            events = await turn(manager, "Hello.")
        finally:
            await manager.cleanup()

        assert [e.content for e in of(events, EventType.THINKING)] == ["Weighing it."]
        tracker = _record(events)
        assert (tracker.cached_tokens, tracker.thinking_tokens) == (800, 25)
        assert tracker.prompt_tokens == 900


# --- a cancelled turn -------------------------------------------------------------------


async def _cancel_mid_tool(manager, text: str) -> None:
    """Start a turn, and cancel it while its tool runs."""
    _HOLD.update(started=asyncio.Event(), release=asyncio.Event())
    running = asyncio.create_task(turn(manager, text))
    await asyncio.wait_for(_HOLD["started"].wait(), timeout=5)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running


class TestACancelledTurn:
    async def test_claude_accepts_the_next_turn(self, hermetic, monkeypatch):
        claude = Claude(**{"Answer.": [calls(("dt_wait", {}, "w1")), say("Back again.")]})
        route(monkeypatch, anthropic, "AsyncClient", claude.api)
        manager = manager_for([AgentConfig(name="solo", prompt="Answer.", tools=[dt_wait])])
        try:
            await _cancel_mid_tool(manager, "Wait for it.")
            events = await turn(manager, "Hello again.")
        finally:
            await manager.cleanup()

        assert claude.refused == []
        assert [e.content for e in of(events, EventType.TEXT)] == ["Back again."]
        sent = claude.api.bodies[-1]["messages"]
        assert unanswered_tool_uses(claude.api.bodies[-1]) == []
        answered = [b for m in sent if isinstance(m["content"], list)
                    for b in m["content"] if b.get("type") == "tool_result"]
        assert [(b["tool_use_id"], b.get("is_error")) for b in answered] == [("w1", True)]

    async def test_openai_responses_accepts_the_next_turn(self, hermetic, monkeypatch):
        """Deep Agents issue #5781: Responses ``function_call`` items."""
        refused: list[list[str]] = []
        replies = [openai_function_call("dt_wait", {}, "call_w1"), openai_response("Back again.")]

        def answer(request):
            body = json.loads(request.content)
            missing = unanswered_function_calls(body)
            if missing:
                refused.append(missing)
                return httpx.Response(400, json={"error": {
                    "message": f"No tool output found for function call {missing[0]}.",
                    "type": "invalid_request_error", "param": "input", "code": None,
                }})
            return replies.pop(0)

        api = FakeAPI(answer)
        route(monkeypatch, openai, "AsyncOpenAI", api)
        manager = manager_for([AgentConfig(name="solo", prompt="Answer.", tools=[dt_wait])],
                              model="openai:gpt-5.5")
        try:
            await _cancel_mid_tool(manager, "Wait for it.")
            events = await turn(manager, "Hello again.")
        finally:
            await manager.cleanup()

        assert refused == []
        assert of(events, EventType.ERROR) == []
        assert [e.content for e in of(events, EventType.TEXT)] == ["Back again."]
        items = api.bodies[-1]["input"]
        assert [i["call_id"] for i in items if i.get("type") == "function_call_output"] == [
            "call_w1"
        ]


# --- a model switch -----------------------------------------------------------------------


class TestAModelSwitch:
    async def test_it_keeps_the_session_and_the_next_turn_runs_on_the_new_model(
        self, hermetic, monkeypatch
    ):
        route_google(monkeypatch, FakeAPI(gemini_reply({"text": "Gemini answer."})))
        claude = Claude(**{"Answer.": [say("Claude answer.")]})
        route(monkeypatch, anthropic, "AsyncClient", claude.api)
        manager = manager_for([AgentConfig(name="solo", prompt="Answer.")],
                              model="google:gemini-2.5-flash")
        try:
            before = await turn(manager, "Hello.")
            await manager.reinitialize(model=CLAUDE, preserve_sessions=True)
            after = await turn(manager, "And now?")
            recent = await manager.recent_messages("s", user_id="u")
        finally:
            await manager.cleanup()

        assert [e.metadata["model"] for e in of(before, EventType.LLM_USAGE)] == [
            "google:gemini-2.5-flash"
        ]
        assert [e.metadata["model"] for e in of(after, EventType.LLM_USAGE)] == [CLAUDE]
        sent = claude.api.bodies[0]["messages"]
        assert [m["role"] for m in sent] == ["user", "assistant", "user"]
        assert "Gemini answer." in json.dumps(sent[1])
        assert [m["content"] for m in recent] == [
            "Hello.", "Gemini answer.", "And now?", "Claude answer."
        ]


# --- sync tools on the executor thread --------------------------------------------------


class TestSyncToolsOnTheWorkerThread:
    async def test_they_see_the_services_the_turn_and_the_input_callback(
        self, hermetic, monkeypatch
    ):
        monkeypatch.setattr("agentic_cli.tools.document.compile._which", lambda name: None)
        source = hermetic / "project" / "report.tex"
        source.write_text("\\documentclass{article}\\begin{document}Hi\\end{document}")
        claude = Claude(**{"Answer.": [
            calls(
                ("execute_python", {"code": "print(6 * 7)"}, "p1"),
                ("sandbox_execute", {"code": "print(6 * 7)"}, "s1"),
                ("compile_document", {"source_path": str(source),
                                      "output_pdf": str(source.with_suffix(".pdf"))}, "c1"),
                ("kb_search", {"query": "anything"}, "k1"),
                ("dt_where", {}, "d1"),
            ),
            say("Ran them all."),
        ]})
        route(monkeypatch, anthropic, "AsyncClient", claude.api)
        manager = manager_for(
            [AgentConfig(name="solo", prompt="Answer.", tools=[
                execute_python, sandbox_execute, compile_document, kb_search, dt_where,
            ])],
            stateful_executor_backend="local",
            knowledge_base_use_mock=True,
        )
        _LOOP["loop"] = asyncio.get_running_loop()
        asked: list[str] = []

        async def callback(request: UserInputRequest) -> str:
            # Every permission prompt is allowed for the session; dt_where's
            # own question is answered "Teal".
            asked.append(request.tool_name)
            return "Teal" if request.tool_name == "dt_where" else ALLOW_SESSION_CHOICE

        manager.set_input_callback(callback)
        try:
            events = await turn(manager, "Run the tools.", user="alice", session="s9")
        finally:
            await manager.cleanup()

        assert of(events, EventType.ERROR) == []
        by_tool = {e.metadata["tool_name"]: e.metadata for e in of(events, EventType.TOOL_RESULT)}
        assert by_tool["execute_python"]["success"] is True
        assert "42" in json.dumps(by_tool["execute_python"]["result"])
        assert by_tool["sandbox_execute"]["success"] is True
        assert "42" in json.dumps(by_tool["sandbox_execute"]["result"])
        # compile_document's own result, not the error middleware's.
        assert by_tool["compile_document"]["success"] is False
        assert "No LaTeX engine on PATH" in by_tool["compile_document"]["result"]["error"]
        assert by_tool["kb_search"]["success"] is True
        assert by_tool["dt_where"]["result"] == {
            "success": True, "worker_thread": True, "workflow": "DeepAgentsWorkflowManager",
            "turn": ["alice", "s9"], "answer": "Teal",
        }
        # The prompts came from the tools' own threads, through the turn's callback.
        assert {"execute_python", "sandbox_execute", "compile_document"} <= set(asked)


# --- Review Focus 1, 2 and 5 --------------------------------------------------------------


def _thinking_blocks(body: dict) -> list[tuple[str, str]]:
    return [
        (block["thinking"], block["signature"])
        for message in body["messages"] if message["role"] == "assistant"
        and isinstance(message["content"], list)
        for block in message["content"] if block.get("type") == "thinking"
    ]


async def test_claude_thinking_round_trips_across_turns(hermetic, monkeypatch):
    """Review Focus 1: Claude's signed thinking survives a tool loop and a second turn."""
    claude = Claude(**{"Answer.": [
        anthropic_reply(
            {"type": "thinking", "thinking": "Look it up.", "signature": "sig-one"},
            {"type": "tool_use", "id": "l1", "name": "dt_lookup", "input": {"topic": "X"}},
            model="claude-opus-4-7",
        ),
        anthropic_reply(
            {"type": "thinking", "thinking": "Found it.", "signature": "sig-two"},
            {"type": "text", "text": "X is 42."},
            model="claude-opus-4-7",
        ),
        say("Still 42."),
    ]})
    route(monkeypatch, anthropic, "AsyncClient", claude.api)
    manager = manager_for([AgentConfig(name="solo", prompt="Answer.", tools=[dt_lookup])])
    try:
        first = await turn(manager, "What is X?")
        await turn(manager, "Are you sure?")
    finally:
        await manager.cleanup()

    assert [e.content for e in of(first, EventType.THINKING)] == ["Look it up.", "Found it."]
    in_the_loop, next_turn = claude.api.bodies[1], claude.api.bodies[2]
    assert _thinking_blocks(in_the_loop) == [("Look it up.", "sig-one")]
    assert _thinking_blocks(next_turn) == [("Look it up.", "sig-one"), ("Found it.", "sig-two")]
    assert claude.refused == []


async def test_parallel_delegation(hermetic, monkeypatch):
    """Review Focus 2: two ``task`` calls in one step."""
    tasks = [{"description": "research", "status": "in_progress"},
             {"description": "draft", "status": "pending"}]
    claude = Claude(**{
        "Lead.": [
            calls(delegate("researcher", "Research X.", "t1"),
                  delegate("writer", "Draft Y.", "t2")),
            say("Both done."),
        ],
        "Research.": [
            calls(("save_tasks", {"tasks": tasks}, "rs1"), ("dt_lookup", {"topic": "X"}, "rl1")),
            say("Research done."),
        ],
        "Write.": [
            calls(("save_tasks", {"tasks": [{**t, "status": "completed"} for t in tasks]}, "ws1")),
            say("Draft done."),
        ],
    })
    route(monkeypatch, anthropic, "AsyncClient", claude.api)
    manager = manager_for([
        AgentConfig(name="lead", prompt="Lead.", sub_agents=["researcher", "writer"]),
        AgentConfig(name="researcher", prompt="Research.", description="Researches.",
                    tools=[dt_lookup]),
        AgentConfig(name="writer", prompt="Write.", description="Writes."),
    ])
    try:
        events = await turn(manager, "Research X and draft Y.")
        state = (await manager._root.aget_state(run_config(manager.session_ref("s", "u")))).values
    finally:
        await manager.cleanup()

    assert of(events, EventType.ERROR) == []
    assert sorted(e.metadata["result"] for e in results(events, "task")) == [
        "Draft done.", "Research done."
    ]
    assert {e.metadata["agent"] for e in results(events, "task")} == {"lead"}
    lead_answers = claude.bodies("Lead.")[1]["messages"][-1]["content"]
    assert sorted((b["tool_use_id"], b["content"]) for b in lead_answers
                  if b.get("type") == "tool_result") == [
        ("t1", "Research done."), ("t2", "Draft done.")
    ]
    # Each sub-agent's events are its own.
    assert [e.metadata["agent"] for e in results(events, "dt_lookup")] == ["researcher"]
    assert sorted(e.metadata["agent"] for e in results(events, "save_tasks")) == [
        "researcher", "writer"
    ]
    assert {e.content: e.metadata["agent"] for e in of(events, EventType.TEXT)} == {
        "Research done.": "researcher", "Draft done.": "writer", "Both done.": "lead",
    }
    # Both wrote tasks in one step: one write stands whole, and nothing failed.
    assert [t["status"] for t in state["tasks"]] in (
        ["in_progress", "pending"], ["completed", "completed"]
    )
    assert of(events, EventType.TASK_PROGRESS)


async def test_subagent_model_error_then_next_turn(hermetic, monkeypatch):
    """Review Focus 5: a sub-agent's provider answers 400 mid-turn."""
    claude = Claude(**{
        "Lead.": [calls(delegate("worker", "Do it.", "t1")), say("Sorry, the worker failed.")],
        "Work.": [bad_request("The worker's request was refused.")],
    })
    route(monkeypatch, anthropic, "AsyncClient", claude.api)
    manager = manager_for([LEAD, AgentConfig(name="worker", prompt="Work.",
                                             description="Works.")])
    try:
        failed = await turn(manager, "Delegate.")
        state = (await manager._root.aget_state(run_config(manager.session_ref("s", "u")))).values
        answered = await turn(manager, "What happened?")
        recent = await manager.recent_messages("s", user_id="u")
    finally:
        await manager.cleanup()

    (error,) = of(failed, EventType.ERROR)
    assert error.metadata["agent"] == "worker"
    assert "refused" in error.content
    assert of(failed, EventType.TEXT) == []
    # The checkpoint holds the question and the task call, nothing half-written.
    assert [m.type for m in state["messages"]] == ["human", "ai"]
    assert state["messages"][1].tool_calls[0]["id"] == "t1"

    # The next turn is accepted: the dangling task call was answered first.
    assert claude.refused == []
    assert [e.content for e in of(answered, EventType.TEXT)] == ["Sorry, the worker failed."]
    patched = claude.bodies("Lead.")[-1]["messages"]
    results_sent = [b for m in patched if isinstance(m["content"], list)
                    for b in m["content"] if b.get("type") == "tool_result"]
    assert [(b["tool_use_id"], b.get("is_error")) for b in results_sent] == [("t1", True)]
    assert [m["content"] for m in recent] == [
        "Delegate.", "What happened?", "Sorry, the worker failed."
    ]
