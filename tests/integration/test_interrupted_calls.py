"""A turn stopped by Ctrl+C, or cut off by a crash: every call gets one result.

Ctrl+C stops the whole turn: the agent loop, its model calls, its tool calls
and its sub-agents. What the model is told afterwards, on both backends
(``workflow/interrupts.py``):

- a call that finished before the cancel keeps its real result;
- a call that was running is answered "Interrupted by the user before it
  finished; it may have partly run. ...";
- a call waiting at its permission prompt is answered "Not run: the user
  cancelled it.";
- a delegation to a sub-agent is answered as interrupted too;
- a cancel during a model call, with no call open, leaves "(Interrupted by
  the user.)" as the main agent's reply, so the next turn answers the new
  message rather than both;
- a call a crash left open (no cancel ran) is answered on the next turn
  "Interrupted before it finished ...". ADK keeps a long-running call
  without a result in every later request, which Claude refuses; it drops
  any other one and logs that over the UI on every request.

The fake Claude refuses, as the real API does, a request with a ``tool_use``
whose ``tool_result`` is missing.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from pathlib import Path

import anthropic
import httpx
import pytest

pytest.importorskip("google.adk")
pytest.importorskip("deepagents")

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.tools.registry import Capability, register_tool  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from agentic_cli.workflow.interrupts import (  # noqa: E402
    INTERRUPTED,
    INTERRUPTED_BY_USER,
    MARKER,
    NOT_RUN,
)
from agentic_cli.workflow.permissions import EXEMPT  # noqa: E402
from agentic_cli.workflow.service_registry import JOB_MANAGER, get_service  # noqa: E402
from tests.integration.test_deepagents_turns import unanswered_tool_uses  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI  # noqa: E402
from tests.workflow.langchain_fakes import anthropic_reply, route  # noqa: E402
from tests.workflow.test_manager_models import route as route_adk  # noqa: E402

CLAUDE = "anthropic:claude-opus-4-7"
BACKENDS = ["adk", "deepagents"]

#: What the test tools signal: ``started`` when ``ic_wait`` runs, ``finished``
#: when ``ic_fast`` returns, ``asked`` when a permission prompt is shown.
SIGNALS: dict[str, asyncio.Event] = {}


@register_tool(capabilities=EXEMPT, description="Waits until cancelled (a test tool).")
async def ic_wait() -> dict:
    """Wait, as a long tool does, until the turn is cancelled."""
    SIGNALS["started"].set()
    await asyncio.Event().wait()
    return {"success": True}


@register_tool(capabilities=EXEMPT, description="Answers at once (a test tool).")
async def ic_fast(topic: str) -> dict:
    """Look a topic up at once."""
    SIGNALS["finished"].set()
    return {"success": True, "fact": f"{topic} is 42"}


@register_tool(
    capabilities=[Capability("filesystem.read", target_arg="path")],
    description="Reads a file (a test tool that needs permission).",
)
async def ic_guarded(path: str) -> dict:
    """Read a file outside the project, which the permission engine asks about."""
    return {"success": True, "read": path}


#: Holds ``ic_start_job``'s job running until the test ends.
JOB_RELEASE = threading.Event()


def _until_released() -> str:
    JOB_RELEASE.wait(timeout=30)
    return "built"


@register_tool(capabilities=EXEMPT, long_running=True, requires=JOB_MANAGER,
               description="Starts a background job, then hangs (a test tool).")
async def ic_start_job(tool_context=None) -> dict:
    """Start a job, then wait until cancelled before returning its pending result."""
    get_service(JOB_MANAGER).submit(
        tool="ic_start_job", backend="inprocess", spec={"target": _until_released},
        resume_on_complete=True, call_id=getattr(tool_context, "function_call_id", None),
    )
    SIGNALS["started"].set()
    await asyncio.Event().wait()
    return {"success": True}


# --- the fake API ---------------------------------------------------------------------


def _system(body: dict) -> str:
    system = body.get("system") or ""
    if isinstance(system, list):
        return "".join(block.get("text", "") for block in system)
    return system


class Claude:
    """Anthropic's API: each agent answers from its own script, picked by a
    word of its system prompt. It refuses (400), as the real API does, a
    ``tool_use`` left without its ``tool_result``; ``hold`` keeps the next
    answer back until the turn is cancelled."""

    def __init__(self, **scripts: list) -> None:
        self.scripts = {word: list(script) for word, script in scripts.items()}
        self.refused: list[list[str]] = []
        self.sent: list[tuple[str, dict]] = []
        self.api = FakeAPI(self._answer)

    async def _answer(self, request):
        body = json.loads(request.content)
        missing = unanswered_tool_uses(body)
        if missing:
            self.refused.append(missing)
            return httpx.Response(400, json={"type": "error", "error": {
                "type": "invalid_request_error",
                "message": f"tool_use ids were found without tool_result blocks: {missing}",
            }})
        system = _system(body)
        word = next(w for w in self.scripts if w in system)
        self.sent.append((word, body))
        reply = self.scripts[word].pop(0)
        if reply == "hold":
            SIGNALS["started"].set()
            await asyncio.Event().wait()
        return reply(request) if callable(reply) else reply

    def last(self, word: str) -> dict:
        return [body for w, body in self.sent if w == word][-1]


def say(text: str):
    return anthropic_reply({"type": "text", "text": text}, model="claude-opus-4-7")


def calls(*uses: tuple[str, dict, str]):
    return anthropic_reply(
        *[{"type": "tool_use", "id": i, "name": n, "input": a} for n, a, i in uses],
        model="claude-opus-4-7",
    )


def results(body: dict) -> dict[str, str]:
    """Each ``tool_result`` of a request, by its call id, as the text sent."""
    found: dict[str, str] = {}
    for message in body["messages"]:
        if not isinstance(message["content"], list):
            continue
        for block in message["content"]:
            if block.get("type") == "tool_result":
                content = block.get("content")
                if isinstance(content, list):
                    content = " ".join(part.get("text", "") for part in content)
                found[block["tool_use_id"]] = content or ""
    return found


def lines(body: dict) -> list[str]:
    """Each text block of a request as ``"<role>: <text>"``."""
    out: list[str] = []
    for message in body["messages"]:
        content = message["content"]
        if isinstance(content, str):
            out.append(f"{message['role']}: {content}")
            continue
        out += [f"{message['role']}: {b['text']}" for b in content if b.get("type") == "text"]
    return out


# --- managers -------------------------------------------------------------------------


@pytest.fixture
def hermetic(tmp_path, monkeypatch) -> Path:
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    monkeypatch.chdir(tmp_path / "project")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for variable in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-test-key")
    SIGNALS.update(started=asyncio.Event(), finished=asyncio.Event(), asked=asyncio.Event())
    return tmp_path


def serve(monkeypatch, claude: Claude, backend: str) -> Claude:
    """ADK builds ``anthropic.AsyncAnthropic`` (and checks its type);
    langchain-anthropic builds ``AsyncClient`` with an HTTP client of its own."""
    if backend == "adk":
        route_adk(monkeypatch, anthropic, "AsyncAnthropic", claude.api)
    else:
        route(monkeypatch, anthropic, "AsyncClient", claude.api)
    return claude


def manager_for(backend: str, configs, **values):
    values.setdefault("session_store", "memory")
    settings = BaseSettings(default_model=CLAUDE, **values)
    if backend == "adk":
        return GoogleADKWorkflowManager(agent_configs=configs, settings=settings)
    return DeepAgentsWorkflowManager(agent_configs=configs, settings=settings)


def solo(backend: str, *tools, **values):
    return manager_for(backend, [AgentConfig(
        name="solo", prompt="Answer.", tools=list(tools), include_state_tools=False,
    )], **values)


async def turn(manager, text: str) -> list:
    return [event async for event in manager.process(text, "u", session_id="s")]


async def cancelled(manager, text: str, until: str = "started") -> None:
    """Start a turn, and cancel it once ``SIGNALS[until]`` is set."""
    running = asyncio.create_task(turn(manager, text))
    await asyncio.wait_for(SIGNALS[until].wait(), timeout=5)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running


# --- a cancel ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
async def test_a_running_call_is_answered_as_interrupted_by_the_user(
    hermetic, monkeypatch, backend
):
    claude = serve(monkeypatch, Claude(Answer=[
        calls(("ic_wait", {}, "toolu_w")), say("Back again."),
    ]), backend)
    manager = solo(backend, ic_wait)
    try:
        await cancelled(manager, "Wait for it.")
        await turn(manager, "Hello again.")
    finally:
        await manager.cleanup()

    assert claude.refused == []
    assert INTERRUPTED_BY_USER in results(claude.last("Answer"))["toolu_w"]


@pytest.mark.parametrize("backend", BACKENDS)
async def test_a_call_that_finished_beside_it_keeps_its_result(hermetic, monkeypatch, backend):
    claude = serve(monkeypatch, Claude(Answer=[
        calls(("ic_fast", {"topic": "tea"}, "toolu_f"), ("ic_wait", {}, "toolu_w")),
        say("Back again."),
    ]), backend)
    manager = solo(backend, ic_fast, ic_wait)
    try:
        running = asyncio.create_task(turn(manager, "Both, please."))
        await asyncio.wait_for(SIGNALS["started"].wait(), timeout=5)
        await asyncio.wait_for(SIGNALS["finished"].wait(), timeout=5)
        await asyncio.sleep(0.05)  # the fast call's result is recorded
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        await turn(manager, "Hello again.")
    finally:
        await manager.cleanup()

    sent = results(claude.last("Answer"))
    assert claude.refused == []
    assert "tea is 42" in sent["toolu_f"] and INTERRUPTED_BY_USER not in sent["toolu_f"]
    assert INTERRUPTED_BY_USER in sent["toolu_w"]


@pytest.mark.parametrize("backend", BACKENDS)
async def test_a_call_waiting_at_its_permission_prompt_is_not_run(
    hermetic, monkeypatch, backend
):
    claude = serve(monkeypatch, Claude(Answer=[
        calls(("ic_guarded", {"path": "/elsewhere/notes.txt"}, "toolu_g")), say("Back again."),
    ]), backend)
    manager = solo(backend, ic_guarded)

    async def never_answered(request) -> str:
        SIGNALS["asked"].set()
        await asyncio.Event().wait()
        return "deny"

    manager.set_input_callback(never_answered)
    try:
        await cancelled(manager, "Read it.", until="asked")
        await turn(manager, "Hello again.")
    finally:
        await manager.cleanup()

    assert claude.refused == []
    assert NOT_RUN in results(claude.last("Answer"))["toolu_g"]


@pytest.mark.parametrize("backend", BACKENDS)
async def test_a_cancel_during_the_model_call_leaves_a_note(hermetic, monkeypatch, backend):
    claude = serve(monkeypatch, Claude(Answer=["hold", say("Back again.")]), backend)
    manager = solo(backend)
    try:
        await cancelled(manager, "Think hard.")
        await turn(manager, "Hello again.")
    finally:
        await manager.cleanup()

    assert lines(claude.last("Answer"))[-3:] == [
        "user: Think hard.", f"assistant: {MARKER}", "user: Hello again.",
    ]


@pytest.mark.parametrize("backend", BACKENDS)
async def test_a_delegation_cut_off_in_its_sub_agent_is_answered_as_interrupted(
    hermetic, monkeypatch, backend
):
    delegation = "researcher" if backend == "adk" else "task"
    brief = {"request": "Look."} if backend == "adk" else {
        "description": "Look.", "subagent_type": "researcher",
    }
    claude = serve(monkeypatch, Claude(
        Coordinate=[calls((delegation, brief, "toolu_d")), say("Back again.")],
        Research=[calls(("ic_wait", {}, "toolu_w"))],
    ), backend)
    manager = manager_for(backend, [
        AgentConfig(name="coordinator", prompt="Coordinate.", sub_agents=["researcher"],
                    include_state_tools=False),
        AgentConfig(name="researcher", prompt="Research.", description="Looks things up.",
                    tools=[ic_wait], include_state_tools=False),
    ])
    try:
        await cancelled(manager, "Find out.")
        await turn(manager, "Hello again.")
    finally:
        await manager.cleanup()

    assert claude.refused == []
    assert INTERRUPTED_BY_USER in results(claude.last("Coordinate"))["toolu_d"]


@pytest.mark.parametrize("backend", BACKENDS)
async def test_a_call_whose_job_was_submitted_is_answered_with_the_running_job(
    hermetic, monkeypatch, backend
):
    """The job is not part of the turn: it runs on, and the model is told so."""
    claude = serve(monkeypatch, Claude(Answer=[
        calls(("ic_start_job", {}, "toolu_j")), say("Back again."),
    ]), backend)
    manager = solo(backend, ic_start_job)
    JOB_RELEASE.clear()
    try:
        await cancelled(manager, "Build it.")
        await turn(manager, "Hello again.")
    finally:
        JOB_RELEASE.set()
        await manager.cleanup()

    sent = json.loads(results(claude.last("Answer"))["toolu_j"])
    assert claude.refused == []
    assert sent["success"] is True and sent["state"] in ("queued", "running")
    assert sent["job_id"] and "result arrives when it ends" in sent["note"]


@register_tool(capabilities=EXEMPT, description="Slow to stop (a test tool).")
async def ic_slow_stop() -> dict:
    """Wait until cancelled, then take a while to clean up, as a process's tool does."""
    SIGNALS["started"].set()
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        await asyncio.sleep(0.5)
        raise
    return {"success": True}


@register_tool(capabilities=EXEMPT, long_running=True, requires=JOB_MANAGER,
               description="Starts a background job and returns nothing (a test tool).")
async def ic_start_job_quietly(tool_context=None) -> None:
    """Start a job and return nothing, a supported starter shape."""
    get_service(JOB_MANAGER).submit(
        tool="ic_start_job_quietly", backend="inprocess", spec={"target": _until_released},
        resume_on_complete=True, call_id=getattr(tool_context, "function_call_id", None),
    )
    SIGNALS["finished"].set()
    return None


@pytest.mark.parametrize("backend", BACKENDS)
async def test_a_cancel_during_a_later_model_call_leaves_a_note(hermetic, monkeypatch, backend):
    """The stop lands while the model answers its tool's result: no call is
    open, and the model is told the turn was stopped all the same."""
    claude = serve(monkeypatch, Claude(Answer=[
        calls(("ic_fast", {"topic": "tea"}, "toolu_f")), "hold", say("Back again."),
    ]), backend)
    manager = solo(backend, ic_fast)
    try:
        await cancelled(manager, "Look tea up and explain.")
        await turn(manager, "Hello again.")
    finally:
        await manager.cleanup()

    assert claude.refused == []
    assert lines(claude.last("Answer"))[-2:] == [f"assistant: {MARKER}", "user: Hello again."]
    assert "tea is 42" in results(claude.last("Answer"))["toolu_f"]


@pytest.mark.parametrize("backend", BACKENDS)
async def test_a_turn_its_consumer_closes_is_closed_inside_it(hermetic, monkeypatch, backend):
    """A library caller may stop reading at an event and close process()
    rather than cancel it: the run stops, and its calls are answered (the
    model is told the user stopped it), before the turn lets go."""
    claude = serve(monkeypatch, Claude(Answer=[
        calls(("ic_wait", {}, "toolu_w")), say("Back again."),
    ]), backend)
    manager = solo(backend, ic_wait, session_store="sqlite")
    try:
        async with contextlib.aclosing(manager.process("Wait for it.", "u", session_id="s")) as events:
            async for event in events:
                if event.type is EventType.TOOL_CALL:
                    break
        closed_inside = not manager._turn_lock.locked()
        await asyncio.wait_for(turn(manager, "Hello again."), timeout=10)
    finally:
        await manager.cleanup()

    # Deep Agents may not have saved the step yet: then no call is open, and
    # the marker says the turn stopped.
    sent = claude.last("Answer")
    assert closed_inside and claude.refused == []
    assert (INTERRUPTED_BY_USER in results(sent).get("toolu_w", "")
            or f"assistant: {MARKER}" in lines(sent))


@pytest.mark.parametrize("store", ["memory", "sqlite"])
async def test_on_adk_a_second_ctrl_c_waits_for_the_answers(hermetic, monkeypatch, caplog, store):
    """Pressed again while a tool takes a while to stop: the turn still ends
    with its calls answered, the finished one with its result."""
    claude = serve(monkeypatch, Claude(Answer=[
        calls(("ic_fast", {"topic": "tea"}, "toolu_f"), ("ic_slow_stop", {}, "toolu_w")),
        say("Back again."),
    ]), "adk")
    manager = solo("adk", ic_fast, ic_slow_stop, session_store=store)
    try:
        with caplog.at_level("WARNING"):
            running = asyncio.create_task(turn(manager, "Both, please."))
            await asyncio.wait_for(SIGNALS["started"].wait(), timeout=5)
            await asyncio.wait_for(SIGNALS["finished"].wait(), timeout=5)
            await asyncio.sleep(0.05)
            running.cancel()
            await asyncio.sleep(0.1)
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
            await turn(manager, "Hello again.")
    finally:
        await manager.cleanup()

    sent = results(claude.last("Answer"))
    assert claude.refused == []
    assert "tea is 42" in sent["toolu_f"] and INTERRUPTED_BY_USER in sent["toolu_w"]
    assert [r.getMessage() for r in caplog.records if r.levelno >= 40] == []


@pytest.mark.parametrize("backend", BACKENDS)
async def test_a_finished_state_tool_keeps_its_change(hermetic, monkeypatch, backend):
    """save_tasks beside a call still running: told it saved, it did save."""
    tasks = [{"description": "Brew tea", "status": "in_progress"}]
    serve(monkeypatch, Claude(Answer=[
        calls(("save_tasks", {"tasks": tasks}, "toolu_t"), ("ic_wait", {}, "toolu_w")),
        say("Back again."),
    ]), backend)
    manager = manager_for(backend, [AgentConfig(
        name="solo", prompt="Answer.", tools=[ic_wait], include_state_tools=True,
    )])
    try:
        running = asyncio.create_task(turn(manager, "Plan and wait."))
        await asyncio.wait_for(SIGNALS["started"].wait(), timeout=5)
        await asyncio.sleep(0.1)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        await turn(manager, "Hello again.")
        if backend == "adk":
            session = await manager._session_service.get_session(
                app_name=manager.app_name, user_id="u", session_id="s")
            saved = session.state.get("tasks")
        else:
            from agentic_cli.workflow.deepagents.sessions import run_config

            state = await manager._root.aget_state(run_config(manager.session_ref("s", "u")))
            saved = state.values.get("tasks")
    finally:
        await manager.cleanup()

    assert [t["description"] for t in saved] == ["Brew tea"]


@pytest.mark.parametrize("backend", BACKENDS)
async def test_a_call_waiting_at_its_question_is_not_run(hermetic, monkeypatch, backend):
    from agentic_cli.tools.interaction_tools import ask_clarification

    claude = serve(monkeypatch, Claude(Answer=[
        calls(("ask_clarification", {"question": "Which file?"}, "toolu_q")), say("Back again."),
    ]), backend)
    manager = solo(backend, ask_clarification)

    async def never_answered(request) -> str:
        SIGNALS["asked"].set()
        await asyncio.Event().wait()
        return "x"

    manager.set_input_callback(never_answered)
    try:
        await cancelled(manager, "Ask me.", until="asked")
        await turn(manager, "Hello again.")
    finally:
        await manager.cleanup()

    assert NOT_RUN in results(claude.last("Answer"))["toolu_q"]


async def test_on_adk_a_quiet_starter_beside_a_cancel_is_answered_with_its_job(
    hermetic, monkeypatch
):
    """A long-running starter that returned nothing gets no result from ADK:
    its job's state answers it, never null."""
    claude = serve(monkeypatch, Claude(Answer=[
        calls(("ic_start_job_quietly", {}, "toolu_j"), ("ic_wait", {}, "toolu_w")),
        say("Back again."),
    ]), "adk")
    manager = solo("adk", ic_start_job_quietly, ic_wait)
    JOB_RELEASE.clear()
    try:
        running = asyncio.create_task(turn(manager, "Build and wait."))
        await asyncio.wait_for(SIGNALS["started"].wait(), timeout=5)
        await asyncio.wait_for(SIGNALS["finished"].wait(), timeout=5)
        await asyncio.sleep(0.05)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        await turn(manager, "Hello again.")
    finally:
        JOB_RELEASE.set()
        await manager.cleanup()

    assert json.loads(results(claude.last("Answer"))["toolu_j"])["job_id"]


async def test_on_a_native_adk_app_nothing_is_answered_for_it(hermetic):
    """ADK leaves some calls open on purpose (a task-mode sub-agent waiting
    for the user): a native app's own agents get ADK's handling alone."""
    from types import SimpleNamespace

    from google.adk.events import Event
    from google.genai import types

    from agentic_cli.workflow.interrupts import OpenCalls

    manager = GoogleADKWorkflowManager.__new__(GoogleADKWorkflowManager)
    manager._adk_config_path = "root_agent.yaml"
    manager._open_calls = OpenCalls()
    appended: list = []

    async def append_event(session, event):
        appended.append(event)

    manager._session_service = SimpleNamespace(append_event=append_event)
    session = SimpleNamespace(events=[Event(
        invocation_id="e-1", author="coordinator",
        content=types.Content(role="model", parts=[types.Part(
            function_call=types.FunctionCall(id="toolu_task", name="booker", args={}),
        )]),
    )])

    await manager._answer_left_open(session)

    assert appended == []


# --- a crash ----------------------------------------------------------------------------


@pytest.mark.parametrize("long_running", [True, False], ids=["long-running", "plain"])
async def test_on_adk_a_call_a_crash_left_open_is_answered_on_the_next_turn(
    hermetic, monkeypatch, caplog, long_running
):
    """No cancel ran, so nothing sealed the call: the next turn answers it."""
    from google.adk.events import Event
    from google.genai import types

    claude = serve(monkeypatch, Claude(Answer=[say("First."), say("Back again.")]), "adk")
    manager = solo("adk", ic_wait)
    try:
        await turn(manager, "Hello.")
        service = manager._session_service
        session = await service.get_session(app_name=manager.app_name, user_id="u",
                                            session_id="s")
        await service.append_event(session, Event(
            invocation_id="e-crashed", author="solo",
            content=types.Content(role="model", parts=[types.Part(
                function_call=types.FunctionCall(id="toolu_c", name="ic_wait", args={}),
            )]),
            long_running_tool_ids={"toolu_c"} if long_running else None,
        ))
        with caplog.at_level("WARNING"):
            await turn(manager, "Hello again.")
    finally:
        await manager.cleanup()

    assert claude.refused == []
    assert INTERRUPTED in results(claude.last("Answer"))["toolu_c"]
    assert not [r for r in caplog.records if "Dropping function calls" in r.getMessage()]


async def test_on_deep_agents_a_call_left_open_across_a_restart_is_answered(
    hermetic, monkeypatch
):
    """The record of the cancel is gone with the process: the call is
    answered as interrupted, by whatever stopped it."""
    claude = serve(monkeypatch, Claude(Answer=[
        calls(("ic_wait", {}, "toolu_w")), say("Back again."),
    ]), "deepagents")
    first = solo("deepagents", ic_wait, session_store="sqlite")
    try:
        await cancelled(first, "Wait for it.")
    finally:
        await first.cleanup()
    second = solo("deepagents", ic_wait, session_store="sqlite")
    try:
        await turn(second, "Hello again.")
    finally:
        await second.cleanup()

    assert claude.refused == []
    assert INTERRUPTED in results(claude.last("Answer"))["toolu_w"]


# --- the ADK it rests on ----------------------------------------------------------------


class TestTheADKItRestsOn:
    def test_a_run_takes_an_abort_signal(self):
        import inspect

        from google.adk import Runner

        parameter = inspect.signature(Runner.run_async).parameters["abort_signal"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY

    def test_its_seals_carry_the_code_the_plugin_reads(self):
        """ADK's private marker of an aborted invocation's events."""
        from google.adk.events import _abort_events

        from agentic_cli.workflow.adk.interrupted_calls_plugin import ABORTED

        assert _abort_events._INVOCATION_ABORTED == ABORTED


async def test_on_adk_a_jobs_resume_answers_its_open_call_first(hermetic, monkeypatch):
    """A starter that returned nothing leaves its call open (ADK sends no
    result for it); the job's notice must not reach Claude beside it."""
    claude = serve(monkeypatch, Claude(Answer=[
        calls(("ic_start_job_quietly", {}, "toolu_j")), say("The build finished."),
    ]), "adk")
    manager = solo("adk", ic_start_job_quietly)
    JOB_RELEASE.set()  # the job ends at once
    try:
        await turn(manager, "Build it.")
        jobs = manager.job_manager
        (record,) = [r for r in jobs.list() if r.call_id == "toolu_j"]
        for _ in range(200):
            if jobs.get(record.job_id).state.value in ("succeeded", "failed"):
                break
            await asyncio.sleep(0.01)
        events = [e async for e in manager.resume_with_job_result(jobs.get(record.job_id))]
    finally:
        await manager.cleanup()

    assert claude.refused == []
    assert [e.content for e in events if e.type is EventType.TEXT] == ["The build finished."]
    assert "toolu_j" in results(claude.last("Answer"))
