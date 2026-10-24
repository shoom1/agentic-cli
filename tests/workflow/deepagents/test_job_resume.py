"""Job resume on Deep Agents (mirrors ``tests/workflow/test_adk_job_resume.py``).

A long-running tool submits its job and returns at once; its call gets that
result and the turn ends. When the job finishes, ``resume_with_job_result``
runs a turn whose input is one ``HumanMessage``: the job notification header,
then the payload ADK sends (``tools/jobs/resume.py``) as JSON, marked in its
``response_metadata`` so ``recent_messages`` leaves it out. The root agent
receives it, whichever agent started the job.

``ActiveToolCallMiddleware`` makes each tool call the active tool call
(``workflow.sessions``), so ``JobManager.submit`` links a job to the call
that started it although a tool written for ADK gets ``tool_context=None``.

Agent models are scripted (``fakes.ScriptedModels``); what reaches each
provider is checked through ``FakeAPI``. Nothing reaches a provider.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import anthropic
import openai
import pytest

pytest.importorskip("deepagents")

from langchain.agents.middleware import ToolCallRequest  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.tools.jobs import JobState  # noqa: E402
from agentic_cli.tools.jobs.backends import TERMINAL_STATES  # noqa: E402
from agentic_cli.tools.jobs.resume import JOB_NOTICE_HEADER, job_result_payload  # noqa: E402
from agentic_cli.tools.registry import register_tool  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.deepagents import manager as manager_module  # noqa: E402
from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager  # noqa: E402
from agentic_cli.workflow.deepagents.middleware import ActiveToolCallMiddleware  # noqa: E402
from agentic_cli.workflow.deepagents.sessions import thread_id  # noqa: E402
from agentic_cli.workflow.deepagents.tools import LONG_RUNNING_NOTE, to_langchain_tools  # noqa: E402
from agentic_cli.workflow.events import EventType, WorkflowEvent  # noqa: E402
from agentic_cli.workflow.permissions import EXEMPT  # noqa: E402
from agentic_cli.workflow.permissions.prompt import ALLOW_ONCE_CHOICE  # noqa: E402
from agentic_cli.workflow.service_registry import JOB_MANAGER, get_service  # noqa: E402
from agentic_cli.workflow.sessions import (  # noqa: E402
    ToolCallRef,
    get_active_tool_call,
    get_active_turn,
    reset_active_tool_call,
    set_active_tool_call,
)
from tests.conftest import MockContext  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI, anthropic_message, openai_response  # noqa: E402
from tests.workflow.deepagents.fakes import ScriptedModels, ai, task  # noqa: E402
from tests.workflow.langchain_fakes import gemini_reply, route, route_google  # noqa: E402

# What the tools below wait on, set by the test that runs them.
_HOLD: dict[str, asyncio.Event | asyncio.Barrier] = {}


def _built(target: str) -> str:
    return f"built {target}"


@register_tool(capabilities=EXEMPT, long_running=True, requires=JOB_MANAGER)
def jr_build(target: str, tool_context=None) -> dict:
    """Builds ``target`` in the background (a long-running exempt test tool).

    Written for ADK: the call id it reads from ``tool_context`` is None here.
    """
    record = get_service(JOB_MANAGER).submit(
        tool="jr_build", backend="inprocess", spec={"target": _built, "args": (target,)},
        resume_on_complete=True, call_id=getattr(tool_context, "function_call_id", None),
    )
    return {"success": True, "job_id": record.job_id, "status": "pending"}


@register_tool(capabilities=EXEMPT, long_running=True, requires=JOB_MANAGER)
async def jr_build_together(target: str) -> dict:
    """Builds ``target`` once its sibling call is running too (a long-running exempt test tool)."""
    await asyncio.wait_for(_HOLD["together"].wait(), timeout=5)
    record = get_service(JOB_MANAGER).submit(
        tool="jr_build_together", backend="inprocess",
        spec={"target": _built, "args": (target,)}, resume_on_complete=True,
    )
    return {"success": True, "job_id": record.job_id, "call": get_active_tool_call().call_id}


@register_tool(capabilities=EXEMPT)
def jr_where() -> dict:
    """Reports the turn and the tool call it runs in (an exempt test tool)."""
    turn, call = get_active_turn(), get_active_tool_call()
    return {
        "success": True,
        "turn": [turn.user_id, turn.session_id] if turn else None,
        "call": [call.call_id, call.name] if call else None,
    }


@register_tool(capabilities=EXEMPT)
async def jr_wait() -> dict:
    """Waits until the test lets it finish (an exempt test tool)."""
    _HOLD["started"].set()
    await _HOLD["release"].wait()
    return {"success": True}


def _jobs_demo():
    """``examples/jobs_demo.py``, loaded once: it registers ``run_shell_job``."""
    module = sys.modules.get("jobs_demo")
    if module is None:
        path = Path(__file__).parents[3] / "examples" / "jobs_demo.py"
        spec = importlib.util.spec_from_file_location("jobs_demo", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["jobs_demo"] = module
        spec.loader.exec_module(module)
    return module


# --- managers and turns -------------------------------------------------------------------


class _Recorder:
    """Stands in for the manager module's structlog logger."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def __getattr__(self, level: str):
        def log(event: str, **fields) -> None:
            self.calls.append((level, event, fields))

        return log

    def events(self) -> list[str]:
        return [event for _, event, _ in self.calls]


class Answers:
    """The input callback: answers each prompt in turn, recording them."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.asked: list = []

    async def __call__(self, request) -> str:
        self.asked.append(request)
        return self.answers.pop(0)


@pytest.fixture
def hermetic(tmp_path, monkeypatch) -> Path:
    """A hermetic cwd and HOME (the job manager keeps its jobs under HOME)."""
    (tmp_path / "home").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path


@pytest.fixture
def ctx(hermetic):
    with MockContext(google_api_key="test-key", session_store="memory") as context:
        yield context


@pytest.fixture
async def cleanup():
    """Clean up every manager a test hands it."""
    managers: list[DeepAgentsWorkflowManager] = []
    yield managers.append
    for manager in managers:
        await manager.cleanup()


def scripted(monkeypatch, **scripts) -> ScriptedModels:
    """Every agent built from now on runs on its scripted model."""
    models = ScriptedModels(**scripts)
    monkeypatch.setattr(manager_module, "build_chat_model", models)
    return models


def _manager(settings, configs, **kwargs) -> DeepAgentsWorkflowManager:
    kwargs.setdefault("model", "gemini-2.5-flash")
    return DeepAgentsWorkflowManager(agent_configs=configs, settings=settings, **kwargs)


async def turn(manager, text: str = "Build it.", *, user: str = "u", session: str = "s") -> list:
    return [event async for event in manager.process(text, user, session_id=session)]


async def resume(manager, record, result=None) -> list:
    return [event async for event in manager.resume_with_job_result(record, result)]


def of(events, kind: EventType, agent: str | None = None) -> list[WorkflowEvent]:
    return [e for e in events
            if e.type is kind and (agent is None or e.metadata.get("agent") == agent)]


def _record(**over) -> SimpleNamespace:
    """A finished job's record, as the CLI hands it to a resume."""
    base = {
        "job_id": "j1", "tool": "jr_build", "name": "jr_build",
        "call_id": "c1", "call_name": "jr_build", "session_id": "s", "user_id": "u",
        "state": JobState.SUCCEEDED, "exit_code": 0, "error": None,
    }
    return SimpleNamespace(**{**base, **over})


def _notice_text(record, result) -> str:
    payload = json.dumps(job_result_payload(record, result), ensure_ascii=False)
    return f"{JOB_NOTICE_HEADER}\n{payload}"


async def _finished(manager, job_id: str):
    """The job's record once it reached a terminal state."""
    for _ in range(250):
        record = manager.job_manager.get(job_id)
        if record.state in TERMINAL_STATES:
            return record
        await asyncio.sleep(0.02)
    raise AssertionError(f"job {job_id} never finished")


def _the_job(manager):
    (record,) = [r for r in manager.job_manager.list() if r.resume_on_complete]
    return record


SOLO = AgentConfig(name="solo", prompt="Build.", tools=[jr_build, jr_where, jr_wait],
                   include_state_tools=False)


# --- the active tool call -------------------------------------------------------------------


def _request(call_id: str = "call-9", name: str = "jr_where") -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": {}, "id": call_id, "type": "tool_call"},
        tool=None,
        state={},
        runtime=SimpleNamespace(tools=[]),
    )


class TestTheActiveToolCall:
    async def test_each_call_runs_as_the_active_tool_call(self):
        seen = []

        async def handler(request):
            seen.append(get_active_tool_call())
            return "result"

        outer = set_active_tool_call(ToolCallRef("task-1", "task"))
        try:
            result = await ActiveToolCallMiddleware().awrap_tool_call(_request(), handler)
            after = get_active_tool_call()
        finally:
            reset_active_tool_call(outer)

        assert result == "result"
        assert seen == [ToolCallRef("call-9", "jr_where")]
        assert after == ToolCallRef("task-1", "task"), "the outer call was not restored"
        assert get_active_tool_call() is None

    async def test_a_raising_call_is_reset_too(self):
        async def handler(request):
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await ActiveToolCallMiddleware().awrap_tool_call(_request(), handler)

        assert get_active_tool_call() is None

    def test_the_sync_path_too(self):
        seen = []

        def handler(request):
            seen.append(get_active_tool_call())
            raise ValueError("bad")

        with pytest.raises(ValueError):
            ActiveToolCallMiddleware().wrap_tool_call(_request("call-3"), handler)

        assert seen == [ToolCallRef("call-3", "jr_where")]
        assert get_active_tool_call() is None

    async def test_two_parallel_long_running_calls_get_their_own_ids(
        self, ctx, monkeypatch, cleanup
    ):
        # Both calls are in flight before either submits its job.
        _HOLD["together"] = asyncio.Barrier(2)
        scripted(monkeypatch, solo=[
            ai(calls=[("jr_build_together", {"target": "a"}, "call-a"),
                      ("jr_build_together", {"target": "b"}, "call-b")]),
            ai("Both started."),
        ])
        manager = _manager(ctx.settings, [AgentConfig(
            name="solo", prompt="Build.", tools=[jr_build_together], include_state_tools=False,
        )])
        cleanup(manager)

        events = await turn(manager)

        results = {e.metadata["tool_call_id"]: e.metadata["result"]
                   for e in of(events, EventType.TOOL_RESULT)}
        assert {call: r["call"] for call, r in results.items()} == {
            "call-a": "call-a", "call-b": "call-b"
        }
        jobs = {r.job_id: r for r in manager.job_manager.list()}
        assert {
            (jobs[r["job_id"]].call_id, jobs[r["job_id"]].call_name)
            for r in results.values()
        } == {("call-a", "jr_build_together"), ("call-b", "jr_build_together")}

    async def test_a_sync_tool_on_the_worker_thread_sees_its_own_call(
        self, ctx, monkeypatch, cleanup
    ):
        scripted(monkeypatch, solo=[ai(calls=[("jr_where", {}, "call-w")]), ai("Done.")])
        manager = _manager(ctx.settings, [SOLO])
        cleanup(manager)

        events = await turn(manager)

        (result,) = of(events, EventType.TOOL_RESULT)
        assert result.metadata["result"]["call"] == ["call-w", "jr_where"]


# --- a long-running call ---------------------------------------------------------------------


class TestALongRunningCall:
    def test_the_demos_run_shell_job_loads_with_tool_context_hidden(self):
        run_shell_job = _jobs_demo().run_shell_job

        (tool,) = to_langchain_tools([run_shell_job])

        assert tool.func is run_shell_job
        assert set(tool.tool_call_schema.model_json_schema()["properties"]) == {
            "command", "name", "cwd"
        }
        assert tool.description.endswith(LONG_RUNNING_NOTE)

    async def test_the_demos_job_is_linked_to_its_call_and_the_turn_ends(
        self, ctx, monkeypatch, cleanup
    ):
        """``examples/jobs_demo.py`` unchanged: gated, pending at once, linked to its call."""
        demo = _jobs_demo()
        (config,) = demo.AGENT_CONFIGS
        models = scripted(monkeypatch, **{config.name: [
            ai(calls=[("run_shell_job", {"command": "echo built"}, "call-sh")]),
            ai("Started it; I will report back."),
        ]})
        manager = _manager(ctx.settings, demo.AGENT_CONFIGS)
        cleanup(manager)
        answers = Answers(ALLOW_ONCE_CHOICE)
        manager.set_input_callback(answers)

        events = await turn(manager, user="alice", session="ops-1")

        assert [a.tool_name for a in answers.asked] == ["run_shell_job"]
        assert of(events, EventType.ERROR) == []
        (result,) = of(events, EventType.TOOL_RESULT)
        assert result.metadata["success"] is True
        assert [e.content for e in of(events, EventType.TEXT)] == [
            "Started it; I will report back."
        ]
        assert len(models.chats[config.name].requests) == 2  # nothing waits for the job
        record = manager.job_manager.get(result.metadata["result"]["job_id"])
        assert (record.call_id, record.call_name) == ("call-sh", "run_shell_job")
        assert (record.session_id, record.user_id) == ("ops-1", "alice")
        assert record.resume_on_complete is True
        await _finished(manager, record.job_id)


# --- the resume turn -------------------------------------------------------------------------


class TestTheResumeTurn:
    async def test_the_root_is_told_in_one_notification(self, ctx, monkeypatch, cleanup):
        models = scripted(monkeypatch, solo=[
            ai(calls=[("jr_build", {"target": "docs"}, "call-b")]),
            ai("Started the build."),
            ai("The build finished: built docs."),
        ])
        manager = _manager(ctx.settings, [SOLO])
        cleanup(manager)
        await turn(manager, "Build the docs.")
        record = await _finished(manager, _the_job(manager).job_id)
        assert (record.call_id, record.call_name) == ("call-b", "jr_build")

        events = await resume(manager, record)  # the result comes from the JobManager

        assert [e.type for e in events] == [EventType.TEXT, EventType.LLM_USAGE]
        assert events[0].content == "The build finished: built docs."
        conversation = models.chats["solo"].conversation(2)
        # The conversation so far (the call, its pending result, the answer), then it.
        assert [type(m).__name__ for m in conversation] == [
            "HumanMessage", "AIMessage", "ToolMessage", "AIMessage", "HumanMessage"
        ]
        assert json.loads(conversation[2].text)["status"] == "pending"
        notice = conversation[-1]
        assert type(notice) is HumanMessage
        assert notice.content == _notice_text(record, "built docs")
        assert notice.content.startswith(
            "[Automatic job notification — not written by the user]\n{"
        )
        assert notice.response_metadata == {
            "agentic_cli": {"kind": "job_result", "job_id": record.job_id}
        }

    async def test_a_result_it_is_given_is_the_one_sent(self, ctx, monkeypatch, cleanup):
        models = scripted(monkeypatch, solo=[ai("Hello."), ai("Noted.")])
        manager = _manager(ctx.settings, [SOLO])
        cleanup(manager)
        await turn(manager, "Hi.")
        record = _record(state=JobState.FAILED, exit_code=3, error="exit 3")

        await resume(manager, record, "the log tail")

        assert models.chats["solo"].conversation(1)[-1].content == _notice_text(
            record, "the log tail"
        )

    async def test_a_result_in_any_script_reaches_the_model_as_written(
        self, ctx, monkeypatch, cleanup
    ):
        models = scripted(monkeypatch, solo=[ai("Hello."), ai("Noted.")])
        manager = _manager(ctx.settings, [SOLO])
        cleanup(manager)
        await turn(manager, "Hi.")

        await resume(manager, _record(), "Сборка готова: 完成")

        notice = models.chats["solo"].conversation(1)[-1].content
        assert "Сборка готова: 完成" in notice
        assert "\\u" not in notice

    async def test_it_is_a_turn_in_the_records_conversation(self, ctx, monkeypatch, cleanup):
        models = scripted(monkeypatch, solo=[
            ai("Alice, hello."), ai("Bob, hello."),
            ai(calls=[("jr_where", {}, "call-w")]), ai("Told."),
            ai("Yes, it finished."),
        ])
        manager = _manager(ctx.settings, [SOLO])
        cleanup(manager)
        await turn(manager, "Hi.", user="alice", session="s")
        await turn(manager, "Hi.", user="bob", session="s")
        record = _record(user_id="alice", session_id="s")

        events = await resume(manager, record, "ok")
        after = await turn(manager, "Did it finish?", user="alice", session="s")

        (where,) = of(events, EventType.TOOL_RESULT)
        assert where.metadata["result"]["turn"] == ["alice", "s"]
        assert get_active_turn() is None
        # Bob's conversation is not Alice's, and the notification stays in hers.
        chat = models.chats["solo"]
        assert [m.text for m in chat.conversation(2)] == [
            "Hi.", "Alice, hello.", _notice_text(record, "ok")
        ]
        assert _notice_text(record, "ok") in [m.text for m in chat.conversation(4)]
        assert [e.content for e in of(after, EventType.TEXT)] == ["Yes, it finished."]

    async def test_it_waits_for_a_running_turn(self, ctx, monkeypatch, cleanup):
        _HOLD.update(started=asyncio.Event(), release=asyncio.Event())
        models = scripted(monkeypatch, solo=[
            ai(calls=[("jr_wait", {}, "call-w")]), ai("Waited."), ai("Resumed."),
        ])
        manager = _manager(ctx.settings, [SOLO])
        cleanup(manager)

        user_turn = asyncio.create_task(turn(manager, "Wait."))
        try:
            await asyncio.wait_for(_HOLD["started"].wait(), timeout=5)
            resumed = asyncio.create_task(resume(manager, _record(), "ok"))
            await asyncio.sleep(0.05)
            calls_meanwhile = len(models.chats["solo"].requests)
        finally:
            _HOLD["release"].set()  # however this ends, the turn may finish
        await asyncio.wait_for(user_turn, timeout=5)
        events = await asyncio.wait_for(resumed, timeout=5)

        assert calls_meanwhile == 1, "the resume ran during a turn"
        assert [e.content for e in of(events, EventType.TEXT)] == ["Resumed."]
        assert [m.text for m in models.chats["solo"].conversation(2)][-2:] == [
            "Waited.", _notice_text(_record(), "ok")
        ]

    async def test_pending_notices_open_it(self, ctx, monkeypatch, cleanup):
        scripted(monkeypatch, solo=[ai("Hi."), ai("Noted."), ai("Again.")])
        manager = _manager(ctx.settings, [SOLO])
        cleanup(manager)
        await turn(manager, "Hi.")
        notice = WorkflowEvent.error("Server x is unavailable.", error_code="mcp_unavailable",
                                     recoverable=True)
        manager._pending_notices.append(notice)

        events = await resume(manager, _record(), "ok")
        after = await turn(manager, "Again.")

        assert events[0] is notice
        assert of(after, EventType.ERROR) == []

    async def test_every_event_passes_the_event_hook(self, ctx, monkeypatch, cleanup):
        scripted(monkeypatch, solo=[ai("Hi."), ai("Noted.")])
        seen = []

        def hook(event: WorkflowEvent) -> WorkflowEvent | None:
            seen.append(event.type)
            return None if event.type is EventType.ERROR else event

        manager = _manager(ctx.settings, [SOLO], on_event=hook)
        cleanup(manager)
        await turn(manager, "Hi.")
        seen.clear()
        manager._pending_notices.append(WorkflowEvent.error(
            "Server x is unavailable.", error_code="mcp_unavailable", recoverable=True
        ))

        events = await resume(manager, _record(), "ok")

        assert seen == [EventType.ERROR, EventType.TEXT, EventType.LLM_USAGE]
        assert [e.type for e in events] == [EventType.TEXT, EventType.LLM_USAGE]

    async def test_raw_logging_records_it(self, ctx, monkeypatch, cleanup):
        scripted(monkeypatch, solo=[ai("Hi."), ai("Noted.")])
        ctx.settings.raw_llm_logging = True
        manager = _manager(ctx.settings, [SOLO])
        cleanup(manager)
        await turn(manager, "Hi.")

        events = await resume(manager, _record(), "ok")

        (request,) = of(events, EventType.LLM_REQUEST)
        assert request.metadata["messages"][-1] == {
            "role": "user", "content": _notice_text(_record(), "ok")
        }
        assert len(of(events, EventType.LLM_RESPONSE)) == 1

    @pytest.mark.parametrize("provider", ["claude", "gemini", "openai"])
    async def test_no_provider_is_sent_the_mark(self, hermetic, monkeypatch, provider):
        """The notification reaches the model as the plain user text it is."""
        record = _record()
        text = _notice_text(record, "ok")
        if provider == "claude":
            api = FakeAPI(anthropic_message({"type": "text", "text": "Hi."},
                                            model="claude-opus-4-7"))
            route(monkeypatch, anthropic, "AsyncClient", api)
            model = "anthropic:claude-opus-4-7"
        elif provider == "gemini":
            api = FakeAPI(gemini_reply({"text": "Hi."}))
            route_google(monkeypatch, api)
            model = "google:gemini-2.5-flash"
        else:
            api = FakeAPI(openai_response("Hi."))
            route(monkeypatch, openai, "AsyncOpenAI", api)
            model = "openai:gpt-5.5"
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        for variable in ("GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.setenv(variable, "a-test-key")
        manager = DeepAgentsWorkflowManager(
            agent_configs=[SOLO],
            settings=BaseSettings(default_model=model, session_store="memory"),
        )
        try:
            await turn(manager, "Hi.")
            events = await resume(manager, record, "ok")
        finally:
            await manager.cleanup()

        assert of(events, EventType.ERROR) == []
        body = api.bodies[-1]
        if provider == "claude":
            assert body["messages"][-1] == {"role": "user", "content": text}
        elif provider == "gemini":
            assert body["contents"][-1] == {"role": "user", "parts": [{"text": text}]}
        else:
            assert body["input"][-1] == {"type": "message", "role": "user", "content": text}


# --- no conversation, no turn -----------------------------------------------------------------


class TestNoConversationNoTurn:
    @pytest.mark.parametrize("missing", ["session_id", "user_id"])
    async def test_a_record_without_its_conversation_runs_nothing(
        self, ctx, monkeypatch, cleanup, missing
    ):
        models = scripted(monkeypatch, solo=[ai("Hi.")])
        recorder = _Recorder()
        monkeypatch.setattr(manager_module, "logger", recorder)
        manager = _manager(ctx.settings, [SOLO])
        cleanup(manager)
        await turn(manager, "Hi.")
        record = _record(**{missing: None})

        assert await manager.can_resume(record) is False
        assert await resume(manager, record, "ok") == []
        assert len(models.chats["solo"].requests) == 1
        assert "job_resume_missing_ids" in recorder.events()

    @pytest.mark.parametrize("whose", ["deleted", "never-there", "another-users"])
    async def test_a_conversation_that_is_not_there_is_not_started(
        self, ctx, monkeypatch, cleanup, whose
    ):
        models = scripted(monkeypatch, solo=[ai("Hi.")])
        recorder = _Recorder()
        monkeypatch.setattr(manager_module, "logger", recorder)
        manager = _manager(ctx.settings, [SOLO])
        cleanup(manager)
        await turn(manager, "Hi.", user="u", session="s")
        record = {
            "deleted": _record(),
            "never-there": _record(session_id="other"),
            "another-users": _record(user_id="mallory"),
        }[whose]
        if whose == "deleted":
            assert await manager.delete_session("s", user_id="u") is True

        assert await manager.can_resume(record) is False
        assert await resume(manager, record, "ok") == []

        assert len(models.chats["solo"].requests) == 1, "a model was called"
        assert "job_resume_session_missing" in recorder.events()
        # No thread was started for it, and the user's own is as it was.
        assert await manager.session_exists(record.session_id, user_id=record.user_id) is False
        assert await manager.list_sessions(user_id="mallory") == []
        assert [
            (entry.session_id, entry.turns)
            for entry in await manager.list_sessions(user_id="u")
        ] == ([] if whose == "deleted" else [("s", 1)])

    async def test_another_apps_conversation_is_not_this_ones(self, hermetic, monkeypatch):
        """Two apps on one sqlite file: a job of one never resumes in the other."""
        models = scripted(monkeypatch, solo=[ai("Hi.")])
        with MockContext(google_api_key="test-key", session_store="sqlite") as context:
            ours = _manager(context.settings, [SOLO], app_name="ours")
            theirs = _manager(context.settings, [SOLO], app_name="theirs")
            try:
                await turn(ours, "Hi.")
                await theirs.initialize_services()
                record = _record()

                assert await ours.can_resume(record) is True
                assert await theirs.can_resume(record) is False
                assert await resume(theirs, record, "ok") == []
                assert await theirs.list_sessions(user_id="u") == []
                assert len(models.chats["solo"].requests) == 1
                assert await ours.recent_messages("s", user_id="u") == [
                    {"role": "user", "content": "Hi."}, {"role": "assistant", "content": "Hi."}
                ]
            finally:
                await ours.cleanup()
                await theirs.cleanup()


# --- can_resume ------------------------------------------------------------------------------


class TestCanResume:
    async def test_on_sqlite_a_fresh_manager_resumes(self, hermetic, monkeypatch):
        models = scripted(monkeypatch, solo=[ai("Started."), ai("It finished.")])
        with MockContext(google_api_key="test-key", session_store="sqlite") as context:
            first = _manager(context.settings, [SOLO])
            try:
                await turn(first, "Start it.")
            finally:
                await first.cleanup()

            second = _manager(context.settings, [SOLO])
            try:
                await second.initialize_services()
                assert await second.can_resume(_record()) is True
                events = await resume(second, _record(), "ok")
            finally:
                await second.cleanup()

        assert [e.content for e in of(events, EventType.TEXT)] == ["It finished."]
        assert [m.text for m in models.chats["solo"].conversation(1)] == [
            "Start it.", "Started.", _notice_text(_record(), "ok")
        ]

    async def test_on_memory_a_fresh_manager_does_not(self, ctx, monkeypatch):
        scripted(monkeypatch, solo=[ai("Started.")])
        first = _manager(ctx.settings, [SOLO])
        try:
            await turn(first, "Start it.")
            assert await first.can_resume(_record()) is True
        finally:
            await first.cleanup()

        second = _manager(ctx.settings, [SOLO])
        try:
            await second.initialize_services()
            assert await second.can_resume(_record()) is False
        finally:
            await second.cleanup()

    async def test_after_the_session_is_deleted_it_does_not(self, ctx, monkeypatch, cleanup):
        scripted(monkeypatch, solo=[ai("Started.")])
        manager = _manager(ctx.settings, [SOLO])
        cleanup(manager)
        await turn(manager, "Start it.")
        assert await manager.can_resume(_record()) is True

        await manager.delete_session("s", user_id="u")

        assert await manager.can_resume(_record()) is False

    async def test_the_call_id_is_not_needed(self, ctx, monkeypatch, cleanup):
        """The root is told in a message of its own; no pending call is answered."""
        scripted(monkeypatch, solo=[ai("Started."), ai("Noted.")])
        manager = _manager(ctx.settings, [SOLO])
        cleanup(manager)
        await turn(manager, "Start it.")
        record = _record(call_id=None, call_name=None)

        assert await manager.can_resume(record) is True
        assert [e.content for e in of(await resume(manager, record, "ok"), EventType.TEXT)] == [
            "Noted."
        ]

    async def test_before_initialization_it_does_not(self, ctx):
        assert await _manager(ctx.settings, [SOLO]).can_resume(_record()) is False


# --- a sub-agent's job ------------------------------------------------------------------------


class TestASubAgentsJob:
    async def test_it_reports_to_the_coordinator(self, ctx, monkeypatch, cleanup):
        models = scripted(
            monkeypatch,
            lead=[
                ai(calls=[task("builder", "Start the docs build.", "call-t")]),
                ai("The builder started it."),
                ai("The docs are built."),
            ],
            builder=[ai(calls=[("jr_build", {"target": "docs"}, "call-b")]), ai("Started.")],
        )
        manager = _manager(ctx.settings, [
            AgentConfig(name="lead", prompt="Lead.", sub_agents=["builder"]),
            AgentConfig(name="builder", prompt="Build.", description="Builds.",
                        tools=[jr_build], include_state_tools=False),
        ])
        cleanup(manager)
        await turn(manager, "Build the docs.")
        record = await _finished(manager, _the_job(manager).job_id)

        events = await resume(manager, record)

        # The sub-agent's own call started the job, not the coordinator's task.
        assert (record.call_id, record.call_name) == ("call-b", "jr_build")
        assert [e.content for e in of(events, EventType.TEXT)] == ["The docs are built."]
        assert {e.metadata["agent"] for e in events} == {"lead"}
        assert len(models.chats["builder"].requests) == 2
        assert models.chats["lead"].conversation(2)[-1].content == _notice_text(
            record, "built docs"
        )


# --- recent_messages -------------------------------------------------------------------------


class TestRecentMessages:
    async def test_the_notification_is_left_out_and_kept_for_the_model(
        self, ctx, monkeypatch, cleanup
    ):
        scripted(monkeypatch, solo=[ai("Started."), ai("It finished.")])
        manager = _manager(ctx.settings, [SOLO])
        cleanup(manager)
        await turn(manager, "Start it.")
        await resume(manager, _record(), "ok")

        assert await manager.recent_messages("s", user_id="u") == [
            {"role": "user", "content": "Start it."},
            {"role": "assistant", "content": "Started."},
            {"role": "assistant", "content": "It finished."},
        ]
        saved = await manager._checkpointer.aget_tuple({"configurable": {
            "thread_id": thread_id(manager.session_ref("s", "u")), "checkpoint_ns": "",
        }})
        assert [m.text for m in saved.checkpoint["channel_values"]["messages"]] == [
            "Start it.", "Started.", _notice_text(_record(), "ok"), "It finished."
        ]

    async def test_the_mark_survives_a_fresh_manager_on_sqlite(self, hermetic, monkeypatch):
        scripted(monkeypatch, solo=[ai("Started."), ai("It finished.")])
        with MockContext(google_api_key="test-key", session_store="sqlite") as context:
            first = _manager(context.settings, [SOLO])
            try:
                await turn(first, "Start it.")
                await resume(first, _record(), "ok")
            finally:
                await first.cleanup()

            second = _manager(context.settings, [SOLO])
            try:
                await second.initialize_services()
                recent = await second.recent_messages("s", user_id="u")
            finally:
                await second.cleanup()

        assert [m["content"] for m in recent] == ["Start it.", "Started.", "It finished."]

    async def test_a_user_message_like_it_is_not_left_out(self, ctx, monkeypatch, cleanup):
        """Only the mark decides: a user who types the header is still the user."""
        scripted(monkeypatch, solo=[ai("Hm.")])
        manager = _manager(ctx.settings, [SOLO])
        cleanup(manager)
        typed = _notice_text(_record(), "ok")

        await turn(manager, typed)

        assert await manager.recent_messages("s", user_id="u") == [
            {"role": "user", "content": typed}, {"role": "assistant", "content": "Hm."}
        ]
