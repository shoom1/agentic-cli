"""One turn protocol, in the base.

``BaseWorkflowManager._turn(session_id, user_id)`` is how every turn-shaped
entry point starts: it names the conversation in the log context, enters
turn admission (initialize outside the turn lock, take it, re-check the
backend, retry once) and the workflow context (the settings, the services
and the active turn), and yields the turn's ``SessionRef``. Each backend
wrote that skeleton itself, six times over, so a change to how a turn starts
had to be made in every copy. A backend written to the base's hooks gets it
whole, and ADK's and Deep Agents' turns enter through it.

A finished job's resume is the base's too: ``resume_with_job_result`` checks
the record names its conversation, enters ``_turn``, checks the conversation
is there (``_conversation_exists``), reads the job's result on a worker
thread (``JobManager.result`` takes the jobs directory's cross-process lock,
so on the event loop it froze the prompt while another CLI held it), builds
the notice (``tools/jobs/resume.py``) and hands it to the backend's
``_deliver_job_notice``. ``supports_job_resume`` is derived from the hook,
as ``supports_compact`` is.

Offline: catalogs never list (``tests/conftest.py``), and no model is called.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable
from typing import Any

import pytest
import structlog

from agentic_cli.tools.jobs import JobManager, JobState
from agentic_cli.tools.jobs.backends import TERMINAL_STATES
from agentic_cli.tools.jobs.resume import job_notice
from agentic_cli.workflow import base_manager as base_manager_module
from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.config import AgentConfig, AgentGraph, validate_agent_graph
from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.service_registry import JOB_MANAGER
from agentic_cli.workflow.sessions import SessionRef, get_active_turn
from tests.conftest import MockContext
from tests.loop_guard import no_blocking_calls
from tests.workflow.test_turn_serialization import _job_record, turn_harness

# ---------------------------------------------------------------------------
# A backend written to the turn's hooks alone
# ---------------------------------------------------------------------------


class _Stub(BaseWorkflowManager):
    """A backend whose every turn holds until the test lets it go.

    Each turn records, as it starts and as it ends, what it is and the active
    turn it sees, and the log context it runs under.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.engine: object | None = None
        self.builds = 0
        self.seen: list[tuple[str, SessionRef | None]] = []
        self.logged_as: list[dict[str, Any]] = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.conversations: set[SessionRef] = set()
        self.told: list[tuple[str, Any, Any, SessionRef]] = []

    @property
    def backend_type(self) -> str:
        return "stub"

    def _get_state_tools(self) -> list[Callable]:
        return []

    def _validate_agent_graph(self) -> AgentGraph:
        return validate_agent_graph(self._agent_configs, backend=self.backend_type)

    async def _do_initialize(self) -> None:
        self.builds += 1
        self.engine = object()

    def _backend_ready(self) -> bool:
        return self._initialized and self.engine is not None

    def _detach_backend(self):
        self.engine = None
        return []

    async def _hold(self, what: str) -> None:
        self.seen.append((f"start {what}", get_active_turn()))
        self.logged_as.append(structlog.contextvars.get_contextvars())
        self.entered.set()
        await self.release.wait()
        self.seen.append((f"end {what}", get_active_turn()))

    async def process(self, message, user_id, session_id=None):
        async with self._turn(session_id, user_id) as ref:
            await self._hold(f"turn {message}")
            yield WorkflowEvent.text(message, ref.session_id)

    async def _conversation_exists(self, ref: SessionRef) -> bool:
        return ref in self.conversations

    async def _deliver_job_notice(self, notice, record, result, ref):
        self.told.append((notice, record, result, ref))
        await self._hold(f"resume {record.job_id}")
        yield WorkflowEvent.text(notice, ref.session_id)


class _Plain(BaseWorkflowManager):
    """A backend with turns and nothing optional."""

    @property
    def backend_type(self) -> str:
        return "plain"

    def _get_state_tools(self) -> list[Callable]:
        return []

    async def _do_initialize(self) -> None:
        return None

    async def process(self, message, user_id, session_id=None):
        async with self._turn(session_id, user_id) as ref:
            yield WorkflowEvent.text(message, ref.session_id)


class _OwnResume(_Plain):
    """A backend written before the template: its own resume, no hook."""

    async def resume_with_job_result(self, record, result=None):
        yield WorkflowEvent.text(f"resumed {record.job_id}")


@pytest.fixture
def ctx():
    with MockContext(google_api_key="test-key") as context:
        yield context


@pytest.fixture
def stub(ctx):
    return _Stub([AgentConfig(name="a", prompt="p")], settings=ctx.settings)


async def _reply(manager, message: str = "hi", user: str = "u", session: str | None = "s"):
    return [event.content async for event in manager.process(message, user, session)]


def _ref(manager: BaseWorkflowManager, user: str = "u", session: str = "s") -> SessionRef:
    return SessionRef(manager.app_name, user, session)


class TestTheTurnEntry:
    async def test_a_turn_runs_as_its_conversation(self, stub):
        assert await _reply(stub, user="alice", session="s1") == ["hi"]

        ref = _ref(stub, "alice", "s1")
        assert stub.seen == [("start turn hi", ref), ("end turn hi", ref)]
        assert get_active_turn() is None

    async def test_without_a_session_it_is_the_managers_current_one(self, stub):
        stub.session_id = "current"

        await _reply(stub, user="alice", session=None)

        assert stub.seen[0] == ("start turn hi", _ref(stub, "alice", "current"))

    async def test_its_log_lines_name_the_conversation(self, stub):
        await _reply(stub, user="alice", session="s1")

        assert stub.logged_as[0]["session_id"] == "s1"
        assert stub.logged_as[0]["user_id"] == "alice"

    async def test_it_initializes_the_manager_and_again_after_a_cleanup(self, stub):
        await _reply(stub)
        assert stub.is_initialized and stub.builds == 1

        await stub.cleanup()
        await _reply(stub)

        assert stub.builds == 2
        await stub.cleanup()

    async def test_turns_run_one_at_a_time(self, stub):
        stub.release.clear()
        first = asyncio.create_task(_reply(stub, "first"))
        await asyncio.wait_for(stub.entered.wait(), timeout=2)

        second = asyncio.create_task(_reply(stub, "second"))
        await asyncio.sleep(0.05)
        assert [what for what, _ in stub.seen] == ["start turn first"], "turns overlapped"

        stub.release.set()
        assert await first == ["first"] and await second == ["second"]
        assert [what for what, _ in stub.seen] == [
            "start turn first", "end turn first", "start turn second", "end turn second",
        ]

    async def test_a_cancelled_turn_frees_the_manager(self, stub):
        stub.release.clear()
        first = asyncio.create_task(_reply(stub, "first"))
        await asyncio.wait_for(stub.entered.wait(), timeout=2)

        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert get_active_turn() is None

        stub.release.set()
        assert await asyncio.wait_for(_reply(stub, "second"), timeout=2) == ["second"]


# ---------------------------------------------------------------------------
# ADK and Deep Agents enter through it
# ---------------------------------------------------------------------------


@pytest.fixture(params=["adk", "deepagents"])
def harness(request):
    h, context = turn_harness(request.param)
    h.release.set()
    try:
        yield h
    finally:
        context.__exit__(None, None, None)


def _record_turns(manager: BaseWorkflowManager) -> list[SessionRef]:
    """Each turn ``manager`` enters through ``_turn``, by its conversation."""
    entered: list[SessionRef] = []
    turn = manager._turn

    @contextlib.asynccontextmanager
    async def recorded(session_id=None, user_id=None):
        async with turn(session_id, user_id) as ref:
            entered.append(ref)
            yield ref

    manager._turn = recorded
    return entered


class TestEachBackendEntersThroughIt:
    async def test_a_turn(self, harness):
        entered = _record_turns(harness.manager)

        assert await harness.drain("sess-a") == ["sess-a"]

        assert entered == [_ref(harness.manager, "u", "sess-a")]

    async def test_a_jobs_resume(self, harness):
        entered = _record_turns(harness.manager)

        events = [
            e.content async for e in harness.manager.resume_with_job_result(_job_record(), "ok")
        ]

        assert events == ["sess-r"]
        assert entered == [_ref(harness.manager, "u", "sess-r")]


# ---------------------------------------------------------------------------
# Job resume
# ---------------------------------------------------------------------------


def _job(user: str = "u", session: str | None = "s", **over: Any) -> Any:
    record = _job_record()
    record.user_id, record.session_id = user, session
    for name, value in over.items():
        setattr(record, name, value)
    return record


async def _resume(manager, record, result: Any = None) -> list[str]:
    return [
        event.content async for event in manager.resume_with_job_result(record, result)
    ]


class _Recorder:
    """Stands in for a module's structlog logger."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def __getattr__(self, level: str):
        return lambda event, **fields: self.events.append(event)


@pytest.fixture
def jobs(tmp_path):
    manager = JobManager(base_dir=tmp_path / "jobs")
    yield manager
    manager.close()


def _finished_job(jobs: JobManager, user: str, session: str):
    """A job of that conversation that ran to its end, with its result."""
    record = jobs.submit(
        tool="run_shell_job", backend="subprocess", spec={"command": "echo built"},
        name="build", resume_on_complete=True, call_id="c1",
        session_id=session, user_id=user,
    )
    deadline = time.monotonic() + 10
    while jobs.get(record.job_id).state not in TERMINAL_STATES:
        assert time.monotonic() < deadline, "the job never finished"
        time.sleep(0.02)
    assert jobs.get(record.job_id).state is JobState.SUCCEEDED
    return jobs.get(record.job_id)


class TestTheResumeTemplate:
    async def test_the_root_is_told_the_notice_in_a_turn_of_the_jobs_conversation(self, stub):
        record = _job(user="alice", session="s1")
        ref = _ref(stub, "alice", "s1")
        stub.conversations.add(ref)

        events = await _resume(stub, record, "built docs")

        notice = job_notice(record, "built docs")
        assert events == [notice]
        assert stub.told == [(notice, record, "built docs", ref)]
        assert stub.seen == [("start resume j1", ref), ("end resume j1", ref)]
        assert stub.logged_as[0]["session_id"] == "s1"
        assert get_active_turn() is None

    async def test_it_waits_for_a_running_turn(self, stub):
        stub.conversations.add(_ref(stub))
        stub.release.clear()
        turn = asyncio.create_task(_reply(stub, "first"))
        await asyncio.wait_for(stub.entered.wait(), timeout=2)

        resume = asyncio.create_task(_resume(stub, _job(), "ok"))
        await asyncio.sleep(0.05)
        assert [what for what, _ in stub.seen] == ["start turn first"], "a resume ran in a turn"

        stub.release.set()
        await turn
        await resume
        assert [what for what, _ in stub.seen] == [
            "start turn first", "end turn first", "start resume j1", "end resume j1",
        ]

    @pytest.mark.parametrize("missing", ["session_id", "user_id"])
    async def test_a_record_without_its_conversation_runs_nothing(
        self, stub, monkeypatch, missing
    ):
        recorder = _Recorder()
        monkeypatch.setattr(base_manager_module, "logger", recorder)

        assert await _resume(stub, _job(**{missing: None}), "ok") == []

        assert stub.told == []
        assert stub.is_initialized is False, "it started a turn"
        assert "job_resume_missing_ids" in recorder.events

    async def test_a_conversation_that_is_not_there_is_not_told(self, stub, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr(base_manager_module, "logger", recorder)

        assert await _resume(stub, _job(), "ok") == []

        assert stub.told == []
        assert "job_resume_session_missing" in recorder.events
        assert get_active_turn() is None

    async def test_without_a_result_the_job_managers_is_read(self, stub, jobs):
        record = _finished_job(jobs, "u", "s")
        stub.conversations.add(_ref(stub))
        await stub.initialize_services()
        stub._services[JOB_MANAGER] = jobs

        with no_blocking_calls():
            events = await _resume(stub, record)

        assert events == [job_notice(record, jobs.result(record.job_id))]
        assert "built" in events[0]
        await stub.cleanup()

    async def test_a_backend_without_it_raises_before_any_turn(self, ctx):
        manager = _Plain([AgentConfig(name="a", prompt="p")], settings=ctx.settings)

        with pytest.raises(NotImplementedError, match="supports_job_resume"):
            await _resume(manager, _job(), "ok")
        assert manager.is_initialized is False


class TestSupportsJobResume:
    def test_it_is_derived_from_the_hook(self, ctx, stub):
        plain = _Plain([AgentConfig(name="a", prompt="p")], settings=ctx.settings)

        assert stub.supports_job_resume is True
        assert plain.supports_job_resume is False

    def test_a_backend_with_its_own_resume_supports_it(self, ctx):
        manager = _OwnResume([AgentConfig(name="a", prompt="p")], settings=ctx.settings)

        assert manager.supports_job_resume is True

    def test_adk_and_deep_agents_support_it_and_langgraph_does_not(self, ctx):
        pytest.importorskip("deepagents")
        pytest.importorskip("langgraph")
        from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager
        from agentic_cli.workflow.deepagents import DeepAgentsWorkflowManager
        from agentic_cli.workflow.langgraph.manager import LangGraphWorkflowManager

        configs = [AgentConfig(name="a", prompt="p")]
        assert GoogleADKWorkflowManager(configs, settings=ctx.settings).supports_job_resume
        assert DeepAgentsWorkflowManager(configs, settings=ctx.settings).supports_job_resume
        with pytest.warns(DeprecationWarning):
            langgraph = LangGraphWorkflowManager(configs, settings=ctx.settings)
        assert langgraph.supports_job_resume is False


class TestEachBackendReadsTheResultOffTheLoop:
    async def test_a_resume_without_its_result(self, harness, jobs):
        record = _finished_job(jobs, "u", "sess-r")
        harness.manager._services[JOB_MANAGER] = jobs

        with no_blocking_calls():
            events = [
                e.content async for e in harness.manager.resume_with_job_result(record)
            ]

        assert events == ["sess-r"]
        assert harness.told == [job_notice(record, jobs.result(record.job_id))]
