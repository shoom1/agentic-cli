"""A manager's services can be handed to a manager of the other backend.

A conversation belongs to one backend, and a run's services belong to the
run. An orchestrator change built a new manager whose services were all new:
the in-process jobs ended, the sandbox kernels restarted, the run's
permission grants were forgotten, the embedding model loaded again, and a
second job manager ran over the jobs directory while the first was live.

A manager now lends its services (``lend_services``) to a new manager, which
builds only what is missing (``initialize_services(services=loan)``). When
the caller gives them over (``give_services``), the new manager owns them,
every reference they hold to their owner moves to it, and the old manager's
release closes none of them. A new manager that fails to start, is cancelled
while it starts, or is abandoned closes nothing it borrowed: the old manager
keeps working with the same objects, and closes them itself.

Real ADK and Deep Agents managers, from one to the other and back, offline:
catalogs never list and no model is called.
"""

from __future__ import annotations

import asyncio
import functools
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("google.adk")
pytest.importorskip("deepagents")

from agentic_cli.memory import MockEmbeddingService  # noqa: E402
from agentic_cli.tools import job_list  # noqa: E402
from agentic_cli.tools.jobs.manager import JobState  # noqa: E402
from agentic_cli.tools.knowledge_tools import kb_search  # noqa: E402
from agentic_cli.tools.registry import Capability  # noqa: E402
from agentic_cli.tools.sandbox import sandbox_execute  # noqa: E402
from agentic_cli.workflow import memory_services  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.base_manager import BaseWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.deepagents import manager as deepagents_module  # noqa: E402
from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager  # noqa: E402
from agentic_cli.workflow.permissions.prompt import ALLOW_SESSION_CHOICE  # noqa: E402
from agentic_cli.workflow.service_registry import (  # noqa: E402
    EMBEDDING_SERVICE,
    JOB_MANAGER,
    KNOWLEDGE_BASE,
    LLM_SUMMARIZER,
    PERMISSION_ENGINE,
    SANDBOX_MANAGER,
    WORKFLOW,
)
from agentic_cli.workflow.sessions import reset_active_turn, set_active_turn  # noqa: E402
from tests.conftest import MockContext  # noqa: E402
from tests.workflow.deepagents.fakes import ScriptedModels  # noqa: E402

READ = [Capability("filesystem.read", target_arg="path")]

#: What a run holds that a hand-over keeps: the session grants (in the
#: engine), the kernels, the jobs, the knowledge bases and the embedding model.
KEPT = (PERMISSION_ENGINE, SANDBOX_MANAGER, JOB_MANAGER, KNOWLEDGE_BASE, EMBEDDING_SERVICE)

#: Each kept service that holds something to release, and how it releases it.
CLOSERS = {SANDBOX_MANAGER: "cleanup", JOB_MANAGER: "close", KNOWLEDGE_BASE: "close"}

_BACKENDS = {"adk": GoogleADKWorkflowManager, "deepagents": DeepAgentsWorkflowManager}


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path / "project")
    monkeypatch.setattr(deepagents_module, "build_chat_model", ScriptedModels(a=[]))
    with MockContext(
        google_api_key="test-key",
        session_store="memory",
        stateful_executor_backend="local",
        knowledge_base_use_mock=False,
    ) as context:
        yield context


@pytest.fixture
def embeddings(monkeypatch) -> list:
    """Every embedding service built: one stands for one load of the model."""
    made: list = []

    def build(settings):
        made.append(MockEmbeddingService(embedding_dim=16))
        return made[-1]

    monkeypatch.setattr(memory_services, "build_embedding_service", build)
    return made


@pytest.fixture(params=[("adk", "deepagents"), ("deepagents", "adk")], ids=lambda p: "-to-".join(p))
def pair(request) -> tuple[type, type]:
    return _BACKENDS[request.param[0]], _BACKENDS[request.param[1]]


def _manager(cls: type, ctx) -> BaseWorkflowManager:
    return cls(
        agent_configs=[
            AgentConfig(name="a", prompt="p", tools=[sandbox_execute, job_list, kb_search])
        ],
        settings=ctx.settings,
    )


class _Approver:
    """Answers every permission prompt with "Allow for this session"."""

    def __init__(self) -> None:
        self.asked = 0

    async def __call__(self, request) -> str:
        self.asked += 1
        return ALLOW_SESSION_CHOICE


async def _read_allowed(manager: BaseWorkflowManager, path: Path) -> bool:
    engine = manager.services[PERMISSION_ENGINE]
    return (await engine.check("read_file", READ, {"path": str(path)})).allowed


def _count_closes(monkeypatch, services: dict[str, Any]) -> Counter:
    """Count each close of the services that hold something to release."""
    closes: Counter = Counter()
    for key, method in CLOSERS.items():
        real = getattr(services[key], method)

        def counted(real=real, key=key) -> None:
            closes[key] += 1
            real()

        monkeypatch.setattr(services[key], method, counted)
    return closes


class _Run:
    """A started manager with a grant, a kernel's variable and a running job."""

    def __init__(self, manager: BaseWorkflowManager, data: Path) -> None:
        self.manager, self.data = manager, data
        self.approver = _Approver()
        self.release_job = threading.Event()
        self.held: dict[str, Any] = {}
        self.job_id = ""

    async def start(self) -> "_Run":
        await self.manager.initialize_services()
        self.manager.set_input_callback(self.approver)
        assert await _read_allowed(self.manager, self.data)
        assert self.approver.asked == 1
        sandbox = self.manager.services[SANDBOX_MANAGER]
        assert sandbox.execute("answer = 41", session_id="s").success
        jobs = self.manager.services[JOB_MANAGER]
        record = jobs.submit(
            tool="t",
            backend="inprocess",
            spec={"target": functools.partial(self.release_job.wait, 10)},
        )
        self.job_id = record.job_id
        self.held = {key: self.manager.services[key] for key in KEPT}
        return self

    async def job_state(self, jobs) -> JobState:
        return (await asyncio.to_thread(jobs.get, self.job_id)).state

    async def finishes(self, jobs) -> bool:
        self.release_job.set()
        for _ in range(200):
            if await self.job_state(jobs) is JobState.SUCCEEDED:
                return True
            await asyncio.sleep(0.02)
        return False

    async def still_works(self, manager: BaseWorkflowManager) -> None:
        """``manager`` holds the run's own services, and they still work."""
        for key, service in self.held.items():
            assert manager.services[key] is service, key
        manager.set_input_callback(self.approver)
        assert await _read_allowed(manager, self.data)
        assert self.approver.asked == 1, "the session grant was lost"
        result = manager.services[SANDBOX_MANAGER].execute("print(answer + 1)", session_id="s")
        assert "42" in result.stdout, result
        assert await self.job_state(manager.services[JOB_MANAGER]) is JobState.RUNNING


@pytest.fixture
async def run(pair, ctx, tmp_path, embeddings):
    lender_cls, _ = pair
    started = await _Run(_manager(lender_cls, ctx), tmp_path / "data" / "notes.txt").start()
    try:
        yield started
    finally:
        started.release_job.set()
        await started.manager.cleanup()


class TestAHandOverThatIsGiven:
    async def test_the_new_manager_holds_the_same_services(
        self, pair, ctx, run, embeddings, monkeypatch
    ):
        _, borrower_cls = pair
        closes = _count_closes(monkeypatch, run.held)
        new = _manager(borrower_cls, ctx)

        loan = run.manager.lend_services()
        await new.initialize_services(services=loan)
        await run.manager.give_services(loan)
        await run.manager.cleanup()
        try:
            assert closes == Counter(), "the old manager closed what it gave"
            assert len(embeddings) == 1, "the embedding model loaded again"
            await run.still_works(new)
            assert await run.finishes(new.services[JOB_MANAGER])
            # The new manager's own entries are itself.
            assert new.services[WORKFLOW] is new
            assert run.manager.services == {}
        finally:
            await new.cleanup()
        assert closes == Counter({key: 1 for key in CLOSERS})
        await new.cleanup()
        assert closes == Counter({key: 1 for key in CLOSERS})

    async def test_what_held_the_old_manager_now_calls_the_new_one(
        self, pair, ctx, run, monkeypatch
    ):
        _, borrower_cls = pair
        new = _manager(borrower_cls, ctx)
        asked: list[str] = []
        for name, manager in (("old", run.manager), ("new", new)):

            async def generate_simple(prompt, max_tokens=500, name=name):
                asked.append(name)
                return "SUMMARY: A summary.\nCLAIMS:\n- A claim.\n"

            monkeypatch.setattr(manager, "generate_simple", generate_simple)
        ref = new.session_ref("s", "u")
        new._outside_marks.mark(ref, "web_fetch")
        engine = run.held[PERMISSION_ENGINE]
        project_kb = run.held[KNOWLEDGE_BASE].manager("project")

        def per_call() -> dict[str, str]:
            token = set_active_turn(ref)
            try:
                return engine._per_call_now()
            finally:
                reset_active_turn(token)

        loan = run.manager.lend_services()
        await new.initialize_services(services=loan)
        # Lent, not given: they still answer the old manager.
        assert per_call() == {}
        await project_kb.generate_sidecar_payload("Body text.", title="T")
        assert asked == ["old"]

        await run.manager.give_services(loan)
        try:
            assert "memory.write" in per_call()
            await project_kb.generate_sidecar_payload("Body text.", title="T")
            assert asked == ["old", "new"]
            assert engine._workflow is new
        finally:
            await new.cleanup()


class TestAHandOverThatFails:
    async def test_a_failed_start_closes_nothing_it_borrowed(self, pair, ctx, run, monkeypatch):
        _, borrower_cls = pair
        closes = _count_closes(monkeypatch, run.held)
        new = _manager(borrower_cls, ctx)

        async def _fails() -> None:
            raise RuntimeError("backend init failed")

        monkeypatch.setattr(new, "_do_initialize", _fails)
        loan = run.manager.lend_services()
        with pytest.raises(RuntimeError, match="backend init failed"):
            await new.initialize_services(services=loan)

        assert closes == Counter()
        assert new.services == {} and new.is_initialized is False
        await run.still_works(run.manager)
        with pytest.raises(RuntimeError, match="borrow"):
            await run.manager.give_services(loan)
        await new.cleanup()
        assert closes == Counter()
        await run.manager.cleanup()
        assert closes == Counter({key: 1 for key in CLOSERS})

    async def test_a_cancel_while_it_starts_closes_nothing_it_borrowed(
        self, pair, ctx, run, monkeypatch
    ):
        _, borrower_cls = pair
        closes = _count_closes(monkeypatch, run.held)
        new = _manager(borrower_cls, ctx)
        building = asyncio.Event()

        async def _hangs() -> None:
            building.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(new, "_do_initialize", _hangs)
        loan = run.manager.lend_services()
        start = asyncio.create_task(new.initialize_services(services=loan))
        await asyncio.wait_for(building.wait(), timeout=10)
        assert new.services[JOB_MANAGER] is run.held[JOB_MANAGER]  # borrowed, published
        start.cancel()
        with pytest.raises(asyncio.CancelledError):
            await start

        assert closes == Counter()
        assert new.services == {}
        await run.still_works(run.manager)
        await run.manager.cleanup()
        assert closes == Counter({key: 1 for key in CLOSERS})

    async def test_an_abandoned_manager_closes_only_what_it_built(
        self, pair, ctx, run, monkeypatch
    ):
        _, borrower_cls = pair
        closes = _count_closes(monkeypatch, run.held)
        new = _manager(borrower_cls, ctx)
        loan = run.manager.lend_services()
        await new.initialize_services(services=loan)
        engine = new.services[PERMISSION_ENGINE]

        await new.cleanup()  # the caller abandons it: never given

        assert engine is run.held[PERMISSION_ENGINE]
        assert closes == Counter()
        await run.still_works(run.manager)
        await run.manager.cleanup()
        assert closes == Counter({key: 1 for key in CLOSERS})


async def _a_turn(manager: BaseWorkflowManager) -> str:
    """Enter a turn of ``manager`` (every turn's entry, ``_turn``), and leave."""
    async with manager._turn("s", "u"):
        return "ran"


class TestNoTurnRunsOnServicesItDoesNotOwn:
    """A manager that holds services another manager owns runs no turn.

    The permission engine's per-call check reads its owner's outside-content
    marks: the borrower's turn before the give would be checked against the
    lender's conversations, and the lender's after it against the
    borrower's, so a memory write from a conversation that read outside
    content went through unasked. The borrower runs turns once it owns them,
    and the lender none after the give, until its release.
    """

    async def test_the_borrower_before_the_give(self, pair, ctx, run):
        _, borrower_cls = pair
        new = _manager(borrower_cls, ctx)
        loan = run.manager.lend_services()
        await new.initialize_services(services=loan)
        try:
            with pytest.raises(RuntimeError, match="does not own"):
                await _a_turn(new)
            assert await _a_turn(run.manager) == "ran"

            await run.manager.give_services(loan)

            assert await _a_turn(new) == "ran"
        finally:
            await new.cleanup()

    async def test_the_lender_after_the_give_until_its_release(self, pair, ctx, run):
        _, borrower_cls = pair
        new = _manager(borrower_cls, ctx)
        loan = run.manager.lend_services()
        await new.initialize_services(services=loan)
        await run.manager.give_services(loan)
        try:
            with pytest.raises(RuntimeError, match="does not own"):
                await _a_turn(run.manager)
            await run.manager.cleanup()
            assert run.manager._service_set.not_owned == {}
        finally:
            await new.cleanup()

    async def test_a_turn_queued_behind_the_give_is_refused(self, pair, ctx, run):
        """asyncio's locks are FIFO: a turn that queued while the give waited
        for the lender's turn lock gets it right after the give, before the
        caller's cleanup."""
        _, borrower_cls = pair
        new = _manager(borrower_cls, ctx)
        loan = run.manager.lend_services()
        await new.initialize_services(services=loan)
        await run.manager._turn_lock.acquire()  # a turn of the lender runs
        try:
            give = asyncio.create_task(run.manager.give_services(loan))
            await asyncio.sleep(0.05)
            turn = asyncio.create_task(_a_turn(run.manager))
            await asyncio.sleep(0.05)
            assert not give.done() and not turn.done()
        finally:
            run.manager._turn_lock.release()
        try:
            await asyncio.wait_for(give, timeout=5)
            with pytest.raises(RuntimeError, match="does not own"):
                await asyncio.wait_for(turn, timeout=5)
        finally:
            await new.cleanup()


class TestTheLoan:
    async def test_it_lends_neither_the_manager_itself_nor_what_it_does_not_own(
        self, pair, ctx, run
    ):
        loan = run.manager.lend_services()

        assert set(loan.services) >= set(KEPT)
        assert WORKFLOW not in loan.services and LLM_SUMMARIZER not in loan.services
        assert all(service is not run.manager for service in loan.services.values())

    async def test_a_manager_that_is_already_started_cannot_borrow(self, pair, ctx, run):
        _, borrower_cls = pair
        new = _manager(borrower_cls, ctx)
        await new.initialize_services()
        try:
            with pytest.raises(RuntimeError, match="initialized"):
                await new.initialize_services(services=run.manager.lend_services())
        finally:
            await new.cleanup()

    async def test_it_is_given_once(self, pair, ctx, run):
        _, borrower_cls = pair
        new = _manager(borrower_cls, ctx)
        loan = run.manager.lend_services()
        await new.initialize_services(services=loan)
        await run.manager.give_services(loan)
        try:
            with pytest.raises(RuntimeError, match="given"):
                await run.manager.give_services(loan)
            with pytest.raises(ValueError, match="did not lend"):
                await new.give_services(loan)
        finally:
            await new.cleanup()

    async def test_it_is_not_given_from_inside_a_turn_of_the_lender(self, pair, ctx, run):
        """The give waits for the lender's running turn: from inside that turn
        it waited for itself, forever."""
        _, borrower_cls = pair
        new = _manager(borrower_cls, ctx)
        loan = run.manager.lend_services()
        await new.initialize_services(services=loan)
        try:
            async with run.manager._turn("s", "u"):
                with pytest.raises(RuntimeError, match="inside a turn"):
                    await asyncio.wait_for(run.manager.give_services(loan), timeout=2)
            assert loan.given is False

            await run.manager.give_services(loan)

            assert loan.given is True
        finally:
            await new.cleanup()
