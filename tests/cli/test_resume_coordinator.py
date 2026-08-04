"""Coordinator: BaseCLIApp.resume_finished_jobs drains awaiting jobs (milestone 3).

Each finished, resume-flagged job becomes one serialized resume turn. Tested on
a bare app (no real ThinkingPromptSession) with fake controller / job manager /
message processor.

Delivery uses the job's resume lifecycle: claim (pending → resuming) *before*
the turn, record delivered/failed *after* it. Marking delivery up-front — the
previous behaviour — silently dropped every resume whose turn then failed.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agentic_cli.cli.app import BaseCLIApp
from agentic_cli.cli.message_processor import TurnResult, TurnStatus


class _FakeJM:
    """Minimal stand-in implementing the resume lifecycle contract."""

    def __init__(self, records: list) -> None:
        self._records = records
        self.claimed: list[str] = []
        self.completed: list[tuple[str, bool, str | None]] = []

    def awaiting_resume(self) -> list:
        return [r for r in self._records if r.job_id not in self.claimed]

    def begin_resume(self, job_id: str) -> bool:
        if job_id in self.claimed:
            return False
        self.claimed.append(job_id)
        return True

    def complete_resume(self, job_id: str, *, delivered: bool, error=None) -> None:
        self.completed.append((job_id, delivered, error))


class _FakeMessageProcessor:
    def __init__(self, result: TurnResult | None = None) -> None:
        self.resumed: list[str] = []
        self._result = result or TurnResult(TurnStatus.COMPLETED)

    async def process_resume(
        self, *, record, workflow_controller, ui, settings, usage_tracker
    ):
        self.resumed.append(record.job_id)
        return self._result


def _app(records: list, *, ready: bool = True, has_jm: bool = True, result=None):
    app = BaseCLIApp.__new__(BaseCLIApp)
    jm = _FakeJM(records) if has_jm else None
    app._workflow_controller = SimpleNamespace(
        is_ready=ready, workflow=SimpleNamespace(job_manager=jm)
    )
    app._turn_lock = asyncio.Lock()
    app._message_processor = _FakeMessageProcessor(result)
    app.session = object()
    app._settings = SimpleNamespace(job_auto_resume=True)
    app._usage_tracker = None
    return app, jm


async def test_resumes_each_awaiting_job_once():
    recs = [SimpleNamespace(job_id="a"), SimpleNamespace(job_id="b")]
    app, jm = _app(recs)
    n = await app.resume_finished_jobs()
    assert n == 2
    assert app._message_processor.resumed == ["a", "b"]
    assert jm.completed == [("a", True, None), ("b", True, None)]


async def test_claims_before_processing_and_records_after():
    order: list = []
    app, jm = _app([SimpleNamespace(job_id="a")])

    real_begin = jm.begin_resume
    jm.begin_resume = lambda jid: (order.append(("claim", jid)), real_begin(jid))[1]
    real_complete = jm.complete_resume
    jm.complete_resume = lambda jid, **kw: (
        order.append(("done", jid, kw["delivered"])),
        real_complete(jid, **kw),
    )[1]

    async def _proc(*, record, **kw):
        order.append(("proc", record.job_id))
        return TurnResult(TurnStatus.COMPLETED)

    app._message_processor.process_resume = _proc

    await app.resume_finished_jobs()
    assert order == [("claim", "a"), ("proc", "a"), ("done", "a", True)]


async def test_failed_resume_is_recorded_as_failed_not_delivered():
    """A resume whose turn failed must not be recorded as delivered."""
    app, jm = _app(
        [SimpleNamespace(job_id="a")],
        result=TurnResult(TurnStatus.FAILED, error="workflow blew up"),
    )

    n = await app.resume_finished_jobs()

    assert n == 1  # picked up...
    assert jm.completed == [("a", False, "workflow blew up")]  # ...but not delivered


async def test_unavailable_conversation_is_recorded_as_failed():
    app, jm = _app(
        [SimpleNamespace(job_id="a")],
        result=TurnResult(TurnStatus.UNAVAILABLE, error="conversation gone"),
    )

    assert await app.resume_finished_jobs() == 1
    assert jm.completed == [("a", False, "conversation gone")]


async def test_unclaimable_job_is_skipped():
    """A job another coordinator already claimed must not be delivered twice."""
    app, jm = _app([SimpleNamespace(job_id="a")])
    jm.begin_resume = lambda jid: False

    assert await app.resume_finished_jobs() == 0
    assert app._message_processor.resumed == []
    assert jm.completed == []


async def test_no_manager_returns_zero():
    app, _ = _app([], has_jm=False)
    assert await app.resume_finished_jobs() == 0


async def test_not_ready_returns_zero():
    app, _ = _app([SimpleNamespace(job_id="a")], ready=False)
    assert await app.resume_finished_jobs() == 0
    assert app._message_processor.resumed == []


class TestShutdownOrder:
    """Fact extraction must run before the controller closes the manager.

    ``background_init``'s exit now calls ``controller.close()``, which cleans up
    and drops the manager. Extraction placed after that block would find
    ``is_ready`` False and silently do nothing.
    """

    def test_extraction_runs_inside_the_controller_context(self):
        import ast
        import inspect
        import textwrap

        from agentic_cli.cli.app import BaseCLIApp

        tree = ast.parse(textwrap.dedent(inspect.getsource(BaseCLIApp.run)))

        def _is_background_init(node: ast.AsyncWith) -> bool:
            return any(
                isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Attribute)
                and item.context_expr.func.attr == "background_init"
                for item in node.items
            )

        blocks = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.AsyncWith) and _is_background_init(n)
        ]
        assert len(blocks) == 1, "run() no longer has a single background_init block"

        def _calls_extraction(node: ast.AST) -> bool:
            return any(
                isinstance(n, ast.Attribute)
                and n.attr == "_extract_session_facts_on_exit"
                for n in ast.walk(node)
            )

        assert _calls_extraction(blocks[0]), (
            "_extract_session_facts_on_exit() must be called inside the "
            "background_init block — the controller closes the workflow on exit"
        )
        outside = [n for n in tree.body[0].body if n is not blocks[0]]
        assert not any(_calls_extraction(n) for n in outside), (
            "_extract_session_facts_on_exit() is also called after cleanup"
        )

    async def test_extraction_is_skipped_once_the_controller_closed(self):
        from agentic_cli.cli.app import BaseCLIApp

        calls: list[str] = []
        app = BaseCLIApp.__new__(BaseCLIApp)
        app._settings = SimpleNamespace(auto_extract_session_facts=True)
        app._workflow_controller = SimpleNamespace(
            is_ready=False,
            workflow=SimpleNamespace(
                on_session_end=lambda: calls.append("extract")
            ),
        )

        await app._extract_session_facts_on_exit()
        assert calls == []


class _StrictJM(_FakeJM):
    """Enforces the real lifecycle: complete only after a claim, once."""

    def __init__(self, records: list) -> None:
        super().__init__(records)
        self.open_claims: set[str] = set()

    def begin_resume(self, job_id: str) -> bool:
        if not super().begin_resume(job_id):
            return False
        self.open_claims.add(job_id)
        return True

    def complete_resume(self, job_id: str, *, delivered: bool, error=None) -> None:
        from agentic_cli.tools.jobs.manager import ResumeStateError

        if job_id not in self.open_claims:
            raise ResumeStateError(f"{job_id} was not claimed")
        self.open_claims.discard(job_id)
        super().complete_resume(job_id, delivered=delivered, error=error)


def _strict_app(records: list, processor=None):
    app = BaseCLIApp.__new__(BaseCLIApp)
    jm = _StrictJM(records)
    app._workflow_controller = SimpleNamespace(
        is_ready=True, workflow=SimpleNamespace(job_manager=jm)
    )
    app._turn_lock = asyncio.Lock()
    app._message_processor = processor or _FakeMessageProcessor()
    app.session = SimpleNamespace(add_error=lambda msg: None)
    app._settings = SimpleNamespace(job_auto_resume=True)
    app._usage_tracker = None
    return app, jm


class TestClaimIsAlwaysClosed:
    """No record may be left RESUMING once the coordinator returns."""

    async def test_processor_exception_closes_the_claim(self):
        class _Raising:
            async def process_resume(self, **kwargs):
                raise RuntimeError("processor exploded")

        app, jm = _strict_app([SimpleNamespace(job_id="a", name="build")], _Raising())

        assert await app.resume_finished_jobs() == 1
        assert jm.open_claims == set(), "the job is stuck RESUMING"
        assert jm.completed == [("a", False, "processor exploded")]

    async def test_can_resume_failure_closes_the_claim(self):
        """``can_resume()`` raising inside process_resume is still a closed claim."""

        class _Raising:
            async def process_resume(self, **kwargs):
                raise ConnectionError("session store unreachable")

        app, jm = _strict_app([SimpleNamespace(job_id="a", name="build")], _Raising())

        await app.resume_finished_jobs()
        assert jm.open_claims == set()
        assert jm.completed[0][1] is False

    async def test_cancellation_closes_the_claim_and_propagates(self):
        started = asyncio.Event()

        class _Hanging:
            async def process_resume(self, **kwargs):
                started.set()
                await asyncio.Event().wait()

        app, jm = _strict_app([SimpleNamespace(job_id="a", name="build")], _Hanging())

        task = asyncio.create_task(app.resume_finished_jobs())
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert jm.open_claims == set(), "cancellation left the job RESUMING"
        assert jm.completed == [("a", False, "resume cancelled")]

    async def test_normal_delivery_closes_the_claim(self):
        app, jm = _strict_app([SimpleNamespace(job_id="a", name="build")])
        assert await app.resume_finished_jobs() == 1
        assert jm.open_claims == set()
        assert jm.completed == [("a", True, None)]

    async def test_duplicate_coordinators_deliver_once(self):
        record = SimpleNamespace(job_id="a", name="build")
        app, jm = _strict_app([record])
        second_app = BaseCLIApp.__new__(BaseCLIApp)
        second_app._workflow_controller = app._workflow_controller
        second_app._turn_lock = asyncio.Lock()
        second_app._message_processor = _FakeMessageProcessor()
        second_app.session = app.session
        second_app._settings = app._settings
        second_app._usage_tracker = None

        counts = await asyncio.gather(
            app.resume_finished_jobs(), second_app.resume_finished_jobs()
        )

        assert sorted(counts) == [0, 1], "both coordinators claimed the same job"
        assert len(jm.completed) == 1
