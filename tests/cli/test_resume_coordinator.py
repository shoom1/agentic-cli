"""Coordinator: BaseCLIApp.resume_finished_jobs drains awaiting jobs (milestone 3).

Each finished, resume-flagged job becomes one serialized resume turn. Tested on
a bare app (no real ThinkingPromptSession) with fake controller / job manager /
message processor.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agentic_cli.cli.app import BaseCLIApp


class _FakeJM:
    def __init__(self, records: list) -> None:
        self._records = records
        self.marked: list[str] = []

    def awaiting_resume(self) -> list:
        return [r for r in self._records if r.job_id not in self.marked]

    def mark_resumed(self, job_id: str) -> None:
        self.marked.append(job_id)


class _FakeMessageProcessor:
    def __init__(self) -> None:
        self.resumed: list[str] = []

    async def process_resume(self, *, record, workflow_controller, ui, settings, usage_tracker):
        self.resumed.append(record.job_id)


def _app(records: list, *, ready: bool = True, has_jm: bool = True):
    app = BaseCLIApp.__new__(BaseCLIApp)
    jm = _FakeJM(records) if has_jm else None
    app._workflow_controller = SimpleNamespace(
        is_ready=ready, workflow=SimpleNamespace(job_manager=jm)
    )
    app._turn_lock = asyncio.Lock()
    app._message_processor = _FakeMessageProcessor()
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
    assert jm.marked == ["a", "b"]


async def test_marks_resumed_before_processing():
    order: list = []
    app, jm = _app([SimpleNamespace(job_id="a")])

    real_mark = jm.mark_resumed
    jm.mark_resumed = lambda jid: (order.append(("mark", jid)), real_mark(jid))[1]

    async def _proc(*, record, **kw):
        order.append(("proc", record.job_id))

    app._message_processor.process_resume = _proc

    await app.resume_finished_jobs()
    assert order == [("mark", "a"), ("proc", "a")]


async def test_no_manager_returns_zero():
    app, _ = _app([], has_jm=False)
    assert await app.resume_finished_jobs() == 0


async def test_not_ready_returns_zero():
    app, _ = _app([SimpleNamespace(job_id="a")], ready=False)
    assert await app.resume_finished_jobs() == 0
    assert app._message_processor.resumed == []
