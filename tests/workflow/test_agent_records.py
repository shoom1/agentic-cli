"""A conversation's sub-agent records, kept beside it in its store.

``RecordsWriter`` (``workflow/agents/records.py``) writes each conversation's
records as they change: one write at a time per conversation, each taking
the records as they are when it starts (changes made while one is written
are written together, by the next), a failure warned of once per
conversation and never raised. A manager (``BaseWorkflowManager``) writes
through its backend's store hooks at each run event that changes the
records, loads them with the conversation (``load_agents``: once, at a
turn's start; again whenever ``load_session`` opens it), has written them
all before a turn returns, and forgets a conversation it deleted.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
from collections.abc import Callable
from typing import Any

import pytest

from agentic_cli.workflow.agents import (
    INSTANCE_IDLE,
    RUN_COMPLETED,
    RUN_INTERRUPTED,
    RUN_RUNNING,
)
from agentic_cli.workflow.agents.delegate import AgentRun, DelegationRequest, RunAnswer
from agentic_cli.workflow.agents.envelope import INTERRUPTED_ERROR
from agentic_cli.workflow import base_manager as base_manager_module
from agentic_cli.workflow.agents import records as records_module
from agentic_cli.workflow.agents.records import RecordsWriter
from agentic_cli.workflow.agents.runs import RECORDS_VERSION
from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.sessions import SessionRef
from tests.conftest import MockContext
from tests.loop_guard import no_blocking_calls

ONE = SessionRef("app", "u", "s1")
TWO = SessionRef("app", "u", "s2")


class _LogRecorder:
    """Stands in for a module's logger and records every call, with its level.

    ``structlog.testing.capture_logs()`` sees nothing once an earlier test has
    run ``configure_logging()`` (a logger is cached on first use).
    """

    def __init__(self) -> None:
        self.events: list[dict] = []

    def __getattr__(self, level: str):
        def log(event: str, **fields) -> None:
            self.events.append({"event": event, "log_level": level, **fields})

        return log


@pytest.fixture
def logged(monkeypatch) -> list[dict]:
    """Every log call of the writer (``agents/records.py``) and the manager (``base_manager.py``)."""
    recorder = _LogRecorder()
    monkeypatch.setattr(records_module, "logger", recorder)
    monkeypatch.setattr(base_manager_module, "logger", recorder)
    return recorder.events


def warned(logged: list[dict]) -> list[tuple[str, str]]:
    """The warnings logged: (event, the conversation's session id)."""
    return [(e["event"], e["session_id"]) for e in logged if e["log_level"] == "warning"]


class Store:
    """A store the writer saves to: what each save wrote, and a gate a test may close."""

    def __init__(self) -> None:
        self.saved: list[tuple[SessionRef, Any]] = []
        self.open = asyncio.Event()
        self.open.set()
        self.entered = asyncio.Event()
        self.fails: Exception | None = None

    async def save(self, conversation: SessionRef, data: Any) -> None:
        self.entered.set()
        await self.open.wait()
        if self.fails is not None:
            raise self.fails
        self.saved.append((conversation, data))


def writer_over(store: Store, records: dict[SessionRef, int]) -> RecordsWriter:
    """A writer whose records of a conversation are the number ``records`` holds for it."""
    return RecordsWriter(export=lambda conversation: records[conversation], save=store.save)


class TestRecordsWriter:
    async def test_a_change_is_written(self):
        store, records = Store(), {ONE: 1}
        writer = writer_over(store, records)

        writer.changed(ONE)
        await writer.flush(ONE)

        assert store.saved == [(ONE, 1)]

    async def test_changes_made_while_one_is_written_are_written_together_the_latest(self):
        store, records = Store(), {ONE: 1}
        writer = writer_over(store, records)
        store.open.clear()
        writer.changed(ONE)
        await store.entered.wait()

        for value in (2, 3, 4):
            records[ONE] = value
            writer.changed(ONE)
        store.open.set()
        await writer.flush(ONE)

        assert store.saved == [(ONE, 1), (ONE, 4)]

    async def test_one_conversations_write_never_waits_for_anothers(self):
        store, records = Store(), {ONE: 1, TWO: 2}
        writer = writer_over(store, records)
        store.open.clear()
        writer.changed(ONE)
        await store.entered.wait()
        other = Store()
        writer._save = lambda conversation, data: (  # TWO's goes to a store that is open
            other.save(conversation, data) if conversation == TWO else store.save(conversation, data)
        )

        writer.changed(TWO)
        await asyncio.wait_for(writer.flush(TWO), 5)

        assert (store.saved, other.saved) == ([], [(TWO, 2)])
        store.open.set()
        await writer.flush(ONE)

    async def test_a_failed_write_is_warned_of_once_per_conversation_and_never_raised(
        self, logged
    ):
        store, records = Store(), {ONE: 1, TWO: 2}
        store.fails = OSError("disk full")
        writer = writer_over(store, records)

        for conversation in (ONE, ONE, TWO):
            writer.changed(conversation)
            await writer.flush(conversation)

        assert warned(logged) == [
            ("subagent_records_not_saved", "s1"), ("subagent_records_not_saved", "s2"),
        ]
        assert [e["error"] for e in logged if e["log_level"] == "warning"][0] == "disk full"

    async def test_a_flush_with_nothing_to_write_returns_at_once(self):
        writer = writer_over(Store(), {})

        await asyncio.wait_for(writer.flush(ONE), 1)
        await asyncio.wait_for(writer.flush_all(), 1)

    async def test_flush_all_waits_for_every_conversation(self):
        store, records = Store(), {ONE: 1, TWO: 2}
        writer = writer_over(store, records)
        store.open.clear()
        writer.changed(ONE)
        writer.changed(TWO)

        waiting = asyncio.ensure_future(writer.flush_all())
        await asyncio.sleep(0)
        assert not waiting.done()
        store.open.set()
        await waiting

        assert sorted(store.saved, key=lambda saved: saved[1]) == [(ONE, 1), (TWO, 2)]

    async def test_a_discard_drops_what_waits_after_the_write_under_way(self):
        store, records = Store(), {ONE: 1}
        writer = writer_over(store, records)
        store.open.clear()
        writer.changed(ONE)
        await store.entered.wait()
        records[ONE] = 2
        writer.changed(ONE)

        discarding = asyncio.ensure_future(writer.discard(ONE))
        await asyncio.sleep(0)
        assert not discarding.done()  # the write under way ends first
        store.open.set()
        await discarding

        assert store.saved == [(ONE, 1)]

    async def test_a_cancelled_flush_leaves_the_write_to_finish(self):
        store, records = Store(), {ONE: 1}
        writer = writer_over(store, records)
        store.open.clear()
        writer.changed(ONE)
        flushing = asyncio.ensure_future(writer.flush(ONE))
        await store.entered.wait()

        flushing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await flushing
        store.open.set()
        await writer.flush(ONE)

        assert store.saved == [(ONE, 1)]

    def test_outside_an_event_loop_nothing_is_written(self):
        store = Store.__new__(Store)  # no event loop to make its events on
        store.saved = []
        writer = RecordsWriter(export=lambda conversation: 1, save=store.save)

        writer.changed(ONE)

        assert store.saved == [] and writer._writing == {}
