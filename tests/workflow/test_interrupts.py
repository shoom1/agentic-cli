"""The record a manager keeps of its calls is kept per conversation.

One manager runs many conversations, and a call a stopped or failed turn left
open is answered at that conversation's next run (Deep Agents), so what one
conversation's calls did must survive a turn in another, and a failed turn's
calls that returned keep their results. A job answers only a call of its own
conversation: the job store is the user's, kept across runs, and a provider
that numbers its call ids repeats them in every conversation.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Iterator

import pytest

from agentic_cli.tools.jobs.backends import JobState
from agentic_cli.workflow.interrupts import (
    INTERRUPTED,
    INTERRUPTED_BY_USER,
    NOT_RUN,
    OpenCalls,
    interrupted,
    job_for_call,
    job_started,
)
from agentic_cli.workflow.service_registry import JOB_MANAGER, set_service_registry
from agentic_cli.workflow.sessions import SessionRef, reset_active_turn, set_active_turn


def ref(session_id: str, user_id: str = "u") -> SessionRef:
    return SessionRef("app", user_id, session_id)


@contextmanager
def turn_of(conversation: SessionRef | None, jobs: object = None) -> Iterator[None]:
    """Run the block as a turn of ``conversation``, with ``jobs`` as the job manager."""
    turn = set_active_turn(conversation)
    registry = set_service_registry({JOB_MANAGER: jobs} if jobs is not None else {})
    try:
        yield
    finally:
        registry.var.reset(registry)
        reset_active_turn(turn)


def cancelled_at_its_prompt(calls: OpenCalls, call_id: str) -> None:
    """``call_id`` waits at its prompt when its turn is cancelled: the mark stays."""
    with pytest.raises(asyncio.CancelledError):
        with calls.asking(call_id):
            raise asyncio.CancelledError


class TestOpenCalls:
    def test_a_turn_that_ends_clears_only_its_own_conversation(self):
        calls = OpenCalls()
        with turn_of(ref("a")):
            calls.finished("call_1", "a's result")
            cancelled_at_its_prompt(calls, "call_2")
        with turn_of(ref("b")):
            calls.finished("call_1", "b's result")

        calls.clear(ref("b"))

        with turn_of(ref("a")):
            assert calls.answer("call_1", by_user=True) == "a's result"
            assert calls.answer("call_2", by_user=True) == interrupted(NOT_RUN)
        with turn_of(ref("b")):
            assert calls.answer("call_1", by_user=True) == interrupted(INTERRUPTED_BY_USER)

    def test_a_failed_turn_keeps_its_finished_calls(self):
        """Its unfinished calls are answered INTERRUPTED at the next run; one
        that waited at its prompt too, since the user did not cancel it."""
        calls = OpenCalls()
        with turn_of(ref("a")):
            calls.finished("call_1", "its result")
            cancelled_at_its_prompt(calls, "call_2")

        calls.failed(ref("a"))

        with turn_of(ref("a")):
            assert calls.answer("call_1", by_user=False) == "its result"
            assert calls.answer("call_2", by_user=False) == interrupted(INTERRUPTED)

    def test_a_call_answered_is_forgotten(self):
        """The answer is in the history now: the record does not grow with
        every conversation a stop leaves behind."""
        calls = OpenCalls()
        with turn_of(ref("a")):
            calls.finished("call_1", "its result")
            cancelled_at_its_prompt(calls, "call_2")
            calls.answer("call_1", by_user=True)
            calls.answer("call_2", by_user=True)

        assert calls._finished == {} and calls._asking == {}

    def test_a_turn_that_ends_leaves_nothing_behind(self):
        calls = OpenCalls()
        with turn_of(ref("a")):
            calls.finished("call_1", "its result")
            cancelled_at_its_prompt(calls, "call_2")

        calls.clear(ref("a"))

        assert calls._finished == {} and calls._asking == {}


def job(job_id: str, call_id: str, session_id: str, user_id: str = "u") -> SimpleNamespace:
    return SimpleNamespace(
        job_id=job_id, call_id=call_id, session_id=session_id, user_id=user_id,
        state=JobState.RUNNING,
    )


class Jobs:
    """The user's job store: every conversation's jobs, from every run."""

    def __init__(self, *records: SimpleNamespace) -> None:
        self.records = list(records)

    def list(self) -> list[SimpleNamespace]:
        return list(self.records)


class TestJobForCall:
    def test_a_job_answers_only_a_call_of_its_own_conversation(self):
        jobs = Jobs(job("j-a", "call_0", "a"), job("j-b", "call_0", "b", user_id="someone"))
        calls = OpenCalls()

        with turn_of(ref("b"), jobs):
            assert calls.answer("call_0", by_user=True) == interrupted(INTERRUPTED_BY_USER)
        with turn_of(ref("a"), jobs):
            assert calls.answer("call_0", by_user=True) == job_started(jobs.records[0])

    def test_outside_a_turn_no_job_answers(self):
        jobs = Jobs(job("j-a", "call_0", "a"))

        with turn_of(None, jobs):
            assert job_for_call("call_0") is None
