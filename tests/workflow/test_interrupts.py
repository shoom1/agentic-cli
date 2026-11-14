"""The record a manager keeps of its calls is kept per conversation.

One manager runs many conversations, and a call a stopped or failed turn left
open is answered at that conversation's next run (Deep Agents), so what one
conversation's calls did must survive a turn in another, and a failed turn's
calls that returned keep their results. A job answers only a call of its own
conversation: the job store is the user's, kept across runs, and a provider
that numbers its call ids repeats them in every conversation. Within a
conversation a call is the run's that made it (the root's own, or a
sub-agent's), since such a provider also gives a sub-agent's call the id of
its parent's delegation.
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
    unrecorded,
)
from agentic_cli.workflow.service_registry import JOB_MANAGER, set_service_registry
from agentic_cli.workflow.sessions import (
    RunRef,
    SessionRef,
    reset_active_run,
    reset_active_turn,
    set_active_run,
    set_active_turn,
)


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

    def test_an_answer_stands_until_its_conversation_s_turn_ends(self):
        """The step that answers a call may be stopped before it is saved, and
        the next run answers it again: the same."""
        calls = OpenCalls()
        with turn_of(ref("a")):
            calls.finished("call_1", "its result")
            cancelled_at_its_prompt(calls, "call_2")
            first = [calls.answer("call_1", by_user=True), calls.answer("call_2", by_user=True)]
            again = [calls.answer("call_1", by_user=True), calls.answer("call_2", by_user=True)]

        assert first == again == ["its result", interrupted(NOT_RUN)]

    def test_a_turn_that_ends_leaves_nothing_behind(self):
        calls = OpenCalls()
        with turn_of(ref("a")):
            calls.finished("call_1", "its result")
            cancelled_at_its_prompt(calls, "call_2")

        calls.clear(ref("a"))

        assert calls._finished == {} and calls._asking == {}


def job(
    job_id: str, call_id: str, session_id: str, user_id: str = "u",
    started_by: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        job_id=job_id, call_id=call_id, session_id=session_id, user_id=user_id,
        state=JobState.RUNNING, started_by=started_by,
    )


class Jobs:
    """The user's job store: every conversation's jobs, from every run."""

    def __init__(self, *records: SimpleNamespace) -> None:
        self.records = list(records)

    def snapshot(self) -> list[SimpleNamespace]:
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


@contextmanager
def run_as(ref: RunRef | None) -> Iterator[None]:
    """Run the block as ``ref``'s calls (None: the root's own)."""
    token = set_active_run(ref)
    try:
        yield
    finally:
        reset_active_run(token)


RESEARCHER = RunRef(
    agent="researcher", instance="researcher-1", run="researcher-1#1", parent_run=None, depth=1
)


class TestEachRunsCallsAreItsOwn:
    """A provider that numbers its call ids per agent (``call_0``, ...) gives a
    sub-agent's first call the id of the delegation that started it: neither
    call ever stands for the other."""

    def test_a_sub_agents_finished_call_never_answers_its_parents(self):
        calls = OpenCalls()
        with turn_of(ref("a")):
            with run_as(RESEARCHER):
                calls.finished("call_0", "the researcher's lookup")

            assert calls.answer("call_0", by_user=True) == interrupted(INTERRUPTED_BY_USER)
            with run_as(RESEARCHER):
                assert calls.answer("call_0", by_user=True) == "the researcher's lookup"

    def test_a_parents_finished_call_never_answers_its_sub_agents(self):
        calls = OpenCalls()
        with turn_of(ref("a")):
            calls.finished("call_0", "the delegation's answer")

            with run_as(RESEARCHER):
                assert calls.answer("call_0", by_user=True) == interrupted(INTERRUPTED_BY_USER)
            assert calls.answer("call_0", by_user=True) == "the delegation's answer"

    def test_a_sub_agents_call_at_its_prompt_is_not_its_parents(self):
        calls = OpenCalls()
        with turn_of(ref("a")):
            with run_as(RESEARCHER):
                cancelled_at_its_prompt(calls, "call_0")

            assert calls.answer("call_0", by_user=True) == interrupted(INTERRUPTED_BY_USER)
            with run_as(RESEARCHER):
                assert calls.answer("call_0", by_user=True) == interrupted(NOT_RUN)

    def test_a_run_named_outright_is_the_run_in_progress(self):
        """ADK words a stopped turn's seals outside the run that made the
        call: it names the run its branch is."""
        calls = OpenCalls()
        with turn_of(ref("a")):
            with run_as(RESEARCHER):
                calls.finished("call_0", "the researcher's lookup")
                cancelled_at_its_prompt(calls, "call_1")

            assert calls.recorded("call_0", run="researcher-1#1") == "the researcher's lookup"
            assert calls.recorded("call_0", run=None) is None
            assert calls.answer("call_0", by_user=True, run="researcher-1#1") == (
                "the researcher's lookup"
            )
            assert calls.answer("call_1", by_user=True, run="researcher-1#1") == interrupted(NOT_RUN)
            assert calls.answer("call_1", by_user=True, run=None) == interrupted(
                INTERRUPTED_BY_USER
            )

    def test_a_turn_that_ends_forgets_every_runs_calls(self):
        calls = OpenCalls()
        with turn_of(ref("a")):
            calls.finished("call_0", "the delegation's answer")
            with run_as(RESEARCHER):
                calls.finished("call_0", "the researcher's lookup")
                cancelled_at_its_prompt(calls, "call_1")

        calls.clear(ref("a"))

        assert calls._finished == {} and calls._asking == {}

    def test_a_failed_turn_keeps_every_runs_finished_calls(self):
        calls = OpenCalls()
        with turn_of(ref("a")):
            with run_as(RESEARCHER):
                calls.finished("call_0", "the researcher's lookup")
                cancelled_at_its_prompt(calls, "call_1")

        calls.failed(ref("a"))

        with turn_of(ref("a")), run_as(RESEARCHER):
            assert calls.answer("call_0", by_user=False) == "the researcher's lookup"
            assert calls.answer("call_1", by_user=False) == interrupted(INTERRUPTED)


class TestUnrecorded:
    def test_a_call_no_record_holds_is_answered_as_interrupted(self):
        with turn_of(ref("a")):
            assert unrecorded("call_0", by_user=True) == interrupted(INTERRUPTED_BY_USER)
            assert unrecorded("call_0", by_user=False) == interrupted(INTERRUPTED)

    def test_one_whose_job_was_submitted_is_answered_with_the_job(self):
        jobs = Jobs(job("j1", "call_0", "a"))
        with turn_of(ref("a"), jobs):
            assert unrecorded("call_0", by_user=True) == job_started(jobs.records[0])


class TestAJobAnswersOnlyItsOwnRunsCall:
    """A provider that numbers its call ids repeats them across agents: a
    sub-agent's job never answers its parent's call with the same id."""

    def test_a_sub_agents_job_does_not_answer_its_parents_call(self):
        jobs = Jobs(job("j-sub", "call_0", "a", started_by="researcher-1"))
        calls = OpenCalls()

        with turn_of(ref("a"), jobs):
            assert calls.answer("call_0", by_user=True) == interrupted(INTERRUPTED_BY_USER)
            assert job_for_call("call_0") is None
            assert job_for_call("call_0", started_by="researcher-1") is jobs.records[0]

    def test_each_run_gets_the_job_its_own_call_started(self):
        jobs = Jobs(
            job("j-root", "call_0", "a"),
            job("j-sub", "call_0", "a", started_by="researcher-1"),
        )
        calls = OpenCalls()

        with turn_of(ref("a"), jobs):
            assert calls.answer("call_0", by_user=True) == job_started(jobs.records[0])
            with run_as(RESEARCHER):
                assert calls.answer("call_0", by_user=True) == job_started(jobs.records[1])

    def test_a_run_named_outright_gets_the_job_its_call_started(self):
        """ADK answers a stopped turn's calls outside the run that made each,
        naming the run its branch is: the job is that run's, not the run in
        progress."""
        jobs = Jobs(
            job("j-root", "call_0", "a"),
            job("j-sub", "call_0", "a", started_by="researcher-1"),
        )
        calls = OpenCalls()

        with turn_of(ref("a"), jobs):
            assert calls.answer("call_0", by_user=True, run="researcher-1#1") == job_started(
                jobs.records[1]
            )
            with run_as(RESEARCHER):
                assert calls.answer("call_0", by_user=True, run=None) == job_started(
                    jobs.records[0]
                )
