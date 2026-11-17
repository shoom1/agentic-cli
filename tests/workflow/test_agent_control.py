"""One sub-agent run's control (``workflow/agents/control.py``), alone.

A ``RunControl`` holds a run's limits, cancels the run's own task (its
parent's stop, the user's, a limit's), counts the run's model calls and
tokens against its limits, and keeps the messages steered to it until its
next model call. Driven here with a plain asyncio task for the run.
"""

from __future__ import annotations

import asyncio

import pytest

from agentic_cli.workflow.agents.control import STEP_LIMIT_NOTE, RunControl
from agentic_cli.workflow.agents.envelope import TIME_LIMIT_ERROR, TOKEN_LIMIT_ERROR
from agentic_cli.workflow.agents.policy import AgentLimits
from agentic_cli.workflow.sessions import RunRef

REF = RunRef(
    agent="researcher", instance="researcher-1", run="researcher-1#1", parent_run=None, depth=1
)


def control(**limits) -> RunControl:
    return RunControl(REF, AgentLimits(**limits))


async def a_run(made: RunControl) -> tuple[asyncio.Task, asyncio.Event]:
    """A run's task bound to ``made``, waiting until cancelled; the event set once it waits."""
    waiting = asyncio.Event()

    async def run() -> str:
        waiting.set()
        await asyncio.Event().wait()
        return "never"  # pragma: no cover

    task = asyncio.create_task(run())
    made.bind(task)
    await waiting.wait()
    return task, waiting


def test_the_step_limit_note():
    assert STEP_LIMIT_NOTE == (
        "You have reached the most model calls this task may make. Answer now with your best "
        "result from what you have; no tools are available for this answer."
    )


def test_it_holds_its_run_and_limits():
    made = control(max_model_calls=3, timeout=2.5)
    assert (made.ref, made.limits, made.cancelled) == (
        REF, AgentLimits(max_model_calls=3, timeout=2.5), None,
    )


class TestCancel:
    async def test_cancels_the_runs_task_and_says_who_and_why(self):
        made = control()
        task, _ = await a_run(made)

        made.cancel(by="parent", error="no longer needed")

        with pytest.raises(asyncio.CancelledError):
            await task
        assert made.cancelled == ("parent", "no longer needed")

    async def test_the_first_cancel_wins(self):
        made = control()
        task, _ = await a_run(made)

        made.cancel(by="limit", error=TIME_LIMIT_ERROR)
        made.cancel(by="parent", error="no longer needed")

        with pytest.raises(asyncio.CancelledError):
            await task
        assert made.cancelled == ("limit", TIME_LIMIT_ERROR)

    async def test_a_run_that_has_ended_is_not_cancelled(self):
        made = control()

        async def run() -> str:
            return "done"

        task = asyncio.create_task(run())
        made.bind(task)
        assert await task == "done"

        made.cancel(by="parent", error="too late")

        assert made.cancelled is None

    async def test_a_cancel_before_the_task_is_bound_cancels_it_then(self):
        made = control()
        made.cancel(by="user", error="stopped")
        task = asyncio.create_task(asyncio.Event().wait())

        made.bind(task)

        with pytest.raises(asyncio.CancelledError):
            await task
        assert made.cancelled == ("user", "stopped")

    async def test_a_task_its_turn_is_unwinding_is_not_cancelled_again(self):
        """A limit that fires while the turn's cancel (Ctrl+C) unwinds the run
        adds no cancel of its own, which would cut the run's cleanup short:
        the hook sees the turn's cancel alone."""
        made = control()
        seen: dict = {}

        async def run() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                seen["own"] = made.own_cancel()
                seen["cancelling"] = asyncio.current_task().cancelling()
                raise

        task = asyncio.create_task(run())
        made.bind(task)
        await asyncio.sleep(0)

        task.cancel()  # the turn's, not yet delivered
        made.cancel(by="limit", error=TIME_LIMIT_ERROR)

        with pytest.raises(asyncio.CancelledError):
            await task
        assert seen == {"own": False, "cancelling": 1}

    def test_who_cancels_is_the_parent_the_user_or_a_limit(self):
        with pytest.raises(ValueError, match="parent, user or limit"):
            control().cancel(by="someone", error="why")


class TestItsOwnCancel:
    """The hook, handling a ``CancelledError`` in the run's task, asks
    ``own_cancel`` whether it is the run's own (then it ends the run and
    returns) or its turn's (then it raises)."""

    async def test_the_controls_cancel_is_the_runs_own_and_is_withdrawn(self):
        made = control()
        seen: dict = {}

        async def run() -> str:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                seen["own"] = made.own_cancel()
                seen["cancelling"] = asyncio.current_task().cancelling()
                await asyncio.sleep(0)  # the run may still finish its work
                return "what it had"
            return "never"  # pragma: no cover

        task = asyncio.create_task(run())
        made.bind(task)
        await asyncio.sleep(0)

        made.cancel(by="parent", error="no longer needed")

        assert await task == "what it had"
        assert seen == {"own": True, "cancelling": 0}

    async def test_another_cancel_is_not(self):
        made = control()
        seen: dict = {}

        async def run() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                seen["own"] = made.own_cancel()
                raise

        task = asyncio.create_task(run())
        made.bind(task)
        await asyncio.sleep(0)

        task.cancel()  # the turn's cancel reaching the run's task

        with pytest.raises(asyncio.CancelledError):
            await task
        assert seen == {"own": False}
        assert made.cancelled is None


class TestTheTimeLimit:
    async def test_past_it_the_run_is_cancelled_by_its_limit(self):
        made = control(timeout=0.05)

        async def run() -> None:
            with made.timing():
                await asyncio.Event().wait()

        task = asyncio.create_task(run())
        made.bind(task)

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert made.cancelled == ("limit", TIME_LIMIT_ERROR)

    async def test_a_run_that_ends_in_time_is_not_cancelled_later(self):
        made = control(timeout=0.05)

        async def run() -> str:
            with made.timing():
                return "done"

        task = asyncio.create_task(run())
        made.bind(task)
        assert await task == "done"
        await asyncio.sleep(0.1)

        assert made.cancelled is None

    async def test_the_clock_starts_with_the_block(self):
        made = control(timeout=1.0)

        async def run() -> str:
            await asyncio.sleep(1.2)  # waiting before it works: not counted
            with made.timing():
                await asyncio.sleep(0.05)
            return "done"

        task = asyncio.create_task(run())
        made.bind(task)

        assert await asyncio.wait_for(task, 5) == "done"
        assert made.cancelled is None


class TestTheStepLimit:
    def test_the_last_allowed_call_is_said_so(self):
        made = control(max_model_calls=3)

        assert [made.model_call() for _ in range(3)] == [False, False, True]
        assert (made.at_step_limit, made.final_call) == (True, True)

    def test_before_it_the_run_is_not_at_its_limit(self):
        made = control(max_model_calls=3)
        made.model_call()

        assert (made.at_step_limit, made.final_call) == (False, False)

    def test_without_a_limit_no_call_is_the_last(self):
        made = control()

        assert [made.model_call() for _ in range(200)] == [False] * 200
        assert made.at_step_limit is False


class TestTheTokenLimit:
    async def test_up_to_it_nothing_happens(self):
        made = control(max_tokens=100)
        made.used(60, 20)
        made.used(15, 5)

        assert (made.cancelled, made.final_call) == (None, False)

    async def test_past_it_the_run_ends_at_that_answer_cancelled_by_its_limit(self):
        made = control(max_tokens=100)
        task, _ = await a_run(made)

        made.used(60, 20)
        made.used(30, 10)

        # Not its task: the answer that went over is the run's last.
        assert (made.cancelled, made.final_call) == (("limit", TOKEN_LIMIT_ERROR), True)
        assert not task.cancelled() and not task.done()
        task.cancel()

    def test_once_past_it_every_call_is_the_last(self):
        """The token limit makes the call in progress the run's last; a call
        after it (should a backend make one) is the last too."""
        made = control(max_tokens=100)
        made.model_call()
        made.used(90, 20)

        made.model_call()

        assert made.final_call is True

    def test_without_a_limit_any_number_is_fine(self):
        made = control()
        made.used(10**9, 10**9)

        assert made.cancelled is None

    async def test_an_answer_past_it_that_makes_no_call_ends_the_run_as_it_would(self):
        made = control(max_tokens=100)
        task, _ = await a_run(made)

        made.used(60, 20)
        made.used(30, 10, calls=False)

        # It is the run's last answer anyway: its result, whole.
        assert (made.cancelled, made.final_call) == (None, False)
        # Counted all the same: an answer past it that would go on is the last.
        made.used(1, 1)
        assert (made.cancelled, made.final_call) == (("limit", TOKEN_LIMIT_ERROR), True)
        task.cancel()


class TestSteering:
    def test_messages_are_taken_oldest_first_once(self):
        made = control()
        made.steer("Also give Z.")
        made.steer("In metric units.")

        assert made.take_steering() == ["Also give Z.", "In metric units."]
        assert made.take_steering() == []

    @pytest.mark.parametrize("message", ["", "   ", None])
    def test_a_message_is_text(self, message):
        with pytest.raises(ValueError, match="text"):
            control().steer(message)
