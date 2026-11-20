"""``Delegations``: the one place a sub-agent run starts, ends and is decided.

Driven with a fake hook and no backend: the brief is read against the
caller's own sub-agents, the run is named and its ``AGENT_RUN`` events are
recorded and reported to the turn's live events, the hook runs in a task of
its own (the run active there, no tool call), and the run ends as the hook's
answer says, with its envelope. An instance outlives its run: ``follow_up``
is its next run through the same hook, ``close`` ends an idle one, and a
run's end says the status it leaves its instance in.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
from collections.abc import Awaitable, Callable, Iterator
from datetime import datetime
from typing import Any

import pytest

from agentic_cli.workflow import live_events
from agentic_cli.workflow.agents import (
    DEFAULT_LIMITS,
    INSTANCE_CLOSED,
    INSTANCE_IDLE,
    INSTANCE_RUNNING,
    INSTANCE_STOPPED,
    RUN_CANCELLED,
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_INPUT_REQUIRED,
    RUN_INTERRUPTED,
    RUN_QUEUED,
    RUN_REFUSED,
    RUN_RUNNING,
    RUN_STARTED,
    AgentLimits,
    AgentRuns,
    SubAgentPolicy,
)
from agentic_cli.workflow.agents import delegate as delegate_module
from agentic_cli.workflow.agents.brief import INHERITED_CONTEXT_NOTE, Brief, task_message
from agentic_cli.workflow.agents.delegate import (
    RUNS_PER_TURN_ERROR,
    AgentRun,
    DelegationRequest,
    Delegations,
    FollowUpRequest,
    RunAnswer,
    RunFailed,
    started_by,
    waiting_on_user,
)
from agentic_cli.workflow.agents.envelope import (
    INTERRUPTED_ERROR,
    STEP_LIMIT_ERROR,
    TIME_LIMIT_ERROR,
    TOKEN_LIMIT_ERROR,
)
from agentic_cli.workflow.agents.outcome import CUT, REFUSAL_ERROR, REFUSED
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.events import EventType, WorkflowEvent
from agentic_cli.workflow.interrupts import OpenCalls
from agentic_cli.workflow.live_events import LiveEvents, merged
from agentic_cli.workflow.sessions import (
    RunRef,
    SessionRef,
    ToolCallRef,
    close_turn_scope,
    get_active_run,
    get_active_tool_call,
    get_active_turn,
    open_turn_scope,
    reset_active_tool_call,
    reset_active_turn,
    set_active_tool_call,
    set_active_turn,
)
from agentic_cli.workflow.usage import (
    PURPOSE_SUBAGENT,
    PURPOSE_SUMMARIZATION,
    PURPOSE_TOOL,
    UsageLedgers,
    UsageTotals,
    llm_usage_event,
)
from tests.loop_guard import no_blocking_calls

CONVERSATION = SessionRef("app", "u", "s")
RESEARCHER = RunRef(
    agent="researcher", instance="researcher-1", run="researcher-1#1", parent_run=None, depth=1
)
TYPES = {
    "coordinator": AgentConfig(name="coordinator", prompt="p", sub_agents=["researcher", "middle",
                                                                            "reader"]),
    "researcher": AgentConfig(name="researcher", prompt="p", description="Finds."),
    "middle": AgentConfig(name="middle", prompt="p", description="Plans.", sub_agents=["leaf"]),
    "leaf": AgentConfig(name="leaf", prompt="p", description="Looks up."),
    "reader": AgentConfig(name="reader", prompt="p", description="Reads.", inherit_context=True),
}
BRIEF = {"agent_type": "researcher", "title": "Find X", "objective": "Find X."}

Hook = Callable[[AgentRun], Awaitable[RunAnswer]]


def answering(text: str = "X is 42.", **fields: Any) -> tuple[Hook, list[AgentRun]]:
    """A hook that answers ``text``; the runs it was asked to run."""
    asked: list[AgentRun] = []

    async def hook(run: AgentRun) -> RunAnswer:
        asked.append(run)
        return RunAnswer(text=text, **fields)

    return hook, asked


def delegations(hook: Hook, **options: Any) -> Delegations:
    return Delegations(
        runs=options.pop("runs", AgentRuns()), policy=options.pop("policy", SubAgentPolicy()),
        types=options.pop("types", TYPES), hook=hook, **options,
    )


def request(arguments: dict | None = None, **fields: Any) -> DelegationRequest:
    return DelegationRequest(
        caller=fields.pop("caller", None), caller_agent=fields.pop("caller_agent", "coordinator"),
        call_id=fields.pop("call_id", "c1"), arguments=BRIEF if arguments is None else arguments,
        **fields,
    )


async def delegated(made: Delegations, asked: DelegationRequest) -> tuple[Any, list[WorkflowEvent]]:
    """``made.delegate(asked)`` in a turn of ``CONVERSATION``: its result, and what it reported."""
    live = LiveEvents()
    out: dict[str, Any] = {}

    async def source():
        token = set_active_turn(CONVERSATION)
        try:
            with live_events.reporting(live):
                out["result"] = await made.delegate(asked)
        finally:
            reset_active_turn(token)
        if False:  # pragma: no cover - a source that yields nothing itself
            yield

    events = [item async for item in merged(source(), live)]
    return out["result"], events


def statuses(events: list[WorkflowEvent]) -> list[tuple[str, str]]:
    return [
        (e.type.value, e.metadata.get("status") or e.content) for e in events
    ]


class TestABriefThatCannotStartARun:
    async def test_is_the_error_result_and_nothing_else(self):
        hook, asked = answering()
        runs = AgentRuns()

        result, events = await delegated(
            delegations(hook, runs=runs), request({"agent_type": "researcher", "title": "Find X"})
        )

        assert result.result == {
            "success": False, "error": "The brief for 'researcher' is missing: objective.",
        }
        assert (result.state, result.run, events, asked) == ({}, None, [], [])
        assert runs.records(CONVERSATION) == []
        # No instance was numbered: the next run is researcher-1.
        assert runs.run_for(CONVERSATION, None, "c9", "researcher").instance == "researcher-1"

    async def test_a_type_that_is_not_the_callers_own_is_unknown(self):
        hook, asked = answering()

        result, _ = await delegated(delegations(hook), request(
            {"agent_type": "leaf", "title": "Look", "objective": "Look X up."}
        ))

        assert result.result["error"] == (
            "Unknown agent_type 'leaf': the agents you can delegate to are "
            "researcher, middle, reader."
        )
        assert asked == []


class TestACompletedRun:
    async def test_starts_runs_and_ends_with_its_envelope(self):
        hook, asked = answering(state={"tasks": [{"description": "find X"}]})
        runs = AgentRuns()

        result, events = await delegated(delegations(hook, runs=runs), request(
            state={"plan": "- [ ] find X"},
        ))

        assert result.result == {
            "success": True, "agent": "researcher-1", "run": "researcher-1#1",
            "status": RUN_COMPLETED, "partial": False, "result": "X is 42.", "error": None,
        }
        assert (result.run, result.state) == (RESEARCHER, {"tasks": [{"description": "find X"}]})
        (run,) = asked
        assert (run.ref, run.config.name, run.conversation) == (
            RESEARCHER, "researcher", CONVERSATION
        )
        assert run.message == task_message(Brief(title="Find X", objective="Find X."))
        assert (run.state, run.history) == ({"plan": "- [ ] find X"}, None)
        started, ended = events
        assert started.metadata == {
            "status": RUN_STARTED, "agent": "researcher", "partial": False, "call_id": "c1",
            "title": "Find X", "brief": {"title": "Find X", "objective": "Find X."},
            "started_by": "agent:root", "mode": "blocking",
            "inherit_context": False, "continuable": True,
        }
        assert (ended.metadata["status"], ended.metadata["result"], ended.run) == (
            RUN_COMPLETED, "X is 42.", RESEARCHER
        )
        # Its instance waits for a follow-up.
        assert ended.metadata["instance_status"] == INSTANCE_IDLE
        assert runs.instance(CONVERSATION, "researcher-1").status == INSTANCE_IDLE
        assert run.follow_up is False
        (record,) = runs.records(CONVERSATION)
        assert (record.status, record.title, record.brief, record.started_by, record.mode) == (
            RUN_COMPLETED, "Find X", {"title": "Find X", "objective": "Find X."},
            "agent:root", "blocking",
        )

    async def test_an_empty_answer_completes_with_an_empty_result(self):
        hook, _ = answering("")

        result, events = await delegated(delegations(hook), request())

        assert (result.result["status"], result.result["result"]) == (RUN_COMPLETED, "")
        assert statuses(events) == [("agent_run", RUN_STARTED), ("agent_run", RUN_COMPLETED)]

    async def test_a_cut_answer_completes_partial_and_is_warned_of_once_after_its_end(self):
        hook, _ = answering("X is\n\n[cut off at the output limit]", outcome=CUT)

        result, events = await delegated(delegations(hook), request())

        assert (result.result["status"], result.result["partial"], result.result["success"]) == (
            RUN_COMPLETED, True, True
        )
        started, ended, warning = events
        assert ended.metadata["partial"] is True
        assert (warning.type, warning.content, warning.run) == (
            EventType.ERROR, "researcher-1's answer was cut off at the output limit", RESEARCHER
        )

    async def test_its_end_carries_what_it_used(self):
        ledgers = UsageLedgers()

        async def hook(run: AgentRun) -> RunAnswer:
            # Recorded where the run's calls are: in its task, the run active.
            ledgers.record(llm_usage_event(
                "anthropic:claude", UsageTotals(input=50, output=5, calls=1)
            ))
            return RunAnswer(text="X is 42.")

        _, events = await delegated(delegations(hook, ledgers=ledgers), request())

        assert events[-1].metadata["usage"] == {"anthropic:claude": {"input": 50, "output": 5}}


class TestARefusedRun:
    async def test_ends_refused_and_its_parent_reads_the_notice(self):
        hook, _ = answering("The subagent stopped because a model refused the request.",
                            outcome=REFUSED)

        result, events = await delegated(delegations(hook), request())

        assert result.result == {
            "success": False, "agent": "researcher-1", "run": "researcher-1#1",
            "status": RUN_REFUSED, "partial": False,
            "result": "The subagent stopped because a model refused the request.",
            "error": REFUSAL_ERROR,
        }
        assert statuses(events) == [("agent_run", RUN_STARTED), ("agent_run", RUN_REFUSED)]


class TestAFailedRun:
    async def test_a_hook_that_raises_fails_it_and_warns_once(self):
        async def hook(run: AgentRun) -> RunAnswer:
            raise RuntimeError("Error code: 400 - overloaded")

        result, events = await delegated(delegations(hook), request())

        assert result.result == {
            "success": False, "agent": "researcher-1", "run": "researcher-1#1",
            "status": RUN_FAILED, "partial": False, "result": "",
            "error": "Error code: 400 - overloaded",
        }
        started, ended, warning = events
        assert (ended.metadata["status"], ended.metadata["error"]) == (
            RUN_FAILED, "Error code: 400 - overloaded"
        )
        assert warning.content == "researcher-1 failed: Error code: 400 - overloaded"
        assert (warning.metadata["recoverable"], warning.metadata["error_code"],
                warning.metadata["agent"], warning.run) == (
            True, "RuntimeError", "researcher", RESEARCHER
        )

    async def test_an_error_without_text_is_named_by_its_type(self):
        async def hook(run: AgentRun) -> RunAnswer:
            raise TimeoutError()

        result, events = await delegated(delegations(hook), request())

        assert result.result["error"] == "TimeoutError"
        assert events[-1].content == "researcher-1 failed: TimeoutError"

    async def test_a_hook_that_answers_an_error_fails_it_with_its_text(self):
        hook, _ = answering("Half an answer.", error="m returned no answer")

        result, events = await delegated(delegations(hook), request())

        assert (result.result["status"], result.result["result"], result.result["error"]) == (
            RUN_FAILED, "Half an answer.", "m returned no answer"
        )
        assert events[-1].content == "researcher-1 failed: m returned no answer"

    async def test_a_hook_that_answers_an_error_warns_with_its_code(self):
        hook, _ = answering("Half an answer.", error="Error code: 400 - overloaded",
                            error_code="BadRequestError")

        result, events = await delegated(delegations(hook), request())

        assert result.result["error"] == "Error code: 400 - overloaded"
        warning = events[-1]
        assert (warning.content, warning.metadata["error_code"]) == (
            "researcher-1 failed: Error code: 400 - overloaded", "BadRequestError"
        )

    async def test_an_error_answered_without_a_code_warns_with_none(self):
        hook, _ = answering("Half an answer.", error="m returned no answer")

        _, events = await delegated(delegations(hook), request())

        assert events[-1].metadata.get("error_code") is None

    async def test_its_own_words_are_said_as_given(self):
        said = "researcher: the request no longer fits m's context window."

        async def hook(run: AgentRun) -> RunAnswer:
            raise RunFailed(said, code="CONTEXT_OVERFLOW")

        result, events = await delegated(delegations(hook), request())

        assert result.result["error"] == said
        warning = events[-1]
        assert (warning.content, warning.metadata["error_code"]) == (said, "CONTEXT_OVERFLOW")
        assert result.result["result"] == ""

    async def test_its_own_words_carry_the_text_it_wrote(self):
        said = "researcher: the request no longer fits m's context window."

        async def hook(run: AgentRun) -> RunAnswer:
            raise RunFailed(said, code="CONTEXT_OVERFLOW", text="X is about 42.")

        result, events = await delegated(delegations(hook), request())

        assert (result.result["status"], result.result["result"], result.result["error"]) == (
            RUN_FAILED, "X is about 42.", said
        )
        started, ended, warning = events
        assert (ended.metadata["status"], ended.metadata["result"]) == (RUN_FAILED, "X is about 42.")
        assert warning.content == said


class TestTheRunsOwnTask:
    async def test_the_run_is_active_there_and_no_tool_call_is(self):
        seen: list[tuple[RunRef | None, ToolCallRef | None]] = []

        async def hook(run: AgentRun) -> RunAnswer:
            seen.append((get_active_run(), get_active_tool_call()))
            return RunAnswer(text="Done.")

        token = set_active_tool_call(ToolCallRef("c1", "task"))
        try:
            await delegated(delegations(hook), request())
            after = (get_active_run(), get_active_tool_call())
        finally:
            reset_active_tool_call(token)

        assert seen == [(RESEARCHER, None)]
        # The caller's own context is as it was.
        assert after == (None, ToolCallRef("c1", "task"))

    async def test_a_nested_run_is_its_callers_child(self):
        hook, asked = answering()
        middle = RunRef(agent="middle", instance="middle-1", run="middle-1#1",
                        parent_run=None, depth=1)

        _, events = await delegated(delegations(hook), request(
            {"agent_type": "leaf", "title": "Look", "objective": "Look X up."},
            caller=middle, caller_agent="middle", call_id="m1",
        ))

        assert asked[0].ref == RunRef(agent="leaf", instance="leaf-1", run="leaf-1#1",
                                      parent_run="middle-1#1", depth=2)
        assert events[0].metadata["started_by"] == "agent:middle-1"
        assert started_by(None) == "agent:root"

    async def test_parallel_runs_each_end_when_their_own_hook_returns(self):
        slow_may_end = asyncio.Event()

        async def hook(run: AgentRun) -> RunAnswer:
            if run.ref.instance == "researcher-2":
                await slow_may_end.wait()
            return RunAnswer(text=run.ref.instance)

        made = delegations(hook)
        live = LiveEvents()
        ended: list[str] = []

        async def source():
            token = set_active_turn(CONVERSATION)
            try:
                with live_events.reporting(live):
                    fast = asyncio.create_task(made.delegate(request(call_id="c1")))
                    slow = asyncio.create_task(made.delegate(request(call_id="c2")))
                    await fast
                    slow_may_end.set()
                    await slow
            finally:
                reset_active_turn(token)
            if False:  # pragma: no cover
                yield

        async for event in merged(source(), live):
            if event.metadata.get("status") == RUN_COMPLETED:
                ended.append(event.run.instance)
                if event.run.instance == "researcher-1":
                    assert not slow_may_end.is_set()  # the slow one still runs

        assert ended == ["researcher-1", "researcher-2"]


class TestAnInheritingType:
    async def test_its_history_is_read_once_and_its_message_ends_with_the_note(self):
        hook, asked = answering()
        read: list[int] = []

        def history() -> list[str]:
            read.append(1)
            return ["the conversation"]

        await delegated(delegations(hook), request(
            {"agent_type": "reader", "title": "Read", "objective": "Read it."}, history=history,
        ))

        (run,) = asked
        assert (run.history, read) == (["the conversation"], [1])
        assert run.message.endswith(INHERITED_CONTEXT_NOTE)

    async def test_a_fresh_type_never_reads_it(self):
        hook, asked = answering()

        def history() -> list[str]:
            raise AssertionError("a fresh run read the conversation")

        await delegated(delegations(hook), request(history=history))

        assert asked[0].history is None


class TestACancel:
    async def test_of_the_callers_task_ends_the_run_interrupted_and_goes_on(self):
        started, stopped = asyncio.Event(), asyncio.Event()

        async def hook(run: AgentRun) -> RunAnswer:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                stopped.set()
                raise
            return RunAnswer()  # pragma: no cover

        runs = AgentRuns()
        made = delegations(hook, runs=runs)
        live = LiveEvents()

        async def caller():
            token = set_active_turn(CONVERSATION)
            try:
                with live_events.reporting(live):
                    await made.delegate(request())
            finally:
                reset_active_turn(token)

        task = asyncio.create_task(caller())
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert stopped.is_set()  # the run stopped before the cancel went on
        (record,) = runs.records(CONVERSATION)
        assert (record.status, record.error) == (RUN_INTERRUPTED, INTERRUPTED_ERROR)

    async def test_of_the_runs_own_task_alone_is_read_by_its_caller(self):
        async def hook(run: AgentRun) -> RunAnswer:
            raise asyncio.CancelledError  # its task stopped, its caller not

        result, events = await delegated(delegations(hook), request())

        assert (result.result["status"], result.result["error"], result.result["success"]) == (
            RUN_INTERRUPTED, INTERRUPTED_ERROR, False
        )
        assert statuses(events) == [("agent_run", RUN_STARTED), ("agent_run", RUN_INTERRUPTED)]


async def test_a_delegation_blocks_no_event_loop():
    hook, _ = answering()

    with no_blocking_calls():
        result, _ = await delegated(delegations(hook), request())

    assert result.result["status"] == RUN_COMPLETED


def storing(*answers: Any) -> tuple[Callable[[SessionRef], Awaitable[dict[str, int]]], list[SessionRef]]:
    """A ``stored`` that answers each ask with the next of ``answers`` (an
    exception is raised, an ``asyncio.Event`` waited on, then the next
    answer given); the conversations it was asked about. Each ask yields to
    the loop before it answers, as a store's read does, so parallel runs
    meet while one asks."""
    asked: list[SessionRef] = []
    left = list(answers)

    async def stored(conversation: SessionRef) -> dict[str, int]:
        asked.append(conversation)
        answer = left.pop(0)
        if isinstance(answer, asyncio.Event):
            await answer.wait()
            answer = left.pop(0)
        await asyncio.sleep(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    return stored, asked


@contextlib.contextmanager
def a_turn() -> Iterator[None]:
    """A turn of ``CONVERSATION``, as a manager's has it: the active turn,
    its span open, its live events reported."""
    token = set_active_turn(CONVERSATION)
    span = open_turn_scope()
    try:
        with live_events.reporting(LiveEvents()):
            yield
    finally:
        close_turn_scope(span)
        reset_active_turn(token)


async def in_turn(made: Delegations, *asked: DelegationRequest) -> list[Any]:
    """Each of ``asked`` delegated at once in a turn of ``CONVERSATION``; their results."""
    with a_turn():
        return list(await asyncio.gather(*(made.delegate(one) for one in asked)))


@pytest.fixture
def logged(monkeypatch) -> list[dict]:
    """Every log call of ``agents/delegate.py``, with its level."""
    events: list[dict] = []

    class Recorder:
        def __getattr__(self, level: str):
            def log(event: str, **fields) -> None:
                events.append({"event": event, "log_level": level, **fields})

            return log

    monkeypatch.setattr(delegate_module, "logger", Recorder())
    return events


class TestNumberedPastTheStored:
    """A conversation's store may keep instances another manager ran (a
    restart, a switch of backend or store, another process that had the
    conversation between turns): ``stored`` says how many of each type,
    asked once per conversation per turn, before the turn's first run is
    named."""

    async def test_the_first_run_is_numbered_past_them(self):
        hook, _ = answering()
        stored, asked = storing({"researcher": 2})

        (result,) = await in_turn(delegations(hook, stored=stored), request())

        assert (result.result["agent"], result.run.run) == ("researcher-3", "researcher-3#1")
        assert asked == [CONVERSATION]

    async def test_it_is_asked_once_per_conversation_per_turn(self):
        """A turn's runs ask once; the next turn asks again, since another
        process may have run the conversation in between."""
        hook, _ = answering()
        stored, asked = storing({"researcher": 2}, {"researcher": 7})
        made = delegations(hook, stored=stored)

        with a_turn():
            first = [await made.delegate(request(call_id=c)) for c in ("c1", "c2")]
        later = await in_turn(made, request(call_id="c3"))

        assert [r.result["agent"] for r in first + later] == [
            "researcher-3", "researcher-4", "researcher-8",
        ]
        assert asked == [CONVERSATION, CONVERSATION]

    async def test_parallel_first_runs_ask_once_and_both_are_numbered_past_them(self):
        hook, _ = answering()
        stored, asked = storing({"researcher": 2})

        results = await in_turn(
            delegations(hook, stored=stored), request(call_id="c1"), request(call_id="c2")
        )

        assert sorted(r.result["agent"] for r in results) == ["researcher-3", "researcher-4"]
        assert asked == [CONVERSATION]

    async def test_a_brief_that_cannot_start_a_run_asks_nothing(self):
        hook, _ = answering()
        stored, asked = storing()

        (result,) = await in_turn(
            delegations(hook, stored=stored), request({"agent_type": "researcher"})
        )

        assert (result.run, asked) == (None, [])

    async def test_a_store_that_cannot_be_read_is_asked_again_at_the_next_run(self):
        hook, _ = answering()
        stored, asked = storing(RuntimeError("database is locked"), {"researcher": 5})
        made = delegations(hook, stored=stored)

        with a_turn():
            results = [await made.delegate(request(call_id=c)) for c in ("c1", "c2")]

        assert [r.result["agent"] for r in results] == ["researcher-1", "researcher-6"]
        assert asked == [CONVERSATION, CONVERSATION]

    async def test_a_store_that_cannot_be_read_is_warned_of_in_the_log(self, logged):
        hook, _ = answering()
        failure = RuntimeError("database is locked")
        stored, _ = storing(failure)

        (result,) = await in_turn(delegations(hook, stored=stored), request())

        assert result.result["status"] == RUN_COMPLETED
        assert [e for e in logged if e["log_level"] == "warning"] == [{
            "event": "subagent_instances_unread", "log_level": "warning",
            "session_id": CONVERSATION.session_id, "error": "database is locked",
        }]
        assert [
            e["exc_info"] for e in logged
            if (e["event"], e["log_level"]) == ("subagent_instances_unread", "debug")
        ] == [failure]

    async def test_an_ask_a_cancel_stopped_starts_no_run_and_is_asked_again(self):
        hook, ran = answering()
        waiting = asyncio.Event()
        stored, asked = storing(waiting, {"researcher": 1})
        runs = AgentRuns()
        made = delegations(hook, runs=runs, stored=stored)

        with a_turn():
            stopped = asyncio.create_task(made.delegate(request(call_id="c1")))
            while not asked:
                await asyncio.sleep(0)
            stopped.cancel()
            with pytest.raises(asyncio.CancelledError):
                await stopped
            result = await made.delegate(request(call_id="c2"))

        assert result.result["agent"] == "researcher-2"
        assert asked == [CONVERSATION, CONVERSATION]
        assert [run.ref.run for run in ran] == ["researcher-2#1"]
        assert [record.ref.run for record in runs.records(CONVERSATION)] == ["researcher-2#1"]

    async def test_without_a_turns_span_every_run_asks(self):
        """A conversation active with no turn's span open (a caller that set
        the active turn alone) has no turn to remember the ask for."""
        hook, _ = answering()
        stored, asked = storing({"researcher": 2}, {"researcher": 2})
        made = delegations(hook, stored=stored)

        token = set_active_turn(CONVERSATION)
        try:
            with live_events.reporting(LiveEvents()):
                results = [await made.delegate(request(call_id=c)) for c in ("c1", "c2")]
        finally:
            reset_active_turn(token)

        assert [r.result["agent"] for r in results] == ["researcher-3", "researcher-4"]
        assert asked == [CONVERSATION, CONVERSATION]

    async def test_the_managers_delegations_ask_its_stored_instances(self):
        """``BaseWorkflowManager._stored_instances``: the base keeps none; a
        backend that stores transcripts says how many of each type."""
        from agentic_cli.workflow.base_manager import BaseWorkflowManager
        from tests.conftest import MockContext

        class Storing(BaseWorkflowManager):
            backend_type = "storing"

            def _get_state_tools(self) -> list:
                return []

            async def _do_initialize(self) -> None:
                return None

            async def process(self, message, user_id, session_id=None):  # pragma: no cover
                yield WorkflowEvent.text(message)

            async def _run_agent(self, run: AgentRun) -> RunAnswer:
                return RunAnswer(text="X is 42.")

        with MockContext(google_api_key="test-key") as context:
            manager = Storing(list(TYPES.values()), settings=context.settings)
            base = await BaseWorkflowManager._stored_instances(manager, CONVERSATION)

            async def stored(conversation: SessionRef) -> dict[str, int]:
                return {"researcher": 4}

            manager._stored_instances = stored
            (result,) = await in_turn(manager._delegations, request())

        assert base == {}
        assert result.result["agent"] == "researcher-5"


# -- each run under its control: its limits, a cancel of the run alone --------------------


def waiting(text: str = "Half done.", *, lets_it_out: bool = False) -> tuple[Hook, asyncio.Event]:
    """A hook that works until cancelled, and the event set once it works.

    On its run's own cancel it returns ``text`` and the tasks it saved (or,
    ``lets_it_out``, lets the cancel out); any other cancel goes on.
    """
    working = asyncio.Event()

    async def hook(run: AgentRun) -> RunAnswer:
        working.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if lets_it_out or not run.control.own_cancel():
                raise
            return RunAnswer(text=text, state={"tasks": [{"description": "half"}]})
        return RunAnswer()  # pragma: no cover

    return hook, working


async def stopped(
    made: Delegations, working: asyncio.Event, *, by: str, error: str,
    asked: DelegationRequest | None = None, run: str = "researcher-1#1",
) -> tuple[Any, list[WorkflowEvent], bool]:
    """A delegation (``asked``, else the researcher's) in a turn of
    ``CONVERSATION`` whose run ``run`` is cancelled once it works: its result,
    what it reported, and what ``cancel`` returned."""
    live = LiveEvents()
    out: dict[str, Any] = {}

    async def source():
        token = set_active_turn(CONVERSATION)
        try:
            with live_events.reporting(live):
                running = asyncio.create_task(made.delegate(asked or request()))
                await working.wait()
                out["cancelled"] = made.cancel(run, by=by, error=error)
                out["result"] = await running
        finally:
            reset_active_turn(token)
        if False:  # pragma: no cover
            yield

    events = [item async for item in merged(source(), live)]
    return out["result"], events, out["cancelled"]


class TestARunsControl:
    async def test_the_hook_runs_with_its_types_limits_over_the_policys(self):
        hook, asked = answering()
        policy = SubAgentPolicy(limits=AgentLimits(max_model_calls=50, timeout=60.0))
        types = {
            **TYPES,
            "researcher": AgentConfig(name="researcher", prompt="p", description="Finds.",
                                      limits=AgentLimits(timeout=5.0)),
        }

        await delegated(delegations(hook, policy=policy, types=types), request())
        await delegated(delegations(hook, policy=policy, types=types), request(
            {"agent_type": "middle", "title": "Plan", "objective": "Plan X."}
        ))

        researcher, middle = asked
        assert (researcher.control.ref, researcher.control.limits) == (
            RESEARCHER, AgentLimits(max_model_calls=50, timeout=5.0)
        )
        assert middle.control.limits == AgentLimits(max_model_calls=50, timeout=60.0)

    async def test_by_default_a_run_makes_at_most_100_model_calls(self):
        hook, asked = answering()

        await delegated(delegations(hook), request())

        assert asked[0].control.limits == DEFAULT_LIMITS == AgentLimits(max_model_calls=100)

    async def test_a_working_runs_control_is_found_until_its_end(self):
        seen: list[Any] = []
        made: Delegations

        async def hook(run: AgentRun) -> RunAnswer:
            seen.append(made.control(run.ref) is run.control)
            seen.append(made.control(None))
            return RunAnswer(text="Done.")

        made = delegations(hook)
        await delegated(made, request())

        assert seen == [True, None]
        token = set_active_turn(CONVERSATION)
        try:
            assert made.control(RESEARCHER) is None
        finally:
            reset_active_turn(token)


class TestCancellingOneRun:
    async def test_a_hook_that_returns_ends_it_cancelled_with_what_it_had(self):
        hook, working = waiting("Half done.")
        runs = AgentRuns()

        result, events, cancelled = await stopped(
            delegations(hook, runs=runs), working, by="parent", error="no longer needed"
        )

        assert cancelled is True
        assert result.result == {
            "success": False, "agent": "researcher-1", "run": "researcher-1#1",
            "status": RUN_CANCELLED, "partial": True, "result": "Half done.",
            "error": "no longer needed", "cancelled_by": "parent",
        }
        assert result.state == {"tasks": [{"description": "half"}]}
        assert statuses(events) == [("agent_run", RUN_STARTED), ("agent_run", RUN_CANCELLED)]
        ended = events[-1]
        assert (ended.metadata["cancelled_by"], ended.metadata["error"],
                ended.metadata["partial"]) == ("parent", "no longer needed", True)
        (record,) = runs.records(CONVERSATION)
        assert (record.status, record.cancelled_by, record.error, record.result) == (
            RUN_CANCELLED, "parent", "no longer needed", "Half done."
        )

    async def test_a_hook_that_lets_the_cancel_out_ends_it_cancelled_with_nothing(self):
        hook, working = waiting(lets_it_out=True)

        result, _, _ = await stopped(delegations(hook), working, by="user", error="stopped")

        assert (result.result["status"], result.result["cancelled_by"], result.result["result"],
                result.result["partial"], result.state) == (
            RUN_CANCELLED, "user", "", False, {}
        )

    async def test_only_a_working_run_is_cancelled(self):
        hook, _ = answering()
        made = delegations(hook)
        await delegated(made, request())

        token = set_active_turn(CONVERSATION)
        try:
            assert made.cancel("researcher-1#1", by="parent", error="late") is False
            assert made.cancel("nobody-1#1", by="parent", error="who") is False
            with pytest.raises(ValueError, match="parent, user or limit"):
                made.cancel("researcher-1#1", by="someone", error="why")
        finally:
            reset_active_turn(token)

    async def test_the_turns_cancel_wins(self):
        """The run's own cancel and its turn's at once: the run is the turn's,
        interrupted, and the cancel goes on."""
        hook, working = waiting()
        runs = AgentRuns()
        made = delegations(hook, runs=runs)

        async def caller():
            token = set_active_turn(CONVERSATION)
            try:
                with live_events.reporting(LiveEvents()):
                    await made.delegate(request())
            finally:
                reset_active_turn(token)

        task = asyncio.create_task(caller())
        await asyncio.wait_for(working.wait(), 5)
        token = set_active_turn(CONVERSATION)
        try:
            made.cancel("researcher-1#1", by="parent", error="no longer needed")
        finally:
            reset_active_turn(token)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        (record,) = runs.records(CONVERSATION)
        assert (record.status, record.error, record.cancelled_by) == (
            RUN_INTERRUPTED, INTERRUPTED_ERROR, None
        )

    async def test_the_turns_cancel_while_the_hook_cleans_up_after_its_own_wins(self):
        """The run's own cancel taken, and its turn stopped while the hook
        cleans up after it: the run is the turn's, interrupted."""
        working, cleaning = asyncio.Event(), asyncio.Event()

        async def hook(run: AgentRun) -> RunAnswer:
            working.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if not run.control.own_cancel():
                    raise
                cleaning.set()
                await asyncio.sleep(0.2)  # stopping its work
                return RunAnswer(text="Half done.")
            return RunAnswer()  # pragma: no cover

        runs = AgentRuns()
        made = delegations(hook, runs=runs)

        async def caller():
            token = set_active_turn(CONVERSATION)
            try:
                with live_events.reporting(LiveEvents()):
                    await made.delegate(request())
            finally:
                reset_active_turn(token)

        task = asyncio.create_task(caller())
        await asyncio.wait_for(working.wait(), 5)
        token = set_active_turn(CONVERSATION)
        try:
            assert made.cancel("researcher-1#1", by="parent", error="no longer needed")
        finally:
            reset_active_turn(token)
        await asyncio.wait_for(cleaning.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        (record,) = runs.records(CONVERSATION)
        assert (record.status, record.cancelled_by) == (RUN_INTERRUPTED, None)

    async def test_a_run_cancelled_already_is_not_cancelled_again(self):
        working, cleaning = asyncio.Event(), asyncio.Event()
        out: dict[str, bool] = {}
        made: Delegations

        async def hook(run: AgentRun) -> RunAnswer:
            working.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if not run.control.own_cancel():
                    raise
                cleaning.set()
                await asyncio.sleep(0.05)
                return RunAnswer(text="Half done.")
            return RunAnswer()  # pragma: no cover

        async def body() -> Any:
            running = asyncio.create_task(made.delegate(request()))
            await working.wait()
            out["first"] = made.cancel("researcher-1#1", by="parent", error="stop")
            await cleaning.wait()
            out["again"] = made.cancel("researcher-1#1", by="user", error="stop it")
            return await running

        made = delegations(hook)
        token = set_active_turn(CONVERSATION)
        try:
            with live_events.reporting(LiveEvents()):
                result = await asyncio.wait_for(body(), 5)
        finally:
            reset_active_turn(token)

        assert out == {"first": True, "again": False}
        assert (result.result["cancelled_by"], result.result["error"]) == ("parent", "stop")

    @pytest.mark.parametrize("by, error", [
        ("parent", "no longer needed"), ("limit", TIME_LIMIT_ERROR), ("user", "stopped"),
    ])
    async def test_a_parent_cancelled_alone_ends_its_working_child_cancelled_by_its_parent(
        self, by, error
    ):
        """A parent run cancelled by its control while its child works (its
        parent's stop, a limit, the user's): the child's caller (the
        parent's call) is stopped, so the child ends cancelled by its
        parent, saying so; the parent ends cancelled with what it had."""
        child_working = asyncio.Event()
        made: Delegations

        async def hook(run: AgentRun) -> RunAnswer:
            if run.ref.agent == "middle":
                try:
                    # As a backend's task tool does: the delegation in a call of its own.
                    await asyncio.create_task(made.delegate(request(
                        {"agent_type": "leaf", "title": "Look", "objective": "Look X up."},
                        caller=run.ref, caller_agent="middle", call_id="m1",
                    )))
                except asyncio.CancelledError:
                    if not run.control.own_cancel():
                        raise
                    return RunAnswer(text="Planned half.")
                return RunAnswer(text="Planned.")  # pragma: no cover
            child_working.set()
            await asyncio.Event().wait()
            return RunAnswer()  # pragma: no cover

        runs = AgentRuns()
        made = delegations(hook, runs=runs)
        working = asyncio.Event()

        async def started() -> None:
            await child_working.wait()
            working.set()

        watcher = asyncio.create_task(started())
        result, _, cancelled = await stopped(
            made, working, by=by, error=error,
            asked=request({"agent_type": "middle", "title": "Plan", "objective": "Plan X."}),
            run="middle-1#1",
        )
        await watcher

        assert cancelled is True
        assert (result.result["status"], result.result["result"]) == (RUN_CANCELLED, "Planned half.")
        assert {
            r.ref.instance: (r.status, r.cancelled_by, r.error) for r in runs.records(CONVERSATION)
        } == {
            "middle-1": (RUN_CANCELLED, by, error),
            "leaf-1": (RUN_CANCELLED, "parent", "its parent's run was stopped"),
        }
        assert runs.instance(CONVERSATION, "leaf-1").status == INSTANCE_IDLE


class TestTheTimeLimit:
    async def test_past_it_the_run_is_cancelled_by_its_limit(self):
        hook, _ = waiting("So far: 42.")
        types = {
            **TYPES,
            "researcher": AgentConfig(name="researcher", prompt="p", description="Finds.",
                                      limits=AgentLimits(timeout=0.05)),
        }

        result, events = await asyncio.wait_for(
            delegated(delegations(hook, types=types), request()), 5
        )

        assert (result.result["status"], result.result["cancelled_by"], result.result["error"],
                result.result["result"]) == (
            RUN_CANCELLED, "limit", TIME_LIMIT_ERROR, "So far: 42."
        )
        assert statuses(events) == [("agent_run", RUN_STARTED), ("agent_run", RUN_CANCELLED)]

    async def test_a_run_whose_work_had_ended_keeps_its_answer(self):
        """The hook's work ended (``RunControl.finish``) and it still reads
        what the run did (ADK reads its child session's state) when the
        time limit fires and its parent stops it: the run completes with its
        answer and its state."""
        cancelled: list[bool] = []
        made: Delegations

        async def hook(run: AgentRun) -> RunAnswer:
            run.control.finish()
            cancelled.append(made.cancel("researcher-1#1", by="parent", error="no longer needed"))
            await asyncio.sleep(0.15)  # the read, past the time limit
            return RunAnswer(text="X is 42.", state={"tasks": [{"description": "find X"}]})

        types = {
            **TYPES,
            "researcher": AgentConfig(name="researcher", prompt="p", description="Finds.",
                                      limits=AgentLimits(timeout=0.05)),
        }
        made = delegations(hook, types=types)

        result, events = await asyncio.wait_for(delegated(made, request()), 5)

        assert cancelled == [False]
        assert (result.result["status"], result.result["result"], result.result["error"]) == (
            RUN_COMPLETED, "X is 42.", None
        )
        assert result.state == {"tasks": [{"description": "find X"}]}
        assert statuses(events) == [("agent_run", RUN_STARTED), ("agent_run", RUN_COMPLETED)]


class TestTheLimitsAtModelCalls:
    """What a backend's run reports to its control at each model call, and how the run ends."""

    async def test_a_run_that_made_its_last_allowed_call_completes_partial(self):
        async def hook(run: AgentRun) -> RunAnswer:
            calls = [run.control.model_call() for _ in range(3)]
            assert calls == [False, False, True]
            return RunAnswer(text="X is about 42.")

        policy = SubAgentPolicy(limits=AgentLimits(max_model_calls=3))
        result, events = await delegated(delegations(hook, policy=policy), request())

        assert result.result == {
            "success": True, "agent": "researcher-1", "run": "researcher-1#1",
            "status": RUN_COMPLETED, "partial": True, "result": "X is about 42.",
            "error": STEP_LIMIT_ERROR,
        }
        # No cut-off warning: the answer was not cut at its output limit.
        assert statuses(events) == [("agent_run", RUN_STARTED), ("agent_run", RUN_COMPLETED)]
        assert (events[-1].metadata["partial"], events[-1].metadata["error"]) == (
            True, STEP_LIMIT_ERROR
        )

    async def test_a_cut_last_answer_is_still_warned_of(self):
        async def hook(run: AgentRun) -> RunAnswer:
            run.control.model_call()
            return RunAnswer(text="X is\n\n[cut off at the output limit]", outcome=CUT)

        policy = SubAgentPolicy(limits=AgentLimits(max_model_calls=1))
        result, events = await delegated(delegations(hook, policy=policy), request())

        assert (result.result["partial"], result.result["error"]) == (True, STEP_LIMIT_ERROR)
        assert events[-1].content == "researcher-1's answer was cut off at the output limit"

    async def test_a_run_over_its_tokens_ends_cancelled_with_that_answer(self):
        async def hook(run: AgentRun) -> RunAnswer:
            run.control.used(80, 20)
            run.control.used(90, 30)
            return RunAnswer(text="Found it.")

        policy = SubAgentPolicy(limits=AgentLimits(max_tokens=150))
        result, events = await delegated(delegations(hook, policy=policy), request())

        assert (result.result["status"], result.result["cancelled_by"], result.result["error"],
                result.result["result"], result.result["partial"]) == (
            RUN_CANCELLED, "limit", TOKEN_LIMIT_ERROR, "Found it.", True
        )
        assert statuses(events) == [("agent_run", RUN_STARTED), ("agent_run", RUN_CANCELLED)]

    async def test_its_side_calls_count_toward_its_tokens(self):
        """The run's total is what its usage record counts: its tools' side
        calls and its summaries, recorded under it, count with its own calls
        (whose own rows are not counted twice); another run's do not."""
        ledgers = UsageLedgers()
        other = RunRef(agent="leaf", instance="leaf-1", run="leaf-1#1",
                       parent_run=None, depth=1)
        seen: list[Any] = []

        def recorded(purpose: str, tokens: int, run: RunRef | None = None) -> None:
            event = llm_usage_event("anthropic:claude", UsageTotals(input=tokens, calls=1))
            event.metadata["purpose"] = purpose
            event.run = run  # None: the run in progress
            ledgers.record(event)

        async def hook(run: AgentRun) -> RunAnswer:
            run.control.used(40, 10)
            recorded(PURPOSE_SUBAGENT, 50)  # that call's own row
            recorded(PURPOSE_TOOL, 30)  # its tool's side call
            recorded(PURPOSE_SUMMARIZATION, 30)  # a summary of its conversation
            recorded(PURPOSE_TOOL, 500, run=other)
            run.control.used(30, 0)  # 80 of its own, 60 besides: 140
            seen.append(run.control.cancelled)
            run.control.used(10, 5)  # 155
            return RunAnswer(text="Found it.")

        policy = SubAgentPolicy(limits=AgentLimits(max_tokens=150))
        result, _ = await delegated(delegations(hook, policy=policy, ledgers=ledgers), request())

        assert seen == [None]
        assert (result.result["status"], result.result["cancelled_by"],
                result.result["error"]) == (RUN_CANCELLED, "limit", TOKEN_LIMIT_ERROR)

    async def test_a_finishing_answer_over_its_tokens_completes_it_whole(self):
        async def hook(run: AgentRun) -> RunAnswer:
            run.control.used(80, 20)
            run.control.used(90, 30, calls=False)
            return RunAnswer(text="Found it.")

        policy = SubAgentPolicy(limits=AgentLimits(max_tokens=150))
        result, events = await delegated(delegations(hook, policy=policy), request())

        assert result.result == {
            "success": True, "agent": "researcher-1", "run": "researcher-1#1",
            "status": RUN_COMPLETED, "partial": False, "result": "Found it.", "error": None,
        }
        assert statuses(events) == [("agent_run", RUN_STARTED), ("agent_run", RUN_COMPLETED)]


# -- the runs one turn may start ------------------------------------------------------


async def in_a_turn(made: Delegations, *asked: DelegationRequest) -> list[Any]:
    """Each of ``asked`` delegated in order, in one turn of ``CONVERSATION``
    (its span open, as a manager's turn has it); their results."""
    live = LiveEvents()
    results: list[Any] = []

    async def source():
        token = set_active_turn(CONVERSATION)
        scope = open_turn_scope()
        try:
            with live_events.reporting(live):
                for one in asked:
                    results.append(await made.delegate(one))
        finally:
            close_turn_scope(scope)
            reset_active_turn(token)
        if False:  # pragma: no cover
            yield

    async for _ in merged(source(), live):
        pass
    return results


class TestTheRunsOfATurn:
    async def test_a_call_past_the_limit_starts_no_run(self):
        hook, asked = answering()
        runs = AgentRuns()
        made = delegations(hook, runs=runs, policy=SubAgentPolicy(max_runs_per_turn=2))

        *ran, refused = await in_a_turn(
            made, request(call_id="c1"), request(call_id="c2"), request(call_id="c3")
        )

        assert [r.result["status"] for r in ran] == [RUN_COMPLETED, RUN_COMPLETED]
        assert refused.result == {
            "success": False,
            "error": "This turn has already started 2 sub-agent runs, the most one turn may "
                     "start. Answer with what you have.",
        }
        assert (refused.run, refused.state) == (None, {})
        assert len(asked) == 2
        assert [r.ref.instance for r in runs.records(CONVERSATION)] == [
            "researcher-1", "researcher-2",
        ]

    async def test_a_brief_that_cannot_start_a_run_counts_nothing(self):
        hook, asked = answering()
        made = delegations(hook, policy=SubAgentPolicy(max_runs_per_turn=1))

        bad, good = await in_a_turn(
            made, request({"agent_type": "researcher", "title": "Find X"}), request()
        )

        assert bad.run is None and "missing: objective" in bad.result["error"]
        assert good.result["status"] == RUN_COMPLETED

    async def test_each_turn_starts_its_own(self):
        hook, asked = answering()
        made = delegations(hook, policy=SubAgentPolicy(max_runs_per_turn=1))

        (first,) = await in_a_turn(made, request(call_id="c1"))
        (second,) = await in_a_turn(made, request(call_id="c2"))

        assert [first.result["status"], second.result["status"]] == [RUN_COMPLETED] * 2

    async def test_a_run_started_by_a_run_counts_too(self):
        made: Delegations

        async def hook(run: AgentRun) -> RunAnswer:
            if run.ref.agent == "middle":
                await made.delegate(request(
                    {"agent_type": "leaf", "title": "Look", "objective": "Look X up."},
                    caller=run.ref, caller_agent="middle", call_id="m1",
                ))
            return RunAnswer(text="Done.")

        made = delegations(hook, policy=SubAgentPolicy(max_runs_per_turn=2))

        middle, refused = await in_a_turn(
            made,
            request({"agent_type": "middle", "title": "Plan", "objective": "Plan X."}),
            request(call_id="c2"),
        )

        assert middle.result["status"] == RUN_COMPLETED
        assert refused.run is None and "already started 2 sub-agent runs" in refused.result["error"]

    async def test_outside_a_turn_nothing_is_counted(self):
        hook, asked = answering()
        made = delegations(hook, policy=SubAgentPolicy(max_runs_per_turn=1))

        await delegated(made, request(call_id="c1"))
        await delegated(made, request(call_id="c2"))

        assert len(asked) == 2


# -- working slots: at most max_active runs of a conversation work at once ----------------

MIDDLE = {"agent_type": "middle", "title": "Plan", "objective": "Plan X."}
LEAF = {"agent_type": "leaf", "title": "Look", "objective": "Look X up."}


async def until(condition: Callable[[], bool], *, within: float = 5.0) -> None:
    """Wait until ``condition()`` holds, letting the other tasks run."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + within
    while not condition():
        if loop.time() > deadline:
            raise AssertionError(f"not within {within} s")
        await asyncio.sleep(0)


class Turn:
    """``work()`` in one turn of ``CONVERSATION`` (its span open), in a task of
    its own, to stop it as Ctrl+C does: what it returned, and every event
    reported meanwhile."""

    def __init__(self, work: Callable[[], Awaitable[Any]]) -> None:
        self.events: list[WorkflowEvent] = []
        self.result: Any = None
        self.task = asyncio.create_task(self._run(work))

    async def _run(self, work: Callable[[], Awaitable[Any]]) -> None:
        live = LiveEvents()

        async def source():
            token = set_active_turn(CONVERSATION)
            scope = open_turn_scope()
            try:
                with live_events.reporting(live):
                    self.result = await work()
            finally:
                close_turn_scope(scope)
                reset_active_turn(token)
            if False:  # pragma: no cover - a source that yields nothing itself
                yield

        async for event in merged(source(), live):
            self.events.append(event)

    def of(self, instance: str) -> list[str]:
        """The statuses ``instance``'s run reported, in order."""
        return [
            event.metadata["status"] for event in self.events
            if event.type is EventType.AGENT_RUN and event.run.instance == instance
        ]

    def order(self, *wanted: str) -> list[tuple[str, str]]:
        """Each run's ``wanted`` statuses as reported, in order: (instance, status)."""
        return [
            (event.run.instance, event.metadata["status"]) for event in self.events
            if event.type is EventType.AGENT_RUN and event.metadata["status"] in wanted
        ]


class Observed(AgentRuns):
    """``AgentRuns`` that also keep each ``AGENT_RUN`` they observe, in order:
    the events a stopped turn reports after its stop reach no consumer."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[tuple[str, str]] = []

    def observe(self, event: WorkflowEvent) -> None:
        self.seen.append((event.run.instance, event.metadata["status"]))
        super().observe(event)

    def of(self, instance: str) -> list[str]:
        """The statuses ``instance``'s run reported, in order."""
        return [status for name, status in self.seen if name == instance]


class Holding:
    """A hook whose runs each work until let go (``go[instance].set()``);
    ``working`` names the runs whose hook ran, in order."""

    def __init__(self) -> None:
        self.working: list[str] = []
        self.go: dict[str, asyncio.Event] = collections.defaultdict(asyncio.Event)

    async def __call__(self, run: AgentRun) -> RunAnswer:
        self.working.append(run.ref.instance)
        await self.go[run.ref.instance].wait()
        return RunAnswer(text=f"{run.ref.instance} done.")


def fanning_out(made: Callable[[], Delegations], worked: list[str]) -> Hook:
    """A hook whose ``middle`` runs each delegate two ``leaf`` runs at once and
    answer with their results; any other run answers at once. ``worked``
    gets each run's work in the order it was done, each with its slot: a
    leaf's answer, a middle's after its leaves (``middle-1 back``)."""

    async def hook(run: AgentRun) -> RunAnswer:
        if run.ref.agent != "middle":
            worked.append(run.ref.instance)
            return RunAnswer(text=f"{run.ref.instance} found it.")
        found = await asyncio.gather(*(
            made().delegate(request(
                LEAF, caller=run.ref, caller_agent="middle", call_id=f"{run.ref.instance}/{n}",
            ))
            for n in (1, 2)
        ))
        worked.append(f"{run.ref.instance} back")
        return RunAnswer(text=" ".join(one.result["result"] for one in found))

    return hook


class TestWorkingSlots:
    async def test_a_run_past_the_slots_waits_queued_then_runs(self):
        hook = Holding()
        runs = AgentRuns()
        made = delegations(hook, runs=runs, policy=SubAgentPolicy(max_active=2))
        pool = made.slots(CONVERSATION)

        turn = Turn(lambda: asyncio.gather(
            *(made.delegate(request(call_id=call)) for call in ("c1", "c2", "c3"))
        ))
        await until(lambda: len(hook.working) == 2 and pool.waiting == 1)
        assert hook.working == ["researcher-1", "researcher-2"]
        assert runs.record(CONVERSATION, "researcher-3").status == RUN_QUEUED
        hook.go["researcher-1"].set()
        await until(lambda: "researcher-3" in hook.working)
        assert runs.record(CONVERSATION, "researcher-3").status == RUN_RUNNING
        hook.go["researcher-2"].set()
        hook.go["researcher-3"].set()
        await asyncio.wait_for(turn.task, 5)

        assert turn.of("researcher-3") == [RUN_STARTED, RUN_QUEUED, RUN_RUNNING, RUN_COMPLETED]
        assert turn.of("researcher-1") == [RUN_STARTED, RUN_COMPLETED]
        assert [one.result["result"] for one in turn.result] == [
            "researcher-1 done.", "researcher-2 done.", "researcher-3 done.",
        ]
        assert (pool.free, pool.waiting) == (2, 0)

    async def test_a_run_holds_a_slot_while_it_works_and_the_root_none(self):
        seen: list[int] = []
        made: Delegations

        async def hook(run: AgentRun) -> RunAnswer:
            seen.append(made.slots(CONVERSATION).free)
            return RunAnswer(text="Done.")

        made = delegations(hook, policy=SubAgentPolicy(max_active=1))
        turn = Turn(lambda: made.delegate(request()))
        await asyncio.wait_for(turn.task, 5)

        assert seen == [0]  # its own; the root's work took none
        assert turn.of("researcher-1") == [RUN_STARTED, RUN_COMPLETED]
        assert made.slots(CONVERSATION).free == 1

    async def test_each_conversation_has_its_own_slots(self):
        hook, _ = answering()
        made = delegations(hook, policy=SubAgentPolicy(max_active=3))

        one, other = made.slots(CONVERSATION), made.slots(SessionRef("app", "u", "other"))

        assert one is made.slots(CONVERSATION) and one is not other
        assert (one.size, other.size) == (3, 3)

    async def test_nested_fan_out_with_one_slot_completes(self):
        """Spec 13: a parent lends its slot while it waits on its children, so
        two runs that each delegate two at once finish with one slot."""
        made = delegations(fanning_out(lambda: made, []), policy=SubAgentPolicy(max_active=1))

        turn = Turn(lambda: asyncio.gather(
            made.delegate(request(MIDDLE, call_id="c1")),
            made.delegate(request(MIDDLE, call_id="c2")),
        ))
        await asyncio.wait_for(turn.task, 5)

        assert [one.result["result"] for one in turn.result] == [
            "leaf-1 found it. leaf-2 found it.", "leaf-3 found it. leaf-4 found it.",
        ]
        pool = made.slots(CONVERSATION)
        assert (pool.free, pool.waiting) == (1, 0)

    async def test_a_parent_whose_children_are_done_goes_ahead_of_runs_that_have_not_started(
        self,
    ):
        """middle-1's last leaf hands its slot back to middle-1 while leaf-3
        and leaf-4 wait: middle-1 does its work before they start."""
        worked: list[str] = []
        made = delegations(fanning_out(lambda: made, worked), policy=SubAgentPolicy(max_active=1))

        turn = Turn(lambda: asyncio.gather(
            made.delegate(request(MIDDLE, call_id="c1")),
            made.delegate(request(MIDDLE, call_id="c2")),
        ))
        await asyncio.wait_for(turn.task, 5)

        # middle-2 takes middle-1's slot, lent while middle-1 waits on its
        # leaves; leaf-1 takes middle-2's; each queued run says when it runs.
        assert turn.order(RUN_RUNNING) == [
            ("middle-2", RUN_RUNNING), ("leaf-1", RUN_RUNNING), ("leaf-2", RUN_RUNNING),
            ("leaf-3", RUN_RUNNING), ("leaf-4", RUN_RUNNING),
        ]
        # Taking its slot back is no new status: the work shows it.
        assert worked == [
            "leaf-1", "leaf-2", "middle-1 back", "leaf-3", "leaf-4", "middle-2 back",
        ]

    async def test_time_spent_queued_is_not_counted_against_the_timeout(self):
        hook = Holding()
        types = {
            **TYPES,
            "researcher": AgentConfig(name="researcher", prompt="p", description="Finds.",
                                      limits=AgentLimits(timeout=0.2)),
        }
        made = delegations(hook, types=types, policy=SubAgentPolicy(max_active=1))
        hook.go["researcher-1"].set()  # answers as soon as it has a slot

        turn = Turn(lambda: asyncio.gather(
            made.delegate(request(MIDDLE, call_id="c1")),
            made.delegate(request(call_id="c2")),
        ))
        await until(lambda: made.slots(CONVERSATION).waiting == 1)
        await asyncio.sleep(0.3)  # past the researcher's time limit, queued
        hook.go["middle-1"].set()
        await asyncio.wait_for(turn.task, 5)

        assert turn.of("researcher-1") == [RUN_STARTED, RUN_QUEUED, RUN_RUNNING, RUN_COMPLETED]

    async def test_queued_runs_block_no_event_loop(self):
        hook = Holding()
        made = delegations(hook, policy=SubAgentPolicy(max_active=1))
        hook.go["researcher-2"].set()

        with no_blocking_calls():
            turn = Turn(lambda: asyncio.gather(
                made.delegate(request(call_id="c1")), made.delegate(request(call_id="c2"))
            ))
            await until(lambda: made.slots(CONVERSATION).waiting == 1)
            hook.go["researcher-1"].set()
            await asyncio.wait_for(turn.task, 5)

        assert turn.of("researcher-2") == [RUN_STARTED, RUN_QUEUED, RUN_RUNNING, RUN_COMPLETED]


class TestACancelWhileQueued:
    async def test_the_turns_leaves_the_heap_and_the_run_never_starts(self):
        hook = Holding()
        runs = AgentRuns()
        made = delegations(hook, runs=runs, policy=SubAgentPolicy(max_active=1))
        pool = made.slots(CONVERSATION)

        turn = Turn(lambda: asyncio.gather(
            made.delegate(request(call_id="c1")), made.delegate(request(call_id="c2"))
        ))
        await until(lambda: hook.working == ["researcher-1"] and pool.waiting == 1)
        turn.task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn.task

        assert hook.working == ["researcher-1"]
        assert [(r.ref.instance, r.status) for r in runs.records(CONVERSATION)] == [
            ("researcher-1", RUN_INTERRUPTED), ("researcher-2", RUN_INTERRUPTED),
        ]
        assert (pool.free, pool.waiting) == (1, 0)

    async def test_its_controls_ends_it_cancelled_with_nothing_and_the_rest_go_on(self):
        hook = Holding()
        made = delegations(hook, policy=SubAgentPolicy(max_active=1))
        pool = made.slots(CONVERSATION)

        turn = Turn(lambda: asyncio.gather(
            made.delegate(request(call_id="c1")), made.delegate(request(call_id="c2"))
        ))
        await until(lambda: hook.working == ["researcher-1"] and pool.waiting == 1)
        token = set_active_turn(CONVERSATION)
        try:
            assert made.cancel("researcher-2#1", by="parent", error="no longer needed") is True
        finally:
            reset_active_turn(token)
        await until(lambda: pool.waiting == 0)
        hook.go["researcher-1"].set()
        await asyncio.wait_for(turn.task, 5)

        first, second = turn.result
        assert first.result["status"] == RUN_COMPLETED
        assert (second.result["status"], second.result["cancelled_by"], second.result["result"],
                second.result["partial"]) == (RUN_CANCELLED, "parent", "", False)
        assert turn.of("researcher-2") == [RUN_STARTED, RUN_QUEUED, RUN_CANCELLED]
        assert hook.working == ["researcher-1"]
        assert (pool.free, pool.waiting) == (1, 0)

    async def test_ctrl_c_with_a_run_queued_and_another_lending_leaves_no_waiter(self):
        """middle-1 lends its slot to researcher-1 while leaf-1
        waits for one; Ctrl+C stops all three, leaf-1 never starts, and the next
        turn's delegation gets a slot at once."""
        hook = Holding()
        runs = AgentRuns()
        made: Delegations

        async def runs_or_delegates(run: AgentRun) -> RunAnswer:
            if run.ref.agent == "middle":
                await made.delegate(request(
                    LEAF, caller=run.ref, caller_agent="middle", call_id="m1",
                ))
            return await hook(run)

        made = delegations(runs_or_delegates, runs=runs, policy=SubAgentPolicy(max_active=1))
        pool = made.slots(CONVERSATION)

        turn = Turn(lambda: asyncio.gather(
            made.delegate(request(MIDDLE, call_id="c1")), made.delegate(request(call_id="c2"))
        ))
        await until(lambda: "researcher-1" in hook.working and pool.waiting == 1)
        assert runs.record(CONVERSATION, "leaf-1").status == RUN_QUEUED
        turn.task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn.task

        assert "leaf-1" not in hook.working
        assert {r.ref.instance: r.status for r in runs.records(CONVERSATION)} == {
            "middle-1": RUN_INTERRUPTED, "researcher-1": RUN_INTERRUPTED,
            "leaf-1": RUN_INTERRUPTED,
        }
        assert (pool.free, pool.waiting) == (1, 0)

        hook.go["researcher-2"].set()
        after = Turn(lambda: made.delegate(request(call_id="c3")))
        await asyncio.wait_for(after.task, 5)

        assert after.of("researcher-2") == [RUN_STARTED, RUN_COMPLETED]


# -- a run waiting on the user lends its slot ------------------------------------------------


class TestWaitingOnTheUser:
    """A sub-agent run that waits on the user (``waiting_on_user``: a
    question, a permission prompt) says ``input_required``, lends its slot
    meanwhile, and takes it back ahead of the runs that have not started
    (``running``), ``queued`` while it waits for it."""

    async def test_a_run_lends_its_slot_and_takes_it_back_ahead_of_new_runs(self):
        hook = Holding()
        runs = AgentRuns()
        answered = asyncio.Event()

        async def asking(run: AgentRun) -> RunAnswer:
            if run.ref.instance != "researcher-1":
                return await hook(run)
            async with waiting_on_user():
                await answered.wait()
            hook.working.append("researcher-1 back")
            return RunAnswer(text="Asked.")

        made = delegations(asking, runs=runs, policy=SubAgentPolicy(max_active=1))
        pool = made.slots(CONVERSATION)
        hook.go["researcher-3"].set()

        turn = Turn(lambda: asyncio.gather(
            *(made.delegate(request(call_id=call)) for call in ("c1", "c2", "c3"))
        ))
        await until(lambda: hook.working == ["researcher-2"] and pool.waiting == 1)
        assert runs.record(CONVERSATION, "researcher-1").status == RUN_INPUT_REQUIRED
        answered.set()
        await until(lambda: pool.waiting == 2)  # researcher-1 wants it back; researcher-3 waits
        # Answered, it waits on its slot, no longer on the user.
        assert runs.record(CONVERSATION, "researcher-1").status == RUN_QUEUED
        hook.go["researcher-2"].set()
        await asyncio.wait_for(turn.task, 5)

        assert hook.working == ["researcher-2", "researcher-1 back", "researcher-3"]
        assert turn.of("researcher-1") == [
            RUN_STARTED, RUN_INPUT_REQUIRED, RUN_QUEUED, RUN_RUNNING, RUN_COMPLETED,
        ]
        assert (pool.free, pool.waiting) == (1, 0)

    async def test_a_question_asked_while_it_waits_for_its_slot_back_is_input_required_again(
        self,
    ):
        """Two calls of one response that each ask: the second asks while the
        first, answered, waits for its slot back. The run waits on the user
        again, and says so; once both are answered and its slot is back, it
        is ``running``."""
        hook = Holding()
        runs = AgentRuns()
        first_answered, second_answered = asyncio.Event(), asyncio.Event()
        second_asks = asyncio.Event()

        async def asks(answered: asyncio.Event) -> None:
            async with waiting_on_user():
                await answered.wait()

        async def asking(run: AgentRun) -> RunAnswer:
            if run.ref.instance != "researcher-1":
                return await hook(run)

            async def second() -> None:
                await second_asks.wait()
                await asks(second_answered)

            await asyncio.gather(asks(first_answered), second())
            return RunAnswer(text="Asked twice.")

        made = delegations(asking, runs=runs, policy=SubAgentPolicy(max_active=1))
        pool = made.slots(CONVERSATION)

        turn = Turn(lambda: asyncio.gather(
            *(made.delegate(request(call_id=call)) for call in ("c1", "c2"))
        ))
        await until(lambda: hook.working == ["researcher-2"])
        first_answered.set()
        await until(lambda: pool.waiting == 1)  # researcher-1 wants its slot back
        assert runs.record(CONVERSATION, "researcher-1").status == RUN_QUEUED
        second_asks.set()
        await until(lambda: runs.record(CONVERSATION, "researcher-1").status
                    == RUN_INPUT_REQUIRED)
        second_answered.set()
        await until(lambda: pool.waiting == 1)
        hook.go["researcher-2"].set()
        await asyncio.wait_for(turn.task, 5)

        assert turn.of("researcher-1") == [
            RUN_STARTED, RUN_INPUT_REQUIRED, RUN_QUEUED, RUN_INPUT_REQUIRED, RUN_QUEUED,
            RUN_RUNNING, RUN_COMPLETED,
        ]
        assert (pool.free, pool.waiting) == (1, 0)

    async def test_nested_waits_count_once(self):
        async def hook(run: AgentRun) -> RunAnswer:
            async with waiting_on_user():
                async with waiting_on_user():  # a prompt, its turn behind another's
                    pass
            return RunAnswer(text="Asked.")

        made = delegations(hook)
        turn = Turn(lambda: made.delegate(request()))
        await asyncio.wait_for(turn.task, 5)

        assert turn.of("researcher-1") == [
            RUN_STARTED, RUN_INPUT_REQUIRED, RUN_RUNNING, RUN_COMPLETED,
        ]

    async def test_the_roots_own_wait_lends_nothing_and_says_nothing(self):
        live = LiveEvents()

        async def source():
            with live_events.reporting(live):
                async with waiting_on_user():
                    pass
            if False:  # pragma: no cover - a source that yields nothing itself
                yield

        assert [event async for event in merged(source(), live)] == []

    async def test_a_child_ending_while_its_parent_waits_on_the_user_gives_its_slot_to_the_pool(
        self,
    ):
        answered = asyncio.Event()
        runs = AgentRuns()
        made: Delegations

        async def hook(run: AgentRun) -> RunAnswer:
            if run.ref.agent != "middle":
                return RunAnswer(text="Found.")

            async def ask() -> None:
                async with waiting_on_user():
                    await answered.wait()

            await asyncio.gather(
                made.delegate(request(LEAF, caller=run.ref, caller_agent="middle", call_id="m1")),
                ask(),
            )
            return RunAnswer(text="Planned.")

        made = delegations(hook, runs=runs, policy=SubAgentPolicy(max_active=1))
        pool = made.slots(CONVERSATION)
        turn = Turn(lambda: made.delegate(request(MIDDLE)))
        await until(lambda: runs.record(CONVERSATION, "leaf-1") is not None
                    and runs.record(CONVERSATION, "leaf-1").status == RUN_COMPLETED)

        assert pool.free == 1  # not handed back: middle-1 still waits on the user
        answered.set()
        await asyncio.wait_for(turn.task, 5)

        assert turn.of("middle-1") == [RUN_STARTED, RUN_INPUT_REQUIRED, RUN_RUNNING, RUN_COMPLETED]
        assert (pool.free, pool.waiting) == (1, 0)

    async def test_ctrl_c_while_a_run_waits_on_the_user_and_another_is_queued(self):
        """researcher-1 lends its slot to researcher-2 while
        the user is slow to answer and researcher-3 waits for a slot; Ctrl+C
        stops all three, researcher-3 never starts, nothing waits, and
        researcher-1 takes nothing back."""
        hook = Holding()
        runs = Observed()

        async def asking(run: AgentRun) -> RunAnswer:
            if run.ref.instance == "researcher-1":
                async with waiting_on_user():
                    await asyncio.Event().wait()  # the user is slow to answer
            return await hook(run)

        made = delegations(asking, runs=runs, policy=SubAgentPolicy(max_active=1))
        pool = made.slots(CONVERSATION)

        turn = Turn(lambda: asyncio.gather(
            *(made.delegate(request(call_id=call)) for call in ("c1", "c2", "c3"))
        ))
        await until(lambda: hook.working == ["researcher-2"] and pool.waiting == 1)
        turn.task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn.task

        assert hook.working == ["researcher-2"]
        assert runs.of("researcher-1") == [RUN_STARTED, RUN_INPUT_REQUIRED, RUN_INTERRUPTED]
        assert runs.of("researcher-3") == [RUN_STARTED, RUN_QUEUED, RUN_INTERRUPTED]
        assert [r.status for r in runs.records(CONVERSATION)] == [RUN_INTERRUPTED] * 3
        assert (pool.free, pool.waiting) == (1, 0)

    async def test_a_wait_on_the_user_blocks_no_event_loop(self):
        async def hook(run: AgentRun) -> RunAnswer:
            async with waiting_on_user():
                await asyncio.sleep(0)
            return RunAnswer(text="Asked.")

        made = delegations(hook, policy=SubAgentPolicy(max_active=1))
        with no_blocking_calls():
            turn = Turn(lambda: made.delegate(request()))
            await asyncio.wait_for(turn.task, 5)

        assert turn.of("researcher-1")[-1] == RUN_COMPLETED


# -- instances: a follow-up is an instance's next run; the status a run leaves it in ---------

CHECKER = AgentConfig(name="checker", prompt="p", description="Checks.", continuable=False)
WITH_CHECKER = {
    **TYPES,
    "coordinator": AgentConfig(
        name="coordinator", prompt="p", sub_agents=["researcher", "middle", "reader", "checker"]
    ),
    "checker": CHECKER,
}
CHECK = {"agent_type": "checker", "title": "Check", "objective": "Check X."}


def follow(
    instance: str = "researcher-1", message: str = "Go on.", **fields: Any
) -> FollowUpRequest:
    """The root's follow-up of ``instance`` (its call ``c2``), unless ``fields`` say otherwise."""
    return FollowUpRequest(
        instance=instance, message=message,
        started_by=fields.pop("started_by", "agent:root"),
        call_id=fields.pop("call_id", "c2"), **fields,
    )


def users(
    instance: str = "researcher-1", message: str = "Go on.", **fields: Any
) -> FollowUpRequest:
    """The user's follow-up of ``instance``: no caller, no call."""
    return FollowUpRequest(instance=instance, message=message, started_by="user", **fields)


async def turned(work: Callable[[], Awaitable[Any]]) -> tuple[Any, list[WorkflowEvent]]:
    """``work()`` in one turn of ``CONVERSATION`` (its span open, its live events
    reported): what it returned, and what was reported."""
    live = LiveEvents()
    out: dict[str, Any] = {}

    async def source():
        token = set_active_turn(CONVERSATION)
        scope = open_turn_scope()
        try:
            with live_events.reporting(live):
                out["result"] = await work()
        finally:
            close_turn_scope(scope)
            reset_active_turn(token)
        if False:  # pragma: no cover
            yield

    events = [item async for item in merged(source(), live)]
    return out["result"], events


def instance_events(events: list[WorkflowEvent]) -> list[tuple[str, str, str | None]]:
    """Each ``AGENT_INSTANCE``: its instance, its status, its reason."""
    return [
        (e.run.instance, e.metadata["status"], e.metadata.get("reason"))
        for e in events if e.type is EventType.AGENT_INSTANCE
    ]


class TestAFollowUp:
    async def test_it_is_the_instances_next_run_through_the_same_hook(self):
        hook, asked = answering()
        runs = AgentRuns()
        made = delegations(hook, runs=runs)

        async def work():
            await made.delegate(request())
            return await made.follow_up(follow(state={"outside_content": "web_fetch"}))

        result, events = await turned(work)

        second = RunRef(agent="researcher", instance="researcher-1", run="researcher-1#2",
                        parent_run=None, depth=1)
        assert result.result == {
            "success": True, "agent": "researcher-1", "run": "researcher-1#2",
            "status": RUN_COMPLETED, "partial": False, "result": "X is 42.", "error": None,
        }
        assert result.run == second
        first_run, second_run = asked
        assert (second_run.ref, second_run.follow_up, second_run.message, second_run.history) == (
            second, True, "Go on.", None
        )
        assert second_run.state == {"outside_content": "web_fetch"}
        assert first_run.follow_up is False
        started = runs_of_status(events, RUN_STARTED)[1]
        assert started.metadata == {
            "status": RUN_STARTED, "agent": "researcher", "partial": False, "call_id": "c2",
            "title": "Go on.", "started_by": "agent:root", "mode": "blocking",
            "input": "Go on.",
        }
        assert runs_of_status(events, RUN_COMPLETED)[1].metadata["instance_status"] == (
            INSTANCE_IDLE
        )
        record = runs.record(CONVERSATION, "researcher-1#2")
        assert (record.input, record.started_by, record.title) == ("Go on.", "agent:root", "Go on.")
        instance = runs.instance(CONVERSATION, "researcher-1")
        assert (instance.status, instance.runs, instance.title) == (
            INSTANCE_IDLE, ["researcher-1#1", "researcher-1#2"], "Find X",
        )

    async def test_its_run_names_the_run_that_sent_it_as_its_parent(self):
        made: Delegations

        async def hook(run: AgentRun) -> RunAnswer:
            if run.ref.run == "middle-1#1":
                await made.delegate(request(LEAF, caller=run.ref, caller_agent="middle",
                                            call_id="m1"))
            elif run.ref.run == "middle-1#2":
                await made.follow_up(follow("leaf-1", caller=run.ref, call_id="m2",
                                            started_by="agent:middle-1"))
            return RunAnswer(text=run.ref.run or "")

        runs = AgentRuns()
        made = delegations(hook, runs=runs)

        async def work():
            await made.delegate(request(MIDDLE))
            return await made.follow_up(follow("middle-1"))

        await turned(work)

        record = runs.record(CONVERSATION, "leaf-1#2")
        assert record.ref == RunRef(agent="leaf", instance="leaf-1", run="leaf-1#2",
                                    parent_run="middle-1#2", depth=2)
        assert record.started_by == "agent:middle-1"
        assert runs.instance(CONVERSATION, "leaf-1").parent == "middle-1"

    async def test_an_agents_follow_up_counts_toward_its_turns_runs_and_the_users_does_not(self):
        hook, asked = answering()
        made = delegations(hook, policy=SubAgentPolicy(max_runs_per_turn=1))

        async def work():
            await made.delegate(request())
            return [await made.follow_up(follow()), await made.follow_up(users())]

        (agents, user), _ = await turned(work)

        assert agents.result == {
            "success": False, "error": RUNS_PER_TURN_ERROR.format(limit=1),
        }
        assert agents.run is None
        assert (user.result["status"], user.run.run) == (RUN_COMPLETED, "researcher-1#2")
        assert len(asked) == 2

    async def test_the_user_may_continue_an_instance_the_user_stopped(self):
        hook, working = waiting()
        runs = AgentRuns()
        made = delegations(hook, runs=runs)
        await stopped(made, working, by="user", error="stopped by the user")
        assert runs.instance(CONVERSATION, "researcher-1").status == INSTANCE_STOPPED

        async def answered(run: AgentRun) -> RunAnswer:
            return RunAnswer(text="Went on.")

        made._hook = answered
        result, _ = await turned(lambda: made.follow_up(users()))

        assert (result.result["status"], result.result["result"]) == (RUN_COMPLETED, "Went on.")
        assert runs.record(CONVERSATION, "researcher-1#2").started_by == "user"
        assert runs.instance(CONVERSATION, "researcher-1").status == INSTANCE_IDLE

    async def test_a_conversation_named_is_the_runs_outside_its_turn(self):
        hook, asked = answering()
        runs = AgentRuns()
        made = delegations(hook, runs=runs)
        await turned(lambda: made.delegate(request()))

        result = await made.follow_up(users(conversation=CONVERSATION))

        assert result.result["run"] == "researcher-1#2"
        assert asked[-1].conversation == CONVERSATION
        assert runs.record(CONVERSATION, "researcher-1#2").status == RUN_COMPLETED
        assert get_active_turn() is None

    async def test_a_follow_up_blocks_no_event_loop(self):
        hook, _ = answering()
        made = delegations(hook)
        await turned(lambda: made.delegate(request()))

        with no_blocking_calls():
            result, _ = await turned(lambda: made.follow_up(follow()))

        assert result.result["status"] == RUN_COMPLETED


def runs_of_status(events: list[WorkflowEvent], status: str) -> list[WorkflowEvent]:
    return [e for e in events if e.type is EventType.AGENT_RUN and e.metadata["status"] == status]


class TestARefusedFollowUp:
    """A follow-up that cannot start a run is its error result, and nothing
    else happens: no run, no record, no event."""

    @staticmethod
    async def refused(made: Delegations, asked: FollowUpRequest, hook_calls: list) -> Any:
        before = len(hook_calls)
        result, events = await turned(lambda: made.follow_up(asked))
        assert (result.run, result.state, events) == (None, {}, [])
        assert len(hook_calls) == before
        assert result.result["success"] is False
        return result.result["error"]

    async def test_an_unknown_instance_with_none_yet(self):
        hook, asked = answering()

        error = await self.refused(delegations(hook), follow("researcher-9"), asked)

        assert error == "'researcher-9' is not one of your sub-agents. You have no sub-agents yet."

    async def test_another_agents_child_names_the_senders_own(self):
        made: Delegations

        async def hook(run: AgentRun) -> RunAnswer:
            if run.ref.agent == "middle":
                await made.delegate(request(LEAF, caller=run.ref, caller_agent="middle",
                                            call_id="m1"))
            return RunAnswer(text="Done.")

        made = delegations(hook)
        await turned(lambda: made.delegate(request(MIDDLE)))
        await turned(lambda: made.delegate(request(call_id="c3")))

        error = await self.refused(made, follow("leaf-1"), [])

        assert error == "'leaf-1' is not one of your sub-agents. Yours: middle-1, researcher-1."

    async def test_the_users_unknown_one_names_every_instance(self):
        hook, asked = answering()
        made = delegations(hook)
        await turned(lambda: made.delegate(request()))

        error = await self.refused(made, users("writer-1"), asked)

        assert error == "'writer-1' is not one of your sub-agents. Yours: researcher-1."

    async def test_a_closed_one(self):
        hook, asked = answering()
        made = delegations(hook, types=WITH_CHECKER)
        await turned(lambda: made.delegate(request(CHECK)))

        error = await self.refused(made, users("checker-1"), asked)

        assert error == "checker-1 is closed; start a new one with task."

    async def test_one_the_user_stopped_sent_by_an_agent(self):
        hook, working = waiting()
        made = delegations(hook)
        await stopped(made, working, by="user", error="stopped by the user")

        error = await self.refused(made, follow(), [])

        assert error == "researcher-1 was stopped by the user; only the user can continue it."

    async def test_a_running_one(self):
        hook, working = waiting()
        runs = AgentRuns()
        made = delegations(hook, runs=runs)
        results: list[Any] = []

        async def work():
            running = asyncio.create_task(made.delegate(request()))
            await working.wait()
            results.append(await made.follow_up(follow()))
            running.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await running

        await turned(work)

        (result,) = results
        assert result.result == {
            "success": False,
            "error": "researcher-1 is running: an instance runs one run at a time.",
        }
        assert [r.ref.run for r in runs.records(CONVERSATION)] == ["researcher-1#1"]

    async def test_a_one_shot_instance_left_idle_is_closed_to_follow_ups(self):
        """A turn that stopped a one-shot run leaves its instance idle in the
        records (``interrupt_open``); it still cannot be continued."""
        hook, asked = answering()
        runs = AgentRuns()
        ref = runs.run_for(CONVERSATION, None, "c1", "checker")
        token = set_active_turn(CONVERSATION)
        try:
            runs.observe(WorkflowEvent.agent_run(
                ref, RUN_STARTED, title="Check", call_id="c1", continuable=False
            ))
            runs.observe(WorkflowEvent.agent_run(
                ref, RUN_COMPLETED, call_id="c1", instance_status=INSTANCE_IDLE
            ))
        finally:
            reset_active_turn(token)
        made = delegations(hook, runs=runs, types=WITH_CHECKER)

        error = await self.refused(made, users("checker-1"), asked)

        assert error == "checker-1 is closed; start a new one with task."

    async def test_an_instance_whose_type_is_gone_is_refused_and_never_left_running(self):
        hook, asked = answering()
        runs = AgentRuns()
        ref = runs.run_for(CONVERSATION, None, "c1", "ghost")
        token = set_active_turn(CONVERSATION)
        try:
            runs.observe(WorkflowEvent.agent_run(ref, RUN_STARTED, title="Boo", call_id="c1"))
            runs.observe(WorkflowEvent.agent_run(
                ref, RUN_COMPLETED, call_id="c1", instance_status=INSTANCE_IDLE
            ))
        finally:
            reset_active_turn(token)
        made = delegations(hook, runs=runs)

        error = await self.refused(made, users("ghost-1"), asked)

        assert error == "ghost-1's type 'ghost' is not one of this app's agents."
        assert runs.instance(CONVERSATION, "ghost-1").status == INSTANCE_IDLE

    async def test_two_at_once_start_one_run(self):
        answered, _ = answering()
        runs = AgentRuns()
        made = delegations(answered, runs=runs)
        await turned(lambda: made.delegate(request()))
        hook, working = waiting()
        made._hook = hook

        async def work():
            both = asyncio.gather(
                made.follow_up(follow(call_id="c2")), made.follow_up(follow(call_id="c3"))
            )
            await working.wait()
            made.cancel("researcher-1#2", by="parent", error="no longer needed")
            # Bounded: were both to start a run, the second would wait for ever.
            return await asyncio.wait_for(both, 5)

        (first, second), _ = await turned(work)

        assert first.result["status"] == RUN_CANCELLED
        assert second.result == {
            "success": False,
            "error": "researcher-1 is running: an instance runs one run at a time.",
        }
        assert [r.ref.run for r in runs.records(CONVERSATION)] == [
            "researcher-1#1", "researcher-1#2",
        ]

    @pytest.mark.parametrize("message", ["", "  \n"])
    async def test_a_blank_message(self, message):
        hook, asked = answering()
        made = delegations(hook)
        await turned(lambda: made.delegate(request()))

        error = await self.refused(made, follow(message=message), asked)

        assert error == "A follow-up needs a message."


class TestTheStatusARunLeavesItsInstanceIn:
    async def test_a_parents_cancel_leaves_it_idle(self):
        hook, working = waiting()
        runs = AgentRuns()

        _, events, _ = await stopped(delegations(hook, runs=runs), working, by="parent",
                                     error="no longer needed")

        assert events[-1].metadata["instance_status"] == INSTANCE_IDLE
        assert runs.instance(CONVERSATION, "researcher-1").status == INSTANCE_IDLE

    async def test_the_users_cancel_leaves_it_stopped(self):
        hook, working = waiting()
        runs = AgentRuns()

        _, events, _ = await stopped(delegations(hook, runs=runs), working, by="user",
                                     error="stopped by the user")

        assert events[-1].metadata["instance_status"] == INSTANCE_STOPPED
        assert runs.instance(CONVERSATION, "researcher-1").status == INSTANCE_STOPPED

    async def test_a_type_that_is_not_continuable_is_closed_at_its_runs_end(self):
        hook, _ = answering()
        runs = AgentRuns()
        made = delegations(hook, runs=runs, types=WITH_CHECKER)

        result, events = await turned(lambda: made.delegate(request(CHECK)))

        assert result.result["status"] == RUN_COMPLETED
        started, ended = events
        assert started.metadata["continuable"] is False
        assert ended.metadata["instance_status"] == INSTANCE_CLOSED
        record = runs.instance(CONVERSATION, "checker-1")
        assert (record.status, record.continuable) == (INSTANCE_CLOSED, False)


class TestMaxOpen:
    async def test_past_it_the_oldest_idle_instance_is_closed(self):
        hook, _ = answering()
        runs = AgentRuns()
        made = delegations(hook, runs=runs, policy=SubAgentPolicy(max_open=2))

        async def work():
            for call_id in ("c1", "c2", "c3"):
                await made.delegate(request(call_id=call_id))

        _, events = await turned(work)

        assert [(i.instance, i.status) for i in runs.instances(CONVERSATION)] == [
            ("researcher-1", INSTANCE_CLOSED), ("researcher-2", INSTANCE_IDLE),
            ("researcher-3", INSTANCE_IDLE),
        ]
        assert instance_events(events) == [("researcher-1", INSTANCE_CLOSED,
            "more than 2 sub-agents open: the oldest idle one is closed")]
        # Closed after the end of the run that took it past the limit.
        third_end = runs_of_status(events, RUN_COMPLETED)[2]
        assert events.index(third_end) < len(events) - 1
        assert third_end.metadata["instance_status"] == INSTANCE_IDLE

    async def test_the_oldest_is_the_one_whose_last_run_ended_first(self):
        hook, _ = answering()
        runs = AgentRuns()
        made = delegations(hook, runs=runs, policy=SubAgentPolicy(max_open=2))

        async def work():
            await made.delegate(request(call_id="c1"))
            await made.delegate(request(call_id="c2"))
            await made.follow_up(follow("researcher-1", call_id="c3"))
            await made.delegate(request(call_id="c4"))

        _, events = await turned(work)

        assert [i.instance for i in runs.instances(CONVERSATION)
                if i.status == INSTANCE_CLOSED] == ["researcher-2"]

    async def test_a_stopped_one_counts_and_is_never_the_one_closed(self):
        hook, working = waiting()
        runs = AgentRuns()
        made = delegations(hook, runs=runs, policy=SubAgentPolicy(max_open=1))
        await stopped(made, working, by="user", error="stopped by the user")

        async def answered(run: AgentRun) -> RunAnswer:
            return RunAnswer(text="Done.")

        made._hook = answered
        _, events = await turned(lambda: made.delegate(request(call_id="c2")))

        # The run's own instance is the only idle one: its end closes it.
        assert runs_of_status(events, RUN_COMPLETED)[0].metadata["instance_status"] == (
            INSTANCE_CLOSED
        )
        assert instance_events(events) == []
        assert [(i.instance, i.status) for i in runs.instances(CONVERSATION)] == [
            ("researcher-1", INSTANCE_STOPPED), ("researcher-2", INSTANCE_CLOSED),
        ]


class TestRankedByTheOrderRunsEnded:
    async def test_equal_timestamps_do_not_make_start_order_the_end_order(self):
        """``max_open`` closes the instance whose last run ended first, by the
        order the ends were recorded in, not by the clock."""
        hook, _ = answering()
        runs = AgentRuns()
        real = runs.observe
        fixed = datetime(2026, 1, 1)

        def same_clock(event: WorkflowEvent) -> None:
            event.timestamp = fixed
            real(event)

        runs.observe = same_clock  # type: ignore[method-assign]
        made = delegations(hook, runs=runs, policy=SubAgentPolicy(max_open=2))

        async def work():
            await made.delegate(request(call_id="c1"))
            await made.delegate(request(call_id="c2"))
            await made.follow_up(follow("researcher-1", call_id="c3"))
            await made.delegate(request(call_id="c4"))

        await turned(work)

        assert [i.instance for i in runs.instances(CONVERSATION)
                if i.status == INSTANCE_CLOSED] == ["researcher-2"]


class TestAClosedInstancesStop:
    """A stop recorded for an instance that can never run again is taken, so its
    calls are freed with the conversation's next clear."""

    async def test_a_one_shot_run_that_was_stopped(self):
        open_calls = OpenCalls()

        async def hook(run: AgentRun) -> RunAnswer:
            open_calls.cancelled(CONVERSATION, instance="checker-1")
            return RunAnswer(text="Done.")

        made = delegations(hook, types=WITH_CHECKER, open_calls=open_calls)

        await turned(lambda: made.delegate(request(CHECK)))

        assert open_calls.take_cancel(CONVERSATION, instance="checker-1") is False

    async def test_an_instance_that_close_closes(self):
        open_calls = OpenCalls()
        hook, _ = answering()
        made = delegations(hook, open_calls=open_calls)
        await turned(lambda: made.delegate(request()))
        open_calls.cancelled(CONVERSATION, instance="researcher-1")

        made.close(CONVERSATION, "researcher-1", reason="done")

        assert open_calls.take_cancel(CONVERSATION, instance="researcher-1") is False


class TestMaxOpenWithStoppedRuns:
    async def test_a_run_the_user_stopped_past_it_stays_stopped_and_the_oldest_idle_one_closes(self):
        hook, _ = answering()
        runs = AgentRuns()
        made = delegations(hook, runs=runs, policy=SubAgentPolicy(max_open=1))
        await turned(lambda: made.delegate(request()))
        waiter, working = waiting()
        made._hook = waiter

        _, events, _ = await stopped(
            made, working, by="user", error="stopped by the user",
            asked=request(call_id="c2"), run="researcher-2#1",
        )

        assert [(i.instance, i.status) for i in runs.instances(CONVERSATION)] == [
            ("researcher-1", INSTANCE_CLOSED), ("researcher-2", INSTANCE_STOPPED),
        ]
        assert events[-2].metadata["instance_status"] == INSTANCE_STOPPED
        assert instance_events(events) == [("researcher-1", INSTANCE_CLOSED,
            "more than 1 sub-agents open: the oldest idle one is closed")]

    async def test_with_none_idle_to_close_it_stays_stopped(self):
        runs = AgentRuns()
        made = delegations(waiting()[0], runs=runs, policy=SubAgentPolicy(max_open=1))
        for call_id, name in (("c1", "researcher-1#1"), ("c2", "researcher-2#1")):
            waiter, working = waiting()
            made._hook = waiter
            _, events, _ = await stopped(
                made, working, by="user", error="stopped by the user",
                asked=request(call_id=call_id), run=name,
            )

        assert [(i.instance, i.status) for i in runs.instances(CONVERSATION)] == [
            ("researcher-1", INSTANCE_STOPPED), ("researcher-2", INSTANCE_STOPPED),
        ]
        assert instance_events(events) == []


class TestClose:
    async def test_it_closes_an_idle_instance_and_its_descendants(self):
        made: Delegations

        async def hook(run: AgentRun) -> RunAnswer:
            if run.ref.agent == "middle":
                await made.delegate(request(LEAF, caller=run.ref, caller_agent="middle",
                                            call_id="m1"))
            return RunAnswer(text="Done.")

        runs = AgentRuns()
        made = delegations(hook, runs=runs)
        await turned(lambda: made.delegate(request(MIDDLE)))
        await turned(lambda: made.delegate(request(call_id="c2")))

        closed, events = await turned(
            lambda: _now(made.close(CONVERSATION, "middle-1", reason="closed by its parent"))
        )

        assert closed == ["middle-1", "leaf-1"]
        assert instance_events(events) == [
            ("middle-1", INSTANCE_CLOSED, "closed by its parent"),
            ("leaf-1", INSTANCE_CLOSED, "closed by its parent"),
        ]
        assert {i.instance: i.status for i in runs.instances(CONVERSATION)} == {
            "middle-1": INSTANCE_CLOSED, "leaf-1": INSTANCE_CLOSED,
            "researcher-1": INSTANCE_IDLE,
        }

    async def test_a_stopped_one_is_closed_too(self):
        hook, working = waiting()
        runs = AgentRuns()
        made = delegations(hook, runs=runs)
        await stopped(made, working, by="user", error="stopped by the user")

        assert made.close(CONVERSATION, "researcher-1", reason="done") == ["researcher-1"]
        assert runs.instance(CONVERSATION, "researcher-1").status == INSTANCE_CLOSED

    async def test_a_running_closed_or_unknown_one_is_not(self):
        hook, working = waiting()
        runs = AgentRuns()
        made = delegations(hook, runs=runs, types=WITH_CHECKER)
        hook_done, _ = answering()
        closes: list[list[str]] = []

        async def work():
            running = asyncio.create_task(made.delegate(request()))
            await working.wait()
            closes.append(made.close(CONVERSATION, "researcher-1", reason="x"))
            running.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await running

        await turned(work)
        made._hook = hook_done
        await turned(lambda: made.delegate(request(CHECK, call_id="c2")))
        closes.append(made.close(CONVERSATION, "checker-1", reason="x"))
        closes.append(made.close(CONVERSATION, "writer-1", reason="x"))

        assert closes == [[], [], []]


async def _now(value: Any) -> Any:
    """``value``, awaited: a synchronous call as ``turned``'s work."""
    return value


# -- the parent's tools: a message, a stop, a look; steering never lost ----------------------

STEERED = "Your message was given to its run before its next step."
FIND_Y = {"agent_type": "researcher", "title": "Find Y", "objective": "Find Y."}
PLAN = {"agent_type": "middle", "title": "Plan", "objective": "Plan X."}


def blocking(*runs: str) -> tuple[Hook, dict[str, asyncio.Event], list[AgentRun]]:
    """A hook whose ``runs`` (by id) work until cancelled, each setting its
    event once it works (and returning ``Half way.`` on its own cancel);
    every other run answers ``Did: <its message>``; the runs it was asked."""
    working = {run: asyncio.Event() for run in runs}
    asked: list[AgentRun] = []

    async def hook(run: AgentRun) -> RunAnswer:
        asked.append(run)
        if run.ref.run not in working:
            return RunAnswer(text=f"Did: {run.message}")
        working[run.ref.run].set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if not run.control.own_cancel():
                raise
            return RunAnswer(text="Half way.")
        return RunAnswer()  # pragma: no cover

    return hook, working, asked


class TestAMessage:
    """``message``, ``message_agent``'s: an idle instance's follow-up; a
    running one's run steered, or with ``interrupt`` stopped and followed
    up; one run per instance at a time."""

    async def test_to_an_idle_instance_is_its_follow_up(self):
        hook, _, asked = blocking()
        made = delegations(hook)

        async def work():
            await made.delegate(request())
            return await made.message(follow(message="Now find Y."))

        result, _ = await turned(work)

        assert result.result == {
            "success": True, "agent": "researcher-1", "run": "researcher-1#2",
            "status": RUN_COMPLETED, "partial": False, "result": "Did: Now find Y.",
            "error": None,
        }
        assert (asked[-1].follow_up, asked[-1].message) == (True, "Now find Y.")

    async def test_to_a_running_one_is_steered_to_its_run(self):
        hook, working, asked = blocking("researcher-1#2")
        runs = AgentRuns()
        made = delegations(hook, runs=runs)

        async def work():
            await made.delegate(request())
            running = asyncio.create_task(made.message(follow()))
            await working["researcher-1#2"].wait()
            told = await made.message(follow(message="Also Z.", call_id="c3"))
            steered = asked[-1].control.unstored()
            made.cancel("researcher-1#2", by="parent", error="enough")
            await running
            return told, steered

        (told, steered), _ = await turned(work)

        assert told.result == {
            "success": True, "agent": "researcher-1", "run": "researcher-1#2",
            "status": "running", "note": STEERED,
        }
        assert (told.run.run, told.state) == ("researcher-1#2", {})
        assert steered == ["Also Z."]
        assert [r.ref.run for r in runs.records(CONVERSATION)] == [
            "researcher-1#1", "researcher-1#2",
        ]

    async def test_two_at_once_to_an_idle_one_start_one_run_and_steer_the_other(self):
        """Two ``message_agent`` calls in one response: the first claims the
        instance as it starts its run, the second is steered into that run,
        and both get a result."""
        taken: list[list[str]] = []

        async def hook(run: AgentRun) -> RunAnswer:
            taken.append(run.control.take_steering())  # its first model call
            run.control.stored()
            return RunAnswer(text=f"Did: {run.message}")

        runs = AgentRuns()
        made = delegations(hook, runs=runs)

        async def work():
            await made.delegate(request())
            return await asyncio.gather(
                made.message(follow(message="Find Y.", call_id="c2")),
                made.message(follow(message="Find Z.", call_id="c3")),
            )

        (first, second), _ = await turned(work)

        assert (first.result["run"], first.result["status"], first.result["result"]) == (
            "researcher-1#2", RUN_COMPLETED, "Did: Find Y."
        )
        assert second.result == {
            "success": True, "agent": "researcher-1", "run": "researcher-1#2",
            "status": "running", "note": STEERED,
        }
        assert taken == [[], ["Find Z."]]
        assert [r.ref.run for r in runs.records(CONVERSATION)] == [
            "researcher-1#1", "researcher-1#2",
        ]

    async def test_with_interrupt_its_run_is_stopped_and_a_new_one_starts_with_it(self):
        hook, working, _ = blocking("researcher-1#2")
        runs = AgentRuns()
        made = delegations(hook, runs=runs)

        async def work():
            await made.delegate(request())
            running = asyncio.create_task(made.message(follow()))
            await working["researcher-1#2"].wait()
            replaced = await made.message(
                follow(message="Do Y instead.", call_id="c3"), interrupt=True
            )
            return await running, replaced

        (stopped_run, replaced), _ = await turned(work)

        assert stopped_run.result == {
            "success": False, "agent": "researcher-1", "run": "researcher-1#2",
            "status": RUN_CANCELLED, "partial": True, "result": "Half way.",
            "error": "stopped by its parent for a new message", "cancelled_by": "parent",
        }
        assert (replaced.result["run"], replaced.result["result"]) == (
            "researcher-1#3", "Did: Do Y instead."
        )
        third = runs.record(CONVERSATION, "researcher-1#3")
        assert (third.input, third.call_id) == ("Do Y instead.", "c3")
        assert runs.instance(CONVERSATION, "researcher-1").status == INSTANCE_IDLE

    async def test_a_message_blocks_no_event_loop(self):
        hook, working, _ = blocking("researcher-1#2")
        made = delegations(hook)

        async def work():
            await made.delegate(request())
            running = asyncio.create_task(made.message(follow()))
            await working["researcher-1#2"].wait()
            steered = await made.message(follow(message="Also Z.", call_id="c3"))
            replaced = await made.message(follow(message="Y.", call_id="c4"), interrupt=True)
            await running
            return steered, replaced

        with no_blocking_calls():
            (steered, replaced), _ = await turned(work)

        assert (steered.result["status"], replaced.result["run"]) == ("running", "researcher-1#3")

    async def test_with_interrupt_an_idle_one_just_runs(self):
        hook, _, _ = blocking()
        made = delegations(hook)

        async def work():
            await made.delegate(request())
            return await made.message(follow(message="Go on."), interrupt=True)

        result, _ = await turned(work)

        assert (result.result["run"], result.result["result"]) == ("researcher-1#2", "Did: Go on.")

    async def test_to_a_run_being_stopped_follows_it_up_once_it_has_ended(self):
        """A run its parent has just stopped takes no more steps: the message
        is not told as given to it, but starts the instance's next run."""
        hook, working, _ = blocking("researcher-1#2")
        runs = AgentRuns()
        made = delegations(hook, runs=runs)

        async def work():
            await made.delegate(request())
            running = asyncio.create_task(made.message(follow()))
            await working["researcher-1#2"].wait()
            stopped_it = made.stop(CONVERSATION, None, "researcher-1")
            told = await made.message(follow(message="Do Y instead.", call_id="c3"))
            return stopped_it, told, await running

        (stopped_it, told, ended), _ = await turned(work)

        assert stopped_it["stopped"] == "researcher-1#2"
        assert ended.result["status"] == RUN_CANCELLED
        assert (told.result["run"], told.result["result"]) == (
            "researcher-1#3", "Did: Do Y instead."
        )
        assert made._unread == {}

    async def test_to_a_run_being_closed_is_refused_as_closed(self):
        hook, working, _ = blocking("researcher-1#2")
        made = delegations(hook)

        async def work():
            await made.delegate(request())
            running = asyncio.create_task(made.message(follow()))
            await working["researcher-1#2"].wait()
            made.stop(CONVERSATION, None, "researcher-1", close=True)
            told = await made.message(follow(message="Do Y instead.", call_id="c3"))
            await running
            return told

        told, _ = await turned(work)

        assert told.result == {
            "success": False, "error": "researcher-1 is closed; start a new one with task.",
        }
        assert made._unread == {}

    async def test_an_interrupt_that_cannot_start_a_follow_up_stops_nothing(self):
        """A one-shot instance's run is not cancelled for a message it could
        not be followed up with."""
        hook, working, _ = blocking("checker-1#1")
        runs = AgentRuns()
        made = delegations(hook, runs=runs, types=WITH_CHECKER)

        async def work():
            running = asyncio.create_task(made.delegate(request(CHECK)))
            await working["checker-1#1"].wait()
            told = await made.message(follow("checker-1", "Y."), interrupt=True)
            still = runs.record(CONVERSATION, "checker-1#1").status
            made.cancel("checker-1#1", by="parent", error="enough")
            await running
            return told, still

        (told, still), _ = await turned(work)

        assert told.result == {
            "success": False, "error": "checker-1 is closed; start a new one with task.",
        }
        assert still == RUN_RUNNING

    async def test_the_users_interrupt_is_the_users_stop(self):
        hook, working, _ = blocking("researcher-1#2")
        made = delegations(hook)

        async def work():
            await made.delegate(request())
            running = asyncio.create_task(made.message(users()))
            await working["researcher-1#2"].wait()
            replaced = await made.message(users(message="Y instead."), interrupt=True)
            return await running, replaced

        (stopped_run, replaced), _ = await turned(work)

        assert (stopped_run.result["cancelled_by"], stopped_run.result["error"]) == (
            "user", "stopped by the user for a new message"
        )
        assert replaced.result["run"] == "researcher-1#3"

    async def test_the_user_may_message_one_the_user_stopped(self):
        hook, working = waiting()
        made = delegations(hook)
        await stopped(made, working, by="user", error="stopped by the user")
        made._hook, _ = answering("Went on.")

        result, _ = await turned(lambda: made.message(users()))

        assert (result.result["run"], result.result["result"]) == ("researcher-1#2", "Went on.")


class TestARefusedMessage:
    """A message that can start no run and steer none is its error result."""

    @staticmethod
    async def refused(made: Delegations, asked: FollowUpRequest) -> str:
        result, events = await turned(lambda: made.message(asked))
        assert (result.run, result.state, events) == (None, {}, [])
        assert result.result["success"] is False
        return result.result["error"]

    @pytest.mark.parametrize("message", ["", "  \n", None])
    async def test_a_blank_one(self, message):
        hook, _ = answering()
        made = delegations(hook)
        await turned(lambda: made.delegate(request()))

        assert await self.refused(made, follow(message=message)) == (
            "message_agent needs a message."
        )

    async def test_to_one_not_yours(self):
        hook, _ = answering()

        assert await self.refused(delegations(hook), follow("researcher-1")) == (
            "'researcher-1' is not one of your sub-agents. You have no sub-agents yet."
        )

    async def test_to_a_running_one_not_yours(self):
        """Another agent's running child is neither steered nor run."""
        leaf_working = asyncio.Event()
        made: Delegations
        refused: list[Any] = []

        async def hook(run: AgentRun) -> RunAnswer:
            if run.ref.agent == "middle":
                child = asyncio.create_task(made.delegate(request(
                    {"agent_type": "leaf", "title": "Look", "objective": "Look X up."},
                    caller=run.ref, caller_agent="middle", call_id="m1",
                )))
                await leaf_working.wait()
                refused.append(await made.message(follow("leaf-1")))
                child.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await child
                return RunAnswer(text="Planned.")
            leaf_working.set()
            await asyncio.Event().wait()
            return RunAnswer()  # pragma: no cover

        made = delegations(hook)
        await turned(lambda: made.delegate(request(PLAN)))

        (result,) = refused
        assert result.result == {
            "success": False,
            "error": "'leaf-1' is not one of your sub-agents. Yours: middle-1.",
        }

    async def test_to_a_closed_one(self):
        hook, _ = answering()
        made = delegations(hook, types=WITH_CHECKER)
        await turned(lambda: made.delegate(request(CHECK)))

        assert await self.refused(made, follow("checker-1")) == (
            "checker-1 is closed; start a new one with task."
        )

    async def test_to_one_the_user_stopped(self):
        hook, working = waiting()
        made = delegations(hook)
        await stopped(made, working, by="user", error="stopped by the user")

        assert await self.refused(made, follow()) == (
            "researcher-1 was stopped by the user; only the user can continue it."
        )

    async def test_past_the_runs_of_its_turn(self):
        hook, _ = answering()
        made = delegations(hook, policy=SubAgentPolicy(max_runs_per_turn=1))

        async def work():
            await made.delegate(request())
            return await made.message(follow())

        result, _ = await turned(work)

        assert result.result == {"success": False, "error": RUNS_PER_TURN_ERROR.format(limit=1)}


def stop_result(agent: str, status: str, stopped: str | None, closed: list[str]) -> dict:
    return {"success": True, "agent": agent, "status": status, "stopped": stopped,
            "closed": closed}


def with_a_leaf(made: Callable[[], Delegations], working: dict[str, asyncio.Event]) -> Hook:
    """A hook in which ``middle``'s first run delegates to ``leaf`` once; the
    runs named in ``working`` work until cancelled; any other answers."""
    async def hook(run: AgentRun) -> RunAnswer:
        if run.ref.run in working:
            working[run.ref.run].set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if not run.control.own_cancel():
                    raise
                return RunAnswer(text="Half way.")
        if run.ref.run == "middle-1#1":
            await made().delegate(request(
                {"agent_type": "leaf", "title": "Look", "objective": "Look X up."},
                caller=run.ref, caller_agent="middle", call_id="m1",
            ))
        return RunAnswer(text="Done.")

    return hook


class TestStop:
    """``stop``, ``stop_agent``'s: a running instance's run cancelled by its
    parent, the instance left idle; ``close`` closes it and its descendants
    (a running one's at its run's end)."""

    async def test_an_idle_one_has_nothing_to_stop(self):
        hook, _ = answering()
        made = delegations(hook)
        await turned(lambda: made.delegate(request()))

        result, events = await turned(lambda: _now(made.stop(CONVERSATION, None, "researcher-1")))

        assert result == stop_result("researcher-1", INSTANCE_IDLE, None, [])
        assert events == []

    async def test_close_closes_an_idle_one_and_its_descendants(self):
        made: Delegations
        runs = AgentRuns()
        made = delegations(with_a_leaf(lambda: made, {}), runs=runs)
        await turned(lambda: made.delegate(request(PLAN)))

        result, events = await turned(
            lambda: _now(made.stop(CONVERSATION, None, "middle-1", close=True))
        )

        assert result == stop_result("middle-1", INSTANCE_CLOSED, None, ["middle-1", "leaf-1"])
        assert instance_events(events) == [
            ("middle-1", INSTANCE_CLOSED, "closed by its parent"),
            ("leaf-1", INSTANCE_CLOSED, "closed by its parent"),
        ]

    async def test_a_running_ones_run_is_cancelled_by_its_parent_and_it_stays_open(self):
        hook, working, _ = blocking("researcher-1#2")
        runs = AgentRuns()
        made = delegations(hook, runs=runs)

        async def work():
            await made.delegate(request())
            running = asyncio.create_task(made.message(follow()))
            await working["researcher-1#2"].wait()
            result = made.stop(CONVERSATION, None, "researcher-1")
            return result, await running

        (result, ran), _ = await turned(work)

        assert result == stop_result("researcher-1", INSTANCE_IDLE, "researcher-1#2", [])
        assert (ran.result["status"], ran.result["cancelled_by"], ran.result["error"]) == (
            RUN_CANCELLED, "parent", "stopped by its parent"
        )
        assert runs.instance(CONVERSATION, "researcher-1").status == INSTANCE_IDLE

    async def test_with_close_a_running_one_is_closed_at_its_runs_end_with_its_descendants(self):
        made: Delegations
        runs = AgentRuns()
        working = {"middle-1#2": asyncio.Event()}
        made = delegations(with_a_leaf(lambda: made, working), runs=runs)

        async def work():
            await made.delegate(request(PLAN))
            running = asyncio.create_task(made.message(follow("middle-1")))
            await working["middle-1#2"].wait()
            result = made.stop(CONVERSATION, None, "middle-1", close=True)
            await running
            return result

        result, events = await turned(work)

        assert result == stop_result("middle-1", INSTANCE_CLOSED, "middle-1#2",
                                     ["middle-1", "leaf-1"])
        ended = runs_of_status(events, RUN_CANCELLED)[-1]
        assert (ended.run.run, ended.metadata["instance_status"]) == ("middle-1#2", INSTANCE_CLOSED)
        assert instance_events(events) == [("leaf-1", INSTANCE_CLOSED, "closed by its parent")]
        assert {i.instance: i.status for i in runs.instances(CONVERSATION)} == {
            "middle-1": INSTANCE_CLOSED, "leaf-1": INSTANCE_CLOSED,
        }

    async def test_closing_an_idle_one_closes_a_running_descendant_at_its_runs_end(self):
        """A descendant that works (the user's follow-up of it) when its
        ancestor is closed is closed as its run ends, not left idle under a
        closed parent."""
        made: Delegations
        runs = AgentRuns()
        working = {"leaf-1#2": asyncio.Event()}
        made = delegations(with_a_leaf(lambda: made, working), runs=runs)

        async def work():
            await made.delegate(request(PLAN))
            running = asyncio.create_task(made.message(users("leaf-1")))
            await working["leaf-1#2"].wait()
            result = made.stop(CONVERSATION, None, "middle-1", close=True)
            assert runs.instance(CONVERSATION, "leaf-1").status == INSTANCE_RUNNING
            made.cancel("leaf-1#2", by="parent", error="enough")
            await running
            return result

        result, events = await turned(work)

        assert result == stop_result("middle-1", INSTANCE_CLOSED, None, ["middle-1", "leaf-1"])
        ended = runs_of_status(events, RUN_CANCELLED)[-1]
        assert (ended.run.run, ended.metadata["instance_status"]) == ("leaf-1#2", INSTANCE_CLOSED)
        assert instance_events(events) == [("middle-1", INSTANCE_CLOSED, "closed by its parent")]
        assert {i.instance: i.status for i in runs.instances(CONVERSATION)} == {
            "middle-1": INSTANCE_CLOSED, "leaf-1": INSTANCE_CLOSED,
        }
        assert made._closing == {}

    async def test_refusals(self):
        hook, working = waiting()
        made = delegations(hook, types=WITH_CHECKER)
        await stopped(made, working, by="user", error="stopped by the user")
        made._hook, _ = answering()
        await turned(lambda: made.delegate(request(CHECK, call_id="c2")))

        token = set_active_turn(CONVERSATION)
        try:
            refusals = [
                made.stop(CONVERSATION, None, name)["error"]
                for name in ("researcher-1", "checker-1", "writer-1")
            ]
        finally:
            reset_active_turn(token)

        assert refusals == [
            "researcher-1 was stopped by the user; only the user can continue it.",
            "checker-1 is closed; start a new one with task.",
            "'writer-1' is not one of your sub-agents. Yours: researcher-1, checker-1.",
        ]


class TestCheck:
    """``check``, ``check_agents``': the caller's sub-agents, newest first,
    each its status, its last run and the tokens of all its runs."""

    @staticmethod
    def used(ledgers: UsageLedgers, tokens: tuple[int, int]) -> None:
        """A model call of the run in progress that used ``tokens``."""
        ledgers.record(llm_usage_event(
            "anthropic:claude", UsageTotals(input=tokens[0], output=tokens[1], calls=1)
        ))

    async def test_the_callers_sub_agents_newest_first(self):
        ledgers = UsageLedgers()
        made: Delegations

        async def hook(run: AgentRun) -> RunAnswer:
            self.used(ledgers, (100, 10))
            if run.ref.run == "middle-1#1":
                await made.delegate(request(
                    {"agent_type": "leaf", "title": "Look", "objective": "Look X up."},
                    caller=run.ref, caller_agent="middle", call_id="m1",
                ))
            return RunAnswer(text=f"{run.ref.run} done.\nSecond line.\nThird.\nFourth.")

        made = delegations(hook, ledgers=ledgers)

        async def work():
            await made.delegate(request())
            await made.message(follow(message="Now find Y."))
            await made.delegate(request(PLAN, call_id="c3"))
            return made.check(CONVERSATION, None)

        result, _ = await turned(work)

        assert result == {"success": True, "agents": [
            {"agent": "middle-1", "type": "middle", "status": INSTANCE_IDLE, "title": "Plan",
             "run": "middle-1#1", "run_status": RUN_COMPLETED,
             "result": "middle-1#1 done.\nSecond line.\nThird.",
             "tokens": {"input": 100, "output": 10}},
            {"agent": "researcher-1", "type": "researcher", "status": INSTANCE_IDLE,
             "title": "Find X", "run": "researcher-1#2", "run_status": RUN_COMPLETED,
             "result": "researcher-1#2 done.\nSecond line.\nThird.",
             "tokens": {"input": 200, "output": 20}},
        ]}

    async def test_one_of_them(self):
        hook, _ = answering()
        made = delegations(hook)

        async def work():
            await made.delegate(request())
            await made.delegate(request(PLAN, call_id="c2"))
            return made.check(CONVERSATION, None, "researcher-1")

        result, _ = await turned(work)

        assert [row["agent"] for row in result["agents"]] == ["researcher-1"]
        assert result["agents"][0]["tokens"] == {"input": 0, "output": 0}

    async def test_a_running_ones_run_says_so(self):
        hook, working, _ = blocking("researcher-1#2")
        made = delegations(hook)

        async def work():
            await made.delegate(request())
            running = asyncio.create_task(made.message(follow()))
            await working["researcher-1#2"].wait()
            result = made.check(CONVERSATION, None)
            made.cancel("researcher-1#2", by="parent", error="enough")
            await running
            return result

        result, _ = await turned(work)

        (row,) = result["agents"]
        assert (row["status"], row["run"], row["run_status"], row["result"]) == (
            "running", "researcher-1#2", RUN_RUNNING, ""
        )

    async def test_a_running_ones_tokens_come_from_the_ledger(self):
        ledgers = UsageLedgers()

        async def hook(run: AgentRun) -> RunAnswer:
            self.used(ledgers, (100, 10) if run.ref.run == "researcher-1#1" else (50, 5))
            if run.ref.run == "researcher-1#2":
                working.set()
                await asyncio.Event().wait()
            return RunAnswer(text="Done.")

        working = asyncio.Event()
        made = delegations(hook, ledgers=ledgers)

        async def work():
            await made.delegate(request())
            running = asyncio.create_task(made.message(follow()))
            await working.wait()
            result = made.check(CONVERSATION, None)
            made.cancel("researcher-1#2", by="parent", error="enough")
            await running
            return result

        result, _ = await turned(work)

        (row,) = result["agents"]
        assert row["tokens"] == {"input": 150, "output": 15}

    async def test_one_not_yours_and_none_yet(self):
        hook, _ = answering()
        made = delegations(hook)

        token = set_active_turn(CONVERSATION)
        try:
            assert made.check(CONVERSATION, None) == {"success": True, "agents": []}
            assert made.check(CONVERSATION, None, "leaf-1") == {
                "success": False,
                "error": "'leaf-1' is not one of your sub-agents. You have no sub-agents yet.",
            }
        finally:
            reset_active_turn(token)


class TestSteerAndCancelByConversation:
    async def test_a_working_run_is_steered_and_cancelled_from_outside_its_turn(self):
        hook, working, asked = blocking("researcher-1#1")
        made = delegations(hook)
        turn = Turn(lambda: made.delegate(request()))
        await asyncio.wait_for(working["researcher-1#1"].wait(), 5)

        assert get_active_turn() is None
        assert made.steer(CONVERSATION, "researcher-1#1", "Also Z.") is True
        assert asked[0].control.unstored() == ["Also Z."]
        assert made.steer(CONVERSATION, "researcher-9#1", "Also Z.") is False
        assert made.cancel(
            "researcher-1#1", by="user", error="stopped by the user", conversation=CONVERSATION
        ) is True
        await asyncio.wait_for(turn.task, 5)

        assert (turn.result.result["status"], turn.result.result["cancelled_by"]) == (
            RUN_CANCELLED, "user"
        )
        assert made.steer(CONVERSATION, "researcher-1#1", "Too late.") is False

    async def test_a_steered_message_is_text(self):
        hook, working, _ = blocking("researcher-1#1")
        made = delegations(hook)
        turn = Turn(lambda: made.delegate(request()))
        await asyncio.wait_for(working["researcher-1#1"].wait(), 5)

        with pytest.raises(ValueError, match="text"):
            made.steer(CONVERSATION, "researcher-1#1", " ")
        made.cancel("researcher-1#1", by="parent", error="enough", conversation=CONVERSATION)
        await asyncio.wait_for(turn.task, 5)


class TestASteeredMessageIsNeverLost:
    """A message steered to a run that its transcript does not hold when the
    run ends (never taken: no model call came after it; or taken by a call
    the run was stopped in before storing it) is given to the instance's
    next run, which takes it at its first model call."""

    @staticmethod
    def steering_hook(
        first: Callable[[AgentRun], Awaitable[None]]
    ) -> tuple[Hook, asyncio.Event, list[list[str]]]:
        working = asyncio.Event()
        later: list[list[str]] = []

        async def hook(run: AgentRun) -> RunAnswer:
            if run.ref.run == "researcher-1#1":
                working.set()
                try:
                    await first(run)
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    if not run.control.own_cancel():
                        raise
                    return RunAnswer(text="Half way.")
            later.append(run.control.take_steering())
            return RunAnswer(text="Went on.")

        return hook, working, later

    async def lost_and_given(
        self, first: Callable[[AgentRun], Awaitable[None]], *, steer_after: bool
    ) -> list[list[str]]:
        hook, working, later = self.steering_hook(first)
        made = delegations(hook)

        async def work():
            running = asyncio.create_task(made.delegate(request()))
            await working.wait()
            if steer_after:
                made.steer(CONVERSATION, "researcher-1#1", "Also Z.")
            await asyncio.sleep(0)
            made.cancel("researcher-1#1", by="parent", error="enough")
            await running
            await made.message(follow(message="Go on."))

        await turned(work)
        return later

    async def test_one_its_run_never_took_goes_to_its_next_run(self):
        async def nothing(run: AgentRun) -> None:
            pass

        assert await self.lost_and_given(nothing, steer_after=True) == [["Also Z."]]

    async def test_one_taken_and_not_stored_goes_to_its_next_run(self):
        async def takes(run: AgentRun) -> None:
            await until(lambda: bool(run.control.unstored()))
            run.control.take_steering()  # a model call stopped before it stored them

        assert await self.lost_and_given(takes, steer_after=True) == [["Also Z."]]

    async def test_one_stored_does_not(self):
        async def stores(run: AgentRun) -> None:
            await until(lambda: bool(run.control.unstored()))
            run.control.take_steering()
            run.control.stored()

        assert await self.lost_and_given(stores, steer_after=True) == [[]]

    async def test_a_closed_instances_are_dropped(self, logged):
        hook, working, _ = blocking("checker-1#1")
        made = delegations(hook, types=WITH_CHECKER)

        async def work():
            running = asyncio.create_task(made.delegate(request(CHECK)))
            await working["checker-1#1"].wait()
            made.steer(CONVERSATION, "checker-1#1", "Also Z.")
            made.cancel("checker-1#1", by="parent", error="enough")
            await running

        await turned(work)

        assert made._unread == {}
        assert {"event": "subagent_steering_dropped", "log_level": "info",
                "instance": "checker-1", "messages": 1} in logged


class TestOpenTasks:
    """A run that did not complete fully names, in its envelope, the tasks
    it left open (``open_tasks``), so its parent can continue it."""

    TASKS = [
        {"description": "find X", "status": "completed"},
        {"description": "find Y", "status": "in_progress"},
        {"description": "find Z", "status": "pending"},
    ]

    async def ended(self, answer: Callable[[AgentRun], Awaitable[RunAnswer]], **options: Any):
        async def hook(run: AgentRun) -> RunAnswer:
            return await answer(run)

        return await delegated(delegations(hook, **options), request())

    async def test_at_its_step_limit(self):
        async def answer(run: AgentRun) -> RunAnswer:
            run.control.model_call()
            return RunAnswer(text="X is 42.", tasks=self.TASKS)

        result, _ = await self.ended(
            answer, policy=SubAgentPolicy(limits=AgentLimits(max_model_calls=1))
        )

        assert (result.result["partial"], result.result["open_tasks"]) == (
            True, ["find Y", "find Z"]
        )

    async def test_cut_off(self):
        async def answer(run: AgentRun) -> RunAnswer:
            return RunAnswer(text="X is", outcome=CUT, tasks=self.TASKS)

        result, _ = await self.ended(answer)

        assert result.result["open_tasks"] == ["find Y", "find Z"]

    async def test_failed(self):
        async def answer(run: AgentRun) -> RunAnswer:
            return RunAnswer(text="X is", error="overloaded", tasks=self.TASKS)

        result, _ = await self.ended(answer)

        assert (result.result["status"], result.result["open_tasks"]) == (
            RUN_FAILED, ["find Y", "find Z"]
        )

    async def test_cancelled_it_keeps_its_plan_and_tasks(self):
        working = asyncio.Event()

        async def hook(run: AgentRun) -> RunAnswer:
            working.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if not run.control.own_cancel():
                    raise
                return RunAnswer(text="Half way.", plan="- [ ] find Y", tasks=self.TASKS)
            return RunAnswer()  # pragma: no cover

        runs = AgentRuns()
        result, events, _ = await stopped(
            delegations(hook, runs=runs), working, by="parent", error="enough"
        )

        assert result.result["open_tasks"] == ["find Y", "find Z"]
        assert list(result.result)[-2:] == ["cancelled_by", "open_tasks"]
        assert (events[-1].metadata["plan"], events[-1].metadata["tasks"]) == (
            "- [ ] find Y", self.TASKS
        )
        (record,) = runs.records(CONVERSATION)
        assert (record.plan, record.tasks) == ("- [ ] find Y", self.TASKS)

    @pytest.mark.parametrize("outcome", [None, REFUSED])
    async def test_none_for_a_run_that_completed_or_was_refused(self, outcome):
        async def answer(run: AgentRun) -> RunAnswer:
            return RunAnswer(text="X is 42.", outcome=outcome, tasks=self.TASKS)

        result, _ = await self.ended(answer)

        assert "open_tasks" not in result.result

    async def test_none_when_every_task_is_done(self):
        async def answer(run: AgentRun) -> RunAnswer:
            run.control.model_call()
            return RunAnswer(text="X is 42.", tasks=self.TASKS[:1])

        result, _ = await self.ended(
            answer, policy=SubAgentPolicy(limits=AgentLimits(max_model_calls=1))
        )

        assert result.result["partial"] is True and "open_tasks" not in result.result


class TestWhyARunStopped:
    """``stop_cause``: what stopped a working run, for a call it left at its
    prompt: its own control's cancel, or its parent's run stopped."""

    async def test_its_own_cancel_and_its_parents(self):
        made: Delegations
        seen: dict[str, Any] = {}

        async def hook(run: AgentRun) -> RunAnswer:
            if run.ref.agent == "middle":
                try:
                    await asyncio.create_task(made.delegate(request(
                        {"agent_type": "leaf", "title": "Look", "objective": "Look X up."},
                        caller=run.ref, caller_agent="middle", call_id="m1",
                    )))
                except asyncio.CancelledError:
                    seen["middle"] = made.stop_cause(run.ref)
                    if not run.control.own_cancel():
                        raise
                    return RunAnswer(text="Planned half.")
                return RunAnswer()  # pragma: no cover
            seen["working"] = True
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                seen["leaf"] = made.stop_cause(run.ref)
                raise
            return RunAnswer()  # pragma: no cover

        made = delegations(hook)
        working = asyncio.Event()

        async def started() -> None:
            await until(lambda: seen.get("working", False))
            working.set()

        watcher = asyncio.create_task(started())
        await stopped(made, working, by="limit", error=TIME_LIMIT_ERROR, asked=request(PLAN),
                      run="middle-1#1")
        await watcher

        assert (seen["middle"], seen["leaf"]) == ("its time limit", "its parent")
        token = set_active_turn(CONVERSATION)
        try:
            assert made.stop_cause(None) is None
            assert made.stop_cause(RESEARCHER) is None  # no working run
        finally:
            reset_active_turn(token)
