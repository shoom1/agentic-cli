"""``Delegations``: the one place a sub-agent run starts, ends and is decided.

Driven with a fake hook and no backend: the brief is read against the
caller's own sub-agents, the run is named and its ``AGENT_RUN`` events are
recorded and reported to the turn's live events, the hook runs in a task of
its own (the run active there, no tool call), and the run ends as the hook's
answer says, with its envelope.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
from collections.abc import Awaitable, Callable, Iterator
from typing import Any

import pytest

from agentic_cli.workflow import live_events
from agentic_cli.workflow.agents import (
    DEFAULT_LIMITS,
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
    AgentRun,
    DelegationRequest,
    Delegations,
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
from agentic_cli.workflow.live_events import LiveEvents, merged
from agentic_cli.workflow.sessions import (
    RunRef,
    SessionRef,
    ToolCallRef,
    close_turn_scope,
    get_active_run,
    get_active_tool_call,
    open_turn_scope,
    reset_active_tool_call,
    reset_active_turn,
    set_active_tool_call,
    set_active_turn,
)
from agentic_cli.workflow.usage import UsageLedgers, UsageTotals, llm_usage_event
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
        }
        assert (ended.metadata["status"], ended.metadata["result"], ended.run) == (
            RUN_COMPLETED, "X is 42.", RESEARCHER
        )
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

    async def test_its_own_words_are_said_as_given(self):
        said = "researcher: the request no longer fits m's context window."

        async def hook(run: AgentRun) -> RunAnswer:
            raise RunFailed(said, code="CONTEXT_OVERFLOW")

        result, events = await delegated(delegations(hook), request())

        assert result.result["error"] == said
        warning = events[-1]
        assert (warning.content, warning.metadata["error_code"]) == (said, "CONTEXT_OVERFLOW")


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

    async def test_a_parent_cancelled_alone_ends_its_working_child_interrupted(self):
        """A parent run cancelled by its control while its child works: the
        child's caller (the parent's call) is stopped, so the child is the
        parent's, interrupted; the parent ends cancelled with what it had."""
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
            made, working, by="parent", error="no longer needed",
            asked=request({"agent_type": "middle", "title": "Plan", "objective": "Plan X."}),
            run="middle-1#1",
        )
        await watcher

        assert cancelled is True
        assert (result.result["status"], result.result["result"]) == (RUN_CANCELLED, "Planned half.")
        assert {r.ref.instance: (r.status, r.cancelled_by) for r in runs.records(CONVERSATION)} == {
            "middle-1": (RUN_CANCELLED, "parent"), "leaf-1": (RUN_INTERRUPTED, None),
        }


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
    (``running``)."""

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
        hook.go["researcher-2"].set()
        await asyncio.wait_for(turn.task, 5)

        assert hook.working == ["researcher-2", "researcher-1 back", "researcher-3"]
        assert turn.of("researcher-1") == [
            RUN_STARTED, RUN_INPUT_REQUIRED, RUN_RUNNING, RUN_COMPLETED,
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
