"""Each Deep Agents event is its run's: the namespace a sub-agent's work streams under names it.

A sub-agent runs inside its coordinator's ``task`` call, a tools task whose
namespace segment ``tools:<task id>`` its events stream under; the ``tasks``
stream announces that task with the call. The event processor names each
event's run from its namespace, follows a ``task`` call and its result with
the ``AGENT_RUN`` that starts and ends the run, and records it;
``ActiveToolCallMiddleware`` makes the run the active one around the
``task`` call, which the whole sub-agent runs inside.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("deepagents")

from langchain.agents.middleware import ToolCallRequest  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402

from agentic_cli.workflow.agents import (  # noqa: E402
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_STARTED,
    AgentRuns,
)
from agentic_cli.workflow.deepagents.event_processor import DeepAgentsEventProcessor  # noqa: E402
from agentic_cli.workflow.deepagents.middleware import ActiveToolCallMiddleware  # noqa: E402
from agentic_cli.workflow.events import EventType  # noqa: E402
from agentic_cli.workflow.sessions import (  # noqa: E402
    RunRef,
    get_active_run,
    reset_active_run,
    root_run,
    set_active_run,
)
from tests.workflow.deepagents.test_event_processor import (  # noqa: E402
    MODELS,
    NESTED,
    SUB,
    _call,
    _delegated,
    _model_update,
    _task_started,
    _tools_update,
    _types,
)

RESEARCHER = RunRef(
    agent="researcher", instance="researcher-1", run="researcher-1#1", parent_run=None, depth=1
)


def _processor(**kwargs) -> DeepAgentsEventProcessor:
    return DeepAgentsEventProcessor(root="coordinator", agent_models=MODELS, **kwargs)


def _delegating(call_id: str = "c1", brief: str = "Find X.\nThen report.") -> dict:
    return _model_update(AIMessage("", tool_calls=[
        _call("task", {"description": brief, "subagent_type": "researcher"}, call_id),
    ]))


class TestEveryEventIsItsRuns:
    def test_the_roots_events_are_its_own(self):
        (event,) = _processor().process((), "updates", _model_update(AIMessage("Hello.")))

        assert event.run == root_run("coordinator")

    def test_a_sub_agents_events_are_its_runs(self):
        processor = _processor()
        _delegated(processor, (), SUB, "researcher")

        (event,) = processor.process((SUB,), "updates", _model_update(AIMessage("It is 42.")))

        assert event.run == RESEARCHER

    def test_a_nested_run_is_its_parents_child(self):
        processor = _processor()
        _delegated(processor, (), SUB, "middle")
        _delegated(processor, (SUB,), NESTED, "leaf")

        (event,) = processor.process((SUB, NESTED), "updates", _model_update(AIMessage("42.")))

        assert event.run == RunRef(
            agent="leaf", instance="leaf-1", run="leaf-1#1", parent_run="middle-1#1", depth=2
        )

    def test_the_tasks_stream_finds_the_run_the_call_started(self):
        processor = _processor()
        _, started = processor.process((), "updates", _delegating(call_id="c-" + SUB))
        _delegated(processor, (), SUB, "researcher")  # announces the call "c-tools:4f2c"

        (event,) = processor.process((SUB,), "updates", _model_update(AIMessage("It is 42.")))

        assert event.run == started.run == RESEARCHER

    def test_a_failed_turns_error_is_the_managers(self):
        assert _processor().turn_failed(RuntimeError("boom")).run is None


class TestDelegationRuns:
    def test_a_task_call_starts_a_run_after_it(self):
        events = _processor().process((), "updates", _delegating())

        assert _types(events) == [EventType.TOOL_CALL, EventType.AGENT_RUN]
        call, started = events
        assert call.run == root_run("coordinator")
        assert started.run == RESEARCHER
        assert started.metadata == {
            "status": RUN_STARTED, "agent": "researcher", "partial": False,
            "call_id": "c1", "title": "Find X.",
        }

    def test_its_result_ends_it(self):
        runs = AgentRuns()
        processor = _processor(runs=runs)
        processor.process((), "updates", _delegating())

        events = processor.process((), "updates", _tools_update(
            ToolMessage("X is 42.", tool_call_id="c1", name="task")
        ))

        assert _types(events) == [EventType.TOOL_RESULT, EventType.AGENT_RUN]
        result, ended = events
        assert result.run == root_run("coordinator")
        assert ended.run == RESEARCHER
        assert (ended.metadata["status"], ended.metadata["result"]) == (RUN_COMPLETED, "X is 42.")
        (record,) = runs.records(None)
        assert (record.status, record.title, record.result) == (RUN_COMPLETED, "Find X.", "X is 42.")

    def test_an_error_result_ends_it_failed(self):
        processor = _processor()
        processor.process((), "updates", _delegating())

        _, ended = processor.process((), "updates", _tools_update(ToolMessage(
            "Error running sub-agent: overloaded", tool_call_id="c1", name="task", status="error"
        )))

        assert ended.metadata["status"] == RUN_FAILED
        assert (ended.metadata["error"], ended.metadata["result"]) == ("overloaded", "")

    def test_the_frameworks_error_result_ends_it_failed_with_its_error(self):
        """The gate's or the error middleware's result: an error message
        whose JSON holds the error."""
        processor = _processor()
        processor.process((), "updates", _delegating())

        _, ended = processor.process((), "updates", _tools_update(ToolMessage(
            json.dumps({"success": False, "error": "researcher failed: boom"}),
            tool_call_id="c1", name="task", status="error",
        )))

        assert ended.metadata["status"] == RUN_FAILED
        assert (ended.metadata["error"], ended.metadata["result"]) == (
            "researcher failed: boom", "",
        )

    def test_an_answer_that_says_it_failed_completes_it(self):
        """The sub-agent's answer is text, never read for success."""
        answer = json.dumps({"success": False, "error": "no X here"})
        processor = _processor()
        processor.process((), "updates", _delegating())

        result, ended = processor.process((), "updates", _tools_update(ToolMessage(
            answer, tool_call_id="c1", name="task",
        )))

        assert (ended.metadata["status"], ended.metadata["result"]) == (RUN_COMPLETED, answer)
        assert (result.metadata["result"], result.metadata["success"]) == (answer, True)

    def test_a_long_result_is_cut(self):
        processor = _processor()
        processor.process((), "updates", _delegating())

        _, ended = processor.process((), "updates", _tools_update(
            ToolMessage("z" * 500, tool_call_id="c1", name="task")
        ))

        assert ended.metadata["result"] == "z" * 200

    def test_another_tool_starts_no_run(self):
        events = _processor().process((), "updates", _model_update(
            AIMessage("", tool_calls=[_call("grep", {"pattern": "x"}, "c1")])
        ))

        assert _types(events) == [EventType.TOOL_CALL]

    def test_a_sub_agents_call_with_its_parents_call_id_is_its_own(self):
        """Call ids are only an agent's own: a provider that numbers them
        (``call_0``, ...) gives a sub-agent's first call its parent's task
        call's id. The sub-agent's result neither ends its own run nor lends
        its name to its parent's result."""
        processor = _processor()
        brief = {"description": "Find X.", "subagent_type": "researcher"}
        processor.process((), "updates", _model_update(
            AIMessage("", tool_calls=[_call("task", brief, "call_0")])
        ))
        processor.process((), "tasks", _task_started("4f2c", _call("task", brief, "call_0")))
        processor.process((SUB,), "updates", _model_update(
            AIMessage("", tool_calls=[_call("lookup", {"q": "X"}, "call_0")])
        ))

        inner = processor.process((SUB,), "updates", _tools_update(
            ToolMessage('{"success": true, "value": 42}', tool_call_id="call_0")
        ))
        outer = processor.process((), "updates", _tools_update(
            ToolMessage("X is 42.", tool_call_id="call_0")
        ))

        assert [(e.type, e.metadata.get("tool_name"), e.run) for e in inner] == [
            (EventType.TOOL_RESULT, "lookup", RESEARCHER),
        ]
        assert [(e.type, e.metadata.get("tool_name")) for e in outer] == [
            (EventType.TOOL_RESULT, "task"), (EventType.AGENT_RUN, None),
        ]
        assert (outer[1].run, outer[1].metadata["result"]) == (RESEARCHER, "X is 42.")


class TestOnlyTheCallersOwnSubAgents:
    """A run starts only for a ``task`` call to one of the calling agent's own
    sub-agents (``sub_agents``); Deep Agents answers any other type that it
    does not exist, and no sub-agent runs."""

    TREE = {"coordinator": ("middle",), "middle": ("leaf",)}

    def test_a_task_call_to_another_agents_sub_agent_is_a_tool_call(self):
        runs = AgentRuns()
        processor = _processor(runs=runs, sub_agents=self.TREE)
        brief = {"description": "Look X up.", "subagent_type": "leaf"}

        called = processor.process((), "updates", _model_update(
            AIMessage("", tool_calls=[_call("task", brief, "c1")])
        ))
        processor.process((), "tasks", _task_started("4f2c", _call("task", brief, "c1")))
        answered = processor.process((), "updates", _tools_update(ToolMessage(
            "We cannot invoke subagent leaf because it does not exist, the only allowed "
            "types are `middle`", tool_call_id="c1", name="task",
        )))

        assert _types(called + answered) == [EventType.TOOL_CALL, EventType.TOOL_RESULT]
        assert runs.records(None) == []
        assert runs.run_for(None, None, "c9", "leaf").run == "leaf-1#1"

    def test_a_sub_agent_delegates_to_its_own(self):
        processor = _processor(sub_agents=self.TREE)
        _delegated(processor, (), SUB, "middle")

        events = processor.process((SUB,), "updates", _model_update(AIMessage("", tool_calls=[
            _call("task", {"description": "Look X up.", "subagent_type": "leaf"}, "m1"),
        ])))

        assert _types(events) == [EventType.TOOL_CALL, EventType.AGENT_RUN]
        assert events[1].run == RunRef(
            agent="leaf", instance="leaf-1", run="leaf-1#1", parent_run="middle-1#1", depth=2
        )


def _request(name: str = "task", args: dict | None = None, call_id: str = "c1") -> ToolCallRequest:
    if args is None:
        args = {"description": "Find X.", "subagent_type": "researcher"}
    return ToolCallRequest(
        tool_call={"name": name, "args": args, "id": call_id, "type": "tool_call"},
        tool=None,
        state={},
        runtime=SimpleNamespace(tools=[]),
    )


class TestTheActiveRun:
    async def test_a_task_call_runs_as_its_sub_agents_run(self):
        seen = []

        async def handler(request):
            seen.append(get_active_run())
            return "done"

        await ActiveToolCallMiddleware(AgentRuns()).awrap_tool_call(_request(), handler)

        assert seen == [RESEARCHER]
        assert get_active_run() is None

    async def test_a_task_call_inside_a_run_starts_its_child(self):
        runs = AgentRuns()
        middle = runs.run_for(None, None, "c0", "middle")
        seen = []

        async def handler(request):
            seen.append(get_active_run())
            return "done"

        token = set_active_run(middle)
        try:
            await ActiveToolCallMiddleware(runs).awrap_tool_call(
                _request(args={"description": "d", "subagent_type": "leaf"}, call_id="m1"), handler
            )
            after = get_active_run()
        finally:
            reset_active_run(token)

        assert seen == [RunRef(
            agent="leaf", instance="leaf-1", run="leaf-1#1", parent_run="middle-1#1", depth=2
        )]
        assert after == middle

    async def test_another_tools_call_keeps_the_run_it_is_in(self):
        seen = []

        async def handler(request):
            seen.append(get_active_run())
            return "done"

        token = set_active_run(RESEARCHER)
        try:
            await ActiveToolCallMiddleware(AgentRuns()).awrap_tool_call(
                _request(name="grep", args={"pattern": "x"}), handler
            )
        finally:
            reset_active_run(token)

        assert seen == [RESEARCHER]

    async def test_without_the_runs_no_run_is_set(self):
        seen = []

        async def handler(request):
            seen.append(get_active_run())
            return "done"

        await ActiveToolCallMiddleware().awrap_tool_call(_request(), handler)

        assert seen == [None]

    async def test_a_raising_call_resets_it_too(self):
        async def handler(request):
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await ActiveToolCallMiddleware(AgentRuns()).awrap_tool_call(_request(), handler)

        assert get_active_run() is None

    async def test_a_task_call_to_another_agents_sub_agent_runs_as_no_run(self):
        runs = AgentRuns()
        seen = []

        async def handler(request):
            seen.append(get_active_run())
            return "We cannot invoke subagent leaf because it does not exist"

        await ActiveToolCallMiddleware(runs, sub_agents=("middle",)).awrap_tool_call(
            _request(args={"description": "d", "subagent_type": "leaf"}), handler
        )

        assert seen == [None]
        assert runs.run_for(None, None, "c9", "leaf").run == "leaf-1#1"

    async def test_a_task_call_to_its_own_sub_agent_runs_as_its_run(self):
        seen = []

        async def handler(request):
            seen.append(get_active_run())
            return "done"

        await ActiveToolCallMiddleware(AgentRuns(), sub_agents=("researcher",)).awrap_tool_call(
            _request(), handler
        )

        assert seen == [RESEARCHER]
