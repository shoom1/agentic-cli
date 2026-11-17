"""Each Deep Agents event is its run's: the root's own, or the sub-agent run whose graph streamed it.

A sub-agent run runs its agent's graph apart from its parent's stream, in
the run's own task (``agents/delegate.py``, ``DeepAgentsWorkflowManager._run_agent``),
through a processor made for that run: every event it makes is that run's.
The root's stream carries the root's own events alone. The ``AGENT_RUN``
events are the framework's. ``ActiveToolCallMiddleware`` makes each tool
call the active one around its tool, the ``task`` call included; the run it
starts is the active run in the run's own task, with no tool call active.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("deepagents")

from langchain.agents.middleware import ToolCallRequest  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from agentic_cli.workflow.deepagents.middleware import ActiveToolCallMiddleware  # noqa: E402
from agentic_cli.workflow.sessions import (  # noqa: E402
    ToolCallRef,
    get_active_run,
    get_active_tool_call,
    reset_active_run,
    root_run,
    set_active_run,
)
from tests.workflow.deepagents.test_event_processor import (  # noqa: E402
    RESEARCHER,
    _call,
    _model_update,
    _processor,
    _run_processor,
)


class TestEveryEventIsItsRuns:
    def test_the_roots_events_are_its_own(self):
        (event,) = _processor().process((), "updates", _model_update(AIMessage("Hello.")))

        assert event.run == root_run("coordinator")

    def test_a_sub_agent_runs_events_are_that_runs(self):
        events = _run_processor().process((), "updates", _model_update(
            AIMessage("Looking.", tool_calls=[_call("grep", {"pattern": "x"}, "r1")])
        ))

        assert [(e.type.value, e.run) for e in events] == [
            ("text", RESEARCHER), ("tool_call", RESEARCHER),
        ]

    def test_a_failed_turns_error_is_the_managers(self):
        assert _processor().turn_failed(RuntimeError("boom")).run is None


def _request(name: str = "task", args: dict | None = None, call_id: str = "c1") -> ToolCallRequest:
    return ToolCallRequest(
        tool_call=_call(name, args or {"agent_type": "researcher"}, call_id),
        tool=None, state={}, runtime=SimpleNamespace(tools=[]),
    )


class TestTheActiveToolCall:
    async def test_each_call_is_the_active_one_around_its_tool(self):
        seen = []

        async def handler(request):
            seen.append((get_active_tool_call(), get_active_run()))
            return "done"

        await ActiveToolCallMiddleware().awrap_tool_call(_request(), handler)

        assert seen == [(ToolCallRef("c1", "task"), None)]
        assert get_active_tool_call() is None

    async def test_a_call_keeps_the_run_it_is_in(self):
        seen = []

        async def handler(request):
            seen.append(get_active_run())
            return "done"

        token = set_active_run(RESEARCHER)
        try:
            await ActiveToolCallMiddleware().awrap_tool_call(
                _request(name="grep", args={"pattern": "x"}), handler
            )
        finally:
            reset_active_run(token)

        assert seen == [RESEARCHER]

    async def test_a_raising_call_resets_it_too(self):
        async def handler(request):
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await ActiveToolCallMiddleware().awrap_tool_call(_request(), handler)

        assert get_active_tool_call() is None
