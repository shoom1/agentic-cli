"""A side call's usage: what it was for, whose run it was, which conversation's.

``generate_simple`` runs outside the agents: the summaries ``web_fetch`` and
the knowledge base ask for (``summarize``: ``tool``), the memory proposals as
a conversation ends (``on_session_end``: ``memory_extraction``, outside any
turn), an app's own (``side_call``). A backend reports its usage through
``_report_side_call``: recorded in the conversation's ledger, and in a turn
reported through its live events, as the run's in progress.
"""

from __future__ import annotations

import asyncio

from agentic_cli.workflow import live_events
from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.sessions import RunRef, reset_active_run, set_active_run
from agentic_cli.workflow.usage import (
    PURPOSE_MEMORY_EXTRACTION,
    PURPOSE_SIDE_CALL,
    PURPOSE_TOOL,
    SideCall,
    current_side_call,
    side_call,
)
from tests.workflow.test_memory_wiring import _manager

RESEARCHER = RunRef(
    agent="researcher", instance="researcher-1", run="researcher-1#1", parent_run=None, depth=1
)


def spent(prompt: int = 30, completion: int = 3) -> WorkflowEvent:
    return WorkflowEvent.llm_usage(model="m", prompt_tokens=prompt, completion_tokens=completion)


class TestWhatACallerSaysItsSideCallIsFor:
    async def test_summarize_s_are_a_tools(self, tmp_path, monkeypatch):
        mgr, _ = _manager(tmp_path)
        seen: list[SideCall | None] = []

        async def generate(prompt, max_tokens=500):
            seen.append(current_side_call())
            return "A summary."

        monkeypatch.setattr(mgr, "generate_simple", generate)
        await mgr.summarize("A page.", "Summarize: A page.")

        assert seen == [SideCall(PURPOSE_TOOL)]

    async def test_the_memory_proposals_are_their_conversations(self, tmp_path, monkeypatch):
        mgr, _ = _manager(tmp_path)
        ref = mgr.session_ref("conv-1", "u")
        seen: list[SideCall | None] = []

        async def generate(prompt, max_tokens=500):
            seen.append(current_side_call())
            return "[]"

        monkeypatch.setattr(mgr, "generate_simple", generate)
        await mgr.on_session_end([{"role": "user", "content": "..."}], session=ref)

        assert seen == [SideCall(PURPOSE_MEMORY_EXTRACTION, ref)]


class TestReportingASideCall:
    async def test_outside_a_turn_it_is_recorded_in_the_conversation_named(self, tmp_path):
        mgr, _ = _manager(tmp_path)
        ref = mgr.session_ref("conv-1", "u")
        live = live_events.LiveEvents()

        with live_events.reporting(live), side_call(PURPOSE_MEMORY_EXTRACTION, ref):
            mgr._report_side_call(spent())

        assert await reported(live) == []  # no turn's stream to go out with
        (row,) = mgr.usage_of(ref).rows
        assert (row.purpose, row.run, row.turn, row.input) == (
            PURPOSE_MEMORY_EXTRACTION, None, None, 30
        )
        assert mgr.usage.rows == ()  # not the open conversation's

    async def test_unsaid_it_is_the_open_conversations_side_call(self, tmp_path):
        mgr, _ = _manager(tmp_path)

        mgr._report_side_call(spent())

        (row,) = mgr.usage.rows
        assert row.purpose == PURPOSE_SIDE_CALL

    async def test_in_a_turn_it_is_the_runs_and_goes_out_with_the_turn(self, tmp_path):
        mgr, _ = _manager(tmp_path)
        live = live_events.LiveEvents()
        event = spent()

        with mgr._workflow_context(session_id="conv-1", user_id="u"), live_events.reporting(live):
            mgr.usage_of(mgr.session_ref("conv-1", "u")).begin_turn()
            token = set_active_run(RESEARCHER)
            try:
                mgr._report_side_call(event)
            finally:
                reset_active_run(token)

        (row,) = mgr.usage_of(mgr.session_ref("conv-1", "u")).rows
        assert (row.run, row.agent, row.purpose, row.turn) == (
            "researcher-1#1", "researcher", PURPOSE_SIDE_CALL, 1
        )
        assert event.run == RESEARCHER
        assert await reported(live) == [event]


    async def test_from_a_task_its_turn_outlived_it_is_recorded_but_not_reported(
        self, tmp_path
    ):
        # A tool's task that outlives its turn: the turn's stream has ended,
        # so the call is recorded in the conversation, with no turn, and
        # handed to no one.
        mgr, _ = _manager(tmp_path)
        live = live_events.LiveEvents()
        go = asyncio.Event()

        async def later() -> None:
            await go.wait()
            mgr._report_side_call(spent())

        with mgr._workflow_context(session_id="conv-1", user_id="u"), live_events.reporting(live):
            mgr.usage_of(mgr.session_ref("conv-1", "u")).begin_turn()
            task = asyncio.create_task(later())
        go.set()
        await task

        (row,) = mgr.usage_of(mgr.session_ref("conv-1", "u")).rows
        assert (row.purpose, row.turn) == (PURPOSE_SIDE_CALL, None)
        assert await reported(live) == []


async def reported(live: live_events.LiveEvents) -> list:
    """What ``live`` was handed, as the turn's stream yields it."""

    async def nothing_else():
        return
        yield

    return [item async for item in live_events.merged(nothing_else(), live)]
