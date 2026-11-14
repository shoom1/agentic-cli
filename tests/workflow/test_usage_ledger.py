"""Token usage: the ledger that keeps a conversation's model calls, a row each.

Each manager keeps a ledger per conversation (``UsageLedgers``), in memory for
the manager's life; it answers totals by run, agent, model, purpose and turn.
A side call's caller says what it is for (``side_call``).
"""

from __future__ import annotations

import contextvars

from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.sessions import (
    RunRef,
    SessionRef,
    close_turn_scope,
    get_active_run,
    open_turn_scope,
    reset_active_run,
    reset_active_turn,
    root_run,
    set_active_run,
    set_active_turn,
)
from agentic_cli.workflow.usage import (
    PURPOSE_MEMORY_EXTRACTION,
    PURPOSE_SUMMARIZATION,
    PURPOSE_TOOL,
    SideCall,
    UsageLedger,
    UsageLedgers,
    UsageTotals,
    current_side_call,
    side_call,
)

RESEARCHER = RunRef(
    agent="researcher", instance="researcher-1", run="researcher-1#1", parent_run=None, depth=1
)
CONVERSATION = SessionRef(app_name="app", user_id="u", session_id="s")


def usage(prompt: int, completion: int, *, agent: str = "coordinator", model: str = "m",
          purpose: str | None = None, run: RunRef | None = None, **parts) -> WorkflowEvent:
    """An ``LLM_USAGE`` as an emitter makes it."""
    event = WorkflowEvent.llm_usage(model=model, prompt_tokens=prompt,
                                    completion_tokens=completion, **parts)
    event.metadata["agent"] = agent
    if purpose is not None:
        event.metadata["purpose"] = purpose
    event.run = run
    return event


def recorded(*events: WorkflowEvent) -> UsageLedger:
    ledger = UsageLedger()
    for event in events:
        ledger.record(event)
    return ledger


class TestTheLedger:
    def test_a_row_per_call(self):
        ledger = recorded(usage(100, 10, run=root_run("coordinator"), cached_tokens=60,
                                cache_creation_tokens=10, thinking_tokens=4, latency_ms=12.5))

        (row,) = ledger.rows
        assert (row.run, row.agent, row.purpose, row.model) == (None, "coordinator", None, "m")
        assert (row.input, row.output, row.cached, row.cache_write, row.thinking) == (
            100, 10, 60, 10, 4
        )
        assert (row.latency_ms, row.turn, row.cost) == (12.5, None, None)

    def test_any_other_event_is_no_call(self):
        ledger = UsageLedger()

        assert ledger.record(WorkflowEvent.text("hi")) is None
        assert ledger.rows == ()

    def test_a_call_is_the_runs_in_progress_and_says_so(self):
        event = usage(50, 5, agent="researcher", purpose="tool")
        ledger = UsageLedger()

        token = set_active_run(RESEARCHER)
        try:
            ledger.record(event)
        finally:
            reset_active_run(token)

        assert ledger.rows[0].run == "researcher-1#1"
        assert event.run == RESEARCHER
        assert get_active_run() is None

    def test_the_agent_is_the_runs_when_the_event_names_none(self):
        event = WorkflowEvent.llm_usage(model="m", prompt_tokens=1, completion_tokens=1)
        event.run = RESEARCHER

        assert recorded(event).rows[0].agent == "researcher"

    def test_totals(self):
        ledger = recorded(
            usage(100, 10, run=root_run("coordinator")),
            usage(50, 5, agent="researcher", purpose="subagent", run=RESEARCHER),
            usage(30, 3, agent="researcher", purpose=PURPOSE_TOOL, run=RESEARCHER, model="small"),
            usage(20, 2, purpose=PURPOSE_SUMMARIZATION, run=root_run("coordinator")),
        )

        assert ledger.totals() == UsageTotals(input=200, output=20, calls=4)
        assert ledger.totals(run="researcher-1#1") == UsageTotals(input=80, output=8, calls=2)
        assert ledger.totals(run=None) == UsageTotals(input=120, output=12, calls=2)
        assert ledger.totals(agent="researcher") == UsageTotals(input=80, output=8, calls=2)
        assert ledger.totals(model="small") == UsageTotals(input=30, output=3, calls=1)
        assert ledger.totals(purpose=PURPOSE_TOOL) == UsageTotals(input=30, output=3, calls=1)

    def test_grouped(self):
        ledger = recorded(
            usage(100, 10, run=root_run("coordinator")),
            usage(50, 5, agent="researcher", purpose="subagent", run=RESEARCHER),
            usage(30, 3, agent="researcher", purpose=PURPOSE_TOOL, run=RESEARCHER, model="small"),
            usage(60, 6, agent="researcher", purpose="subagent", run=RESEARCHER),
            usage(20, 2, purpose=PURPOSE_SUMMARIZATION, run=root_run("coordinator")),
        )

        assert ledger.by_agent() == {
            "coordinator": UsageTotals(input=100, output=10, calls=1),
            "researcher": UsageTotals(input=110, output=11, calls=2),
        }
        assert ledger.by_purpose() == {
            PURPOSE_TOOL: UsageTotals(input=30, output=3, calls=1),
            PURPOSE_SUMMARIZATION: UsageTotals(input=20, output=2, calls=1),
        }
        assert ledger.by_model(run="researcher-1#1") == {
            "m": UsageTotals(input=110, output=11, calls=2),
            "small": UsageTotals(input=30, output=3, calls=1),
        }
        assert ledger.run_usage("researcher-1#1") == {
            "m": {"input": 110, "output": 11}, "small": {"input": 30, "output": 3},
        }
        assert ledger.run_usage("nobody-1#1") == {}

    def test_a_call_in_a_turn_carries_its_number(self):
        ledger = UsageLedger(CONVERSATION)
        ledger.record(usage(1, 1))  # before any turn
        token = set_active_turn(CONVERSATION)
        try:
            ledger.begin_turn()
            ledger.record(usage(10, 1))
            ledger.begin_turn()
            ledger.record(usage(20, 2))
        finally:
            reset_active_turn(token)
        ledger.record(usage(5, 5))  # outside a turn: the memory proposals

        assert [row.turn for row in ledger.rows] == [None, 1, 2, None]
        assert ledger.totals(turn=2) == UsageTotals(input=20, output=2, calls=1)
        assert ledger.turns == 2

    def test_a_call_in_another_conversations_turn_carries_none(self):
        # The memory proposals of a conversation left, made while another's turn runs.
        ledger = UsageLedger(CONVERSATION)
        ledger.begin_turn()
        token = set_active_turn(SessionRef(app_name="app", user_id="u", session_id="t"))
        try:
            ledger.record(usage(10, 1))
        finally:
            reset_active_turn(token)

        assert ledger.rows[0].turn is None

    def test_a_call_after_its_turn_has_ended_carries_none(self):
        # A task the turn started that outlived it: its context is the turn's, ended.
        ledger = UsageLedger(CONVERSATION)
        turn = set_active_turn(CONVERSATION)
        scope = open_turn_scope()
        ledger.begin_turn()
        outlived = contextvars.copy_context()
        close_turn_scope(scope)
        reset_active_turn(turn)

        outlived.run(ledger.record, usage(10, 1))

        assert ledger.rows[0].turn is None


class TestTheLedgers:
    def test_one_per_conversation_by_the_turn_a_call_is_made_in(self):
        ledgers = UsageLedgers()
        other = SessionRef(app_name="app", user_id="u", session_id="t")

        token = set_active_turn(CONVERSATION)
        try:
            ledgers.record(usage(10, 1))
        finally:
            reset_active_turn(token)
        ledgers.ledger(other).record(usage(20, 2))

        assert ledgers.ledger(CONVERSATION).totals() == UsageTotals(input=10, output=1, calls=1)
        assert ledgers.ledger(other).totals() == UsageTotals(input=20, output=2, calls=1)
        assert ledgers.ledger(CONVERSATION) is ledgers.ledger(CONVERSATION)


class TestSideCall:
    def test_the_caller_names_what_its_side_calls_are_for(self):
        assert current_side_call() is None
        with side_call(PURPOSE_MEMORY_EXTRACTION, CONVERSATION) as made:
            assert current_side_call() == made == SideCall(PURPOSE_MEMORY_EXTRACTION, CONVERSATION)
            with side_call(PURPOSE_TOOL):
                assert current_side_call() == SideCall(PURPOSE_TOOL)
            assert current_side_call() == made
        assert current_side_call() is None
