"""The parent's tools as both backends call them (``workflow/agents/follow_ups.py``).

``call_follow_up_tool`` reads a call's arguments as the model wrote them and
hands them to the manager's ``Delegations`` as the run in progress: the
calling agent's run (None for the root), in the active turn's conversation.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentic_cli.workflow.agents import RUN_COMPLETED, AgentRuns, SubAgentPolicy
from agentic_cli.workflow.agents.delegate import (
    AgentRun,
    DelegationRequest,
    Delegations,
    RunAnswer,
)
from agentic_cli.workflow.agents.follow_ups import call_follow_up_tool
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.sessions import (
    SessionRef,
    reset_active_run,
    reset_active_turn,
    set_active_run,
    set_active_turn,
)

CONVERSATION = SessionRef("app", "u", "s")
TYPES = {
    "coordinator": AgentConfig(name="coordinator", prompt="p", sub_agents=["researcher"]),
    "researcher": AgentConfig(name="researcher", prompt="p", description="Finds."),
}


async def with_one_instance() -> tuple[Delegations, AgentRuns, list[AgentRun]]:
    """Delegations whose root has delegated once (``researcher-1``, idle)."""
    asked: list[AgentRun] = []

    async def hook(run: AgentRun) -> RunAnswer:
        asked.append(run)
        return RunAnswer(text=f"Did: {run.message}")

    runs = AgentRuns()
    made = Delegations(runs=runs, policy=SubAgentPolicy(), types=TYPES, hook=hook)
    await made.delegate(DelegationRequest(
        caller=None, caller_agent="coordinator", call_id="c1",
        arguments={"agent_type": "researcher", "title": "Find X", "objective": "Find X."},
    ))
    return made, runs, asked


@pytest.fixture
def a_turn():
    token = set_active_turn(CONVERSATION)
    yield
    reset_active_turn(token)


async def call(made: Delegations, name: str, **arguments: Any) -> Any:
    return await call_follow_up_tool(
        made, name, arguments, call_id="c2", state={"outside_content": "web_fetch"}
    )


@pytest.mark.usefixtures("a_turn")
class TestCallFollowUpTool:
    async def test_message_agent_sends_the_root_its_follow_up(self):
        made, runs, asked = await with_one_instance()

        delegated = await call(made, "message_agent", agent=" researcher-1 ", message="Go on.")

        assert (delegated.result["run"], delegated.result["status"]) == (
            "researcher-1#2", RUN_COMPLETED
        )
        assert (asked[-1].message, asked[-1].state) == ("Go on.", {"outside_content": "web_fetch"})
        record = runs.record(CONVERSATION, "researcher-1#2")
        assert (record.started_by, record.call_id, record.ref.parent_run) == (
            "agent:root", "c2", None
        )

    async def test_a_sub_agent_is_the_caller_in_its_own_run(self):
        made, _, _ = await with_one_instance()
        token = set_active_run(made._runs.record(CONVERSATION, "researcher-1").ref)
        try:
            delegated = await call(made, "check_agents")
        finally:
            reset_active_run(token)

        # The researcher has no sub-agents of its own.
        assert delegated.result == {"success": True, "agents": []}

    async def test_only_true_interrupts_or_closes(self):
        made, _, _ = await with_one_instance()

        stopped = await call(made, "stop_agent", agent="researcher-1", close="true")

        assert stopped.result == {"success": True, "agent": "researcher-1", "status": "idle",
                                  "stopped": None, "closed": []}
        assert (stopped.state, stopped.run) == ({}, None)

    async def test_an_agent_that_is_not_text_is_none(self):
        made, _, _ = await with_one_instance()

        checked = await call(made, "check_agents", agent=" ")
        messaged = await call(made, "message_agent", agent=7, message="Go on.")

        assert [row["agent"] for row in checked.result["agents"]] == ["researcher-1"]
        assert messaged.result == {
            "success": False,
            "error": "'' is not one of your sub-agents. Yours: researcher-1.",
        }

    async def test_no_other_tool(self):
        made, _, _ = await with_one_instance()

        with pytest.raises(ValueError, match="'task' is not one of the parent's tools"):
            await call(made, "task")
