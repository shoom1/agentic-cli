"""A sub-agent run's start and end, as an event (``AGENT_RUN``), and the run an event names.

``WorkflowEvent.run`` is the run of the agent that wrote the event: the
root's own (``root_run``) or a sub-agent's; None for what the manager says
itself (a failed turn's error). ``WorkflowEvent.agent_run`` is a run's start
(``started``, with its title) or its end (``completed``, ``failed``,
``refused``, ``cancelled``, ``interrupted``, with the first of its result).
"""

from __future__ import annotations

from agentic_cli.workflow.events import EventType, WorkflowEvent
from agentic_cli.workflow.sessions import RunRef

RESEARCHER = RunRef(
    agent="researcher", instance="researcher-1", run="researcher-1#1", parent_run=None, depth=1
)


class TestAgentRun:
    def test_its_value(self):
        assert EventType.AGENT_RUN.value == "agent_run"

    def test_a_start_names_its_run_title_and_call(self):
        event = WorkflowEvent.agent_run(RESEARCHER, "started", title="Find X.", call_id="c1")

        assert event.type is EventType.AGENT_RUN
        assert event.run == RESEARCHER
        assert event.content == "researcher-1 started"
        assert event.metadata == {
            "status": "started", "agent": "researcher", "partial": False,
            "title": "Find X.", "call_id": "c1",
        }

    def test_an_end_carries_its_result_and_error(self):
        event = WorkflowEvent.agent_run(
            RESEARCHER, "failed", call_id="c1", result="", error="overloaded"
        )

        assert event.content == "researcher-1 failed"
        assert event.metadata == {
            "status": "failed", "agent": "researcher", "partial": False,
            "call_id": "c1", "result": "", "error": "overloaded",
        }

    def test_an_end_carries_what_the_run_used(self):
        used = {"anthropic:claude-sonnet-4-6": {"input": 140, "output": 14}}

        event = WorkflowEvent.agent_run(RESEARCHER, "completed", result="X is 42.", usage=used)

        assert event.metadata["usage"] == used

    def test_a_partial_end_says_so(self):
        event = WorkflowEvent.agent_run(RESEARCHER, "completed", result="X is", partial=True)

        assert event.metadata["partial"] is True


class TestTheRunAnEventNames:
    def test_none_unless_given(self):
        assert WorkflowEvent.text("hi").run is None
        assert WorkflowEvent.tool_call("grep", {}).run is None

    def test_it_is_a_field(self):
        assert WorkflowEvent(type=EventType.TEXT, content="hi", run=RESEARCHER).run == RESEARCHER
