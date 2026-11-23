"""A sub-agent run's start and end, as an event (``AGENT_RUN``), and the run an event names.

``WorkflowEvent.run`` is the run of the agent that wrote the event: the
root's own (``root_run``) or a sub-agent's; None for what the manager says
itself (a failed turn's error). ``WorkflowEvent.agent_run`` is a run's start
(``started``, with its title) or its end (``completed``, ``failed``,
``refused``, ``cancelled``, ``interrupted``, with the first of its result and
its instance's status after it). ``WorkflowEvent.agent_instance`` is a change
of an instance's status without a run.
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


class TestAnInstancesStatus:
    """An instance's status rides on its runs' events (``instance_status`` on
    an end), and on ``AGENT_INSTANCE`` for a change without a run."""

    def test_an_end_carries_its_instances_status(self):
        event = WorkflowEvent.agent_run(
            RESEARCHER, "completed", result="X is 42.", instance_status="idle"
        )

        assert event.metadata["instance_status"] == "idle"

    def test_none_unless_given(self):
        event = WorkflowEvent.agent_run(RESEARCHER, "completed", result="X is 42.")

        assert "instance_status" not in event.metadata

    def test_a_first_runs_start_carries_what_its_instance_keeps(self):
        event = WorkflowEvent.agent_run(
            RESEARCHER, "started", title="Find X.", inherit_context=True, continuable=False
        )

        assert (event.metadata["inherit_context"], event.metadata["continuable"]) == (True, False)

    def test_a_first_runs_start_carries_the_tools_and_model_it_was_given(self):
        event = WorkflowEvent.agent_run(
            RESEARCHER, "started", title="Find X.", tools=("lookup", "fetch"),
            model="fast"
        )

        assert (event.metadata["tools"], event.metadata["model"]) == (
            ["lookup", "fetch"], "fast"
        )

    def test_none_of_its_own_tools_is_said_too(self):
        event = WorkflowEvent.agent_run(RESEARCHER, "started", title="Find X.",
                                        tools=[])

        assert event.metadata["tools"] == []
        assert "model" not in event.metadata

    def test_no_tools_or_model_unless_given(self):
        event = WorkflowEvent.agent_run(RESEARCHER, "started", title="Find X.")

        assert "tools" not in event.metadata and "model" not in event.metadata

    def test_a_general_purpose_runs_start_carries_its_spawner_type(self):
        ref = RunRef("general_purpose", "general_purpose-1", "general_purpose-1#1", None, 1)

        event = WorkflowEvent.agent_run(ref, "started", title="Sum it up.", based_on="lead")

        assert event.metadata["based_on"] == "lead"
        assert "based_on" not in WorkflowEvent.agent_run(ref, "started").metadata

    def test_a_follow_ups_start_carries_its_input(self):
        second = RunRef("researcher", "researcher-1", "researcher-1#2", None, 1)

        event = WorkflowEvent.agent_run(second, "started", title="Go on.", input="Go on.\nAnd on.")

        assert event.metadata["input"] == "Go on.\nAnd on."
        assert "inherit_context" not in event.metadata and "continuable" not in event.metadata


class TestAgentInstance:
    def test_its_value(self):
        assert EventType.AGENT_INSTANCE.value == "agent_instance"

    def test_it_names_the_instance_by_its_last_run_and_says_its_status(self):
        event = WorkflowEvent.agent_instance(RESEARCHER, "closed", reason="closed by its parent")

        assert event.type is EventType.AGENT_INSTANCE
        assert event.run == RESEARCHER
        assert event.content == "researcher-1 closed"
        assert event.metadata == {
            "status": "closed", "agent": "researcher", "reason": "closed by its parent",
        }

    def test_no_reason_unless_given(self):
        event = WorkflowEvent.agent_instance(RESEARCHER, "stopped")

        assert event.metadata == {"status": "stopped", "agent": "researcher"}
