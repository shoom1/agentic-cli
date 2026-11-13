"""A delegation's result on Deep Agents, shown as it is on ADK.

Deep Agents' ``task`` returns the sub-agent's whole answer as text, which
the generic rendering showed up to 200 characters, lines and all. The event
processor knows the call is a delegation (it ends the run's ``AGENT_RUN``
there), so it shows the answer's first line
(``tool_summaries.delegation_summary``), and a sub-agent that failed as a
failed result; ``metadata["result"]`` stays the text the parent read.
"""

from __future__ import annotations

import pytest

pytest.importorskip("deepagents")

from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402

from agentic_cli.workflow.events import EventType  # noqa: E402
from tests.workflow.deepagents.test_event_processor import (  # noqa: E402
    _call,
    _model_update,
    _processor,
    _tools_update,
)


def _result_of(message: ToolMessage, *, name: str = "task"):
    """The ``TOOL_RESULT`` the coordinator's call ``c1`` of ``name`` comes back with."""
    processor = _processor()
    args = {"description": "Find X.", "subagent_type": "researcher"} if name == "task" else {}
    processor.process((), "updates", _model_update(AIMessage("", tool_calls=[
        _call(name, args, "c1"),
    ])))
    events = processor.process((), "updates", _tools_update(message))
    return next(e for e in events if e.type is EventType.TOOL_RESULT)


class TestADelegationsResult:
    def test_shows_the_answers_first_line(self):
        result = _result_of(ToolMessage(
            "X is 42.\nThe handbook says so.", tool_call_id="c1", name="task"
        ))

        assert result.content == "X is 42."
        assert result.metadata["success"] is True
        # What the parent read stays as Deep Agents gave it.
        assert result.metadata["result"] == "X is 42.\nThe handbook says so."

    def test_an_empty_answer_is_no_answer(self):
        result = _result_of(ToolMessage("", tool_call_id="c1", name="task"))

        assert result.content == "No answer"

    def test_a_sub_agent_that_failed_is_a_failed_result(self):
        result = _result_of(ToolMessage(
            "Error running sub-agent: 529 overloaded",
            tool_call_id="c1", name="task", status="error",
        ))

        assert result.content == "Failed: 529 overloaded"
        assert result.metadata["success"] is False
        assert result.metadata["error"] == "Error running sub-agent: 529 overloaded"

    def test_another_tools_result_is_rendered_as_before(self):
        result = _result_of(
            ToolMessage("X is 42.\nMore.", tool_call_id="c1", name="lookup"), name="lookup"
        )

        assert result.content == "X is 42.\nMore."
