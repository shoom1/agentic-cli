"""The rules each backend's ``_run_agent`` reads a sub-agent run's answer by.

Both managers hand a run's shared-state changes back by one rule, Deep
Agents' own ``task``'s (``deepagents.middleware.subagents._is_unchanged``):
a key returned exactly as it was received, or equal to it, is no change, and
an ambiguous comparison (an array's) counts as a change. Deep Agents also
reads a run's answer only from the messages after its task message, the
last of them with text.

The contract suite runs whole turns; these pin the rules themselves, which
the builder tests' fake hook only re-implements.
"""

from __future__ import annotations

import numpy as np
import pytest


def _adk_changed():
    pytest.importorskip("google.adk")
    from agentic_cli.workflow.adk.manager import _changed

    return _changed


def _deepagents_changed():
    pytest.importorskip("deepagents")
    from agentic_cli.workflow.deepagents.manager import _changed

    return _changed


@pytest.fixture(params=["adk", "deepagents"])
def changed(request):
    """The backend's ``_changed(state at the end, state at the start)``."""
    return _adk_changed() if request.param == "adk" else _deepagents_changed()


class TestTheChangedKeys:
    def test_an_unchanged_key_is_never_handed_back(self, changed):
        tasks = [{"description": "find X", "status": "pending"}]
        start = {"plan": "- [ ] find X", "tasks": tasks}

        # The same objects, and equal copies.
        assert changed({"plan": start["plan"], "tasks": tasks}, start) == {}
        assert changed({"plan": "- [ ] find X", "tasks": [dict(tasks[0])]}, start) == {}

    def test_a_changed_key_and_a_new_one_are(self, changed):
        start = {"plan": "- [ ] find X"}
        end = {"plan": "- [x] find X", "tasks": [{"description": "find X"}]}

        assert changed(end, start) == end

    def test_a_key_that_is_not_shared_is_never_handed_back(self, changed):
        assert changed({"messages": ["hi"], "files": {"a": 1}}, {}) == {}

    def test_an_ambiguous_comparison_is_a_change(self, changed):
        # An array's == is an array, neither True nor False.
        start = {"tasks": np.array([1, 2])}
        end = {"tasks": np.array([1, 2])}

        (key,) = changed(end, start)
        assert key == "tasks"
        # The same object is no change, whatever its == says.
        assert changed(start, start) == {}


class TestDeepAgentsAnswer:
    @pytest.fixture(autouse=True)
    def _deepagents(self):
        pytest.importorskip("deepagents")

    def test_the_last_text_is_the_last_message_with_any(self):
        from langchain_core.messages import AIMessage

        from agentic_cli.workflow.deepagents.manager import _last_text

        answers = [AIMessage("First."), AIMessage("X is 42.  \n", tool_calls=[
            {"name": "contract_lookup", "args": {}, "id": "r1"},
        ]), AIMessage(""), AIMessage([])]

        assert _last_text(answers) == "X is 42."
        assert _last_text([AIMessage(""), AIMessage("   ")]) == ""
        assert _last_text([]) == ""

    def test_its_own_messages_are_those_after_its_task_message(self):
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        from agentic_cli.workflow.deepagents.manager import _own_messages

        history = [HumanMessage("My name is Ada.", id="h1"), AIMessage("Noted, Ada.", id="a1"),
                   ToolMessage("answered", tool_call_id="x", id="t1")]
        task = HumanMessage("# Task: Find the code", id="task-1")
        own = [AIMessage("", id="a2")]

        assert _own_messages([*history, task, *own], "task-1") == own
        assert _own_messages([*history, task], "task-1") == []

    def test_without_its_task_message_none_is_its_own(self):
        from langchain_core.messages import AIMessage, HumanMessage

        from agentic_cli.workflow.deepagents.manager import _own_messages

        messages = [HumanMessage("Hi.", id="h1"), AIMessage("The caller's reply.", id="a1")]

        assert _own_messages(messages, "task-1") == []
        assert _own_messages(messages, None) == []
