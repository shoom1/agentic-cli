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
def backend(request):
    return request.param


@pytest.fixture
def changed(backend):
    """The backend's ``_changed(state at the end, state at the start)``."""
    return _adk_changed() if backend == "adk" else _deepagents_changed()


@pytest.fixture
def mark(backend):
    """The backend's one shared key: the outside-content mark."""
    if backend == "adk":
        pytest.importorskip("google.adk")
        from agentic_cli.workflow.adk.task_tool import SHARED_KEYS
    else:
        pytest.importorskip("deepagents")
        from agentic_cli.workflow.deepagents.task_tool import SHARED_KEYS
    (key,) = SHARED_KEYS
    return key


class TestTheChangedKeys:
    def test_an_unchanged_key_is_never_handed_back(self, changed, mark):
        marked = [{"source": "a page"}]

        # The same object, and an equal copy.
        assert changed({mark: marked}, {mark: marked}) == {}
        assert changed({mark: [dict(marked[0])]}, {mark: marked}) == {}

    def test_a_changed_key_and_a_new_one_are(self, changed, mark):
        assert changed({mark: True}, {mark: False}) == {mark: True}
        assert changed({mark: True}, {}) == {mark: True}

    def test_a_key_that_is_not_shared_is_never_handed_back(self, changed):
        # The plan and the tasks are each run's own, not shared.
        end = {"messages": ["hi"], "files": {"a": 1}, "plan": "- [ ] x", "tasks": [{"d": 1}]}
        assert changed(end, {}) == {}

    def test_an_ambiguous_comparison_is_a_change(self, changed, mark):
        # An array's == is an array, neither True nor False.
        start = {mark: np.array([1, 2])}
        end = {mark: np.array([1, 2])}

        assert list(changed(end, start)) == [mark]
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
