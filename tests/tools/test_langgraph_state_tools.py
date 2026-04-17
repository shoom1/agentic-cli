"""Tests for LangGraph-native state tools (agentic_cli.tools.langgraph.state_tools).

These tools return Command objects with state updates and ToolMessages.
"""

import json

import pytest

from langgraph.types import Command
from langchain_core.messages import ToolMessage

from agentic_cli.tools.langgraph.state_tools import (
    save_plan,
    get_plan,
    save_tasks,
    get_tasks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_result(command: Command) -> dict:
    """Extract the tool result dict from a Command's ToolMessage."""
    messages = command.update.get("messages", [])
    assert len(messages) == 1
    msg = messages[0]
    assert isinstance(msg, ToolMessage)
    return json.loads(msg.content)


# ---------------------------------------------------------------------------
# save_plan
# ---------------------------------------------------------------------------


class TestLangGraphSavePlan:
    """Tests for LangGraph save_plan."""

    def test_returns_command(self):
        result = save_plan(content="## Plan\n- [ ] Step 1", tool_call_id="tc_1")
        assert isinstance(result, Command)

    def test_updates_plan_in_state(self):
        result = save_plan(content="My plan content", tool_call_id="tc_1")
        assert result.update["plan"] == "My plan content"

    def test_tool_message_has_correct_tool_call_id(self):
        result = save_plan(content="Plan", tool_call_id="tc_42")
        msg = result.update["messages"][0]
        assert msg.tool_call_id == "tc_42"

    def test_result_reports_success(self):
        result = save_plan(content="Plan", tool_call_id="tc_1")
        data = _extract_result(result)
        assert data["success"] is True

    def test_result_includes_checkbox_summary(self):
        result = save_plan(
            content="- [x] Done\n- [ ] Todo\n- [ ] Also todo",
            tool_call_id="tc_1",
        )
        data = _extract_result(result)
        assert "3 tasks" in data["message"]
        assert "1 done" in data["message"]
        assert "2 pending" in data["message"]

    def test_result_plain_message_when_no_checkboxes(self):
        result = save_plan(content="Just text", tool_call_id="tc_1")
        data = _extract_result(result)
        assert data["message"] == "Plan saved"


# ---------------------------------------------------------------------------
# get_plan
# ---------------------------------------------------------------------------


class TestLangGraphGetPlan:
    """Tests for LangGraph get_plan."""

    def test_returns_command(self):
        result = get_plan(state={"plan": "My plan"}, tool_call_id="tc_1")
        assert isinstance(result, Command)

    def test_returns_existing_plan(self):
        result = get_plan(state={"plan": "## Plan\n- [x] Done"}, tool_call_id="tc_1")
        data = _extract_result(result)
        assert data["success"] is True
        assert data["content"] == "## Plan\n- [x] Done"

    def test_returns_no_plan_message_when_empty(self):
        result = get_plan(state={}, tool_call_id="tc_1")
        data = _extract_result(result)
        assert data["success"] is True
        assert data["content"] == ""
        assert "No plan created yet" in data["message"]

    def test_returns_no_plan_message_when_empty_string(self):
        result = get_plan(state={"plan": ""}, tool_call_id="tc_1")
        data = _extract_result(result)
        assert data["message"] == "No plan created yet"

    def test_does_not_update_plan_state(self):
        """get_plan should only update messages, not plan."""
        result = get_plan(state={"plan": "Existing"}, tool_call_id="tc_1")
        assert "plan" not in result.update


# ---------------------------------------------------------------------------
# save_tasks
# ---------------------------------------------------------------------------


class TestLangGraphSaveTasks:
    """Tests for LangGraph save_tasks."""

    def test_returns_command(self):
        result = save_tasks(
            tasks=[{"description": "Task 1"}],
            tool_call_id="tc_1",
        )
        assert isinstance(result, Command)

    def test_updates_tasks_in_state(self):
        result = save_tasks(
            tasks=[
                {"description": "Task 1"},
                {"description": "Task 2"},
            ],
            tool_call_id="tc_1",
        )
        assert len(result.update["tasks"]) == 2
        assert result.update["tasks"][0]["description"] == "Task 1"

    def test_result_reports_success(self):
        result = save_tasks(
            tasks=[{"description": "Task 1"}],
            tool_call_id="tc_1",
        )
        data = _extract_result(result)
        assert data["success"] is True
        assert data["count"] == 1
        assert len(data["task_ids"]) == 1

    def test_empty_tasks_clears(self):
        result = save_tasks(tasks=[], tool_call_id="tc_1")
        assert result.update["tasks"] == []
        data = _extract_result(result)
        assert data["success"] is True
        assert data["count"] == 0
        assert data["message"] == "Tasks cleared"

    def test_validation_error_returns_command_without_tasks_update(self):
        result = save_tasks(
            tasks=[{"status": "pending"}],  # missing description
            tool_call_id="tc_1",
        )
        assert isinstance(result, Command)
        data = _extract_result(result)
        assert data["success"] is False
        assert "description" in data["error"].lower()
        # Should NOT update tasks in state
        assert "tasks" not in result.update

    def test_validation_error_on_invalid_status(self):
        result = save_tasks(
            tasks=[{"description": "Task", "status": "bogus"}],
            tool_call_id="tc_1",
        )
        data = _extract_result(result)
        assert data["success"] is False
        assert "bogus" in data["error"]

    def test_normalized_tasks_have_all_fields(self):
        result = save_tasks(
            tasks=[{"description": "Minimal"}],
            tool_call_id="tc_1",
        )
        task = result.update["tasks"][0]
        assert "id" in task
        assert "status" in task
        assert "priority" in task
        assert "tags" in task
        assert "created_at" in task
        assert "completed_at" in task

    def test_completed_task_gets_completed_at(self):
        result = save_tasks(
            tasks=[{"description": "Done", "status": "completed"}],
            tool_call_id="tc_1",
        )
        task = result.update["tasks"][0]
        assert task["completed_at"] != ""


# ---------------------------------------------------------------------------
# get_tasks
# ---------------------------------------------------------------------------


class TestLangGraphGetTasks:
    """Tests for LangGraph get_tasks."""

    def test_returns_command(self):
        result = get_tasks(state={}, tool_call_id="tc_1")
        assert isinstance(result, Command)

    def test_returns_empty_when_no_tasks(self):
        result = get_tasks(state={}, tool_call_id="tc_1")
        data = _extract_result(result)
        assert data["success"] is True
        assert data["tasks"] == []
        assert data["count"] == 0

    def test_returns_all_tasks(self):
        tasks_data = [
            {"id": "1", "description": "Task 1", "status": "pending", "priority": "medium", "tags": [], "created_at": "", "completed_at": ""},
            {"id": "2", "description": "Task 2", "status": "completed", "priority": "high", "tags": [], "created_at": "", "completed_at": ""},
        ]
        result = get_tasks(state={"tasks": tasks_data}, tool_call_id="tc_1")
        data = _extract_result(result)
        assert data["count"] == 2

    def test_filter_by_status(self):
        tasks_data = [
            {"id": "1", "description": "Active", "status": "in_progress", "priority": "medium", "tags": [], "created_at": "", "completed_at": ""},
            {"id": "2", "description": "Pending", "status": "pending", "priority": "medium", "tags": [], "created_at": "", "completed_at": ""},
        ]
        result = get_tasks(status="in_progress", state={"tasks": tasks_data}, tool_call_id="tc_1")
        data = _extract_result(result)
        assert data["count"] == 1
        assert data["tasks"][0]["id"] == "1"

    def test_filter_by_priority(self):
        tasks_data = [
            {"id": "1", "description": "Low", "status": "pending", "priority": "low", "tags": [], "created_at": "", "completed_at": ""},
            {"id": "2", "description": "High", "status": "pending", "priority": "high", "tags": [], "created_at": "", "completed_at": ""},
        ]
        result = get_tasks(priority="high", state={"tasks": tasks_data}, tool_call_id="tc_1")
        data = _extract_result(result)
        assert data["count"] == 1

    def test_filter_by_tag(self):
        tasks_data = [
            {"id": "1", "description": "Tagged", "status": "pending", "priority": "medium", "tags": ["bug"], "created_at": "", "completed_at": ""},
            {"id": "2", "description": "Plain", "status": "pending", "priority": "medium", "tags": [], "created_at": "", "completed_at": ""},
        ]
        result = get_tasks(tag="bug", state={"tasks": tasks_data}, tool_call_id="tc_1")
        data = _extract_result(result)
        assert data["count"] == 1

    def test_empty_string_filters_are_ignored(self):
        tasks_data = [
            {"id": "1", "description": "A", "status": "pending", "priority": "medium", "tags": [], "created_at": "", "completed_at": ""},
        ]
        result = get_tasks(status="", priority="", tag="", state={"tasks": tasks_data}, tool_call_id="tc_1")
        data = _extract_result(result)
        assert data["count"] == 1

    def test_none_state_returns_empty(self):
        result = get_tasks(state=None, tool_call_id="tc_1")
        data = _extract_result(result)
        assert data["count"] == 0

    def test_does_not_update_tasks_state(self):
        """get_tasks should only update messages, not tasks."""
        tasks_data = [{"id": "1", "description": "A", "status": "pending", "priority": "medium", "tags": [], "created_at": "", "completed_at": ""}]
        result = get_tasks(state={"tasks": tasks_data}, tool_call_id="tc_1")
        assert "tasks" not in result.update
