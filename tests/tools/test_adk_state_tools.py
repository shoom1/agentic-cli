"""Tests for ADK-native state tools (agentic_cli.tools.adk.state_tools).

Uses a mock ToolContext with a dict-like .state attribute to avoid
importing the real Google ADK ToolContext.
"""

import pytest

from agentic_cli.tools.adk.state_tools import (
    save_plan,
    get_plan,
    save_tasks,
    get_tasks,
)


class MockState(dict):
    """Dict-like state object mimicking ToolContext.state."""
    pass


class MockToolContext:
    """Minimal mock for google.adk.tools.tool_context.ToolContext.

    Only provides the ``.state`` attribute used by state tools.
    """

    def __init__(self, state: dict | None = None):
        self.state = MockState(state or {})


# ---------------------------------------------------------------------------
# save_plan / get_plan
# ---------------------------------------------------------------------------


class TestADKSavePlan:
    """Tests for ADK save_plan."""

    def test_saves_plan_to_state(self):
        ctx = MockToolContext()
        result = save_plan("## My Plan\n- [ ] Task 1", tool_context=ctx)
        assert result["success"] is True
        assert ctx.state["plan"] == "## My Plan\n- [ ] Task 1"

    def test_message_includes_summary_when_checkboxes_present(self):
        ctx = MockToolContext()
        result = save_plan("- [x] Done\n- [ ] Todo", tool_context=ctx)
        assert "2 tasks" in result["message"]
        assert "1 done" in result["message"]
        assert "1 pending" in result["message"]

    def test_message_plain_when_no_checkboxes(self):
        ctx = MockToolContext()
        result = save_plan("Just a text plan.", tool_context=ctx)
        assert result["message"] == "Plan saved"

    def test_overwrites_existing_plan(self):
        ctx = MockToolContext({"plan": "Old plan"})
        save_plan("New plan", tool_context=ctx)
        assert ctx.state["plan"] == "New plan"


class TestADKGetPlan:
    """Tests for ADK get_plan."""

    def test_returns_existing_plan(self):
        ctx = MockToolContext({"plan": "## My Plan\n- [ ] Task 1"})
        result = get_plan(tool_context=ctx)
        assert result["success"] is True
        assert result["content"] == "## My Plan\n- [ ] Task 1"

    def test_returns_no_plan_message_when_empty(self):
        ctx = MockToolContext()
        result = get_plan(tool_context=ctx)
        assert result["success"] is True
        assert result["content"] == ""
        assert "No plan created yet" in result["message"]

    def test_returns_no_plan_message_when_empty_string(self):
        ctx = MockToolContext({"plan": ""})
        result = get_plan(tool_context=ctx)
        assert result["message"] == "No plan created yet"


# ---------------------------------------------------------------------------
# save_tasks / get_tasks
# ---------------------------------------------------------------------------


class TestADKSaveTasks:
    """Tests for ADK save_tasks."""

    def test_saves_tasks_to_state(self):
        ctx = MockToolContext()
        result = save_tasks(
            tasks=[
                {"description": "Task 1"},
                {"description": "Task 2"},
            ],
            tool_context=ctx,
        )
        assert result["success"] is True
        assert result["count"] == 2
        assert len(result["task_ids"]) == 2
        assert len(ctx.state["tasks"]) == 2

    def test_empty_tasks_clears_state(self):
        ctx = MockToolContext({"tasks": [{"id": "old", "description": "Old"}]})
        result = save_tasks(tasks=[], tool_context=ctx)
        assert result["success"] is True
        assert result["count"] == 0
        assert ctx.state["tasks"] == []

    def test_validation_error_on_missing_description(self):
        ctx = MockToolContext()
        result = save_tasks(
            tasks=[{"status": "pending"}],
            tool_context=ctx,
        )
        assert result["success"] is False
        assert "description" in result["error"].lower()

    def test_validation_error_on_invalid_status(self):
        ctx = MockToolContext()
        result = save_tasks(
            tasks=[{"description": "Task", "status": "invalid"}],
            tool_context=ctx,
        )
        assert result["success"] is False
        assert "invalid" in result["error"]

    def test_validation_error_on_invalid_priority(self):
        ctx = MockToolContext()
        result = save_tasks(
            tasks=[{"description": "Task", "priority": "critical"}],
            tool_context=ctx,
        )
        assert result["success"] is False
        assert "critical" in result["error"]

    def test_normalized_tasks_have_all_fields(self):
        ctx = MockToolContext()
        save_tasks(
            tasks=[{"description": "Minimal task"}],
            tool_context=ctx,
        )
        task = ctx.state["tasks"][0]
        assert "id" in task
        assert "status" in task
        assert "priority" in task
        assert "tags" in task
        assert "created_at" in task
        assert "completed_at" in task

    def test_validation_does_not_write_on_error(self):
        ctx = MockToolContext()
        save_tasks(
            tasks=[{"description": "Good"}, {"description": "Bad", "status": "bogus"}],
            tool_context=ctx,
        )
        # Tasks should NOT be in state because validation failed
        assert ctx.state.get("tasks") is None or ctx.state.get("tasks") == []


class TestADKGetTasks:
    """Tests for ADK get_tasks."""

    def test_returns_empty_when_no_tasks(self):
        ctx = MockToolContext()
        result = get_tasks(tool_context=ctx)
        assert result["success"] is True
        assert result["tasks"] == []
        assert result["count"] == 0

    def test_returns_all_tasks(self):
        tasks_data = [
            {"id": "1", "description": "Task 1", "status": "pending", "priority": "medium", "tags": [], "created_at": "", "completed_at": ""},
            {"id": "2", "description": "Task 2", "status": "completed", "priority": "high", "tags": [], "created_at": "", "completed_at": ""},
        ]
        ctx = MockToolContext({"tasks": tasks_data})
        result = get_tasks(tool_context=ctx)
        assert result["count"] == 2

    def test_filter_by_status(self):
        tasks_data = [
            {"id": "1", "description": "Active", "status": "in_progress", "priority": "medium", "tags": [], "created_at": "", "completed_at": ""},
            {"id": "2", "description": "Pending", "status": "pending", "priority": "medium", "tags": [], "created_at": "", "completed_at": ""},
        ]
        ctx = MockToolContext({"tasks": tasks_data})
        result = get_tasks(status="in_progress", tool_context=ctx)
        assert result["count"] == 1
        assert result["tasks"][0]["id"] == "1"

    def test_filter_by_priority(self):
        tasks_data = [
            {"id": "1", "description": "Low", "status": "pending", "priority": "low", "tags": [], "created_at": "", "completed_at": ""},
            {"id": "2", "description": "High", "status": "pending", "priority": "high", "tags": [], "created_at": "", "completed_at": ""},
        ]
        ctx = MockToolContext({"tasks": tasks_data})
        result = get_tasks(priority="high", tool_context=ctx)
        assert result["count"] == 1
        assert result["tasks"][0]["id"] == "2"

    def test_filter_by_tag(self):
        tasks_data = [
            {"id": "1", "description": "Tagged", "status": "pending", "priority": "medium", "tags": ["feature"], "created_at": "", "completed_at": ""},
            {"id": "2", "description": "Untagged", "status": "pending", "priority": "medium", "tags": [], "created_at": "", "completed_at": ""},
        ]
        ctx = MockToolContext({"tasks": tasks_data})
        result = get_tasks(tag="feature", tool_context=ctx)
        assert result["count"] == 1
        assert result["tasks"][0]["id"] == "1"

    def test_empty_string_filters_are_ignored(self):
        tasks_data = [
            {"id": "1", "description": "A", "status": "pending", "priority": "medium", "tags": [], "created_at": "", "completed_at": ""},
        ]
        ctx = MockToolContext({"tasks": tasks_data})
        result = get_tasks(status="", priority="", tag="", tool_context=ctx)
        assert result["count"] == 1

    def test_none_tool_context_returns_empty(self):
        result = get_tasks(tool_context=None)
        assert result["success"] is True
        assert result["count"] == 0
