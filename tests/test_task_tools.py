"""Tests for task management tools."""

from typing import AsyncGenerator
from unittest.mock import patch

import pytest

from agentic_cli.tools.task_tools import TaskStore, TaskItem
from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.events import WorkflowEvent, EventType


class TestTaskItem:
    """Tests for TaskItem dataclass."""

    def test_to_dict(self):
        item = TaskItem(
            id="abc123",
            description="Test task",
            status="pending",
            priority="high",
            tags=["feature"],
            created_at="2024-01-01T00:00:00",
        )
        d = item.to_dict()
        assert d["id"] == "abc123"
        assert d["description"] == "Test task"
        assert d["status"] == "pending"
        assert d["priority"] == "high"
        assert d["tags"] == ["feature"]
        assert d["created_at"] == "2024-01-01T00:00:00"
        assert d["completed_at"] == ""

    def test_from_dict(self):
        data = {
            "id": "abc123",
            "description": "Test task",
            "status": "in_progress",
            "priority": "low",
            "tags": ["bug"],
            "created_at": "2024-01-01T00:00:00",
            "completed_at": "",
        }
        item = TaskItem.from_dict(data)
        assert item.id == "abc123"
        assert item.description == "Test task"
        assert item.status == "in_progress"
        assert item.priority == "low"
        assert item.tags == ["bug"]

    def test_from_dict_defaults(self):
        data = {"id": "x", "description": "Minimal"}
        item = TaskItem.from_dict(data)
        assert item.status == "pending"
        assert item.priority == "medium"
        assert item.tags == []

    def test_from_dict_invalid_status_defaults_to_pending(self):
        data = {"id": "x", "description": "Bad status", "status": "bogus"}
        item = TaskItem.from_dict(data)
        assert item.status == "pending"

    def test_from_dict_invalid_priority_defaults_to_medium(self):
        data = {"id": "x", "description": "Bad priority", "priority": "critical"}
        item = TaskItem.from_dict(data)
        assert item.priority == "medium"

    def test_roundtrip(self):
        item = TaskItem(
            id="rt1",
            description="Roundtrip",
            status="completed",
            priority="medium",
            tags=["a", "b"],
            created_at="2024-06-01T12:00:00",
            completed_at="2024-06-01T13:00:00",
        )
        restored = TaskItem.from_dict(item.to_dict())
        assert restored.id == item.id
        assert restored.description == item.description
        assert restored.status == item.status
        assert restored.completed_at == item.completed_at


class TestTaskStore:
    """Tests for TaskStore class."""

    def test_replace_all_creates_tasks(self, mock_context):
        store = TaskStore(mock_context.settings)
        ids = store.replace_all([
            {"description": "Implement auth module", "priority": "high"},
            {"description": "Write tests"},
        ])
        assert len(ids) == 2
        task = store.get(ids[0])
        assert task is not None
        assert task.description == "Implement auth module"
        assert task.priority == "high"
        assert task.status == "pending"

    def test_replace_all_with_tags(self, mock_context):
        store = TaskStore(mock_context.settings)
        ids = store.replace_all([
            {"description": "Fix bug", "tags": ["bug", "urgent"]},
        ])
        task = store.get(ids[0])
        assert task.tags == ["bug", "urgent"]

    def test_replace_all_with_statuses(self, mock_context):
        store = TaskStore(mock_context.settings)
        ids = store.replace_all([
            {"description": "Done task", "status": "completed"},
            {"description": "Active task", "status": "in_progress"},
            {"description": "Pending task"},
        ])
        assert store.get(ids[0]).status == "completed"
        assert store.get(ids[1]).status == "in_progress"
        assert store.get(ids[2]).status == "pending"

    def test_replace_all_completed_sets_timestamp(self, mock_context):
        store = TaskStore(mock_context.settings)
        ids = store.replace_all([
            {"description": "Test task", "status": "completed"},
        ])
        task = store.get(ids[0])
        assert task.status == "completed"
        assert task.completed_at != ""

    def test_replace_all_preserves_existing_id(self, mock_context):
        store = TaskStore(mock_context.settings)
        ids = store.replace_all([
            {"id": "custom-id", "description": "Task with ID"},
        ])
        assert ids == ["custom-id"]
        assert store.get("custom-id") is not None

    def test_replace_all_clears_previous(self, mock_context):
        store = TaskStore(mock_context.settings)
        store.replace_all([{"description": "Old task"}])
        assert len(store.list_tasks()) == 1
        store.replace_all([{"description": "New task 1"}, {"description": "New task 2"}])
        assert len(store.list_tasks()) == 2
        tasks = store.list_tasks()
        assert tasks[0].description == "New task 1"

    def test_replace_all_empty_clears(self, mock_context):
        store = TaskStore(mock_context.settings)
        store.replace_all([{"description": "Task"}])
        assert not store.is_empty()
        store.replace_all([])
        assert store.is_empty()

    def test_list_all(self, mock_context):
        store = TaskStore(mock_context.settings)
        store.replace_all([
            {"description": "Task 1"},
            {"description": "Task 2"},
            {"description": "Task 3"},
        ])
        assert len(store.list_tasks()) == 3

    def test_list_filter_status(self, mock_context):
        store = TaskStore(mock_context.settings)
        store.replace_all([
            {"description": "Task 1", "status": "in_progress"},
            {"description": "Task 2", "status": "pending"},
        ])
        assert len(store.list_tasks(status="in_progress")) == 1
        assert len(store.list_tasks(status="pending")) == 1

    def test_list_filter_priority(self, mock_context):
        store = TaskStore(mock_context.settings)
        store.replace_all([
            {"description": "Low", "priority": "low"},
            {"description": "High", "priority": "high"},
        ])
        assert len(store.list_tasks(priority="high")) == 1

    def test_list_filter_tag(self, mock_context):
        store = TaskStore(mock_context.settings)
        store.replace_all([
            {"description": "Tagged", "tags": ["important"]},
            {"description": "Untagged"},
        ])
        assert len(store.list_tasks(tag="important")) == 1

    def test_is_empty(self, mock_context):
        store = TaskStore(mock_context.settings)
        assert store.is_empty()
        store.replace_all([{"description": "Task"}])
        assert not store.is_empty()

    def test_clear_empties_store(self, mock_context):
        """clear() removes all tasks from memory."""
        store = TaskStore(mock_context.settings)
        store.replace_all([
            {"description": "Task 1"},
            {"description": "Task 2"},
        ])
        assert not store.is_empty()
        store.clear()
        assert store.is_empty()
        assert len(store.list_tasks()) == 0

    def test_clear_on_empty_store(self, mock_context):
        """clear() on an empty store is a no-op (no exception)."""
        store = TaskStore(mock_context.settings)
        store.clear()  # should not raise
        assert store.is_empty()

    def test_in_memory_only(self, mock_context):
        """TaskStore has no file persistence attributes."""
        store = TaskStore(mock_context.settings)
        assert not hasattr(store, "_storage_path")
        assert not hasattr(store, "_tasks_dir")


class TestTaskTools:
    """Tests for save_tasks and get_tasks tool functions."""

    def test_save_tasks_bulk_create(self, mock_context):
        from agentic_cli.tools.task_tools import save_tasks
        from agentic_cli.workflow.context import set_context_task_store

        store = TaskStore(mock_context.settings)
        token = set_context_task_store(store)
        try:
            result = save_tasks(tasks=[
                {"description": "Task 1"},
                {"description": "Task 2"},
                {"description": "Task 3"},
            ])
            assert result["success"] is True
            assert result["count"] == 3
            assert len(result["task_ids"]) == 3
        finally:
            token.var.reset(token)

    def test_save_tasks_replaces_existing(self, mock_context):
        from agentic_cli.tools.task_tools import save_tasks
        from agentic_cli.workflow.context import set_context_task_store

        store = TaskStore(mock_context.settings)
        token = set_context_task_store(store)
        try:
            save_tasks(tasks=[{"description": "Old task"}])
            result = save_tasks(tasks=[
                {"description": "New task 1"},
                {"description": "New task 2"},
            ])
            assert result["count"] == 2
            assert len(store.list_tasks()) == 2
        finally:
            token.var.reset(token)

    def test_save_tasks_with_statuses(self, mock_context):
        from agentic_cli.tools.task_tools import save_tasks
        from agentic_cli.workflow.context import set_context_task_store

        store = TaskStore(mock_context.settings)
        token = set_context_task_store(store)
        try:
            result = save_tasks(tasks=[
                {"description": "Done", "status": "completed"},
                {"description": "Active", "status": "in_progress"},
                {"description": "Todo"},
            ])
            assert result["success"] is True
            assert len(store.list_tasks(status="completed")) == 1
            assert len(store.list_tasks(status="in_progress")) == 1
            assert len(store.list_tasks(status="pending")) == 1
        finally:
            token.var.reset(token)

    def test_save_tasks_empty_clears(self, mock_context):
        from agentic_cli.tools.task_tools import save_tasks
        from agentic_cli.workflow.context import set_context_task_store

        store = TaskStore(mock_context.settings)
        token = set_context_task_store(store)
        try:
            save_tasks(tasks=[{"description": "Task"}])
            result = save_tasks(tasks=[])
            assert result["success"] is True
            assert result["count"] == 0
            assert store.is_empty()
        finally:
            token.var.reset(token)

    def test_save_tasks_missing_description(self, mock_context):
        from agentic_cli.tools.task_tools import save_tasks
        from agentic_cli.workflow.context import set_context_task_store

        store = TaskStore(mock_context.settings)
        token = set_context_task_store(store)
        try:
            result = save_tasks(tasks=[
                {"description": "Valid"},
                {"status": "pending"},  # missing description
            ])
            assert result["success"] is False
            assert "index 1" in result["error"]
            assert "description" in result["error"].lower()
        finally:
            token.var.reset(token)

    def test_save_tasks_invalid_status_returns_error(self, mock_context):
        from agentic_cli.tools.task_tools import save_tasks
        from agentic_cli.workflow.context import set_context_task_store

        store = TaskStore(mock_context.settings)
        token = set_context_task_store(store)
        try:
            result = save_tasks(tasks=[
                {"description": "Valid task"},
                {"description": "Bad status", "status": "bogus"},
            ])
            assert result["success"] is False
            assert "index 1" in result["error"]
            assert "bogus" in result["error"]
        finally:
            token.var.reset(token)

    def test_save_tasks_invalid_priority_returns_error(self, mock_context):
        from agentic_cli.tools.task_tools import save_tasks
        from agentic_cli.workflow.context import set_context_task_store

        store = TaskStore(mock_context.settings)
        token = set_context_task_store(store)
        try:
            result = save_tasks(tasks=[
                {"description": "Bad priority", "priority": "critical"},
            ])
            assert result["success"] is False
            assert "index 0" in result["error"]
            assert "critical" in result["error"]
        finally:
            token.var.reset(token)

    def test_save_tasks_no_store(self):
        from agentic_cli.tools.task_tools import save_tasks
        from agentic_cli.workflow.context import set_context_task_store

        token = set_context_task_store(None)
        try:
            result = save_tasks(tasks=[{"description": "Test"}])
            assert result["success"] is False
            assert "not available" in result["error"].lower()
        finally:
            token.var.reset(token)

    def test_get_tasks(self, mock_context):
        from agentic_cli.tools.task_tools import get_tasks
        from agentic_cli.workflow.context import set_context_task_store

        store = TaskStore(mock_context.settings)
        store.replace_all([
            {"description": "Task 1"},
            {"description": "Task 2"},
        ])
        token = set_context_task_store(store)
        try:
            result = get_tasks()
            assert result["success"] is True
            assert result["count"] == 2
            assert len(result["tasks"]) == 2
        finally:
            token.var.reset(token)

    def test_get_tasks_with_filter(self, mock_context):
        from agentic_cli.tools.task_tools import get_tasks
        from agentic_cli.workflow.context import set_context_task_store

        store = TaskStore(mock_context.settings)
        store.replace_all([
            {"description": "Task 1", "priority": "high", "status": "in_progress"},
            {"description": "Task 2", "priority": "low"},
        ])
        token = set_context_task_store(store)
        try:
            result = get_tasks(status="in_progress")
            assert result["count"] == 1
            assert result["tasks"][0]["priority"] == "high"
        finally:
            token.var.reset(token)

    def test_get_tasks_no_store(self):
        from agentic_cli.tools.task_tools import get_tasks
        from agentic_cli.workflow.context import set_context_task_store

        token = set_context_task_store(None)
        try:
            result = get_tasks()
            assert result["success"] is False
        finally:
            token.var.reset(token)


class TestTaskStoreProgress:
    """Tests for TaskStore progress/display helper methods."""

    def test_get_progress_empty(self, mock_context):
        store = TaskStore(mock_context.settings)
        progress = store.get_progress()
        assert progress == {"total": 0, "pending": 0, "in_progress": 0, "completed": 0, "cancelled": 0}

    def test_get_progress_mixed(self, mock_context):
        store = TaskStore(mock_context.settings)
        store.replace_all([
            {"description": "Task 1", "status": "completed"},
            {"description": "Task 2", "status": "in_progress"},
            {"description": "Task 3", "status": "pending"},
            {"description": "Task 4", "status": "cancelled"},
        ])
        progress = store.get_progress()
        assert progress["total"] == 4
        assert progress["pending"] == 1
        assert progress["in_progress"] == 1
        assert progress["completed"] == 1
        assert progress["cancelled"] == 1

    def test_get_progress_all_completed(self, mock_context):
        store = TaskStore(mock_context.settings)
        store.replace_all([
            {"description": "Task 1", "status": "completed"},
            {"description": "Task 2", "status": "completed"},
        ])
        progress = store.get_progress()
        assert progress["total"] == 2
        assert progress["completed"] == 2
        assert progress["pending"] == 0

    def test_to_compact_display_empty(self, mock_context):
        store = TaskStore(mock_context.settings)
        assert store.to_compact_display() == ""

    def test_to_compact_display_mixed(self, mock_context):
        store = TaskStore(mock_context.settings)
        store.replace_all([
            {"description": "Gather data", "status": "completed"},
            {"description": "Analyze results", "status": "in_progress"},
            {"description": "Write report", "status": "pending"},
        ])
        display = store.to_compact_display()
        assert "[✓] Gather data" in display
        assert "[▸] Analyze results" in display
        assert "[ ] Write report" in display

    def test_to_compact_display_cancelled(self, mock_context):
        store = TaskStore(mock_context.settings)
        store.replace_all([
            {"description": "Dropped task", "status": "cancelled"},
        ])
        display = store.to_compact_display()
        assert "[-] Dropped task" in display

    def test_to_compact_display_sorted_order(self, mock_context):
        """In-progress tasks appear first, then pending, cancelled, completed."""
        store = TaskStore(mock_context.settings)
        store.replace_all([
            {"description": "Done task", "status": "completed"},
            {"description": "Waiting task", "status": "pending"},
            {"description": "Active task", "status": "in_progress"},
            {"description": "Dropped task", "status": "cancelled"},
        ])
        display = store.to_compact_display()
        lines = display.strip().splitlines()
        assert lines[0] == "[▸] Active task"
        assert lines[1] == "[ ] Waiting task"
        assert lines[2] == "[-] Dropped task"
        assert lines[3] == "[✓] Done task"

    def test_to_compact_display_preserves_order_within_status(self, mock_context):
        """Tasks with the same status preserve insertion order."""
        store = TaskStore(mock_context.settings)
        store.replace_all([
            {"description": "First pending"},
            {"description": "Second pending"},
            {"description": "Third pending"},
        ])
        display = store.to_compact_display()
        lines = display.strip().splitlines()
        assert lines[0] == "[ ] First pending"
        assert lines[1] == "[ ] Second pending"
        assert lines[2] == "[ ] Third pending"

    def test_get_current_task_none(self, mock_context):
        store = TaskStore(mock_context.settings)
        assert store.get_current_task() is None

    def test_get_current_task_only_pending(self, mock_context):
        store = TaskStore(mock_context.settings)
        store.replace_all([{"description": "Pending task"}])
        assert store.get_current_task() is None

    def test_get_current_task_found(self, mock_context):
        store = TaskStore(mock_context.settings)
        ids = store.replace_all([
            {"description": "Pending"},
            {"description": "Active", "status": "in_progress"},
        ])
        current = store.get_current_task()
        assert current is not None
        assert current.id == ids[1]
        assert current.description == "Active"

    def test_get_current_task_returns_first_in_progress(self, mock_context):
        store = TaskStore(mock_context.settings)
        ids = store.replace_all([
            {"description": "First active", "status": "in_progress"},
            {"description": "Second active", "status": "in_progress"},
        ])
        current = store.get_current_task()
        assert current.id == ids[0]

    def test_all_done_empty(self, mock_context):
        """all_done() returns False for empty store."""
        store = TaskStore(mock_context.settings)
        assert store.all_done() is False

    def test_all_done_all_completed(self, mock_context):
        """all_done() returns True when all tasks are completed."""
        store = TaskStore(mock_context.settings)
        store.replace_all([
            {"description": "Task 1", "status": "completed"},
            {"description": "Task 2", "status": "completed"},
        ])
        assert store.all_done() is True

    def test_all_done_mixed_terminal(self, mock_context):
        """all_done() returns True for mix of completed and cancelled."""
        store = TaskStore(mock_context.settings)
        store.replace_all([
            {"description": "Task 1", "status": "completed"},
            {"description": "Task 2", "status": "cancelled"},
        ])
        assert store.all_done() is True

    def test_all_done_with_pending(self, mock_context):
        """all_done() returns False when any task is pending."""
        store = TaskStore(mock_context.settings)
        store.replace_all([
            {"description": "Task 1", "status": "completed"},
            {"description": "Task 2", "status": "pending"},
        ])
        assert store.all_done() is False

    def test_all_done_with_in_progress(self, mock_context):
        """all_done() returns False when any task is in_progress."""
        store = TaskStore(mock_context.settings)
        store.replace_all([
            {"description": "Task 1", "status": "completed"},
            {"description": "Task 2", "status": "in_progress"},
        ])
        assert store.all_done() is False


class _MinimalWorkflowManager(BaseWorkflowManager):
    """Minimal concrete subclass for testing base class methods."""

    async def _do_initialize(self) -> None:
        pass

    async def process(
        self, message: str, user_id: str, session_id: str | None = None
    ) -> AsyncGenerator[WorkflowEvent, None]:
        if False:
            yield  # type: ignore[misc]

    async def reinitialize(self, model: str | None = None, preserve_sessions: bool = True) -> None:
        pass

    async def cleanup(self) -> None:
        pass


class TestEmitTaskProgressEvent:
    """Tests for BaseWorkflowManager._emit_task_progress_event()."""

    def test_returns_none_when_no_store(self, mock_context):
        mgr = _MinimalWorkflowManager(agent_configs=[], settings=mock_context.settings)
        mgr._task_store = None
        assert mgr._emit_task_progress_event() is None

    def test_returns_none_when_empty(self, mock_context):
        store = TaskStore(mock_context.settings)
        mgr = _MinimalWorkflowManager(agent_configs=[], settings=mock_context.settings)
        mgr._task_store = store
        assert mgr._emit_task_progress_event() is None

    def test_returns_task_progress_event(self, mock_context):
        store = TaskStore(mock_context.settings)
        ids = store.replace_all([
            {"description": "Research topic", "status": "in_progress"},
            {"description": "Write summary"},
        ])

        mgr = _MinimalWorkflowManager(agent_configs=[], settings=mock_context.settings)
        mgr._task_store = store

        event = mgr._emit_task_progress_event()
        assert event is not None
        assert event.type == EventType.TASK_PROGRESS
        assert "[▸] Research topic" in event.content
        assert "[ ] Write summary" in event.content
        assert event.metadata["progress"]["total"] == 2
        assert event.metadata["progress"]["in_progress"] == 1
        assert event.metadata["current_task_id"] == ids[0]
        assert event.metadata["current_task_description"] == "Research topic"

    def test_works_without_in_progress_task(self, mock_context):
        store = TaskStore(mock_context.settings)
        store.replace_all([{"description": "Pending task"}])

        mgr = _MinimalWorkflowManager(agent_configs=[], settings=mock_context.settings)
        mgr._task_store = store

        event = mgr._emit_task_progress_event()
        assert event is not None
        assert event.type == EventType.TASK_PROGRESS
        assert "[ ] Pending task" in event.content
        assert "current_task_id" not in event.metadata
        assert event.metadata["progress"]["pending"] == 1

    def test_falls_back_to_plan_store(self, mock_context):
        """When TaskStore is None, uses PlanStore checkboxes."""
        from agentic_cli.tools.planning_tools import PlanStore

        plan_store = PlanStore()
        plan_store.save(
            "## Research\n- [x] Gather data\n- [ ] Analyze results\n"
            "## Writing\n- [ ] Draft report"
        )

        mgr = _MinimalWorkflowManager(agent_configs=[], settings=mock_context.settings)
        mgr._task_store = None
        mgr._plan_store = plan_store

        event = mgr._emit_task_progress_event()
        assert event is not None
        assert event.type == EventType.TASK_PROGRESS
        assert "Research:" in event.content
        assert "[✓] Gather data" in event.content
        assert "[ ] Analyze results" in event.content
        assert "Writing:" in event.content
        assert "[ ] Draft report" in event.content
        assert event.metadata["progress"]["total"] == 3
        assert event.metadata["progress"]["completed"] == 1
        assert event.metadata["progress"]["pending"] == 2

    def test_plan_progress_all_done_returns_none(self, mock_context):
        """When all plan checkboxes are checked, returns None."""
        from agentic_cli.tools.planning_tools import PlanStore

        plan_store = PlanStore()
        plan_store.save("- [x] Task A\n- [x] Task B")

        mgr = _MinimalWorkflowManager(agent_configs=[], settings=mock_context.settings)
        mgr._task_store = None
        mgr._plan_store = plan_store

        assert mgr._emit_task_progress_event() is None

    def test_plan_progress_no_checkboxes_returns_none(self, mock_context):
        """When plan has no checkboxes, returns None."""
        from agentic_cli.tools.planning_tools import PlanStore

        plan_store = PlanStore()
        plan_store.save("## Plan\nJust some text without checkboxes.")

        mgr = _MinimalWorkflowManager(agent_configs=[], settings=mock_context.settings)
        mgr._task_store = None
        mgr._plan_store = plan_store

        assert mgr._emit_task_progress_event() is None

    def test_plan_progress_empty_plan_returns_none(self, mock_context):
        """When PlanStore is empty, returns None."""
        from agentic_cli.tools.planning_tools import PlanStore

        plan_store = PlanStore()

        mgr = _MinimalWorkflowManager(agent_configs=[], settings=mock_context.settings)
        mgr._task_store = None
        mgr._plan_store = plan_store

        assert mgr._emit_task_progress_event() is None

    def test_task_store_takes_priority(self, mock_context):
        """When TaskStore has tasks, PlanStore is ignored."""
        from agentic_cli.tools.planning_tools import PlanStore

        task_store = TaskStore(mock_context.settings)
        task_store.replace_all([
            {"description": "TaskStore item", "status": "in_progress"},
        ])

        plan_store = PlanStore()
        plan_store.save("- [ ] PlanStore item")

        mgr = _MinimalWorkflowManager(agent_configs=[], settings=mock_context.settings)
        mgr._task_store = task_store
        mgr._plan_store = plan_store

        event = mgr._emit_task_progress_event()
        assert event is not None
        assert "TaskStore item" in event.content
        assert "PlanStore item" not in event.content

    def test_plan_progress_includes_sections(self, mock_context):
        """Section headers from plan are included in display."""
        from agentic_cli.tools.planning_tools import PlanStore

        plan_store = PlanStore()
        plan_store.save(
            "## Phase 1\n"
            "- [x] Setup\n"
            "- [ ] Configure\n"
            "\n"
            "### Phase 2\n"
            "- [ ] Build\n"
        )

        mgr = _MinimalWorkflowManager(agent_configs=[], settings=mock_context.settings)
        mgr._task_store = None
        mgr._plan_store = plan_store

        event = mgr._emit_task_progress_event()
        assert event is not None
        assert "Phase 1:" in event.content
        assert "Phase 2:" in event.content
        assert "[✓] Setup" in event.content
        assert "[ ] Configure" in event.content
        assert "[ ] Build" in event.content
