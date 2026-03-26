"""Tests for task management tools."""

from unittest.mock import patch

import pytest

from agentic_cli.tools.task_tools import TaskStore, TaskItem


@pytest.fixture
def task_registry_ctx(mock_context):
    """Provide a service registry for task tools, auto-cleanup."""
    from agentic_cli.workflow.service_registry import set_service_registry

    registry = {}
    token = set_service_registry(registry)
    yield registry
    token.var.reset(token)


@pytest.fixture
def no_task_registry_ctx():
    """Set service registry to None (no registry), auto-cleanup."""
    from agentic_cli.workflow.service_registry import clear_service_registry

    token = clear_service_registry()
    yield
    token.var.reset(token)


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

    def test_save_tasks_bulk_create(self, task_registry_ctx):
        from agentic_cli.tools.task_tools import save_tasks

        result = save_tasks(tasks=[
            {"description": "Task 1"},
            {"description": "Task 2"},
            {"description": "Task 3"},
        ])
        assert result["success"] is True
        assert result["count"] == 3
        assert len(result["task_ids"]) == 3

    def test_save_tasks_replaces_existing(self, task_registry_ctx):
        from agentic_cli.tools.task_tools import save_tasks

        save_tasks(tasks=[{"description": "Old task"}])
        result = save_tasks(tasks=[
            {"description": "New task 1"},
            {"description": "New task 2"},
        ])
        assert result["count"] == 2
        # Verify registry has the right number of tasks
        assert len(task_registry_ctx.get("tasks", [])) == 2

    def test_save_tasks_with_statuses(self, task_registry_ctx):
        from agentic_cli.tools.task_tools import save_tasks, get_tasks

        result = save_tasks(tasks=[
            {"description": "Done", "status": "completed"},
            {"description": "Active", "status": "in_progress"},
            {"description": "Todo"},
        ])
        assert result["success"] is True
        # Verify via get_tasks
        completed = get_tasks(status="completed")
        assert completed["count"] == 1
        in_progress = get_tasks(status="in_progress")
        assert in_progress["count"] == 1
        pending = get_tasks(status="pending")
        assert pending["count"] == 1

    def test_save_tasks_empty_clears(self, task_registry_ctx):
        from agentic_cli.tools.task_tools import save_tasks

        save_tasks(tasks=[{"description": "Task"}])
        result = save_tasks(tasks=[])
        assert result["success"] is True
        assert result["count"] == 0
        assert task_registry_ctx.get("tasks", []) == []

    def test_save_tasks_missing_description(self, task_registry_ctx):
        from agentic_cli.tools.task_tools import save_tasks

        result = save_tasks(tasks=[
            {"description": "Valid"},
            {"status": "pending"},  # missing description
        ])
        assert result["success"] is False
        assert "index 1" in result["error"]
        assert "description" in result["error"].lower()

    def test_save_tasks_invalid_status_returns_error(self, task_registry_ctx):
        from agentic_cli.tools.task_tools import save_tasks

        result = save_tasks(tasks=[
            {"description": "Valid task"},
            {"description": "Bad status", "status": "bogus"},
        ])
        assert result["success"] is False
        assert "index 1" in result["error"]
        assert "bogus" in result["error"]

    def test_save_tasks_invalid_priority_returns_error(self, task_registry_ctx):
        from agentic_cli.tools.task_tools import save_tasks

        result = save_tasks(tasks=[
            {"description": "Bad priority", "priority": "critical"},
        ])
        assert result["success"] is False
        assert "index 0" in result["error"]
        assert "critical" in result["error"]

    def test_save_tasks_no_registry(self, no_task_registry_ctx):
        from agentic_cli.tools.task_tools import save_tasks

        # With no registry, get_service_registry() auto-creates one,
        # so the tool should succeed
        result = save_tasks(tasks=[{"description": "Test"}])
        assert result["success"] is True

    def test_get_tasks(self, task_registry_ctx):
        from agentic_cli.tools.task_tools import save_tasks, get_tasks

        save_tasks(tasks=[
            {"description": "Task 1"},
            {"description": "Task 2"},
        ])
        result = get_tasks()
        assert result["success"] is True
        assert result["count"] == 2
        assert len(result["tasks"]) == 2

    def test_get_tasks_with_filter(self, task_registry_ctx):
        from agentic_cli.tools.task_tools import save_tasks, get_tasks

        save_tasks(tasks=[
            {"description": "Task 1", "priority": "high", "status": "in_progress"},
            {"description": "Task 2", "priority": "low"},
        ])
        result = get_tasks(status="in_progress")
        assert result["count"] == 1
        assert result["tasks"][0]["priority"] == "high"

    def test_get_tasks_empty_registry(self, task_registry_ctx):
        from agentic_cli.tools.task_tools import get_tasks

        result = get_tasks()
        assert result["success"] is True
        assert result["count"] == 0


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


class TestEmitTaskProgressEvent:
    """Tests for task progress data (formerly _emit_task_progress_event).

    The base manager no longer has this method — task progress is handled
    per-backend (ADK plugin, LangGraph graph state).  See:
    - tests/workflow/test_task_progress_plugin.py (ADK)
    - tests/tools/test_core_tasks.py (shared task_progress_data)
    """

    def test_progress_data_returns_none_when_empty(self):
        from agentic_cli.tools._core.tasks import task_progress_data
        assert task_progress_data([]) is None

    def test_progress_data_with_tasks(self):
        from agentic_cli.tools._core.tasks import task_progress_data, normalize_tasks
        normalized, ids = normalize_tasks([
            {"description": "Research topic", "status": "in_progress"},
            {"description": "Write summary"},
        ])
        result = task_progress_data(normalized)
        assert result is not None
        assert "[▸] Research topic" in result["display"]
        assert "[ ] Write summary" in result["display"]
        assert result["progress"]["total"] == 2
        assert result["progress"]["in_progress"] == 1
        assert result["current_task_id"] == ids[0]
        assert result["current_task_description"] == "Research topic"

    def test_progress_data_without_in_progress(self):
        from agentic_cli.tools._core.tasks import task_progress_data, normalize_tasks
        normalized, _ = normalize_tasks([{"description": "Pending task"}])
        result = task_progress_data(normalized)
        assert result is not None
        assert "[ ] Pending task" in result["display"]
        assert result["current_task_id"] is None
        assert result["progress"]["pending"] == 1
