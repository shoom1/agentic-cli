"""Tests for task management tools."""

import json
from unittest.mock import patch

import pytest

from agentic_cli.tasks.store import TaskStore, TaskItem


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

    def test_create_task(self, mock_context):
        store = TaskStore(mock_context.settings)
        task_id = store.create("Implement auth module", priority="high")
        assert task_id
        task = store.get(task_id)
        assert task is not None
        assert task.description == "Implement auth module"
        assert task.priority == "high"
        assert task.status == "pending"

    def test_create_with_tags(self, mock_context):
        store = TaskStore(mock_context.settings)
        task_id = store.create("Fix bug", tags=["bug", "urgent"])
        task = store.get(task_id)
        assert task.tags == ["bug", "urgent"]

    def test_update_status(self, mock_context):
        store = TaskStore(mock_context.settings)
        task_id = store.create("Test task")
        assert store.update_status(task_id, "in_progress")
        assert store.get(task_id).status == "in_progress"

    def test_update_status_completed_sets_timestamp(self, mock_context):
        store = TaskStore(mock_context.settings)
        task_id = store.create("Test task")
        store.update_status(task_id, "completed")
        task = store.get(task_id)
        assert task.status == "completed"
        assert task.completed_at != ""

    def test_update_status_not_found(self, mock_context):
        store = TaskStore(mock_context.settings)
        assert store.update_status("nonexistent", "completed") is False

    def test_update_fields(self, mock_context):
        store = TaskStore(mock_context.settings)
        task_id = store.create("Original")
        store.update(task_id, description="Updated", priority="high")
        task = store.get(task_id)
        assert task.description == "Updated"
        assert task.priority == "high"

    def test_update_not_found(self, mock_context):
        store = TaskStore(mock_context.settings)
        assert store.update("nonexistent", description="x") is False

    def test_delete(self, mock_context):
        store = TaskStore(mock_context.settings)
        task_id = store.create("To delete")
        assert store.delete(task_id)
        assert store.get(task_id) is None

    def test_delete_not_found(self, mock_context):
        store = TaskStore(mock_context.settings)
        assert store.delete("nonexistent") is False

    def test_list_all(self, mock_context):
        store = TaskStore(mock_context.settings)
        store.create("Task 1")
        store.create("Task 2")
        store.create("Task 3")
        assert len(store.list_tasks()) == 3

    def test_list_filter_status(self, mock_context):
        store = TaskStore(mock_context.settings)
        id1 = store.create("Task 1")
        store.create("Task 2")
        store.update_status(id1, "in_progress")
        assert len(store.list_tasks(status="in_progress")) == 1
        assert len(store.list_tasks(status="pending")) == 1

    def test_list_filter_priority(self, mock_context):
        store = TaskStore(mock_context.settings)
        store.create("Low", priority="low")
        store.create("High", priority="high")
        assert len(store.list_tasks(priority="high")) == 1

    def test_list_filter_tag(self, mock_context):
        store = TaskStore(mock_context.settings)
        store.create("Tagged", tags=["important"])
        store.create("Untagged")
        assert len(store.list_tasks(tag="important")) == 1

    def test_is_empty(self, mock_context):
        store = TaskStore(mock_context.settings)
        assert store.is_empty()
        store.create("Task")
        assert not store.is_empty()

    def test_persistence(self, mock_context):
        """Test that tasks persist across store instances."""
        store1 = TaskStore(mock_context.settings)
        task_id = store1.create("Persistent task")

        store2 = TaskStore(mock_context.settings)
        task = store2.get(task_id)
        assert task is not None
        assert task.description == "Persistent task"

    def test_persistence_corrupt_file(self, mock_context):
        """Test that corrupt JSON file is handled gracefully."""
        store = TaskStore(mock_context.settings)
        store.create("Task")
        # Corrupt the file
        store._storage_path.write_text("not json")
        store2 = TaskStore(mock_context.settings)
        assert store2.is_empty()


class TestTaskTools:
    """Tests for save_tasks and get_tasks tool functions."""

    def test_save_tasks_create(self, mock_context):
        from agentic_cli.tasks.store import TaskStore
        from agentic_cli.tools.task_tools import save_tasks
        from agentic_cli.workflow.context import set_context_task_store

        store = TaskStore(mock_context.settings)
        token = set_context_task_store(store)
        try:
            result = save_tasks(operation="create", description="New task")
            assert result["success"] is True
            assert "task_id" in result
        finally:
            token.var.reset(token)

    def test_save_tasks_create_missing_description(self, mock_context):
        from agentic_cli.tasks.store import TaskStore
        from agentic_cli.tools.task_tools import save_tasks
        from agentic_cli.workflow.context import set_context_task_store

        store = TaskStore(mock_context.settings)
        token = set_context_task_store(store)
        try:
            result = save_tasks(operation="create")
            assert result["success"] is False
            assert "description" in result["error"].lower()
        finally:
            token.var.reset(token)

    def test_save_tasks_update(self, mock_context):
        from agentic_cli.tasks.store import TaskStore
        from agentic_cli.tools.task_tools import save_tasks
        from agentic_cli.workflow.context import set_context_task_store

        store = TaskStore(mock_context.settings)
        task_id = store.create("Test")
        token = set_context_task_store(store)
        try:
            result = save_tasks(operation="update", task_id=task_id, status="in_progress")
            assert result["success"] is True
            assert store.get(task_id).status == "in_progress"
        finally:
            token.var.reset(token)

    def test_save_tasks_delete(self, mock_context):
        from agentic_cli.tasks.store import TaskStore
        from agentic_cli.tools.task_tools import save_tasks
        from agentic_cli.workflow.context import set_context_task_store

        store = TaskStore(mock_context.settings)
        task_id = store.create("To delete")
        token = set_context_task_store(store)
        try:
            result = save_tasks(operation="delete", task_id=task_id)
            assert result["success"] is True
            assert store.get(task_id) is None
        finally:
            token.var.reset(token)

    def test_save_tasks_unknown_operation(self, mock_context):
        from agentic_cli.tasks.store import TaskStore
        from agentic_cli.tools.task_tools import save_tasks
        from agentic_cli.workflow.context import set_context_task_store

        store = TaskStore(mock_context.settings)
        token = set_context_task_store(store)
        try:
            result = save_tasks(operation="invalid")
            assert result["success"] is False
            assert "unknown" in result["error"].lower()
        finally:
            token.var.reset(token)

    def test_save_tasks_no_store(self):
        from agentic_cli.tools.task_tools import save_tasks
        from agentic_cli.workflow.context import set_context_task_store

        token = set_context_task_store(None)
        try:
            result = save_tasks(operation="create", description="Test")
            assert result["success"] is False
            assert "not available" in result["error"].lower()
        finally:
            token.var.reset(token)

    def test_get_tasks(self, mock_context):
        from agentic_cli.tasks.store import TaskStore
        from agentic_cli.tools.task_tools import get_tasks
        from agentic_cli.workflow.context import set_context_task_store

        store = TaskStore(mock_context.settings)
        store.create("Task 1")
        store.create("Task 2")
        token = set_context_task_store(store)
        try:
            result = get_tasks()
            assert result["success"] is True
            assert result["count"] == 2
            assert len(result["tasks"]) == 2
        finally:
            token.var.reset(token)

    def test_get_tasks_with_filter(self, mock_context):
        from agentic_cli.tasks.store import TaskStore
        from agentic_cli.tools.task_tools import get_tasks
        from agentic_cli.workflow.context import set_context_task_store

        store = TaskStore(mock_context.settings)
        id1 = store.create("Task 1", priority="high")
        store.create("Task 2", priority="low")
        store.update_status(id1, "in_progress")
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
