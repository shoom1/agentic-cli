"""Tests for task management pure functions."""

from unittest.mock import patch

import pytest

from agentic_cli.tools._core.tasks import (
    TaskStatus,
    TaskPriority,
    normalize_tasks,
    filter_tasks,
    task_progress_data,
)


class TestNormalizeTasks:
    """Tests for normalize_tasks (replaces TestTaskItem)."""

    def test_normalize_basic(self):
        normalized, ids = normalize_tasks([
            {"description": "Test task", "priority": "high", "tags": ["feature"]},
        ])
        assert len(normalized) == 1
        t = normalized[0]
        assert t["description"] == "Test task"
        assert t["status"] == "pending"
        assert t["priority"] == "high"
        assert t["tags"] == ["feature"]
        assert t["created_at"] != ""
        assert t["completed_at"] == ""
        assert ids[0] == t["id"]

    def test_normalize_all_fields(self):
        normalized, ids = normalize_tasks([{
            "id": "abc123",
            "description": "Test task",
            "status": "in_progress",
            "priority": "low",
            "tags": ["bug"],
            "created_at": "2024-01-01T00:00:00",
            "completed_at": "",
        }])
        t = normalized[0]
        assert t["id"] == "abc123"
        assert t["description"] == "Test task"
        assert t["status"] == "in_progress"
        assert t["priority"] == "low"
        assert t["tags"] == ["bug"]

    def test_normalize_defaults(self):
        normalized, _ = normalize_tasks([{"description": "Minimal"}])
        t = normalized[0]
        assert t["status"] == "pending"
        assert t["priority"] == "medium"
        assert t["tags"] == []

    def test_normalize_invalid_status_defaults_to_pending(self):
        normalized, _ = normalize_tasks([
            {"description": "Bad status", "status": "bogus"},
        ])
        assert normalized[0]["status"] == "pending"

    def test_normalize_invalid_priority_defaults_to_medium(self):
        normalized, _ = normalize_tasks([
            {"description": "Bad priority", "priority": "critical"},
        ])
        assert normalized[0]["priority"] == "medium"

    def test_normalize_roundtrip(self):
        original = [{
            "id": "rt1",
            "description": "Roundtrip",
            "status": "completed",
            "priority": "medium",
            "tags": ["a", "b"],
            "created_at": "2024-06-01T12:00:00",
            "completed_at": "2024-06-01T13:00:00",
        }]
        normalized, ids = normalize_tasks(original)
        t = normalized[0]
        assert t["id"] == "rt1"
        assert t["description"] == "Roundtrip"
        assert t["status"] == "completed"
        assert t["completed_at"] == "2024-06-01T13:00:00"


class TestFilterTasks:
    """Tests for filter_tasks and normalize_tasks (replaces TestTaskStore)."""

    def test_normalize_creates_tasks(self):
        normalized, ids = normalize_tasks([
            {"description": "Implement auth module", "priority": "high"},
            {"description": "Write tests"},
        ])
        assert len(ids) == 2
        assert normalized[0]["description"] == "Implement auth module"
        assert normalized[0]["priority"] == "high"
        assert normalized[0]["status"] == "pending"

    def test_normalize_with_tags(self):
        normalized, ids = normalize_tasks([
            {"description": "Fix bug", "tags": ["bug", "urgent"]},
        ])
        assert normalized[0]["tags"] == ["bug", "urgent"]

    def test_normalize_with_statuses(self):
        normalized, ids = normalize_tasks([
            {"description": "Done task", "status": "completed"},
            {"description": "Active task", "status": "in_progress"},
            {"description": "Pending task"},
        ])
        assert normalized[0]["status"] == "completed"
        assert normalized[1]["status"] == "in_progress"
        assert normalized[2]["status"] == "pending"

    def test_normalize_completed_sets_timestamp(self):
        normalized, _ = normalize_tasks([
            {"description": "Test task", "status": "completed"},
        ])
        assert normalized[0]["status"] == "completed"
        assert normalized[0]["completed_at"] != ""

    def test_normalize_preserves_existing_id(self):
        normalized, ids = normalize_tasks([
            {"id": "custom-id", "description": "Task with ID"},
        ])
        assert ids == ["custom-id"]
        assert normalized[0]["id"] == "custom-id"

    def test_filter_by_status(self):
        normalized, _ = normalize_tasks([
            {"description": "Task 1", "status": "in_progress"},
            {"description": "Task 2", "status": "pending"},
        ])
        assert len(filter_tasks(normalized, status="in_progress")) == 1
        assert len(filter_tasks(normalized, status="pending")) == 1

    def test_filter_by_priority(self):
        normalized, _ = normalize_tasks([
            {"description": "Low", "priority": "low"},
            {"description": "High", "priority": "high"},
        ])
        assert len(filter_tasks(normalized, priority="high")) == 1

    def test_filter_by_tag(self):
        normalized, _ = normalize_tasks([
            {"description": "Tagged", "tags": ["important"]},
            {"description": "Untagged"},
        ])
        assert len(filter_tasks(normalized, tag="important")) == 1


class TestTaskProgressData:
    """Tests for task_progress_data (replaces TestTaskStoreProgress)."""

    def test_progress_empty(self):
        assert task_progress_data([]) is None

    def test_progress_mixed(self):
        normalized, _ = normalize_tasks([
            {"description": "Task 1", "status": "completed"},
            {"description": "Task 2", "status": "in_progress"},
            {"description": "Task 3", "status": "pending"},
            {"description": "Task 4", "status": "cancelled"},
        ])
        result = task_progress_data(normalized)
        progress = result["progress"]
        assert progress["total"] == 4
        assert progress["pending"] == 1
        assert progress["in_progress"] == 1
        assert progress["completed"] == 1
        assert progress["cancelled"] == 1

    def test_progress_all_completed(self):
        normalized, _ = normalize_tasks([
            {"description": "Task 1", "status": "completed"},
            {"description": "Task 2", "status": "completed"},
        ])
        result = task_progress_data(normalized)
        assert result["progress"]["total"] == 2
        assert result["progress"]["completed"] == 2
        assert result["progress"]["pending"] == 0

    def test_all_done_empty(self):
        """task_progress_data returns None for empty list."""
        assert task_progress_data([]) is None

    def test_all_done_all_completed(self):
        normalized, _ = normalize_tasks([
            {"description": "Task 1", "status": "completed"},
            {"description": "Task 2", "status": "completed"},
        ])
        result = task_progress_data(normalized)
        assert result["all_done"] is True

    def test_all_done_mixed_terminal(self):
        normalized, _ = normalize_tasks([
            {"description": "Task 1", "status": "completed"},
            {"description": "Task 2", "status": "cancelled"},
        ])
        result = task_progress_data(normalized)
        assert result["all_done"] is True

    def test_all_done_with_pending(self):
        normalized, _ = normalize_tasks([
            {"description": "Task 1", "status": "completed"},
            {"description": "Task 2", "status": "pending"},
        ])
        result = task_progress_data(normalized)
        assert result["all_done"] is False

    def test_all_done_with_in_progress(self):
        normalized, _ = normalize_tasks([
            {"description": "Task 1", "status": "completed"},
            {"description": "Task 2", "status": "in_progress"},
        ])
        result = task_progress_data(normalized)
        assert result["all_done"] is False


class TestEmitTaskProgressEvent:
    """Tests for task progress data (formerly _emit_task_progress_event).

    The base manager no longer has this method — task progress is handled
    per-backend (ADK plugin, LangGraph graph state).  See:
    - tests/workflow/test_task_progress_plugin.py (ADK)
    - tests/tools/test_core_tasks.py (shared task_progress_data)
    """

    def test_progress_data_returns_none_when_empty(self):
        assert task_progress_data([]) is None

    def test_progress_data_with_tasks(self):
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
        normalized, _ = normalize_tasks([{"description": "Pending task"}])
        result = task_progress_data(normalized)
        assert result is not None
        assert "[ ] Pending task" in result["display"]
        assert result["current_task_id"] is None
        assert result["progress"]["pending"] == 1
