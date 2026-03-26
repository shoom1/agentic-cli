"""Tests for agentic_cli.tools._core.tasks — pure task functions."""

import pytest

from agentic_cli.tools._core.tasks import (
    validate_tasks,
    normalize_tasks,
    filter_tasks,
    format_task_checklist,
    task_progress_data,
)


# ---------------------------------------------------------------------------
# validate_tasks
# ---------------------------------------------------------------------------


class TestValidateTasks:
    """Tests for validate_tasks()."""

    def test_valid_tasks_returns_none(self):
        tasks = [
            {"description": "Task 1", "status": "pending", "priority": "high"},
            {"description": "Task 2"},
        ]
        assert validate_tasks(tasks) is None

    def test_empty_list_returns_none(self):
        assert validate_tasks([]) is None

    def test_missing_description_returns_error(self):
        result = validate_tasks([{"status": "pending"}])
        assert result is not None
        assert result["success"] is False
        assert "index 0" in result["error"]
        assert "description" in result["error"].lower()

    def test_empty_description_returns_error(self):
        result = validate_tasks([{"description": ""}])
        assert result is not None
        assert result["success"] is False

    def test_missing_description_at_index_2(self):
        tasks = [
            {"description": "A"},
            {"description": "B"},
            {"status": "pending"},  # missing description
        ]
        result = validate_tasks(tasks)
        assert result is not None
        assert "index 2" in result["error"]

    def test_invalid_status_returns_error(self):
        result = validate_tasks([{"description": "Task", "status": "bogus"}])
        assert result is not None
        assert result["success"] is False
        assert "bogus" in result["error"]

    def test_invalid_priority_returns_error(self):
        result = validate_tasks([{"description": "Task", "priority": "critical"}])
        assert result is not None
        assert result["success"] is False
        assert "critical" in result["error"]

    def test_valid_statuses_accepted(self):
        for status in ("pending", "in_progress", "completed", "cancelled"):
            assert validate_tasks([{"description": "T", "status": status}]) is None

    def test_valid_priorities_accepted(self):
        for priority in ("low", "medium", "high"):
            assert validate_tasks([{"description": "T", "priority": priority}]) is None

    def test_defaults_are_valid(self):
        """Tasks with only description should pass (defaults are valid)."""
        assert validate_tasks([{"description": "Minimal task"}]) is None


# ---------------------------------------------------------------------------
# normalize_tasks
# ---------------------------------------------------------------------------


class TestNormalizeTasks:
    """Tests for normalize_tasks()."""

    def test_assigns_ids(self):
        normalized, ids = normalize_tasks([{"description": "A"}, {"description": "B"}])
        assert len(ids) == 2
        assert all(isinstance(tid, str) and len(tid) > 0 for tid in ids)
        assert normalized[0]["id"] == ids[0]
        assert normalized[1]["id"] == ids[1]

    def test_preserves_existing_id(self):
        normalized, ids = normalize_tasks([{"id": "my-id", "description": "A"}])
        assert ids == ["my-id"]
        assert normalized[0]["id"] == "my-id"

    def test_defaults_status_to_pending(self):
        normalized, _ = normalize_tasks([{"description": "A"}])
        assert normalized[0]["status"] == "pending"

    def test_defaults_priority_to_medium(self):
        normalized, _ = normalize_tasks([{"description": "A"}])
        assert normalized[0]["priority"] == "medium"

    def test_defaults_tags_to_empty(self):
        normalized, _ = normalize_tasks([{"description": "A"}])
        assert normalized[0]["tags"] == []

    def test_sets_created_at(self):
        normalized, _ = normalize_tasks([{"description": "A"}])
        assert normalized[0]["created_at"] != ""

    def test_preserves_existing_created_at(self):
        normalized, _ = normalize_tasks([
            {"description": "A", "created_at": "2024-01-01T00:00:00"}
        ])
        assert normalized[0]["created_at"] == "2024-01-01T00:00:00"

    def test_completed_sets_completed_at(self):
        normalized, _ = normalize_tasks([
            {"description": "A", "status": "completed"}
        ])
        assert normalized[0]["completed_at"] != ""

    def test_completed_preserves_existing_completed_at(self):
        normalized, _ = normalize_tasks([
            {"description": "A", "status": "completed", "completed_at": "2024-06-01T12:00:00"}
        ])
        assert normalized[0]["completed_at"] == "2024-06-01T12:00:00"

    def test_pending_has_empty_completed_at(self):
        normalized, _ = normalize_tasks([{"description": "A"}])
        assert normalized[0]["completed_at"] == ""

    def test_invalid_status_defaults_to_pending(self):
        normalized, _ = normalize_tasks([{"description": "A", "status": "bogus"}])
        assert normalized[0]["status"] == "pending"

    def test_invalid_priority_defaults_to_medium(self):
        normalized, _ = normalize_tasks([{"description": "A", "priority": "critical"}])
        assert normalized[0]["priority"] == "medium"

    def test_all_fields_present(self):
        normalized, _ = normalize_tasks([{"description": "A"}])
        expected_keys = {"id", "description", "status", "priority", "tags", "created_at", "completed_at"}
        assert set(normalized[0].keys()) == expected_keys

    def test_empty_list(self):
        normalized, ids = normalize_tasks([])
        assert normalized == []
        assert ids == []


# ---------------------------------------------------------------------------
# filter_tasks
# ---------------------------------------------------------------------------


class TestFilterTasks:
    """Tests for filter_tasks()."""

    @pytest.fixture
    def sample_tasks(self):
        return [
            {"id": "1", "description": "High pending", "status": "pending", "priority": "high", "tags": ["feature"]},
            {"id": "2", "description": "Low active", "status": "in_progress", "priority": "low", "tags": ["bug"]},
            {"id": "3", "description": "Medium done", "status": "completed", "priority": "medium", "tags": ["feature", "bug"]},
            {"id": "4", "description": "High active", "status": "in_progress", "priority": "high", "tags": []},
        ]

    def test_no_filters_returns_all(self, sample_tasks):
        result = filter_tasks(sample_tasks)
        assert len(result) == 4

    def test_filter_by_status(self, sample_tasks):
        result = filter_tasks(sample_tasks, status="in_progress")
        assert len(result) == 2
        assert all(t["status"] == "in_progress" for t in result)

    def test_filter_by_priority(self, sample_tasks):
        result = filter_tasks(sample_tasks, priority="high")
        assert len(result) == 2
        assert all(t["priority"] == "high" for t in result)

    def test_filter_by_tag(self, sample_tasks):
        result = filter_tasks(sample_tasks, tag="feature")
        assert len(result) == 2
        assert all("feature" in t["tags"] for t in result)

    def test_filter_combined(self, sample_tasks):
        result = filter_tasks(sample_tasks, status="in_progress", priority="high")
        assert len(result) == 1
        assert result[0]["id"] == "4"

    def test_filter_no_match(self, sample_tasks):
        result = filter_tasks(sample_tasks, status="cancelled")
        assert len(result) == 0

    def test_filter_empty_list(self):
        result = filter_tasks([], status="pending")
        assert result == []


# ---------------------------------------------------------------------------
# format_task_checklist
# ---------------------------------------------------------------------------


class TestFormatTaskChecklist:
    """Tests for format_task_checklist()."""

    def test_empty_list(self):
        assert format_task_checklist([]) == ""

    def test_single_pending(self):
        result = format_task_checklist([
            {"description": "My task", "status": "pending"},
        ])
        assert result == "[ ] My task"

    def test_single_completed(self):
        result = format_task_checklist([
            {"description": "Done task", "status": "completed"},
        ])
        assert result == "[✓] Done task"

    def test_single_in_progress(self):
        result = format_task_checklist([
            {"description": "Active task", "status": "in_progress"},
        ])
        assert result == "[▸] Active task"

    def test_single_cancelled(self):
        result = format_task_checklist([
            {"description": "Dropped", "status": "cancelled"},
        ])
        assert result == "[-] Dropped"

    def test_sorted_by_status_priority(self):
        """in_progress first, then pending, cancelled, completed."""
        tasks = [
            {"description": "Done", "status": "completed"},
            {"description": "Waiting", "status": "pending"},
            {"description": "Active", "status": "in_progress"},
            {"description": "Dropped", "status": "cancelled"},
        ]
        result = format_task_checklist(tasks)
        lines = result.strip().splitlines()
        assert lines[0] == "[▸] Active"
        assert lines[1] == "[ ] Waiting"
        assert lines[2] == "[-] Dropped"
        assert lines[3] == "[✓] Done"

    def test_defaults_to_pending_icon_for_unknown_status(self):
        result = format_task_checklist([
            {"description": "Mystery", "status": "unknown"},
        ])
        assert result == "[ ] Mystery"


# ---------------------------------------------------------------------------
# task_progress_data
# ---------------------------------------------------------------------------


class TestTaskProgressData:
    """Tests for task_progress_data()."""

    def test_empty_list_returns_none(self):
        assert task_progress_data([]) is None

    def test_single_pending_task(self):
        tasks = [{"id": "1", "description": "Task 1", "status": "pending"}]
        result = task_progress_data(tasks)
        assert result is not None
        assert result["progress"]["total"] == 1
        assert result["progress"]["pending"] == 1
        assert result["current_task_id"] is None
        assert result["all_done"] is False

    def test_in_progress_task_reported(self):
        tasks = [
            {"id": "1", "description": "Active", "status": "in_progress"},
            {"id": "2", "description": "Pending", "status": "pending"},
        ]
        result = task_progress_data(tasks)
        assert result["current_task_id"] == "1"
        assert result["current_task_description"] == "Active"

    def test_first_in_progress_wins(self):
        tasks = [
            {"id": "1", "description": "First active", "status": "in_progress"},
            {"id": "2", "description": "Second active", "status": "in_progress"},
        ]
        result = task_progress_data(tasks)
        assert result["current_task_id"] == "1"

    def test_all_done_when_all_completed(self):
        tasks = [
            {"id": "1", "description": "A", "status": "completed"},
            {"id": "2", "description": "B", "status": "completed"},
        ]
        result = task_progress_data(tasks)
        assert result["all_done"] is True

    def test_all_done_when_mix_of_completed_cancelled(self):
        tasks = [
            {"id": "1", "description": "A", "status": "completed"},
            {"id": "2", "description": "B", "status": "cancelled"},
        ]
        result = task_progress_data(tasks)
        assert result["all_done"] is True

    def test_not_done_when_pending_exists(self):
        tasks = [
            {"id": "1", "description": "A", "status": "completed"},
            {"id": "2", "description": "B", "status": "pending"},
        ]
        result = task_progress_data(tasks)
        assert result["all_done"] is False

    def test_display_contains_checklist(self):
        tasks = [
            {"id": "1", "description": "Active", "status": "in_progress"},
            {"id": "2", "description": "Waiting", "status": "pending"},
        ]
        result = task_progress_data(tasks)
        assert "[▸] Active" in result["display"]
        assert "[ ] Waiting" in result["display"]

    def test_progress_counts(self):
        tasks = [
            {"id": "1", "description": "A", "status": "pending"},
            {"id": "2", "description": "B", "status": "in_progress"},
            {"id": "3", "description": "C", "status": "completed"},
            {"id": "4", "description": "D", "status": "cancelled"},
        ]
        result = task_progress_data(tasks)
        p = result["progress"]
        assert p["total"] == 4
        assert p["pending"] == 1
        assert p["in_progress"] == 1
        assert p["completed"] == 1
        assert p["cancelled"] == 1
