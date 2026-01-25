"""Tests for planning module."""

import pytest


class TestTaskGraph:
    def test_add_task(self):
        from agentic_cli.planning import TaskGraph
        graph = TaskGraph()
        task_id = graph.add_task("Analyze documents")
        assert task_id is not None
        task = graph.get_task(task_id)
        assert task.description == "Analyze documents"

    def test_add_task_with_dependencies(self):
        from agentic_cli.planning import TaskGraph, TaskStatus
        graph = TaskGraph()
        task1 = graph.add_task("Gather data")
        task2 = graph.add_task("Process data", dependencies=[task1])
        t2 = graph.get_task(task2)
        assert task1 in t2.dependencies

    def test_add_subtask(self):
        from agentic_cli.planning import TaskGraph
        graph = TaskGraph()
        parent_id = graph.add_task("Main task")
        child_id = graph.add_task("Subtask", parent=parent_id)
        parent = graph.get_task(parent_id)
        assert child_id in parent.subtasks

    def test_update_task_status(self):
        from agentic_cli.planning import TaskGraph, TaskStatus
        graph = TaskGraph()
        task_id = graph.add_task("Test task")
        graph.update_status(task_id, TaskStatus.IN_PROGRESS)
        task = graph.get_task(task_id)
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.started_at is not None

    def test_complete_task_with_result(self):
        from agentic_cli.planning import TaskGraph, TaskStatus
        graph = TaskGraph()
        task_id = graph.add_task("Calculate")
        graph.update_status(task_id, TaskStatus.COMPLETED, result={"answer": 42})
        task = graph.get_task(task_id)
        assert task.status == TaskStatus.COMPLETED
        assert task.result == {"answer": 42}
        assert task.completed_at is not None

    def test_get_ready_tasks(self):
        from agentic_cli.planning import TaskGraph, TaskStatus
        graph = TaskGraph()
        task1 = graph.add_task("First")
        task2 = graph.add_task("Second", dependencies=[task1])
        task3 = graph.add_task("Third")
        ready = graph.get_ready_tasks()
        ready_ids = [t.id for t in ready]
        assert task1 in ready_ids
        assert task3 in ready_ids
        assert task2 not in ready_ids
        # Complete task1
        graph.update_status(task1, TaskStatus.COMPLETED)
        ready = graph.get_ready_tasks()
        ready_ids = [t.id for t in ready]
        assert task2 in ready_ids

    def test_get_progress(self):
        from agentic_cli.planning import TaskGraph, TaskStatus
        graph = TaskGraph()
        graph.add_task("Task 1")
        task2 = graph.add_task("Task 2")
        task3 = graph.add_task("Task 3")
        graph.update_status(task2, TaskStatus.COMPLETED)
        graph.update_status(task3, TaskStatus.IN_PROGRESS)
        progress = graph.get_progress()
        assert progress["total"] == 3
        assert progress["pending"] == 1
        assert progress["in_progress"] == 1
        assert progress["completed"] == 1

    def test_to_display(self):
        from agentic_cli.planning import TaskGraph, TaskStatus
        graph = TaskGraph()
        task1 = graph.add_task("First task")
        graph.add_task("Second task", dependencies=[task1])
        graph.update_status(task1, TaskStatus.COMPLETED)
        display = graph.to_display()
        assert "First task" in display
        assert "Second task" in display
        assert "✓" in display


class TestTaskStatus:
    def test_all_statuses_exist(self):
        from agentic_cli.planning import TaskStatus
        assert TaskStatus.PENDING
        assert TaskStatus.IN_PROGRESS
        assert TaskStatus.COMPLETED
        assert TaskStatus.BLOCKED
        assert TaskStatus.FAILED
        assert TaskStatus.SKIPPED


class TestTask:
    def test_task_creation(self):
        from agentic_cli.planning import Task, TaskStatus
        from datetime import datetime
        task = Task(
            id="task_1",
            description="Test task",
            created_at=datetime.now()
        )
        assert task.id == "task_1"
        assert task.description == "Test task"
        assert task.status == TaskStatus.PENDING
        assert task.dependencies == []
        assert task.subtasks == []
        assert task.parent is None
        assert task.result is None
        assert task.error is None
        assert task.started_at is None
        assert task.completed_at is None
        assert task.metadata == {}

    def test_task_to_dict(self):
        from agentic_cli.planning import Task, TaskStatus
        from datetime import datetime
        created = datetime.now()
        task = Task(
            id="task_1",
            description="Test task",
            status=TaskStatus.COMPLETED,
            dependencies=["dep_1"],
            subtasks=["sub_1"],
            parent="parent_1",
            result={"key": "value"},
            created_at=created,
            metadata={"priority": "high"}
        )
        data = task.to_dict()
        assert data["id"] == "task_1"
        assert data["description"] == "Test task"
        assert data["status"] == "completed"
        assert data["dependencies"] == ["dep_1"]
        assert data["subtasks"] == ["sub_1"]
        assert data["parent"] == "parent_1"
        assert data["result"] == {"key": "value"}
        assert data["metadata"] == {"priority": "high"}
        assert "created_at" in data

    def test_task_from_dict(self):
        from agentic_cli.planning import Task, TaskStatus
        from datetime import datetime
        data = {
            "id": "task_1",
            "description": "Test task",
            "status": "in_progress",
            "dependencies": ["dep_1"],
            "subtasks": ["sub_1"],
            "parent": "parent_1",
            "result": None,
            "error": None,
            "created_at": datetime.now().isoformat(),
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "metadata": {"priority": "high"}
        }
        task = Task.from_dict(data)
        assert task.id == "task_1"
        assert task.description == "Test task"
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.dependencies == ["dep_1"]
        assert task.subtasks == ["sub_1"]
        assert task.parent == "parent_1"
        assert task.metadata == {"priority": "high"}


class TestTaskGraphSerialization:
    def test_to_dict_empty(self):
        from agentic_cli.planning import TaskGraph
        graph = TaskGraph()
        data = graph.to_dict()
        assert "tasks" in data
        assert data["tasks"] == {}

    def test_to_dict_with_tasks(self):
        from agentic_cli.planning import TaskGraph, TaskStatus
        graph = TaskGraph()
        task1 = graph.add_task("Task 1")
        task2 = graph.add_task("Task 2", dependencies=[task1])
        graph.update_status(task1, TaskStatus.COMPLETED)
        data = graph.to_dict()
        assert task1 in data["tasks"]
        assert task2 in data["tasks"]
        assert data["tasks"][task1]["status"] == "completed"

    def test_from_dict(self):
        from agentic_cli.planning import TaskGraph, TaskStatus
        graph = TaskGraph()
        task1 = graph.add_task("Task 1")
        task2 = graph.add_task("Task 2", dependencies=[task1])
        graph.update_status(task1, TaskStatus.COMPLETED, result=42)
        data = graph.to_dict()
        restored = TaskGraph.from_dict(data)
        t1 = restored.get_task(task1)
        t2 = restored.get_task(task2)
        assert t1.description == "Task 1"
        assert t1.status == TaskStatus.COMPLETED
        assert t1.result == 42
        assert t2.description == "Task 2"
        assert task1 in t2.dependencies


class TestTaskGraphRevise:
    def test_revise_add_task(self):
        from agentic_cli.planning import TaskGraph
        graph = TaskGraph()
        changes = [
            {"action": "add", "description": "New task", "metadata": {"priority": "high"}}
        ]
        graph.revise(changes)
        progress = graph.get_progress()
        assert progress["total"] == 1
        tasks = list(graph._tasks.values())
        assert tasks[0].description == "New task"
        assert tasks[0].metadata == {"priority": "high"}

    def test_revise_remove_task(self):
        from agentic_cli.planning import TaskGraph
        graph = TaskGraph()
        task_id = graph.add_task("Task to remove")
        changes = [
            {"action": "remove", "task_id": task_id}
        ]
        graph.revise(changes)
        assert graph.get_task(task_id) is None

    def test_revise_update_task(self):
        from agentic_cli.planning import TaskGraph, TaskStatus
        graph = TaskGraph()
        task_id = graph.add_task("Original description")
        changes = [
            {"action": "update", "task_id": task_id, "description": "Updated description", "status": "in_progress"}
        ]
        graph.revise(changes)
        task = graph.get_task(task_id)
        assert task.description == "Updated description"
        assert task.status == TaskStatus.IN_PROGRESS

    def test_revise_multiple_changes(self):
        from agentic_cli.planning import TaskGraph
        graph = TaskGraph()
        task1 = graph.add_task("Task 1")
        changes = [
            {"action": "add", "description": "Task 2"},
            {"action": "update", "task_id": task1, "description": "Updated Task 1"},
        ]
        graph.revise(changes)
        progress = graph.get_progress()
        assert progress["total"] == 2
        t1 = graph.get_task(task1)
        assert t1.description == "Updated Task 1"


class TestTaskGraphDisplay:
    def test_display_with_all_statuses(self):
        from agentic_cli.planning import TaskGraph, TaskStatus
        graph = TaskGraph()
        t1 = graph.add_task("Pending task")
        t2 = graph.add_task("In progress task")
        t3 = graph.add_task("Completed task")
        t4 = graph.add_task("Failed task")
        t5 = graph.add_task("Blocked task", dependencies=[t1])
        t6 = graph.add_task("Skipped task")

        graph.update_status(t2, TaskStatus.IN_PROGRESS)
        graph.update_status(t3, TaskStatus.COMPLETED)
        graph.update_status(t4, TaskStatus.FAILED, error="Something went wrong")
        graph.update_status(t5, TaskStatus.BLOCKED)
        graph.update_status(t6, TaskStatus.SKIPPED)

        display = graph.to_display()

        # Check status icons are present
        assert "☐" in display  # pending
        assert "◐" in display  # in_progress
        assert "✓" in display  # completed
        assert "✗" in display  # failed
        assert "⊘" in display  # blocked
        assert "⊝" in display  # skipped

    def test_display_with_subtasks(self):
        from agentic_cli.planning import TaskGraph
        graph = TaskGraph()
        parent = graph.add_task("Parent task")
        child1 = graph.add_task("Child 1", parent=parent)
        child2 = graph.add_task("Child 2", parent=parent)

        display = graph.to_display()

        assert "Parent task" in display
        assert "Child 1" in display
        assert "Child 2" in display


class TestTaskGraphMetadata:
    def test_add_task_with_metadata(self):
        from agentic_cli.planning import TaskGraph
        graph = TaskGraph()
        task_id = graph.add_task(
            "Task with metadata",
            priority="high",
            assigned_to="agent_1",
            custom_field=42
        )
        task = graph.get_task(task_id)
        assert task.metadata["priority"] == "high"
        assert task.metadata["assigned_to"] == "agent_1"
        assert task.metadata["custom_field"] == 42


class TestTaskGraphEdgeCases:
    def test_get_nonexistent_task(self):
        from agentic_cli.planning import TaskGraph
        graph = TaskGraph()
        assert graph.get_task("nonexistent") is None

    def test_update_nonexistent_task_raises(self):
        from agentic_cli.planning import TaskGraph, TaskStatus
        graph = TaskGraph()
        with pytest.raises(ValueError):
            graph.update_status("nonexistent", TaskStatus.COMPLETED)

    def test_empty_graph_get_ready_tasks(self):
        from agentic_cli.planning import TaskGraph
        graph = TaskGraph()
        ready = graph.get_ready_tasks()
        assert ready == []

    def test_empty_graph_progress(self):
        from agentic_cli.planning import TaskGraph
        graph = TaskGraph()
        progress = graph.get_progress()
        assert progress["total"] == 0
        assert progress["pending"] == 0

    def test_circular_dependency_handling(self):
        from agentic_cli.planning import TaskGraph
        graph = TaskGraph()
        task1 = graph.add_task("Task 1")
        task2 = graph.add_task("Task 2", dependencies=[task1])
        # Task 1 depends on task 2 which depends on task 1
        # This should still work, just means both are blocked
        task1_obj = graph.get_task(task1)
        task1_obj.dependencies.append(task2)
        ready = graph.get_ready_tasks()
        # Neither should be ready since they depend on each other
        assert task1 not in [t.id for t in ready]
        assert task2 not in [t.id for t in ready]
