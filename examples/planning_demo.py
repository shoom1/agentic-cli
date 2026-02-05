#!/usr/bin/env python
"""Standalone demo for the TaskGraph planning system.

This demo tests the task planning system:
1. Task creation with descriptions
2. Dependency management between tasks
3. Status tracking (pending, in_progress, completed, failed)
4. Subtask hierarchy
5. Progress monitoring and display

Usage:
    conda run -n agenticcli python examples/planning_demo.py
"""

import sys
from datetime import datetime

from agentic_cli.planning.task_graph import TaskGraph, TaskStatus, STATUS_ICONS


# =============================================================================
# Demo Functions
# =============================================================================


def demo_basic_task_creation():
    """Demo basic task creation and retrieval."""
    print("\n" + "=" * 60)
    print("Basic Task Creation Demo")
    print("=" * 60)

    graph = TaskGraph()

    # Create some tasks
    task1_id = graph.add_task("Research topic")
    task2_id = graph.add_task("Write outline")
    task3_id = graph.add_task("Draft content")

    print(f"  Created 3 tasks:")
    for task_id in [task1_id, task2_id, task3_id]:
        task = graph.get_task(task_id)
        icon = STATUS_ICONS[task.status]
        print(f"    {icon} [{task_id}] {task.description}")

    print()
    print(f"  Total tasks: {len(graph.all_tasks())}")
    print()


def demo_dependencies():
    """Demo task dependencies and ready task detection."""
    print("\n" + "=" * 60)
    print("Task Dependencies Demo")
    print("=" * 60)

    graph = TaskGraph()

    # Create tasks with dependencies
    research_id = graph.add_task("Research topic")
    outline_id = graph.add_task("Write outline", dependencies=[research_id])
    draft_id = graph.add_task("Draft content", dependencies=[outline_id])
    review_id = graph.add_task("Review draft", dependencies=[draft_id])

    print("  Task dependency chain:")
    print(f"    Research -> Outline -> Draft -> Review")
    print()

    # Check initial ready tasks
    ready = graph.get_ready_tasks()
    print(f"  Initial ready tasks: {[t.description for t in ready]}")
    print()

    # Complete research task
    graph.update_status(research_id, TaskStatus.COMPLETED)
    ready = graph.get_ready_tasks()
    print(f"  After completing 'Research': {[t.description for t in ready]}")

    # Complete outline task
    graph.update_status(outline_id, TaskStatus.COMPLETED)
    ready = graph.get_ready_tasks()
    print(f"  After completing 'Outline': {[t.description for t in ready]}")
    print()


def demo_status_tracking():
    """Demo status transitions and progress tracking."""
    print("\n" + "=" * 60)
    print("Status Tracking Demo")
    print("=" * 60)

    graph = TaskGraph()

    # Create tasks
    ids = [
        graph.add_task("Task A"),
        graph.add_task("Task B"),
        graph.add_task("Task C"),
        graph.add_task("Task D"),
    ]

    print("  Initial status:")
    print_progress(graph)

    # Simulate workflow
    print("\n  Simulating workflow...")

    # Start Task A
    graph.update_status(ids[0], TaskStatus.IN_PROGRESS)
    print(f"    Started Task A")

    # Complete Task A
    graph.update_status(ids[0], TaskStatus.COMPLETED, result="Done!")
    print(f"    Completed Task A with result")

    # Start and fail Task B
    graph.update_status(ids[1], TaskStatus.IN_PROGRESS)
    graph.update_status(ids[1], TaskStatus.FAILED, error="Something went wrong")
    print(f"    Task B failed with error")

    # Skip Task C
    graph.update_status(ids[2], TaskStatus.SKIPPED)
    print(f"    Skipped Task C")

    print("\n  Final status:")
    print_progress(graph)

    # Verify timestamps
    task_a = graph.get_task(ids[0])
    print(f"\n  Task A timestamps:")
    print(f"    Created:   {task_a.created_at.strftime('%H:%M:%S')}")
    print(f"    Started:   {task_a.started_at.strftime('%H:%M:%S') if task_a.started_at else 'N/A'}")
    print(f"    Completed: {task_a.completed_at.strftime('%H:%M:%S') if task_a.completed_at else 'N/A'}")
    print()


def print_progress(graph: TaskGraph):
    """Print progress statistics."""
    progress = graph.get_progress()
    for status in ["pending", "in_progress", "completed", "failed", "skipped"]:
        count = progress.get(status, 0)
        if count > 0:
            print(f"    {status}: {count}")


def demo_subtasks():
    """Demo parent-child task hierarchy."""
    print("\n" + "=" * 60)
    print("Subtask Hierarchy Demo")
    print("=" * 60)

    graph = TaskGraph()

    # Create parent task
    parent_id = graph.add_task("Build feature")

    # Create subtasks
    sub1_id = graph.add_task("Design API", parent=parent_id)
    sub2_id = graph.add_task("Implement logic", parent=parent_id)
    sub3_id = graph.add_task("Write tests", parent=parent_id)

    print("  Created hierarchy:")
    print(graph.to_display())
    print()

    # Update subtask statuses
    graph.update_status(sub1_id, TaskStatus.COMPLETED)
    graph.update_status(sub2_id, TaskStatus.IN_PROGRESS)

    print("  After updates:")
    print(graph.to_display())
    print()


def demo_display_formats():
    """Demo different display format options."""
    print("\n" + "=" * 60)
    print("Display Formats Demo")
    print("=" * 60)

    graph = TaskGraph()

    # Create a more complex graph
    research = graph.add_task("Research machine learning algorithms")
    data_prep = graph.add_task("Prepare training dataset", dependencies=[research])
    model = graph.add_task("Train model", dependencies=[data_prep])
    eval_task = graph.add_task("Evaluate model performance", dependencies=[model])
    deploy = graph.add_task("Deploy to production", dependencies=[eval_task])
    docs = graph.add_task("Write documentation")
    tests = graph.add_task("Write integration tests", dependencies=[deploy])

    # Set various statuses
    graph.update_status(research, TaskStatus.COMPLETED)
    graph.update_status(data_prep, TaskStatus.COMPLETED)
    graph.update_status(model, TaskStatus.IN_PROGRESS)
    graph.update_status(docs, TaskStatus.IN_PROGRESS)

    print("  Full display format:")
    print("-" * 40)
    print(graph.to_display())
    print()

    print("  Compact display format (for status line):")
    print("-" * 40)
    print(graph.to_compact_display(max_tasks=4))
    print()


def demo_serialization():
    """Demo saving and loading task graphs."""
    print("\n" + "=" * 60)
    print("Serialization Demo")
    print("=" * 60)

    # Create original graph
    graph = TaskGraph()
    t1 = graph.add_task("First task", priority="high")
    t2 = graph.add_task("Second task", dependencies=[t1])
    graph.update_status(t1, TaskStatus.COMPLETED)

    print("  Original graph:")
    print(graph.to_display())

    # Serialize to dict
    data = graph.to_dict()
    print(f"\n  Serialized to dict with {len(data['tasks'])} tasks")

    # Restore from dict
    restored = TaskGraph.from_dict(data)
    print("\n  Restored graph:")
    print(restored.to_display())

    # Verify metadata preserved
    task = restored.get_task(t1)
    print(f"\n  Metadata preserved: priority={task.metadata.get('priority')}")
    print()


def demo_revise():
    """Demo graph revision (add/remove/update tasks)."""
    print("\n" + "=" * 60)
    print("Graph Revision Demo")
    print("=" * 60)

    graph = TaskGraph()
    t1 = graph.add_task("Original task 1")
    t2 = graph.add_task("Original task 2")

    print("  Initial graph:")
    print(graph.to_display())

    # Apply revisions
    changes = [
        {"action": "add", "description": "New task added"},
        {"action": "update", "task_id": t1, "description": "Updated task 1"},
        {"action": "update", "task_id": t2, "status": "completed"},
    ]

    graph.revise(changes)

    print("\n  After revisions:")
    print(graph.to_display())
    print()


def main():
    """Run all demos."""
    print("\n" + "#" * 60)
    print("#  TaskGraph Planning System Demo")
    print("#" * 60)

    print("\nStatus icons:")
    for status, icon in STATUS_ICONS.items():
        print(f"  {icon} = {status.value}")

    # Run demos
    demo_basic_task_creation()
    demo_dependencies()
    demo_status_tracking()
    demo_subtasks()
    demo_display_formats()
    demo_serialization()
    demo_revise()

    print("\n" + "#" * 60)
    print("#  Demo Complete!")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
