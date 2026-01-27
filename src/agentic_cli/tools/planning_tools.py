"""Planning tools for agentic workflows.

These tools provide task graph management for multi-step planning.
The TaskGraph is auto-created by the workflow manager when these tools are used.

Example:
    from agentic_cli.tools import planning_tools

    # In agent config
    AgentConfig(
        tools=[planning_tools.create_task, planning_tools.get_next_tasks, ...],
    )
"""

from typing import Any

from agentic_cli.tools import requires
from agentic_cli.workflow.context import get_context_task_graph


@requires("task_graph")
def create_task(
    description: str,
    dependencies: list[str] | None = None,
    parent: str | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    """Create a new task in the plan.

    Args:
        description: Human-readable description of what needs to be done.
        dependencies: List of task IDs this task depends on.
        parent: Optional parent task ID for subtask relationships.
        **metadata: Additional key-value pairs to store with the task.

    Returns:
        A dict with the created task ID.
    """
    graph = get_context_task_graph()
    if graph is None:
        return {"success": False, "error": "Task graph not available"}

    task_id = graph.add_task(
        description=description,
        dependencies=dependencies,
        parent=parent,
        **metadata,
    )

    return {
        "success": True,
        "task_id": task_id,
        "description": description,
        "message": f"Created task: {description}",
    }


@requires("task_graph")
def update_task_status(
    task_id: str,
    status: str,
    result: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Update task status.

    Args:
        task_id: The task ID to update.
        status: New status ("pending", "in_progress", "completed", "blocked", "failed", "skipped").
        result: Optional result or notes about completion.
        error: Optional error message (for failed status).

    Returns:
        A dict with the updated task status.
    """
    graph = get_context_task_graph()
    if graph is None:
        return {"success": False, "error": "Task graph not available"}

    from agentic_cli.planning import TaskStatus

    try:
        task_status = TaskStatus(status)
    except ValueError:
        valid = [s.value for s in TaskStatus]
        return {"success": False, "error": f"Invalid status: {status}. Valid: {', '.join(valid)}"}

    try:
        graph.update_status(task_id, task_status, result=result, error=error)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    return {
        "success": True,
        "task_id": task_id,
        "status": status,
        "message": f"Task {task_id} updated to {status}",
    }


@requires("task_graph")
def get_next_tasks(limit: int = 3) -> dict[str, Any]:
    """Get the next tasks ready to work on (no unmet dependencies).

    A task is ready if its status is PENDING and all dependencies are COMPLETED.

    Args:
        limit: Maximum number of tasks to return.

    Returns:
        A dict with the list of ready tasks.
    """
    graph = get_context_task_graph()
    if graph is None:
        return {"success": False, "error": "Task graph not available"}

    ready_tasks = graph.get_ready_tasks()[:limit]

    if not ready_tasks:
        progress = graph.get_progress()
        if progress["completed"] == progress["total"] and progress["total"] > 0:
            return {
                "success": True,
                "tasks": [],
                "message": "All tasks completed!",
            }
        return {
            "success": True,
            "tasks": [],
            "message": "No tasks ready (dependencies not met or no tasks exist)",
        }

    tasks = [
        {
            "id": task.id,
            "description": task.description,
            "status": task.status.value,
            "dependencies": task.dependencies,
        }
        for task in ready_tasks
    ]

    return {
        "success": True,
        "tasks": tasks,
        "count": len(tasks),
    }


@requires("task_graph")
def get_task(task_id: str) -> dict[str, Any]:
    """Get details of a specific task.

    Args:
        task_id: The task ID to look up.

    Returns:
        A dict with the task details.
    """
    graph = get_context_task_graph()
    if graph is None:
        return {"success": False, "error": "Task graph not available"}

    task = graph.get_task(task_id)
    if task is None:
        return {"success": False, "error": f"Task '{task_id}' not found"}

    return {
        "success": True,
        "task": task.to_dict(),
    }


@requires("task_graph")
def get_plan_summary() -> dict[str, Any]:
    """Get a summary of the current plan.

    Returns:
        A dict with progress statistics and formatted display.
    """
    graph = get_context_task_graph()
    if graph is None:
        return {"success": False, "error": "Task graph not available"}

    progress = graph.get_progress()
    display = graph.to_display()

    return {
        "success": True,
        "progress": progress,
        "display": display,
    }


@requires("task_graph")
def create_plan(
    topic: str,
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a complete task plan for a project.

    Organizes work into a structured task graph with dependencies.
    This is a convenience function that creates multiple tasks at once.

    Args:
        topic: The project/research topic.
        tasks: List of task definitions, each with:
            - description: What needs to be done
            - depends_on: Optional list of task indices (0-based) this depends on

    Returns:
        A dict with the created task IDs and plan summary.

    Example:
        >>> create_plan("Python history", [
        ...     {"description": "Gather initial sources"},
        ...     {"description": "Review key events", "depends_on": [0]},
        ...     {"description": "Write summary", "depends_on": [1]},
        ... ])
    """
    graph = get_context_task_graph()
    if graph is None:
        return {"success": False, "error": "Task graph not available"}

    # Clear existing tasks for a fresh plan
    graph._tasks.clear()

    # Create tasks and track their IDs
    task_ids: list[str] = []

    for i, task_def in enumerate(tasks):
        description = task_def.get("description", f"Task {i + 1}")
        depends_on = task_def.get("depends_on", [])

        # Convert dependency indices to task IDs
        dependencies = [task_ids[idx] for idx in depends_on if idx < len(task_ids)]

        task_id = graph.add_task(
            description=description,
            dependencies=dependencies,
            topic=topic,
        )
        task_ids.append(task_id)

    return {
        "success": True,
        "topic": topic,
        "task_count": len(task_ids),
        "task_ids": task_ids,
        "display": graph.to_display(),
        "message": f"Created plan with {len(task_ids)} tasks",
    }


@requires("task_graph")
def revise_plan(changes: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply changes to the task plan.

    Each change is a dictionary with an "action" key:
    - "add": Add a new task. Keys: description, dependencies, parent, metadata
    - "remove": Remove a task. Keys: task_id
    - "update": Update a task. Keys: task_id, and any fields to update

    Args:
        changes: List of change dictionaries to apply.

    Returns:
        A dict with the updated plan summary.
    """
    graph = get_context_task_graph()
    if graph is None:
        return {"success": False, "error": "Task graph not available"}

    graph.revise(changes)

    return {
        "success": True,
        "changes_applied": len(changes),
        "display": graph.to_display(),
        "progress": graph.get_progress(),
    }
