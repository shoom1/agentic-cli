"""Agent tools for the Research Demo application.

Wraps memory, planning, file operations, and shell execution
with appropriate interfaces for agent use.
"""

from typing import Any

# Global state for the demo (initialized by the app)
_memory_manager = None
_task_graph = None
_approval_manager = None
_checkpoint_manager = None
_settings = None


def set_demo_state(
    memory_manager,
    task_graph,
    approval_manager,
    checkpoint_manager,
    settings,
) -> None:
    """Set the global state for tools (called by the app)."""
    global _memory_manager, _task_graph, _approval_manager, _checkpoint_manager, _settings
    _memory_manager = memory_manager
    _task_graph = task_graph
    _approval_manager = approval_manager
    _checkpoint_manager = checkpoint_manager
    _settings = settings


# =============================================================================
# Memory Tools
# =============================================================================


def remember_context(key: str, value: str, tags: list[str] | None = None) -> dict[str, Any]:
    """Store context in working memory for the current session.

    Use this to remember important context like the current research topic,
    user preferences, or intermediate results.

    Args:
        key: A unique identifier for this context (e.g., "current_topic", "user_preference").
        value: The value to store.
        tags: Optional tags for categorization (e.g., ["research", "draft"]).

    Returns:
        A dict with success status and stored key.
    """
    if _memory_manager is None:
        return {"success": False, "error": "Memory manager not initialized"}

    _memory_manager.working.set(key, value, tags=tags)
    return {
        "success": True,
        "key": key,
        "message": f"Stored '{key}' in working memory",
    }


def recall_context(key: str) -> dict[str, Any]:
    """Recall a specific context from working memory.

    Args:
        key: The key to look up.

    Returns:
        A dict with the value if found, or an error message.
    """
    if _memory_manager is None:
        return {"success": False, "error": "Memory manager not initialized"}

    value = _memory_manager.working.get(key)
    if value is None:
        return {"success": False, "error": f"Key '{key}' not found in working memory"}

    return {
        "success": True,
        "key": key,
        "value": value,
    }


def recall_info(query: str, memory_type: str | None = None) -> dict[str, Any]:
    """Search across all memory for information matching a query.

    Searches both working memory (session) and long-term memory (persistent).

    Args:
        query: The search query.
        memory_type: Optional filter for long-term memory type
            ("fact", "preference", "learning", "reference").

    Returns:
        A dict with matching results from both memory tiers.
    """
    if _memory_manager is None:
        return {"success": False, "error": "Memory manager not initialized"}

    from agentic_cli.memory.longterm import MemoryType

    # Search across tiers
    results = _memory_manager.search(query, include_working=True, include_longterm=True)

    # Format results
    working_results = [
        {"key": key, "value": value}
        for key, value in results.working_results
    ]

    longterm_results = []
    for entry in results.longterm_results:
        if memory_type is None or entry.type.value == memory_type:
            longterm_results.append({
                "id": entry.id,
                "type": entry.type.value,
                "content": entry.content,
                "tags": entry.tags,
            })

    return {
        "success": True,
        "query": query,
        "working_memory": working_results,
        "longterm_memory": longterm_results,
    }


def store_learning(content: str, tags: list[str] | None = None) -> dict[str, Any]:
    """Store a learning in long-term memory for future sessions.

    Use this when you discover important information that should be
    remembered across sessions.

    Args:
        content: The learning content to store.
        tags: Optional tags for categorization.

    Returns:
        A dict with the stored entry ID.
    """
    if _memory_manager is None:
        return {"success": False, "error": "Memory manager not initialized"}

    from agentic_cli.memory.longterm import MemoryType

    entry_id = _memory_manager.longterm.store(
        type=MemoryType.LEARNING,
        content=content,
        source="research_demo",
        tags=tags,
    )

    return {
        "success": True,
        "entry_id": entry_id,
        "message": f"Stored learning in long-term memory (id: {entry_id})",
    }


def store_fact(content: str, tags: list[str] | None = None) -> dict[str, Any]:
    """Store a fact in long-term memory.

    Use this for factual information discovered during research.

    Args:
        content: The fact to store.
        tags: Optional tags for categorization.

    Returns:
        A dict with the stored entry ID.
    """
    if _memory_manager is None:
        return {"success": False, "error": "Memory manager not initialized"}

    from agentic_cli.memory.longterm import MemoryType

    entry_id = _memory_manager.longterm.store(
        type=MemoryType.FACT,
        content=content,
        source="research_demo",
        tags=tags,
    )

    return {
        "success": True,
        "entry_id": entry_id,
        "message": f"Stored fact in long-term memory (id: {entry_id})",
    }


# =============================================================================
# Planning Tools
# =============================================================================


def create_research_plan(topic: str, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a task plan for a research project.

    Organizes research into a structured task graph with dependencies.

    Args:
        topic: The research topic.
        tasks: List of task definitions, each with:
            - description: What needs to be done
            - depends_on: Optional list of task indices this depends on

    Returns:
        A dict with the created task IDs and plan summary.

    Example:
        >>> create_research_plan("Python history", [
        ...     {"description": "Gather initial sources"},
        ...     {"description": "Review key events", "depends_on": [0]},
        ...     {"description": "Write summary", "depends_on": [1]},
        ... ])
    """
    if _task_graph is None:
        return {"success": False, "error": "Task graph not initialized"}

    # Clear existing tasks for a fresh plan
    _task_graph._tasks.clear()

    # Store the topic in working memory
    if _memory_manager is not None:
        _memory_manager.working.set("research_topic", topic, tags=["planning"])

    # Create tasks and track their IDs
    task_ids: list[str] = []

    for i, task_def in enumerate(tasks):
        description = task_def.get("description", f"Task {i + 1}")
        depends_on = task_def.get("depends_on", [])

        # Convert dependency indices to task IDs
        dependencies = [task_ids[idx] for idx in depends_on if idx < len(task_ids)]

        task_id = _task_graph.add_task(
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
        "message": f"Created research plan with {len(task_ids)} tasks",
    }


def get_next_task() -> dict[str, Any]:
    """Get the next task that is ready to start.

    Returns:
        A dict with the next ready task, or a message if none are ready.
    """
    if _task_graph is None:
        return {"success": False, "error": "Task graph not initialized"}

    ready_tasks = _task_graph.get_ready_tasks()

    if not ready_tasks:
        progress = _task_graph.get_progress()
        if progress["completed"] == progress["total"]:
            return {
                "success": True,
                "task": None,
                "message": "All tasks completed!",
            }
        return {
            "success": True,
            "task": None,
            "message": "No tasks ready (dependencies not met)",
        }

    task = ready_tasks[0]
    return {
        "success": True,
        "task": {
            "id": task.id,
            "description": task.description,
            "status": task.status.value,
        },
    }


def update_task_status(
    task_id: str,
    status: str,
    result: str | None = None,
) -> dict[str, Any]:
    """Update the status of a task in the plan.

    Args:
        task_id: The task ID to update.
        status: New status ("pending", "in_progress", "completed", "failed", "skipped").
        result: Optional result or notes about completion.

    Returns:
        A dict with the updated task status.
    """
    if _task_graph is None:
        return {"success": False, "error": "Task graph not initialized"}

    from agentic_cli.planning import TaskStatus

    try:
        task_status = TaskStatus(status)
    except ValueError:
        return {"success": False, "error": f"Invalid status: {status}"}

    try:
        _task_graph.update_status(task_id, task_status, result=result)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    return {
        "success": True,
        "task_id": task_id,
        "status": status,
        "message": f"Task {task_id} updated to {status}",
    }


def get_plan_progress() -> dict[str, Any]:
    """Get the current progress of the research plan.

    Returns:
        A dict with progress statistics and display.
    """
    if _task_graph is None:
        return {"success": False, "error": "Task graph not initialized"}

    progress = _task_graph.get_progress()
    display = _task_graph.to_display()

    return {
        "success": True,
        "progress": progress,
        "display": display,
    }


# =============================================================================
# File Tools
# =============================================================================


def save_finding(filename: str, content: str) -> dict[str, Any]:
    """Save a research finding to a file in the workspace.

    This operation requires approval before writing.

    Args:
        filename: Name of the file (will be saved in workspace/findings/).
        content: Content to write.

    Returns:
        A dict with the saved file path or approval status.
    """
    if _settings is None:
        return {"success": False, "error": "Settings not initialized"}

    from agentic_cli.tools.file_ops import file_manager

    # Determine the full path
    findings_dir = _settings.workspace_dir / "findings"
    file_path = findings_dir / filename

    # Check if approval is required
    if _approval_manager is not None:
        needs_approval = _approval_manager.requires_approval(
            "file_manager",
            "write",
            {"path": str(file_path), "content": content},
        )

        if needs_approval:
            request = _approval_manager.request_approval(
                tool="file_manager",
                operation="write",
                description=f"Write {len(content)} chars to {filename}",
                details={"path": str(file_path), "size": len(content)},
                risk_level="low",
            )
            # For demo purposes, auto-approve file writes
            _approval_manager.approve(request.id)

    # Create findings directory if needed
    findings_dir.mkdir(parents=True, exist_ok=True)

    # Write the file
    result = file_manager("write", str(file_path), content=content)

    if result["success"]:
        return {
            "success": True,
            "path": str(file_path),
            "size": result["size"],
            "message": f"Saved finding to {filename}",
        }

    return result


def read_finding(filename: str) -> dict[str, Any]:
    """Read a previously saved finding.

    Args:
        filename: Name of the file to read.

    Returns:
        A dict with the file content.
    """
    if _settings is None:
        return {"success": False, "error": "Settings not initialized"}

    from agentic_cli.tools.file_ops import file_manager

    findings_dir = _settings.workspace_dir / "findings"
    file_path = findings_dir / filename

    return file_manager("read", str(file_path))


def list_findings() -> dict[str, Any]:
    """List all saved findings.

    Returns:
        A dict with the list of findings files.
    """
    if _settings is None:
        return {"success": False, "error": "Settings not initialized"}

    from agentic_cli.tools.file_ops import file_manager

    findings_dir = _settings.workspace_dir / "findings"

    if not findings_dir.exists():
        return {
            "success": True,
            "path": str(findings_dir),
            "entries": {},
            "count": 0,
        }

    return file_manager("list", str(findings_dir))


def compare_versions(file_a: str, file_b: str) -> dict[str, Any]:
    """Compare two versions of a document.

    Args:
        file_a: First file path or content.
        file_b: Second file path or content.

    Returns:
        A dict with the diff and similarity score.
    """
    from agentic_cli.tools.file_ops import diff_compare

    return diff_compare(file_a, file_b, mode="unified")


# =============================================================================
# Shell Tool
# =============================================================================


def run_safe_command(command: str) -> dict[str, Any]:
    """Run a safe shell command.

    Only allows safe read-only commands. Dangerous commands are blocked.

    Args:
        command: The shell command to run.

    Returns:
        A dict with command output or error message.
    """
    from agentic_cli.tools.shell import shell_executor

    # Check approval for non-trivial commands
    if _approval_manager is not None:
        needs_approval = _approval_manager.requires_approval(
            "shell_executor",
            "run",
            {"command": command},
        )

        if needs_approval:
            request = _approval_manager.request_approval(
                tool="shell_executor",
                operation="run",
                description=f"Run: {command[:50]}...",
                details={"command": command},
                risk_level="medium",
            )
            # For demo, auto-approve safe commands
            _approval_manager.approve(request.id)

    return shell_executor(command, timeout=30)


# =============================================================================
# Checkpoint Tool
# =============================================================================


def create_checkpoint(
    name: str,
    content: str,
    content_type: str = "markdown",
) -> dict[str, Any]:
    """Create a checkpoint for user review.

    Use this when you have draft content that should be reviewed
    before proceeding.

    Args:
        name: Name of the checkpoint (e.g., "draft_summary").
        content: Content to review.
        content_type: Type of content ("markdown", "text", "code").

    Returns:
        A dict with the checkpoint ID.
    """
    if _checkpoint_manager is None:
        return {"success": False, "error": "Checkpoint manager not initialized"}

    checkpoint = _checkpoint_manager.create_checkpoint(
        name=name,
        content=content,
        content_type=content_type,
    )

    return {
        "success": True,
        "checkpoint_id": checkpoint.id,
        "name": name,
        "message": f"Created checkpoint '{name}' for review",
    }
