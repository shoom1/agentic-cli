"""Todo list middleware factory for LangGraph workflows.

Provides factory function to create todo list middleware for
task tracking within agentic workflows.

Note: This module provides a placeholder for future integration with
LangChain's native TodoListMiddleware. Currently, task tracking is
handled via the TaskGraph context manager.

Example:
    from agentic_cli.workflow.langgraph.middleware import create_todo_list_middleware

    middleware = create_todo_list_middleware(settings)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings

logger = Loggers.workflow()


def create_todo_list_middleware(settings: "BaseSettings") -> Any | None:
    """Create todo list middleware for task tracking.

    This function creates middleware for tracking tasks and subtasks
    within agentic workflows, enabling structured work management.

    Note: Currently returns None as task tracking is handled via
    the TaskGraph context manager. This is a placeholder for future
    integration with LangChain's native TodoListMiddleware.

    Args:
        settings: Application settings (reserved for future configuration).

    Returns:
        Todo list middleware instance, or None if not configured.
    """
    logger.debug("todo_list_middleware_config")

    # Placeholder for future middleware integration
    # LangChain middleware imports would go here when available:
    #
    # try:
    #     from langchain.agents.middleware import TodoListMiddleware
    #
    #     return TodoListMiddleware(
    #         persist_path=settings.workspace_dir / "todos.json",
    #     )
    # except ImportError:
    #     logger.warning("todo_list_middleware_not_available")
    #     return None

    # Currently, task tracking is handled via TaskGraph context
    return None
