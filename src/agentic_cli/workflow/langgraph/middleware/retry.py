"""Retry middleware factory for LangGraph workflows.

Provides factory function to create retry middleware for handling
transient errors in tool and model invocations.

Note: This module provides a placeholder for future integration with
LangChain's native retry middleware. Currently, LangGraph handles retries
via RetryPolicy on nodes (used in LangGraphWorkflowManager._build_graph).

Example:
    from agentic_cli.workflow.langgraph.middleware import create_retry_middleware

    middlewares = create_retry_middleware(settings)
    # Returns list of middleware instances
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings

logger = Loggers.workflow()


def create_retry_middleware(settings: "BaseSettings") -> list[Any]:
    """Create retry middleware instances for tool and model retries.

    This function creates middleware for handling transient errors
    with configurable retry behavior including exponential backoff.

    Note: Currently returns an empty list as LangGraph handles retries
    via RetryPolicy on nodes. This is a placeholder for future integration
    with LangChain's native middleware when it becomes available.

    Args:
        settings: Application settings containing retry configuration:
            - retry_max_attempts: Maximum number of retry attempts
            - retry_initial_delay: Initial delay in seconds
            - retry_backoff_factor: Multiplier for exponential backoff

    Returns:
        List of middleware instances (currently empty).
    """
    logger.debug(
        "retry_middleware_config",
        max_attempts=settings.retry_max_attempts,
        initial_delay=settings.retry_initial_delay,
        backoff_factor=settings.retry_backoff_factor,
    )

    # Placeholder for future middleware integration
    # LangChain middleware imports would go here when available:
    #
    # try:
    #     from langchain.agents.middleware import (
    #         ToolRetryMiddleware,
    #         ModelRetryMiddleware,
    #     )
    #
    #     return [
    #         ToolRetryMiddleware(
    #             max_retries=settings.retry_max_attempts,
    #             backoff_factor=settings.retry_backoff_factor,
    #             initial_delay=settings.retry_initial_delay,
    #         ),
    #         ModelRetryMiddleware(
    #             max_retries=settings.retry_max_attempts,
    #             backoff_factor=settings.retry_backoff_factor,
    #         ),
    #     ]
    # except ImportError:
    #     logger.warning("retry_middleware_not_available")
    #     return []

    # Currently, retry is handled via RetryPolicy in _build_graph
    return []
