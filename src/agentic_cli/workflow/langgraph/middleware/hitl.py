"""Human-in-the-loop middleware factory for LangGraph workflows.

Provides factory function to create HITL middleware for requiring
human approval before executing certain tools or actions.

Note: This module provides a placeholder for future integration with
LangChain's native HumanInTheLoopMiddleware. Currently, HITL is handled
via LangGraph's interrupt mechanism in the manager.

Example:
    from agentic_cli.workflow.langgraph.middleware import create_hitl_middleware

    middleware = create_hitl_middleware(
        tools_requiring_approval=["shell_execute", "file_write"],
        settings=settings,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings

logger = Loggers.workflow()


def create_hitl_middleware(
    tools_requiring_approval: list[str],
    settings: "BaseSettings",
) -> Any | None:
    """Create HITL middleware for specified tools.

    This function creates middleware that interrupts workflow execution
    to request human approval before executing specified tools.

    Note: Currently returns None as LangGraph handles interrupts
    natively. This is a placeholder for future integration with
    LangChain's native HumanInTheLoopMiddleware.

    Args:
        tools_requiring_approval: List of tool names that require
            human approval before execution.
        settings: Application settings containing HITL configuration:
            - hitl_enabled: Whether HITL is globally enabled
            - hitl_checkpoint_enabled: Whether to create checkpoints
            - hitl_feedback_enabled: Whether to collect feedback

    Returns:
        HITL middleware instance, or None if disabled.
    """
    if not settings.hitl_enabled:
        logger.debug("hitl_middleware_disabled")
        return None

    if not tools_requiring_approval:
        logger.debug("hitl_middleware_no_tools")
        return None

    logger.debug(
        "hitl_middleware_config",
        tools=tools_requiring_approval,
        checkpoint_enabled=settings.hitl_checkpoint_enabled,
        feedback_enabled=settings.hitl_feedback_enabled,
    )

    # Placeholder for future middleware integration
    # LangChain middleware imports would go here when available:
    #
    # try:
    #     from langchain.agents.middleware import HumanInTheLoopMiddleware
    #
    #     # Build interrupt configuration per tool
    #     interrupt_on = {
    #         tool: ["approve", "edit", "reject"]
    #         for tool in tools_requiring_approval
    #     }
    #
    #     return HumanInTheLoopMiddleware(
    #         interrupt_on=interrupt_on,
    #         checkpoint_enabled=settings.hitl_checkpoint_enabled,
    #         feedback_enabled=settings.hitl_feedback_enabled,
    #     )
    # except ImportError:
    #     logger.warning("hitl_middleware_not_available")
    #     return None

    # Currently, HITL is handled via LangGraph interrupts in the manager
    return None


def get_hitl_interrupt_config(
    tools_requiring_approval: list[str],
) -> dict[str, list[str]]:
    """Get interrupt configuration for HITL tools.

    This helper creates the interrupt configuration dict that can be
    used with LangGraph's native interrupt mechanism.

    Args:
        tools_requiring_approval: List of tool names requiring approval.

    Returns:
        Dict mapping tool names to list of interrupt actions
        (approve, edit, reject).
    """
    return {
        tool: ["approve", "edit", "reject"]
        for tool in tools_requiring_approval
    }
