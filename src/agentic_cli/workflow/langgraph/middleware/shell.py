"""Shell execution middleware factory for LangGraph workflows.

Provides factory function to create shell middleware with configurable
execution policies (host or Docker sandbox).

Note: This module provides a placeholder for future integration with
LangChain's native ShellToolMiddleware. Currently, shell execution
is handled via custom tools with pattern blocking.

Example:
    from agentic_cli.workflow.langgraph.middleware import create_shell_middleware

    middleware = create_shell_middleware(settings)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path

from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings

logger = Loggers.workflow()


def create_shell_middleware(settings: "BaseSettings") -> Any | None:
    """Create shell middleware with appropriate execution policy.

    This function creates middleware for executing shell commands
    with configurable sandboxing (host execution or Docker).

    Note: Currently returns None as shell execution is handled
    via custom tools with pattern blocking. This is a placeholder
    for future integration with LangChain's native ShellToolMiddleware.

    Args:
        settings: Application settings containing shell configuration:
            - shell_sandbox_type: "host" or "docker"
            - shell_docker_image: Docker image for sandbox execution
            - shell_timeout: Default timeout in seconds
            - workspace_dir: Working directory for shell commands

    Returns:
        Shell middleware instance, or None if not configured.
    """
    sandbox_type = settings.shell_sandbox_type
    workspace_dir = settings.workspace_dir

    logger.debug(
        "shell_middleware_config",
        sandbox_type=sandbox_type,
        docker_image=settings.shell_docker_image if sandbox_type == "docker" else None,
        timeout=settings.shell_timeout,
        workspace=str(workspace_dir),
    )

    # Placeholder for future middleware integration
    # LangChain middleware imports would go here when available:
    #
    # try:
    #     from langchain.agents.middleware import ShellToolMiddleware
    #     from langchain.agents.middleware import (
    #         HostExecutionPolicy,
    #         DockerExecutionPolicy,
    #     )
    #
    #     if sandbox_type == "docker":
    #         policy = DockerExecutionPolicy(
    #             image=settings.shell_docker_image,
    #             timeout=settings.shell_timeout,
    #         )
    #     else:
    #         policy = HostExecutionPolicy(
    #             timeout=settings.shell_timeout,
    #         )
    #
    #     return ShellToolMiddleware(
    #         workspace_root=workspace_dir,
    #         execution_policy=policy,
    #     )
    # except ImportError:
    #     logger.warning("shell_middleware_not_available")
    #     return None

    # Currently, shell execution is handled via custom tools
    return None


def get_blocked_patterns() -> list[str]:
    """Get list of blocked shell command patterns.

    Returns patterns that should be blocked for security reasons.
    These patterns prevent destructive or dangerous commands.

    Returns:
        List of regex patterns to block.
    """
    return [
        r"rm\s+-rf\s+/",  # Recursive delete from root
        r"rm\s+-rf\s+~",  # Recursive delete home
        r"rm\s+-rf\s+\*",  # Recursive delete all
        r":\(\)\{\s*:\|:&\s*\};:",  # Fork bomb
        r"dd\s+if=/dev/zero",  # Disk wipe
        r"mkfs\.",  # Format filesystem
        r"chmod\s+-R\s+777\s+/",  # Dangerous permissions
        r"curl.*\|\s*bash",  # Pipe to shell
        r"wget.*\|\s*bash",  # Pipe to shell
        r">\s*/dev/sd",  # Write to disk device
    ]


def is_command_safe(command: str) -> tuple[bool, str | None]:
    """Check if a shell command is safe to execute.

    Args:
        command: Shell command to check.

    Returns:
        Tuple of (is_safe, reason). If is_safe is False,
        reason contains explanation of why command was blocked.
    """
    import re

    patterns = get_blocked_patterns()

    for pattern in patterns:
        if re.search(pattern, command, re.IGNORECASE):
            return False, f"Command matches blocked pattern: {pattern}"

    return True, None
