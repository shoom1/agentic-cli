"""Shell execution tools for LangGraph workflows.

Thin wrapper around the main shell executor that provides
LangChain-compatible @tool decorators for LangGraph agents.

Example:
    from agentic_cli.workflow.langgraph.tools import create_shell_tool

    shell_tool = create_shell_tool(
        workspace_dir="/path/to/workspace",
        timeout=60,
    )
    llm_with_tools = llm.bind_tools([shell_tool])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_cli.logging import Loggers
from agentic_cli.tools.shell import shell_executor as main_shell_executor, is_shell_enabled

logger = Loggers.workflow()


# Re-export safety utilities from main shell module
def get_blocked_patterns() -> list[str]:
    """Get list of blocked shell command patterns.

    Delegates to the main shell security module.
    """
    from agentic_cli.tools.shell.config import get_strict_config
    config = get_strict_config()
    # Return a representative list for backward compatibility
    return [
        r"rm\s+-rf\s+/",
        r"rm\s+-rf\s+~",
        r"rm\s+-rf\s+\*",
        r":\(\)\{\s*:\|:&\s*\};:",
        r"dd\s+if=/dev/zero",
        r"mkfs\.",
        r"chmod\s+-R\s+777\s+/",
        r"curl.*\|\s*bash",
        r"wget.*\|\s*bash",
        r">\s*/dev/sd",
    ]


def is_command_safe(command: str) -> tuple[bool, str | None]:
    """Check if a shell command is safe to execute.

    Delegates to the main shell security analysis.
    """
    from agentic_cli.tools.shell import analyze_command
    analysis = analyze_command(command)
    if analysis.get("blocked"):
        return False, analysis.get("reason", "Command blocked by security policy")
    return True, None


def shell_execute(
    command: str,
    working_dir: str | Path | None = None,
    timeout: int = 60,
    capture_stderr: bool = True,
) -> dict[str, Any]:
    """Execute a shell command with safety checks.

    Delegates to the main shell executor.

    Args:
        command: Shell command to execute.
        working_dir: Working directory for command execution.
        timeout: Maximum execution time in seconds.
        capture_stderr: Whether to capture stderr separately.

    Returns:
        Dictionary with success, stdout, stderr, return_code, error keys.
    """
    cwd = str(working_dir) if working_dir else None
    return main_shell_executor(command, working_dir=cwd, timeout=timeout)


async def shell_execute_async(
    command: str,
    working_dir: str | Path | None = None,
    timeout: int = 60,
    capture_stderr: bool = True,
) -> dict[str, Any]:
    """Execute a shell command asynchronously with safety checks.

    Delegates to the main shell executor (runs sync in executor).

    Args:
        command: Shell command to execute.
        working_dir: Working directory for command execution.
        timeout: Maximum execution time in seconds.
        capture_stderr: Whether to capture stderr separately.

    Returns:
        Dictionary with success, stdout, stderr, return_code, error keys.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: shell_execute(command, working_dir, timeout, capture_stderr),
    )


def create_shell_tool(
    workspace_dir: str | Path | None = None,
    timeout: int = 60,
    require_approval: bool = False,
) -> Any:
    """Create a shell execution tool for binding to LangGraph agents.

    Creates a LangChain-compatible tool instance that can be bound
    to language models via llm.bind_tools().

    Args:
        workspace_dir: Base directory for shell commands.
        timeout: Default timeout in seconds for commands.
        require_approval: Whether this tool requires human approval.

    Returns:
        Shell tool instance.
    """
    try:
        from langchain_core.tools import tool
    except ImportError:
        logger.warning("langchain_core_not_installed")
        return None

    base_dir = str(workspace_dir) if workspace_dir else None
    default_timeout = timeout

    @tool
    def shell(
        command: str,
        working_dir: str | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Execute a shell command.

        Runs a shell command with safety validation. Dangerous commands
        (rm -rf /, format disk, etc.) are automatically blocked.

        Args:
            command: Shell command to execute
            working_dir: Working directory (defaults to workspace)
            timeout: Timeout in seconds (defaults to configured value)

        Returns:
            Dictionary with stdout, stderr, return_code, and success status
        """
        cwd = working_dir or base_dir
        cmd_timeout = timeout or default_timeout

        return shell_execute(
            command=command,
            working_dir=cwd,
            timeout=cmd_timeout,
        )

    # Add metadata for HITL middleware
    if require_approval:
        shell.metadata = {"requires_approval": True}

    return shell
