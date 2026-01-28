"""Shell execution tools for LangGraph workflows.

Provides shell execution tools with safety checks and configurable
execution policies. Commands are validated against blocked patterns
before execution.

Example:
    from agentic_cli.workflow.langgraph.tools import create_shell_tool

    shell_tool = create_shell_tool(
        workspace_dir="/path/to/workspace",
        timeout=60,
    )
    llm_with_tools = llm.bind_tools([shell_tool])
"""

from __future__ import annotations

import asyncio
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentic_cli.logging import Loggers

logger = Loggers.workflow()


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
    patterns = get_blocked_patterns()

    for pattern in patterns:
        if re.search(pattern, command, re.IGNORECASE):
            return False, f"Command matches blocked pattern: {pattern}"

    return True, None


def shell_execute(
    command: str,
    working_dir: str | Path | None = None,
    timeout: int = 60,
    capture_stderr: bool = True,
) -> dict[str, Any]:
    """Execute a shell command with safety checks.

    Validates the command against blocked patterns before execution.
    Returns structured output with stdout, stderr, and return code.

    Args:
        command: Shell command to execute.
        working_dir: Working directory for command execution.
        timeout: Maximum execution time in seconds.
        capture_stderr: Whether to capture stderr separately.

    Returns:
        Dictionary with keys:
        - success: Whether command succeeded (return code 0)
        - stdout: Standard output
        - stderr: Standard error (if capture_stderr=True)
        - return_code: Process return code
        - error: Error message if command was blocked or failed

    Example:
        >>> shell_execute("ls -la", "/path/to/dir")
        {'success': True, 'stdout': '...', 'return_code': 0}
    """
    # Validate command safety
    is_safe, reason = is_command_safe(command)
    if not is_safe:
        logger.warning(
            "shell_command_blocked",
            command=command[:100],
            reason=reason,
        )
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "return_code": -1,
            "error": f"Command blocked: {reason}",
        }

    cwd = Path(working_dir) if working_dir else None
    if cwd and not cwd.exists():
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "return_code": -1,
            "error": f"Working directory does not exist: {cwd}",
        }

    logger.debug(
        "shell_execute_start",
        command=command[:100],
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
    )

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        success = result.returncode == 0
        output = {
            "success": success,
            "stdout": result.stdout,
            "return_code": result.returncode,
        }

        if capture_stderr:
            output["stderr"] = result.stderr

        logger.debug(
            "shell_execute_complete",
            success=success,
            return_code=result.returncode,
            stdout_len=len(result.stdout),
        )

        return output

    except subprocess.TimeoutExpired:
        logger.warning("shell_execute_timeout", command=command[:100], timeout=timeout)
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "return_code": -1,
            "error": f"Command timed out after {timeout} seconds",
        }

    except Exception as e:
        logger.error("shell_execute_error", command=command[:100], error=str(e))
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "return_code": -1,
            "error": str(e),
        }


async def shell_execute_async(
    command: str,
    working_dir: str | Path | None = None,
    timeout: int = 60,
    capture_stderr: bool = True,
) -> dict[str, Any]:
    """Execute a shell command asynchronously with safety checks.

    Async version of shell_execute for use in async workflows.

    Args:
        command: Shell command to execute.
        working_dir: Working directory for command execution.
        timeout: Maximum execution time in seconds.
        capture_stderr: Whether to capture stderr separately.

    Returns:
        Dictionary with keys:
        - success: Whether command succeeded (return code 0)
        - stdout: Standard output
        - stderr: Standard error (if capture_stderr=True)
        - return_code: Process return code
        - error: Error message if command was blocked or failed
    """
    # Validate command safety
    is_safe, reason = is_command_safe(command)
    if not is_safe:
        logger.warning(
            "shell_command_blocked",
            command=command[:100],
            reason=reason,
        )
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "return_code": -1,
            "error": f"Command blocked: {reason}",
        }

    cwd = Path(working_dir) if working_dir else None
    if cwd and not cwd.exists():
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "return_code": -1,
            "error": f"Working directory does not exist: {cwd}",
        }

    logger.debug(
        "shell_execute_async_start",
        command=command[:100],
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
    )

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE if capture_stderr else None,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            logger.warning(
                "shell_execute_async_timeout",
                command=command[:100],
                timeout=timeout,
            )
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "error": f"Command timed out after {timeout} seconds",
            }

        success = process.returncode == 0
        output = {
            "success": success,
            "stdout": stdout.decode("utf-8", errors="replace") if stdout else "",
            "return_code": process.returncode or 0,
        }

        if capture_stderr and stderr:
            output["stderr"] = stderr.decode("utf-8", errors="replace")

        logger.debug(
            "shell_execute_async_complete",
            success=success,
            return_code=process.returncode,
        )

        return output

    except Exception as e:
        logger.error("shell_execute_async_error", command=command[:100], error=str(e))
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "return_code": -1,
            "error": str(e),
        }


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
            Defaults to current working directory.
        timeout: Default timeout in seconds for commands.
        require_approval: Whether this tool requires human approval.
            (Used as metadata hint for HITL middleware)

    Returns:
        Shell tool instance.

    Example:
        shell = create_shell_tool("/path/to/project", timeout=120)
        llm_with_tools = llm.bind_tools([shell])
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
