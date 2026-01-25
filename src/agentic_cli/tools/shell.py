"""Shell executor tool with safety controls.

Provides a safe shell command execution interface with built-in protection
against dangerous commands.
"""

import re
import subprocess
import time
from typing import Any

# Maximum output length before truncation
MAX_OUTPUT_LENGTH = 50000

# Dangerous command patterns to block
DANGEROUS_PATTERNS = [
    # Destructive file operations
    r"rm\s+-[rf]*\s+/\s*$",  # rm -rf /
    r"rm\s+-[rf]*\s+/\s+",  # rm -rf / with more args
    r"rm\s+-rf\s+\*",  # rm -rf *
    r"rm\s+-fr\s+\*",  # rm -fr *
    r"rm\s+-rf\s+/[a-zA-Z]*$",  # rm -rf /boot, /etc, etc.
    # Filesystem operations
    r"mkfs\.",  # mkfs.ext4, mkfs.xfs, etc.
    r"dd\s+.*of=/dev/",  # dd of=/dev/
    # Fork bombs
    r":\(\)\s*\{\s*:\|:&\s*\}\s*;:",  # :(){ :|:& };:
    r"fork\s*\(\s*\)\s*while",  # fork() while loops
    # Permission changes on root
    r"chmod\s+777\s+/\s*$",  # chmod 777 /
    r"chmod\s+-R\s+777\s+/",  # chmod -R 777 /
    r"chown\s+.*\s+/\s*$",  # chown ... /
    # Remote code execution via pipe
    r"curl\s+.*\|\s*sh",  # curl | sh
    r"curl\s+.*\|\s*bash",  # curl | bash
    r"wget\s+.*\|\s*sh",  # wget | sh
    r"wget\s+.*\|\s*bash",  # wget | bash
    # Additional dangerous patterns
    r">\s*/dev/sd[a-z]",  # Writing directly to disk devices
    r"cat\s+/dev/zero\s*>\s*/",  # cat /dev/zero > /
    r"echo\s+.*>\s*/dev/sd",  # echo > /dev/sd*
]

# Compile patterns for efficiency
_compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in DANGEROUS_PATTERNS]


def is_dangerous_command(command: str) -> bool:
    """Check if a command matches any dangerous patterns.

    Args:
        command: The shell command to check.

    Returns:
        True if the command is potentially dangerous, False otherwise.
    """
    for pattern in _compiled_patterns:
        if pattern.search(command):
            return True
    return False


def truncate_output(output: str, max_length: int = MAX_OUTPUT_LENGTH) -> str:
    """Truncate output if it exceeds maximum length.

    Args:
        output: The output string to potentially truncate.
        max_length: Maximum allowed length.

    Returns:
        Original or truncated output with indicator.
    """
    if len(output) <= max_length:
        return output

    truncated = output[:max_length]
    return f"{truncated}\n... [OUTPUT TRUNCATED - exceeded {max_length} characters]"


def shell_executor(
    command: str,
    working_dir: str | None = None,
    timeout: int = 60,
    capture_output: bool = True,
) -> dict[str, Any]:
    """Execute a shell command with safety controls.

    Executes the given shell command with built-in safety controls that block
    potentially dangerous operations like rm -rf /, fork bombs, and remote
    code execution via piped scripts.

    Args:
        command: The shell command to execute.
        working_dir: Optional working directory for command execution.
        timeout: Maximum execution time in seconds (default: 60).
        capture_output: Whether to capture stdout/stderr (default: True).

    Returns:
        A dictionary containing:
            - success: bool - True if return_code == 0
            - stdout: str - Standard output from the command
            - stderr: str - Standard error from the command
            - return_code: int - Exit code of the command
            - duration: float - Execution time in seconds
            - error: str | None - Error message if failed

    Examples:
        >>> result = shell_executor("echo hello")
        >>> result["success"]
        True
        >>> result["stdout"].strip()
        'hello'

        >>> result = shell_executor("rm -rf /")
        >>> result["success"]
        False
        >>> "blocked" in result["error"].lower()
        True
    """
    # Check for dangerous commands
    if is_dangerous_command(command):
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "return_code": -1,
            "duration": 0.0,
            "error": "Command blocked: potentially dangerous operation detected",
        }

    start_time = time.time()

    try:
        result = subprocess.run(
            command,
            shell=True,
            timeout=timeout,
            capture_output=capture_output,
            text=True,
            cwd=working_dir,
        )

        duration = time.time() - start_time

        stdout = result.stdout if result.stdout else ""
        stderr = result.stderr if result.stderr else ""

        # Truncate output if needed
        stdout = truncate_output(stdout)
        stderr = truncate_output(stderr)

        return {
            "success": result.returncode == 0,
            "stdout": stdout,
            "stderr": stderr,
            "return_code": result.returncode,
            "duration": duration,
            "error": None if result.returncode == 0 else f"Command exited with code {result.returncode}",
        }

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "return_code": -1,
            "duration": duration,
            "error": f"Command timeout after {timeout} seconds",
        }

    except Exception as e:
        duration = time.time() - start_time
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "return_code": -1,
            "duration": duration,
            "error": f"Execution error: {str(e)}",
        }
