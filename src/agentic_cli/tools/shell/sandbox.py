"""Execution sandbox with resource limits.

Layer 7: Resource Limits & Execution Sandbox
- Constrain execution to prevent resource exhaustion
- Cross-platform support via ulimit
- Configurable limits for memory, processes, files
"""

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExecutionLimits:
    """Resource limits for command execution.

    Attributes:
        timeout_seconds: Maximum execution time.
        max_output_bytes: Maximum output size before truncation.
        max_memory_mb: Maximum memory usage in MB.
        max_cpu_percent: Maximum CPU percentage (informational).
        max_processes: Maximum number of processes.
        max_open_files: Maximum number of open files.
        enable_ulimits: Whether to apply ulimit restrictions (may cause issues on some systems).
    """
    timeout_seconds: int = 60
    max_output_bytes: int = 50000
    max_memory_mb: int = 512
    max_cpu_percent: int = 80  # Informational - not enforced by ulimit
    max_processes: int = 1000  # Higher default to avoid fork issues
    max_open_files: int = 256
    enable_ulimits: bool = False  # Disabled by default - use subprocess timeout instead


@dataclass
class ExecutionResult:
    """Result of sandboxed command execution.

    Attributes:
        success: Whether the command completed successfully.
        stdout: Standard output (may be truncated).
        stderr: Standard error (may be truncated).
        return_code: Exit code of the command.
        duration_ms: Execution duration in milliseconds.
        error: Error message if execution failed.
        truncated: Whether output was truncated.
        resource_limit_hit: Which limit was hit, if any.
    """
    success: bool
    stdout: str
    stderr: str
    return_code: int
    duration_ms: int
    error: str | None = None
    truncated: bool = False
    resource_limit_hit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_code": self.return_code,
            "duration": self.duration_ms / 1000,  # Convert to seconds for compatibility
            "error": self.error,
            "truncated": self.truncated,
            "resource_limit_hit": self.resource_limit_hit,
        }


class ExecutionSandbox:
    """Sandboxed command executor with resource limits.

    Phase 3 implementation uses ulimit for cross-platform resource limits.
    Phase 4 will extend to use:
    - Linux: cgroups v2
    - macOS: sandbox-exec
    """

    def __init__(self, limits: ExecutionLimits | None = None):
        """Initialize sandbox with resource limits.

        Args:
            limits: Resource limits to apply.
        """
        self.limits = limits or ExecutionLimits()

    def execute(
        self,
        command: str,
        working_dir: Path | str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute a command with resource limits.

        Args:
            command: The shell command to execute.
            working_dir: Working directory for execution.
            env: Additional environment variables.

        Returns:
            ExecutionResult with execution details.
        """
        # For now, use ulimit-based limits (cross-platform for Unix-like systems)
        # Future: dispatch to platform-specific implementations
        return self._execute_with_ulimit(command, working_dir, env)

    def _execute_with_ulimit(
        self,
        command: str,
        working_dir: Path | str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute command with optional ulimit-based resource limits.

        Uses ulimit shell builtin to set limits before executing the command.
        Works on Linux and macOS. ulimits are disabled by default since they
        can cause issues (fork failures) on some systems.
        """
        if working_dir is not None:
            working_dir = Path(working_dir)

        # Build the command (with optional ulimit wrapper)
        if self.limits.enable_ulimits:
            # Build the wrapper command with ulimit settings
            # Note: Some ulimit settings may require elevated privileges
            # We set what we can and ignore failures
            ulimit_commands = []

            # -t: CPU time in seconds
            ulimit_commands.append(f"ulimit -t {self.limits.timeout_seconds} 2>/dev/null || true")

            # -v: Virtual memory in KB (max_memory_mb * 1024)
            # Note: On macOS, -v is not supported, use -m instead
            if sys.platform == "darwin":
                # macOS: -m sets resident set size (not perfect but closest)
                ulimit_commands.append(
                    f"ulimit -m {self.limits.max_memory_mb * 1024} 2>/dev/null || true"
                )
            else:
                # Linux: -v sets virtual memory
                ulimit_commands.append(
                    f"ulimit -v {self.limits.max_memory_mb * 1024} 2>/dev/null || true"
                )

            # -u: Maximum number of processes
            ulimit_commands.append(
                f"ulimit -u {self.limits.max_processes} 2>/dev/null || true"
            )

            # -n: Maximum number of open files
            ulimit_commands.append(
                f"ulimit -n {self.limits.max_open_files} 2>/dev/null || true"
            )

            # Combine ulimit settings with the actual command
            # Using a subshell to contain the limit settings
            wrapper = f"""(
{chr(10).join(ulimit_commands)}
{command}
)"""
        else:
            # No ulimit wrapper - just run the command directly
            wrapper = command

        start_time = time.time()
        truncated = False
        resource_limit_hit = None

        try:
            # Use subprocess.Popen for more control over output handling
            process = subprocess.Popen(
                wrapper,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=working_dir,
                env={**dict(subprocess.os.environ), **(env or {})},
            )

            try:
                stdout_bytes, stderr_bytes = process.communicate(
                    timeout=self.limits.timeout_seconds
                )
                return_code = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                stdout_bytes, stderr_bytes = process.communicate()
                duration_ms = int((time.time() - start_time) * 1000)
                return ExecutionResult(
                    success=False,
                    stdout=self._decode_and_truncate(stdout_bytes),
                    stderr=self._decode_and_truncate(stderr_bytes),
                    return_code=-1,
                    duration_ms=duration_ms,
                    error=f"Command timeout after {self.limits.timeout_seconds} seconds",
                    resource_limit_hit="timeout",
                )

            duration_ms = int((time.time() - start_time) * 1000)

            # Decode and truncate output
            stdout, stdout_truncated = self._decode_and_truncate(
                stdout_bytes, return_truncated=True
            )
            stderr, stderr_truncated = self._decode_and_truncate(
                stderr_bytes, return_truncated=True
            )
            truncated = stdout_truncated or stderr_truncated

            # Check for resource limit signals
            # SIGXCPU (24 on Linux, 24 on macOS) - CPU time limit exceeded
            # SIGKILL (9) - often from OOM killer
            if return_code == -9 or return_code == 137:  # 128 + 9 = SIGKILL
                resource_limit_hit = "memory"
            elif return_code == -24 or return_code == 152:  # 128 + 24 = SIGXCPU
                resource_limit_hit = "cpu_time"

            return ExecutionResult(
                success=return_code == 0,
                stdout=stdout,
                stderr=stderr,
                return_code=return_code,
                duration_ms=duration_ms,
                error=(
                    None if return_code == 0
                    else f"Command exited with code {return_code}"
                ),
                truncated=truncated,
                resource_limit_hit=resource_limit_hit,
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                return_code=-1,
                duration_ms=duration_ms,
                error=f"Execution error: {str(e)}",
            )

    def _decode_and_truncate(
        self,
        data: bytes,
        return_truncated: bool = False,
    ) -> str | tuple[str, bool]:
        """Decode bytes and truncate if necessary.

        Args:
            data: Bytes to decode.
            return_truncated: Whether to return truncation flag.

        Returns:
            Decoded string, or tuple of (string, truncated_flag).
        """
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = str(data)

        truncated = False
        if len(text) > self.limits.max_output_bytes:
            text = text[:self.limits.max_output_bytes]
            text += f"\n... [OUTPUT TRUNCATED - exceeded {self.limits.max_output_bytes} bytes]"
            truncated = True

        if return_truncated:
            return text, truncated
        return text

    def get_limits_description(self) -> str:
        """Get human-readable description of current limits."""
        return (
            f"Timeout: {self.limits.timeout_seconds}s, "
            f"Memory: {self.limits.max_memory_mb}MB, "
            f"Processes: {self.limits.max_processes}, "
            f"Open files: {self.limits.max_open_files}, "
            f"Output: {self.limits.max_output_bytes} bytes"
        )


# Convenience function for simple execution
def execute_sandboxed(
    command: str,
    working_dir: Path | str | None = None,
    limits: ExecutionLimits | None = None,
) -> ExecutionResult:
    """Execute a command in a sandbox with resource limits.

    Convenience wrapper around ExecutionSandbox.

    Args:
        command: The shell command to execute.
        working_dir: Working directory for execution.
        limits: Resource limits to apply.

    Returns:
        ExecutionResult with execution details.
    """
    sandbox = ExecutionSandbox(limits)
    return sandbox.execute(command, working_dir)
