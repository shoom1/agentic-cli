"""Abstract base and no-op implementations for OS-level sandboxing.

Provides the OSSandbox ABC that platform-specific implementations extend,
plus OSSandboxResult for returning wrapped commands, and NoOpSandbox as
the fallback when no sandbox tool is available.
"""

from __future__ import annotations

import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_cli.tools.shell.os_sandbox.policy import OSSandboxPolicy


@dataclass
class OSSandboxResult:
    """Result of wrapping a command with OS sandbox.

    Attributes:
        command: The wrapped command string (ready for shell execution).
        env: Environment variable overrides to set.
        success: Whether wrapping succeeded.
        error: Error message if wrapping failed.
        sandbox_type: Identifier for which sandbox was used.
    """

    command: str
    env: dict[str, str] = field(default_factory=dict)
    success: bool = True
    error: str | None = None
    sandbox_type: str = "none"


class OSSandbox(ABC):
    """Abstract base for OS-level sandbox implementations.

    Implementations wrap commands with platform-specific sandboxing
    (Seatbelt on macOS, bubblewrap on Linux). The wrapped command
    is then executed by the existing subprocess infrastructure.
    """

    @abstractmethod
    def wrap_shell_command(
        self,
        command: str,
        working_dir: Path,
        policy: OSSandboxPolicy,
    ) -> OSSandboxResult:
        """Wrap a shell command with OS-level sandboxing.

        Args:
            command: The raw shell command to execute.
            working_dir: Working directory for execution.
            policy: Sandbox policy defining allowed/denied paths.

        Returns:
            OSSandboxResult with the wrapped command.
        """
        ...

    @abstractmethod
    def wrap_python_command(
        self,
        python_args: list[str],
        working_dir: Path,
        policy: OSSandboxPolicy,
    ) -> OSSandboxResult:
        """Wrap a Python subprocess command with OS-level sandboxing.

        Args:
            python_args: e.g. [sys.executable, "-c", script]
            working_dir: Working directory for execution.
            policy: Sandbox policy.

        Returns:
            OSSandboxResult with the wrapped command.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the sandbox tool is available on this system."""
        ...

    @property
    @abstractmethod
    def sandbox_type(self) -> str:
        """Return identifier: 'seatbelt', 'bubblewrap', 'none'."""
        ...


class NoOpSandbox(OSSandbox):
    """Pass-through sandbox that applies no OS-level isolation.

    Used as fallback when sandbox tools are not available.
    Commands are returned unchanged.
    """

    def wrap_shell_command(
        self,
        command: str,
        working_dir: Path,
        policy: OSSandboxPolicy,
    ) -> OSSandboxResult:
        return OSSandboxResult(command=command, sandbox_type="none")

    def wrap_python_command(
        self,
        python_args: list[str],
        working_dir: Path,
        policy: OSSandboxPolicy,
    ) -> OSSandboxResult:
        return OSSandboxResult(
            command=shlex.join(python_args),
            sandbox_type="none",
        )

    def is_available(self) -> bool:
        return True

    @property
    def sandbox_type(self) -> str:
        return "none"
