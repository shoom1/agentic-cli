"""Linux sandbox using bubblewrap (bwrap) with namespace isolation.

Wraps commands via:
    bwrap [args] -- /bin/bash -c '<command>'

Bubblewrap creates a lightweight user namespace with:
- Read-only root filesystem
- Writable bind mounts for allowed paths
- PID namespace isolation
- Network namespace isolation (no network)
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from agentic_cli.tools.shell.os_sandbox.base import OSSandbox, OSSandboxResult

if TYPE_CHECKING:
    from agentic_cli.tools.shell.os_sandbox.policy import OSSandboxPolicy


class BubblewrapSandbox(OSSandbox):
    """Linux sandbox using bubblewrap (bwrap) with namespace isolation."""

    def wrap_shell_command(
        self,
        command: str,
        working_dir: Path,
        policy: OSSandboxPolicy,
    ) -> OSSandboxResult:
        """Wrap a shell command with bwrap.

        Args:
            command: The raw shell command.
            working_dir: Working directory for execution.
            policy: Sandbox policy.

        Returns:
            OSSandboxResult with the wrapped command.
        """
        try:
            bwrap_args = self._build_bwrap_args(working_dir, policy)
            escaped_command = command.replace("'", "'\\''")
            wrapped = f"{shlex.join(bwrap_args)} -- /bin/bash -c '{escaped_command}'"
            return OSSandboxResult(command=wrapped, sandbox_type="bubblewrap")
        except Exception as e:
            return OSSandboxResult(
                command=command,
                success=False,
                error=f"Failed to build bwrap arguments: {e}",
                sandbox_type="bubblewrap",
            )

    def wrap_python_command(
        self,
        python_args: list[str],
        working_dir: Path,
        policy: OSSandboxPolicy,
    ) -> OSSandboxResult:
        """Wrap a Python command with bwrap.

        Args:
            python_args: e.g. [sys.executable, "-c", script]
            working_dir: Working directory.
            policy: Sandbox policy.

        Returns:
            OSSandboxResult with the wrapped command.
        """
        try:
            bwrap_args = self._build_bwrap_args(working_dir, policy)
            wrapped = shlex.join(bwrap_args + ["--"] + python_args)
            return OSSandboxResult(command=wrapped, sandbox_type="bubblewrap")
        except Exception as e:
            return OSSandboxResult(
                command=shlex.join(python_args),
                success=False,
                error=f"Failed to build bwrap arguments: {e}",
                sandbox_type="bubblewrap",
            )

    def _build_bwrap_args(
        self, working_dir: Path, policy: OSSandboxPolicy
    ) -> list[str]:
        """Build bwrap command-line arguments from policy.

        Argument ordering matters — bwrap processes bind mounts in order,
        so more specific mounts must come after less specific ones.

        Structure:
        1. Isolation flags (session, die-with-parent, PID namespace)
        2. Network namespace (if blocking network)
        3. Read-only root (--ro-bind / /)
        4. Writable bind mounts for allowed paths
        5. Read-only overrides for denied paths within writable regions
        6. Block nonexistent deny targets
        7. Hide denied read paths
        8. Device and proc filesystems

        Args:
            working_dir: Working directory (always writable).
            policy: Sandbox policy.

        Returns:
            List of bwrap arguments (without the trailing command).
        """
        args = [
            "bwrap",
            "--new-session",
            "--die-with-parent",
            "--unshare-pid",
        ]

        # Network isolation
        if not policy.allow_network:
            args.append("--unshare-net")

        # Read-only root filesystem
        args.extend(["--ro-bind", "/", "/"])

        # Writable bind mounts for allowed paths
        writable = policy.resolved_writable_paths(working_dir)
        for wp in writable:
            if wp.exists():
                args.extend(["--bind", str(wp), str(wp)])
            else:
                # Create a tmpfs for nonexistent writable paths so they
                # can be written to inside the sandbox
                args.extend(["--tmpfs", str(wp)])

        # Deny writes within writable regions: re-mount as read-only
        deny_writes = policy.resolved_deny_write_paths(working_dir)
        for dp in deny_writes:
            if dp.exists():
                args.extend(["--ro-bind", str(dp), str(dp)])
            elif dp.parent.exists():
                # Block creation of nonexistent deny targets by binding
                # /dev/null over them
                args.extend(["--ro-bind", "/dev/null", str(dp)])

        # Hide denied read paths with empty tmpfs overlays
        deny_reads = policy.resolved_deny_read_paths()
        for drp in deny_reads:
            if drp.is_dir():
                args.extend(["--tmpfs", str(drp)])
            elif drp.is_file():
                args.extend(["--ro-bind", "/dev/null", str(drp)])

        # Standard device nodes and proc
        args.extend(["--dev", "/dev"])
        args.extend(["--proc", "/proc"])

        return args

    def is_available(self) -> bool:
        from agentic_cli.tools.shell.os_sandbox.detect import (
            detect_sandbox_capabilities,
        )

        return detect_sandbox_capabilities().bubblewrap_available

    @property
    def sandbox_type(self) -> str:
        return "bubblewrap"
