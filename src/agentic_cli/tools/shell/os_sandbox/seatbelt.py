"""macOS sandbox using sandbox-exec with Seatbelt SBPL profiles.

Wraps commands via:
    sandbox-exec -p '<profile>' /bin/bash -c '<command>'

The profile is generated dynamically based on the OSSandboxPolicy.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from agentic_cli.tools.shell.os_sandbox.base import OSSandbox, OSSandboxResult
from agentic_cli.tools.shell.os_sandbox.seatbelt_profile import (
    generate_seatbelt_profile,
)

if TYPE_CHECKING:
    from agentic_cli.tools.shell.os_sandbox.policy import OSSandboxPolicy


def _escape_single_quotes(s: str) -> str:
    """Escape single quotes for embedding in a single-quoted shell string.

    In shell, 'it'\''s' produces: it's
    We end the single-quoted string, add an escaped quote, and restart.

    Args:
        s: String to escape.

    Returns:
        Escaped string safe for single-quoting.
    """
    return s.replace("'", "'\\''")


class SeatbeltSandbox(OSSandbox):
    """macOS sandbox using sandbox-exec with dynamically generated SBPL profiles."""

    def wrap_shell_command(
        self,
        command: str,
        working_dir: Path,
        policy: OSSandboxPolicy,
    ) -> OSSandboxResult:
        """Wrap a shell command with sandbox-exec.

        Generates an SBPL profile from the policy and wraps the command as:
            sandbox-exec -p '<profile>' /bin/bash -c '<command>'

        Args:
            command: The raw shell command.
            working_dir: Working directory for execution.
            policy: Sandbox policy.

        Returns:
            OSSandboxResult with the wrapped command.
        """
        try:
            profile = self._build_profile(working_dir, policy)
            escaped_profile = _escape_single_quotes(profile)
            escaped_command = _escape_single_quotes(command)
            wrapped = (
                f"sandbox-exec -p '{escaped_profile}' "
                f"/bin/bash -c '{escaped_command}'"
            )
            return OSSandboxResult(command=wrapped, sandbox_type="seatbelt")
        except Exception as e:
            return OSSandboxResult(
                command=command,
                success=False,
                error=f"Failed to generate Seatbelt profile: {e}",
                sandbox_type="seatbelt",
            )

    def wrap_python_command(
        self,
        python_args: list[str],
        working_dir: Path,
        policy: OSSandboxPolicy,
    ) -> OSSandboxResult:
        """Wrap a Python command with sandbox-exec.

        For Python, we invoke the interpreter directly (not via /bin/bash)
        to preserve stdin piping:
            sandbox-exec -p '<profile>' /path/to/python -c '<script>'

        Args:
            python_args: e.g. [sys.executable, "-c", script]
            working_dir: Working directory.
            policy: Sandbox policy.

        Returns:
            OSSandboxResult with the wrapped command.
        """
        try:
            profile = self._build_profile(working_dir, policy)
            escaped_profile = _escape_single_quotes(profile)
            args_str = shlex.join(python_args)
            wrapped = f"sandbox-exec -p '{escaped_profile}' {args_str}"
            return OSSandboxResult(command=wrapped, sandbox_type="seatbelt")
        except Exception as e:
            return OSSandboxResult(
                command=shlex.join(python_args),
                success=False,
                error=f"Failed to generate Seatbelt profile: {e}",
                sandbox_type="seatbelt",
            )

    def _build_profile(self, working_dir: Path, policy: OSSandboxPolicy) -> str:
        """Build the SBPL profile from policy.

        Args:
            working_dir: Working directory (always writable).
            policy: Sandbox policy with path lists.

        Returns:
            SBPL profile string.
        """
        return generate_seatbelt_profile(
            writable_paths=policy.resolved_writable_paths(working_dir),
            deny_write_paths=policy.resolved_deny_write_paths(working_dir),
            readable_paths=policy.resolved_readable_paths(),
            deny_read_paths=policy.resolved_deny_read_paths(),
            allow_network=policy.allow_network,
        )

    def is_available(self) -> bool:
        from agentic_cli.tools.shell.os_sandbox.detect import (
            detect_sandbox_capabilities,
        )

        return detect_sandbox_capabilities().seatbelt_available

    @property
    def sandbox_type(self) -> str:
        return "seatbelt"
