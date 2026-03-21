"""OS-native sandboxing for shell and Python execution.

Provides platform-specific sandbox wrappers:
- macOS: Seatbelt (sandbox-exec) with SBPL profiles
- Linux: bubblewrap (bwrap) with namespace isolation
- Fallback: NoOpSandbox when tools are unavailable

Usage:
    from agentic_cli.tools.shell.os_sandbox import get_os_sandbox, OSSandboxPolicy

    sandbox = get_os_sandbox()
    policy = OSSandboxPolicy(writable_paths=["/tmp"])
    result = sandbox.wrap_shell_command("echo hello", Path.cwd(), policy)
    # result.command is now "sandbox-exec -p '...' /bin/bash -c 'echo hello'"
"""

from __future__ import annotations

import structlog

from agentic_cli.tools.shell.os_sandbox.base import (
    NoOpSandbox,
    OSSandbox,
    OSSandboxResult,
)
from agentic_cli.tools.shell.os_sandbox.detect import (
    SandboxCapabilities,
    detect_sandbox_capabilities,
)
from agentic_cli.tools.shell.os_sandbox.policy import OSSandboxPolicy

logger = structlog.get_logger(__name__)

_cached_sandbox: OSSandbox | None = None


def get_os_sandbox(force_noop: bool = False) -> OSSandbox:
    """Get the OS sandbox appropriate for the current platform.

    Factory function that returns:
    - SeatbeltSandbox on macOS if sandbox-exec is available
    - BubblewrapSandbox on Linux if bwrap is available
    - NoOpSandbox otherwise (with a logged warning)

    Results are cached for the process lifetime. Use force_noop=True
    to bypass caching and always return NoOpSandbox (for testing).

    Args:
        force_noop: If True, always return NoOpSandbox.

    Returns:
        An OSSandbox implementation.
    """
    if force_noop:
        return NoOpSandbox()

    global _cached_sandbox
    if _cached_sandbox is not None:
        return _cached_sandbox

    caps = detect_sandbox_capabilities()

    if caps.seatbelt_available:
        from agentic_cli.tools.shell.os_sandbox.seatbelt import SeatbeltSandbox

        _cached_sandbox = SeatbeltSandbox()
        logger.info("os_sandbox.initialized", sandbox_type="seatbelt")

    elif caps.bubblewrap_available:
        from agentic_cli.tools.shell.os_sandbox.bubblewrap import BubblewrapSandbox

        _cached_sandbox = BubblewrapSandbox()
        logger.info("os_sandbox.initialized", sandbox_type="bubblewrap")

    else:
        _cached_sandbox = NoOpSandbox()
        logger.warning(
            "os_sandbox.no_sandbox_available",
            platform=caps.platform,
            msg="No OS sandbox tool found; using no-op fallback",
        )

    return _cached_sandbox


def reset_cached_sandbox() -> None:
    """Reset the cached sandbox instance. Useful for testing."""
    global _cached_sandbox
    _cached_sandbox = None


__all__ = [
    "get_os_sandbox",
    "reset_cached_sandbox",
    "OSSandbox",
    "OSSandboxResult",
    "OSSandboxPolicy",
    "NoOpSandbox",
    "SandboxCapabilities",
    "detect_sandbox_capabilities",
]
