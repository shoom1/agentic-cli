"""Detect available OS sandbox tools.

Checks for platform-specific sandbox tools (sandbox-exec on macOS,
bwrap on Linux) and caches the result for the process lifetime.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from functools import lru_cache

import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class SandboxCapabilities:
    """What OS sandbox tools are available on this system.

    Attributes:
        platform: sys.platform value (darwin, linux, win32).
        seatbelt_available: Whether macOS sandbox-exec is available.
        bubblewrap_available: Whether Linux bwrap is available.
    """

    platform: str
    seatbelt_available: bool
    bubblewrap_available: bool


@lru_cache(maxsize=1)
def detect_sandbox_capabilities() -> SandboxCapabilities:
    """Detect available OS sandbox tools.

    Returns cached result after first call. Logs warnings if
    expected tools are not found for the current platform.

    Returns:
        SandboxCapabilities describing what's available.
    """
    platform = sys.platform
    seatbelt = False
    bubblewrap = False

    if platform == "darwin":
        seatbelt = shutil.which("sandbox-exec") is not None
        if not seatbelt:
            logger.warning(
                "os_sandbox.seatbelt_unavailable",
                msg="sandbox-exec not found; OS sandboxing disabled on macOS",
            )

    elif platform == "linux":
        bubblewrap = shutil.which("bwrap") is not None
        if not bubblewrap:
            logger.warning(
                "os_sandbox.bubblewrap_unavailable",
                msg="bwrap not found; install bubblewrap for OS sandboxing",
            )

    elif platform == "win32":
        logger.info(
            "os_sandbox.windows_unsupported",
            msg="OS sandboxing not yet supported on Windows",
        )

    return SandboxCapabilities(
        platform=platform,
        seatbelt_available=seatbelt,
        bubblewrap_available=bubblewrap,
    )


def clear_detection_cache() -> None:
    """Clear the cached detection result. Useful for testing."""
    detect_sandbox_capabilities.cache_clear()
