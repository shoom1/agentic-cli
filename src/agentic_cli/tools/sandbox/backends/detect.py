"""Detect an available container runtime (docker, then podman)."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache

from agentic_cli.logging import Loggers

logger = Loggers.tools()

_RUNTIMES = ("docker", "podman")


@dataclass(frozen=True)
class DockerAvailability:
    available: bool
    runtime: str
    detail: str


def _probe_daemon(exe: str) -> bool:
    """Return True if `<exe> info` succeeds (daemon reachable)."""
    try:
        proc = subprocess.run(
            [exe, "info"], capture_output=True, timeout=10, check=False,
        )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@lru_cache(maxsize=1)
def detect_docker() -> DockerAvailability:
    details: list[str] = []
    for exe in _RUNTIMES:
        if shutil.which(exe) is None:
            continue
        if _probe_daemon(exe):
            return DockerAvailability(True, exe, f"{exe} available")
        # CLI present but daemon down — keep probing the remaining runtimes
        # (the "docker, then podman" design) instead of giving up here.
        details.append(f"{exe} CLI found but daemon not reachable")
    if details:
        return DockerAvailability(False, "", "; ".join(details))
    return DockerAvailability(False, "", "docker/podman not found in PATH")


def docker_available() -> bool:
    return detect_docker().available


def clear_detection_cache() -> None:
    detect_docker.cache_clear()
