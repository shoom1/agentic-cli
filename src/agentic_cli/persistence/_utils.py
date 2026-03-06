"""Shared persistence utilities."""

import fcntl
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator


class FileLockTimeout(TimeoutError):
    """Raised when a file lock cannot be acquired within the timeout."""


@contextmanager
def file_lock(path: Path, timeout: float = 10.0) -> Generator[None, None, None]:
    """Acquire an exclusive file lock for cross-process safety.

    Uses a .lock file adjacent to the target path.
    The lock is advisory (relies on all writers using this utility).

    Args:
        path: The file path to lock.
        timeout: Maximum seconds to wait for the lock (default 10s).
                 Use 0 for non-blocking, None for indefinite (old behavior).

    Raises:
        FileLockTimeout: If the lock cannot be acquired within the timeout.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = open(lock_path, "w")  # noqa: SIM115
    try:
        if timeout is None:
            # Indefinite blocking (legacy behavior)
            fcntl.flock(fd, fcntl.LOCK_EX)
        else:
            deadline = time.monotonic() + timeout
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise FileLockTimeout(
                            f"Could not acquire lock on {lock_path} "
                            f"within {timeout}s"
                        )
                    time.sleep(0.05)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename.

    Replaces any character that isn't alphanumeric, hyphen, or underscore with underscore.
    """
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Write JSON data to a file atomically.

    Writes to a temporary file first, then renames to the target path.
    This prevents data corruption if the process crashes mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=indent))
    tmp_path.replace(path)


def atomic_write_text(path: Path, content: str) -> None:
    """Write text to a file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content)
    tmp_path.replace(path)
