"""Shared persistence utilities."""

import fcntl
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator


@contextmanager
def file_lock(path: Path) -> Generator[None, None, None]:
    """Acquire an exclusive file lock for cross-process safety.

    Uses a .lock file adjacent to the target path.
    The lock is advisory (relies on all writers using this utility).
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = open(lock_path, "w")  # noqa: SIM115
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
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
