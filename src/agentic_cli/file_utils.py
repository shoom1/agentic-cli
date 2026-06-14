"""Shared file utilities — atomic writes, file locking, filename sanitization."""

import fcntl
import json
import os
import tempfile
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
    with open(lock_path, "w") as fd:
        if timeout is None:
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
        fcntl.flock(fd, fcntl.LOCK_UN)


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename.

    Replaces any character that isn't alphanumeric, hyphen, or underscore with underscore.
    """
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def _atomic_write(path: Path, content: str) -> None:
    """Write content to a file atomically and durably.

    Writes to a uniquely-named temp file in the same directory, flushes and
    fsyncs it, then atomically renames it over the target. Notes:
    - Unique temp name (tempfile.mkstemp) so two concurrent writers can't
      truncate each other's temp file the way a fixed ``.tmp`` name allows.
    - fsync before the rename so a crash can't persist the rename ahead of the
      data and leave a torn/empty file in place of the previously-good one.
    - Explicit UTF-8 so output doesn't depend on the locale (a C/POSIX locale
      would otherwise raise UnicodeEncodeError on non-ASCII content).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Write JSON data to a file atomically."""
    _atomic_write(path, json.dumps(data, indent=indent))


def atomic_write_text(path: Path, content: str) -> None:
    """Write text to a file atomically."""
    _atomic_write(path, content)
