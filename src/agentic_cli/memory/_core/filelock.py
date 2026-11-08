"""One process at a time in a store's folder.

Several processes can open one store: every CLI session of an application
shares the user's memory store and knowledge base. Each change is a read of
the store's files, the change, and a rewrite of them, so two processes doing
that at once, or one working from what it read earlier, write over each
other's changes. A store holds this lock (an ``flock`` on a file in its
folder) around each read-and-change, and reads again first whatever another
process wrote (``file_stamp``).

``flock`` is POSIX. Where it is unavailable the lock is a no-op and the
store keeps only its own threads apart, as before.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - not POSIX
    fcntl = None  # type: ignore[assignment]


@contextmanager
def folder_lock(
    path: Path, *, timeout: float | None = None, on_wait: Callable[[], None] | None = None
) -> Iterator[None]:
    """Hold an exclusive lock on ``path`` (created private, 0600) for the block.

    ``flock`` belongs to the open file, so the lock is not re-entrant: a
    holder must not take it again, even in the same process. A symlink at
    ``path`` is not followed (``OSError``): opening it would create, or
    open for writing, the file it points to.

    With ``timeout``, a lock another process holds is waited for that many
    seconds at most (``on_wait`` is called once, as the wait starts), then
    ``TimeoutError`` is raised.
    """
    if fcntl is None:  # pragma: no cover - not POSIX
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        if timeout is None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        else:
            _lock_within(fd, path, timeout, on_wait)
        yield
    finally:
        os.close(fd)  # closing the file releases the lock


def _lock_within(fd: int, path: Path, timeout: float, on_wait: Callable[[], None] | None) -> None:
    """Take the lock on ``fd`` without blocking, trying again until ``timeout``."""
    deadline = time.monotonic() + timeout
    waited = False
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            pass
        if not waited and on_wait is not None:
            on_wait()
        waited = True
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{path} stayed locked for {timeout:g} s")
        time.sleep(0.05)


def file_stamp(path: Path) -> tuple[int, int, int] | None:
    """What identifies ``path``'s current contents: inode, size, mtime.

    An atomic write replaces the file, so its inode changes; None when the
    file does not exist.
    """
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return (st.st_ino, st.st_size, st.st_mtime_ns)
