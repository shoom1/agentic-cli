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
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - not POSIX
    fcntl = None  # type: ignore[assignment]


@contextmanager
def folder_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on ``path`` (created private, 0600) for the block.

    ``flock`` belongs to the open file, so the lock is not re-entrant: a
    holder must not take it again, even in the same process.
    """
    if fcntl is None:  # pragma: no cover - not POSIX
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)  # closing the file releases the lock


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
