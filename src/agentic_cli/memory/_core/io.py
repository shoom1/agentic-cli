"""Reads, appends and atomic writes for the memory package's own files.

The files are private (0600): they hold the package's state, not files a user
edits. Same behavior as ``agentic_cli.file_utils`` without ``preserve_mode``;
the package keeps its own copy so it imports nothing from agentic-cli.

A project can ship its knowledge base, and git keeps symlinks, so a name in a
store's folder may be a symlink to a file elsewhere, a FIFO (opening one
blocks until the other end appears) or a link to a device such as
``/dev/zero`` (reading it never ends). A file is therefore read or appended
to only when it is a regular file, never through a symlink, and an atomic
write replaces whatever the name held (a link included) instead of writing
through it. Only the last component is checked: the caller makes sure the
folders on the way are its own.
"""

from __future__ import annotations

import errno
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, BinaryIO

from agentic_cli.memory._core.log import get_logger

logger = get_logger("agentic_cli.memory")

_NO_FOLLOW = getattr(os, "O_NOFOLLOW", 0)
_NON_BLOCK = getattr(os, "O_NONBLOCK", 0)
# What os.open answers for a name that is not a regular file: a symlink under
# O_NOFOLLOW (ELOOP), a FIFO without a reader under O_NONBLOCK|O_WRONLY
# (ENXIO), a folder opened for writing (EISDIR).
_REFUSED_ERRNOS = {errno.ELOOP, errno.ENXIO, errno.EISDIR}


class NotARegularFile(OSError):
    """The name is a symbolic link, or not a regular file, so it is not opened."""


def _open(path: Path, flags: int, mode: int = 0o600) -> int:
    """``os.open`` that never follows a final symlink nor waits on a FIFO."""
    try:
        fd = os.open(path, flags | _NO_FOLLOW | _NON_BLOCK, mode)
    except OSError as e:
        if e.errno in _REFUSED_ERRNOS:
            raise NotARegularFile(e.errno, "a symbolic link or not a regular file", str(path)) from None
        raise
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise NotARegularFile(errno.EINVAL, "not a regular file", str(path))
    except BaseException:
        os.close(fd)
        raise
    return fd


def open_regular_file(path: Path) -> BinaryIO:
    """``path`` opened for reading in binary mode, if it is a regular file.

    Raises:
        FileNotFoundError: nothing is at ``path``.
        NotARegularFile: ``path`` is a symlink or not a regular file.
        OSError: it cannot be opened.
    """
    return os.fdopen(_open(path, os.O_RDONLY), "rb")


def read_regular_file(path: Path, *, limit: int | None = None) -> bytes:
    """The bytes of ``path``, if it is a regular file (see :func:`open_regular_file`).

    Raises:
        OSError: as :func:`open_regular_file`, or the file holds more than
            ``limit`` bytes.
    """
    with open_regular_file(path) as f:
        data = f.read() if limit is None else f.read(limit + 1)
    if limit is not None and len(data) > limit:
        raise OSError(errno.EFBIG, f"more than {limit} bytes", str(path))
    return data


def open_if_regular(path: Path) -> BinaryIO | None:
    """``path`` opened as :func:`open_regular_file` does, or None when nothing
    is there or it is refused.

    A refused name (a symlink, a FIFO, a folder) counts as no file, with a
    warning (``memory_file_refused``); the next atomic write replaces it.
    Other errors are raised.
    """
    try:
        return open_regular_file(path)
    except FileNotFoundError:
        return None
    except NotARegularFile as e:
        logger.warning("memory_file_refused", path=str(path), error=str(e))
        return None


def read_if_regular(path: Path) -> bytes | None:
    """The bytes of ``path``, or None (see :func:`open_if_regular`)."""
    f = open_if_regular(path)
    if f is None:
        return None
    with f:
        return f.read()


def is_regular_file(path: Path) -> bool:
    """Whether ``path`` itself (not what a symlink there points to) is a regular file."""
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except OSError:
        return False


def append_text(path: Path, text: str) -> None:
    """Append ``text`` to ``path`` (created private, 0600), if it is a regular file.

    Raises:
        NotARegularFile: ``path`` is a symlink (dangling or not) or not a
            regular file; nothing is written or created.
        OSError: it could not be written.
    """
    fd = _open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    try:
        data = memoryview(text.encode("utf-8"))
        while data:
            data = data[os.write(fd, data):]
    finally:
        os.close(fd)


def _atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` to a temp file beside ``path``, fsync, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Write ``data`` as JSON, atomically."""
    _atomic_write(path, json.dumps(data, indent=indent).encode("utf-8"))


def atomic_write_text(path: Path, content: str) -> None:
    """Write ``content``, atomically."""
    _atomic_write(path, content.encode("utf-8"))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data``, atomically."""
    _atomic_write(path, data)
