"""Shared file utilities — atomic writes, filename sanitization."""

import json
import os
import re
import secrets
import stat
import tempfile
from pathlib import Path
from typing import Any


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename.

    Replaces any character that isn't alphanumeric, hyphen, or underscore with underscore.
    """
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


#: The most patterns one brace pattern may stand for.
MAX_BRACE_ALTERNATIVES = 100


def expand_braces(pattern: str) -> list[str]:
    """The patterns a brace pattern stands for, as a shell expands it.

    ``*.{js,ts}`` is ``*.js`` and ``*.ts``; braces nest (``{a,b{1,2}}``). A
    brace with no comma inside (``{x}``), one never closed, and one inside a
    ``[...]`` class are literal. An empty alternative names no file and is
    dropped (``{a,}`` is ``a``), as are duplicates, the first one kept.

    Raises:
        ValueError: the pattern stands for more than
            ``MAX_BRACE_ALTERNATIVES`` patterns (duplicates and empty ones
            count: they cost the same work), or for none.
    """
    expanded: list[str] = []
    produced = 0  # patterns finished, duplicates and empty ones included
    pending = [pattern]
    while pending:
        current = pending.pop(0)
        group = _first_brace_group(current)
        if group is None:
            produced += 1
            if current and current not in expanded:
                expanded.append(current)
            continue
        start, end, commas = group
        # Each pending pattern stands for at least one, so this is a lower
        # bound of the total: checked before the group's patterns are built.
        if produced + len(pending) + len(commas) + 1 > MAX_BRACE_ALTERNATIVES:
            raise ValueError(
                f"Pattern stands for more than {MAX_BRACE_ALTERNATIVES} alternatives: {pattern!r}"
            )
        bounds = [start, *commas, end]
        pending[:0] = [
            current[:start] + current[a + 1:b] + current[end + 1:]
            for a, b in zip(bounds, bounds[1:])
        ]
    if not expanded:
        raise ValueError(f"Pattern names no file: {pattern!r}")
    return expanded


def _first_brace_group(pattern: str) -> tuple[int, int, list[int]] | None:
    """The ``{…,…}`` group that starts first in ``pattern``: where it starts
    and ends, and where its top-level commas are; None when there is none.

    One pass with a stack of open braces, so unclosed ones cost nothing more.
    """
    found: tuple[int, int, list[int]] | None = None
    open_braces: list[tuple[int, list[int]]] = []  # (start, its commas)
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == "[":
            close = pattern.find("]", i + 2)  # "[]...]" holds a "]"
            if close != -1:
                i = close + 1
                continue
        elif char == "{":
            open_braces.append((i, []))
        elif char == "," and open_braces:
            open_braces[-1][1].append(i)
        elif char == "}" and open_braces:
            start, commas = open_braces.pop()
            if commas and (found is None or start < found[0]):
                found = (start, i, commas)
        i += 1
    return found


def glob_pattern_escapes_root(pattern: str) -> bool:
    """True if a glob pattern would search outside its base directory.

    Rejects absolute patterns and any pattern containing a ``..`` path
    component, in any of the patterns its braces stand for (``{x,../*}``).
    ``pathlib.Path.glob`` does not normalize ``..``, so a pattern like
    ``../*`` escapes the base directory even though the permission engine
    only authorized that base. Callers should reject such patterns before
    globbing. A pattern that stands for too many patterns counts as one that
    escapes.
    """
    try:
        patterns = expand_braces(pattern)
    except ValueError:
        return True
    return any(
        os.path.isabs(p) or ".." in re.split(r"[\\/]+", p) for p in patterns
    )


def path_is_within(path: Path, root: Path) -> bool:
    """True if ``path`` resolves to a location inside ``root``.

    Both operands are fully resolved (following symlinks) before the check, so
    a symlink under ``root`` that points outside is correctly rejected. Returns
    False on any resolution error (loop, permission) — fail closed.
    """
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError, RuntimeError):
        return False


def open_dir_no_follow(name: str, *, dir_fd: int, create: bool = False) -> int:
    """Open the directory ``name`` inside ``dir_fd`` without following a symlink.

    For a directory another party can replace with a symlink at any moment
    (a sandbox kernel's ``inputs``/``outputs``): open it once with this and do
    every later step relative to the descriptor, so a swap changes the path,
    not the directory being read or written. ``create`` makes it (default
    mode) when it is missing.

    Raises OSError when ``name`` is a symlink (dangling or not), not a
    directory, or missing (without ``create``).
    """
    if create:
        try:
            os.mkdir(name, dir_fd=dir_fd)
        except FileExistsError:
            pass
    return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dir_fd)


def copy_regular_file_no_follow(
    src: str | os.PathLike[str],
    dst: str | os.PathLike[str],
    *,
    src_dir_fd: int | None = None,
    dst_dir_fd: int | None = None,
) -> None:
    """Copy a regular file src -> dst without following symlinks at either end.

    Source: opened O_RDONLY | O_NOFOLLOW | O_NONBLOCK (final component); fstat
    must be a regular file (rejects symlink / dir / FIFO / socket / device).
    O_NOFOLLOW closes the check->open TOCTOU on the final component; O_NONBLOCK
    ensures a FIFO/device source fails fast instead of blocking the open
    (regular files ignore O_NONBLOCK for reads).
    Dest: written to a private (0600) temp file beside dst, fsync'd, then
    os.replace()'d over dst — a pre-planted symlink at dst is replaced, not
    written through.

    O_NOFOLLOW guards only the final component, so a directory another
    party can swap for a symlink (a sandbox's ``inputs``/``outputs``) must be
    opened first and passed as a descriptor: ``src_dir_fd`` makes ``src``
    relative to it, and ``dst_dir_fd`` makes ``dst`` a file name in it (a
    bare name, no separator), where the temp file is created and replaced.
    The copy then reads and writes the directories that were opened, whatever
    their paths have become since.

    Raises OSError (source open incl. ELOOP for a final-component symlink) or
    ValueError (source not a regular file; with ``dst_dir_fd``, a ``dst``
    that is not a file name). POSIX (macOS/Linux).
    """
    dst = Path(dst)
    if dst_dir_fd is not None and (str(dst) != dst.name or dst.name in ("", ".", "..")):
        raise ValueError(f"not a file name: {str(dst)!r}")
    fd = os.open(src, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=src_dir_fd)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"not a regular file: {src}")
        if dst_dir_fd is None:
            dst.parent.mkdir(parents=True, exist_ok=True)
        tfd, tmp_path = _open_temp_beside(dst, mode=0o600, dir_fd=dst_dir_fd)
        try:
            with os.fdopen(tfd, "wb") as out:
                while True:
                    chunk = os.read(fd, 1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp_path, dst, src_dir_fd=dst_dir_fd, dst_dir_fd=dst_dir_fd)
        except BaseException:
            try:
                os.unlink(tmp_path, dir_fd=dst_dir_fd)
            except FileNotFoundError:
                pass
            raise
    finally:
        os.close(fd)


def _fchmod(fd: int, mode: int) -> None:
    """``os.fchmod(fd, mode)`` where the platform has it.

    Python on Windows has none before 3.13, and there the permission bits
    mean nothing but read-only: the mode is then left as it is.
    """
    fchmod = getattr(os, "fchmod", None)
    if fchmod is not None:
        fchmod(fd, mode)


def make_private_file(path: Path, *, tighten: bool = True) -> None:
    """Make ``path`` an owner-only file (0600) before a library opens it.

    For a file that holds what the user said, such as a session database,
    which its library would create with the process's umask (0644 under
    022). A missing file is created empty: SQLite takes an empty file for an
    empty database. A missing directory is created owner-only (0700); one
    that exists is left as it is, since it may be the user's own. An
    existing file is tightened, dropping its group and other bits (one an
    older version created 0644), unless ``tighten`` is False: a file at a
    location the user names may be shared on purpose. A symlink, and what
    it points at, are left alone, as is a file this process may not change
    (and every file where the platform has no ``fchmod``: Windows' Python
    before 3.13).
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow, 0o600)
    except FileExistsError:
        if not tighten:
            return
    else:
        os.close(fd)
        return
    try:
        fd = os.open(path, os.O_RDONLY | nofollow | getattr(os, "O_NONBLOCK", 0))
    except OSError:  # a symlink (ELOOP), or nothing this process may open
        return
    try:
        status = os.fstat(fd)
        mode = stat.S_IMODE(status.st_mode)
        if stat.S_ISREG(status.st_mode) and mode & 0o077:
            _fchmod(fd, mode & 0o700)
    except OSError:  # not this process's file
        pass
    finally:
        os.close(fd)


def _open_temp_beside(
    path: Path, *, mode: int = 0o666, dir_fd: int | None = None
) -> tuple[int, Path]:
    """Create a uniquely-named temp file beside ``path``, opened for writing.

    ``mode`` less the umask (the kernel applies it): the default 0o666 is
    what a plain ``open()`` gives a new file, where ``mkstemp`` would force
    0600. With ``dir_fd``, ``path`` is relative to that directory."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    for _ in range(100):
        tmp_path = path.parent / f".{path.name}.{secrets.token_hex(6)}.tmp"
        try:
            return os.open(tmp_path, flags, mode, dir_fd=dir_fd), tmp_path
        except FileExistsError:
            continue
    raise FileExistsError(f"could not create a temp file beside {path}")


def _atomic_write(path: Path, content: str | bytes, *, preserve_mode: bool = False) -> None:
    """Write content to a file atomically and durably.

    Writes to a uniquely-named temp file in the same directory, flushes and
    fsyncs it, then atomically renames it over the target. Notes:
    - Unique temp name so two concurrent writers can't truncate each other's
      temp file the way a fixed ``.tmp`` name allows.
    - fsync before the rename so a crash can't persist the rename ahead of the
      data and leave a torn/empty file in place of the previously-good one.
    - Explicit UTF-8 for text, so output doesn't depend on the locale (a
      C/POSIX locale would otherwise raise UnicodeEncodeError on non-ASCII
      content).
    - Permissions: by default the file is private (0600), right for the
      framework's own state. ``preserve_mode=True`` is for files the user
      owns: an existing file keeps its permission bits and a new one gets the
      process default, as with a plain write. Setuid/setgid are not carried
      over (the kernel clears them when a non-root process writes a file).
      Where the platform has no ``fchmod`` (Windows' Python before 3.13,
      whose bits mean nothing), the bits are not copied.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = None
    if preserve_mode:
        try:
            existing = stat.S_IMODE(os.stat(path).st_mode) & 0o777
        except FileNotFoundError:
            pass
        fd, tmp_path = _open_temp_beside(path)
    else:
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        tmp_path = Path(tmp_name)
    data = content.encode("utf-8") if isinstance(content, str) else content
    try:
        with os.fdopen(fd, "wb") as f:
            if existing is not None:
                _fchmod(f.fileno(), existing)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Write JSON data to a file atomically."""
    _atomic_write(path, json.dumps(data, indent=indent))


def atomic_write_text(path: Path, content: str, *, preserve_mode: bool = False) -> None:
    """Write text to a file atomically.

    The file is private (0600) unless ``preserve_mode``: then an existing
    file keeps its permission bits and a new one gets the process default.
    """
    _atomic_write(path, content, preserve_mode=preserve_mode)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes to a file atomically. The file is private (0600)."""
    _atomic_write(path, data)
