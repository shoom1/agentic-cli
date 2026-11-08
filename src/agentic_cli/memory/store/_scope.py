"""One scope's folder: its memory files, their history, its lock (spec §5).

Every call lists the folder and reads again only the files whose inode, size
or mtime changed, so a person's edit is seen at the next call. A file that
cannot be used is a ``Problem`` and is never written, moved or deleted here.
A change holds the folder's lock (``.lock``) and replaces files atomically,
so a reader, which takes no lock, sees a file either before or after it.
"""

from __future__ import annotations

import contextlib
import errno
import os
import stat
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agentic_cli.memory._core.filelock import folder_lock
from agentic_cli.memory._core.io import atomic_write_text, read_regular_file
from agentic_cli.memory._core.log import get_logger
from agentic_cli.memory.store._files import (
    FILE_MAX_BYTES,
    FieldError,
    FormatError,
    file_name,
    memory_fields,
    memory_from,
    parse,
    render,
    valid_id,
)
from agentic_cli.memory.store.models import Memory, MemoryBusy, Problem, Scope, UnsafeMemoryStore

logger = get_logger("agentic_cli.memory.store")

LOCK_TIMEOUT_S = 10.0
HISTORY_DAYS = 30
HISTORY_KEEP = 20
INBOX = ".inbox"
HISTORY = ".history"
LOCK = ".lock"
PROJECT_NOTE = "project-path.txt"
_STAMP = "%Y%m%dT%H%M%S.%fZ"


@dataclass(frozen=True)
class Entry:
    """A memory as read, with the keys the store does not know and the file's mtime."""

    memory: Memory
    extra: dict[str, str]
    mtime_ns: int


def resolve_duplicates(entries: list[Entry]) -> tuple[dict[str, Entry], list[Problem]]:
    """Each id's memory, and a problem for every other file claiming it.

    The memory is the one with the later ``updated``, then the later mtime.
    Ids that differ only by case are one id: on a case-insensitive file
    system they share a history folder and file names.
    """
    by_id: dict[str, list[Entry]] = {}
    for entry in entries:
        by_id.setdefault(entry.memory.id.casefold(), []).append(entry)
    winners: dict[str, Entry] = {}
    problems: list[Problem] = []
    for group in by_id.values():
        group.sort(key=lambda e: (e.memory.updated, e.mtime_ns), reverse=True)
        winner = group[0]
        winners[winner.memory.id] = winner
        problems += [
            Problem(other.memory.path, "duplicate_id",
                    f"the id {other.memory.id} is also {winner.memory.path.name}'s")
            for other in group[1:]
        ]
    return winners, problems


def _private_dirs(path: Path) -> None:
    """Create ``path`` and any missing folder above it, each 0700."""
    missing: list[Path] = []
    current = path
    while not os.path.lexists(current):
        missing.append(current)
        current = current.parent
    for folder in reversed(missing):
        with contextlib.suppress(FileExistsError):
            folder.mkdir(mode=0o700)


class ScopeFolder:
    """The folder of one scope (``user`` or ``project``).

    Args:
        path: The folder; it is created when first written.
        scope: Which scope it is.
        note: A file and its text written when the folder is created (the
            project scope's ``project-path.txt``).
    """

    def __init__(self, path: Path, scope: Scope, *, note: tuple[Path, str] | None = None) -> None:
        self.path = path
        self.scope = scope
        self._note = note
        self._cache: dict[str, tuple[tuple[int, int, int], Entry | Problem]] = {}

    def check(self) -> None:
        """Refuse a folder, ``.inbox`` or ``.history`` that is a link or not a folder.

        Raises:
            UnsafeMemoryStore: one is.
        """
        for path in (self.path, self.path / INBOX, self.path / HISTORY):
            try:
                mode = os.lstat(path).st_mode
            except FileNotFoundError:
                continue
            if not stat.S_ISDIR(mode):
                raise UnsafeMemoryStore(f"{path} is a symbolic link or not a folder")

    def ensure(self, sub: str | None = None) -> Path:
        """The folder (or ``sub`` inside it), created 0700 when missing."""
        created = not os.path.lexists(self.path)
        _private_dirs(self.path)
        if created and self._note is not None:
            note, text = self._note
            atomic_write_text(note, text + "\n")
        target = self.path / sub if sub else self.path
        _private_dirs(target)
        self.check()
        return target

    @contextlib.contextmanager
    def lock(self) -> Iterator[None]:
        """Hold the folder's lock for the block.

        Raises:
            MemoryBusy: another process held it for ``LOCK_TIMEOUT_S``.
        """
        self.ensure()
        with contextlib.ExitStack() as stack:
            try:
                stack.enter_context(folder_lock(self.path / LOCK, timeout=LOCK_TIMEOUT_S))
            except TimeoutError as e:
                raise MemoryBusy(
                    f"memory is busy: {self.path} stayed locked for {LOCK_TIMEOUT_S:g} s; try again"
                ) from e
            yield

    def target(self, description: str, memory_id: str) -> Path:
        """Where a memory with this description and id is written."""
        return self.path / file_name(description, memory_id)

    def scan(self) -> tuple[list[Entry], list[Problem]]:
        """Every memory file directly in the folder, and every problem."""
        try:
            listing = list(os.scandir(self.path))
        except FileNotFoundError:
            self._cache.clear()
            return [], []
        entries: list[Entry] = []
        problems: list[Problem] = []
        seen: set[str] = set()
        for item in listing:
            name = item.name
            if name.startswith(".") or not name.endswith(".md"):
                continue
            path = Path(item.path)
            try:
                st = os.lstat(path)
            except FileNotFoundError:
                continue
            seen.add(name)
            key = (st.st_ino, st.st_size, st.st_mtime_ns)
            cached = self._cache.get(name)
            if cached is None or cached[0] != key:
                cached = (key, self._read(path, st))
                self._cache[name] = cached
                if isinstance(cached[1], Problem):
                    logger.warning("memory_file_problem", path=str(path),
                                   kind=cached[1].kind, detail=cached[1].detail)
            result = cached[1]
            if isinstance(result, Entry):
                entries.append(result)
            else:
                problems.append(result)
        for name in [n for n in self._cache if n not in seen]:
            del self._cache[name]
        return entries, problems

    def _read(self, path: Path, st: os.stat_result) -> Entry | Problem:
        if not stat.S_ISREG(st.st_mode):
            return Problem(path, "unreadable", "a symbolic link or not a regular file")
        try:
            data = read_regular_file(path, limit=FILE_MAX_BYTES)
        except OSError as e:
            if e.errno == errno.EFBIG:
                return Problem(path, "too_large", f"more than {FILE_MAX_BYTES} bytes")
            return Problem(path, "unreadable", str(e))
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return Problem(path, "unreadable", "not UTF-8 text")
        try:
            fields, content = parse(text)
        except FormatError as e:
            return Problem(path, "bad_frontmatter", str(e))
        mtime = datetime.fromtimestamp(st.st_mtime, timezone.utc)
        try:
            memory, extra = memory_from(fields, content, scope=self.scope, path=path, mtime=mtime)
        except FieldError as e:
            claimed = fields.get("id")
            return Problem(path, "bad_field", str(e), memory_id=claimed if valid_id(claimed) else None)
        return Entry(memory, extra, st.st_mtime_ns)

    def write(self, memory: Memory, extra: Mapping[str, str]) -> None:
        """Write ``memory`` to ``memory.path``, atomically. Hold the lock."""
        self.ensure()
        atomic_write_text(memory.path, render(memory_fields(memory, extra), memory.content))

    def keep_version(self, entry: Entry, saved: datetime) -> Path:
        """Copy ``entry`` into ``.history/<id>/<stamp>.md``. Hold the lock."""
        folder = self.ensure(f"{HISTORY}/{entry.memory.id}")
        moment = saved.astimezone(timezone.utc)
        path = folder / f"{moment.strftime(_STAMP)}.md"
        while os.path.lexists(path):
            moment += timedelta(microseconds=1)
            path = folder / f"{moment.strftime(_STAMP)}.md"
        atomic_write_text(path, render(memory_fields(entry.memory, entry.extra), entry.memory.content))
        return path

    def history_ids(self) -> set[str]:
        """The ids that have a history folder."""
        root = self.path / HISTORY
        try:
            return {
                item.name for item in os.scandir(root)
                if item.is_dir(follow_symlinks=False) and valid_id(item.name)
            }
        except FileNotFoundError:
            return set()

    def versions(self, memory_id: str | None = None) -> list[tuple[str, datetime, Path]]:
        """``(memory id, saved, file)`` of every version (of ``memory_id`` only, if given)."""
        ids = [memory_id] if memory_id is not None else sorted(self.history_ids())
        found: list[tuple[str, datetime, Path]] = []
        for one in ids:
            if not valid_id(one):
                continue
            folder = self.path / HISTORY / one
            try:
                items = list(os.scandir(folder))
            except (FileNotFoundError, NotADirectoryError):
                continue
            for item in items:
                if not item.name.endswith(".md") or not item.is_file(follow_symlinks=False):
                    continue
                try:
                    saved = datetime.strptime(item.name[:-3], _STAMP).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                found.append((one, saved, Path(item.path)))
        return found

    def read_version(self, path: Path) -> Entry | None:
        """A version file as an entry; None when it cannot be read."""
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            return None
        result = self._read(path, st)
        return result if isinstance(result, Entry) else None

    def prune_history(self, now: datetime, memory_id: str | None = None) -> None:
        """Drop versions older than ``HISTORY_DAYS`` and all but each memory's newest
        ``HISTORY_KEEP``. Hold the lock."""
        cutoff = now - timedelta(days=HISTORY_DAYS)
        by_id: dict[str, list[tuple[datetime, Path]]] = {}
        for one, saved, path in self.versions(memory_id):
            by_id.setdefault(one, []).append((saved, path))
        for one, versions in by_id.items():
            versions.sort(reverse=True)
            for index, (saved, path) in enumerate(versions):
                if index >= HISTORY_KEEP or saved < cutoff:
                    path.unlink(missing_ok=True)
            with contextlib.suppress(OSError):
                (self.path / HISTORY / one).rmdir()  # only when it is empty
