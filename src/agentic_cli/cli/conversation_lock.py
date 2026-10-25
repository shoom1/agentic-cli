"""One process at a time in a stored conversation.

Two CLIs on one conversation lose a turn: Deep Agents keeps only one of two
runs that start from the same checkpoint, and ADK fails the second writer's
turn. So the CLI holds a lock on the conversation it has open, and refuses
one another process holds (``BaseCLIApp``: ``--resume <id>``, ``/resume``,
``--continue``).

A lock is an ``flock`` on a private file under ``<sessions_dir>/locks/``,
named by a hash of the conversation's key (the backend, the store, the app,
the user and the id: the same id in another backend is another
conversation). The OS releases it when the process ends, however it ends,
so a crash leaves nothing held. The holder writes its process id into the
file, for the refusal to name. ``flock`` is POSIX; where it is unavailable
nothing is locked, as before. It guards one machine: a shared Postgres store
is not covered.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - not POSIX
    fcntl = None  # type: ignore[assignment]

#: What a holder's process id reads as when its file holds none.
UNKNOWN_HOLDER = 0


class ConversationLock:
    """The lock on the one conversation this process has open.

    Args:
        folder: Where the lock files live.
    """

    def __init__(self, folder: Path) -> None:
        self._folder = folder
        self._key: tuple | None = None
        self._fd: int | None = None

    def take(self, key: tuple) -> int | None:
        """Hold ``key``'s conversation, letting go of the one held before.

        Returns:
            None once it is held (already, or now); the process id of
            another process that holds it (``UNKNOWN_HOLDER`` when it wrote
            none), and the conversation held before stays held.
        """
        taken = self.acquire(key)
        if isinstance(taken, int):
            return taken
        self.commit(taken)
        return None

    def acquire(self, key: tuple) -> "Taken | int":
        """Hold ``key``'s conversation as well as the one held now.

        A switch that may still fail holds both: ``commit`` lets the one
        before go once it is done, ``abandon`` lets the new one go else.

        Returns:
            The new hold, or the process id of another process that holds it.
        """
        if fcntl is None or key == self._key:
            return Taken(key, None)
        self._folder.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path(key), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = _holder(fd)
            os.close(fd)
            return holder
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        return Taken(key, fd)

    def commit(self, taken: "Taken") -> None:
        """Keep ``taken`` alone: the one held before is let go."""
        if taken.fd is None:  # held already, or nothing locks here
            return
        self.release()
        self._key, self._fd = taken.key, taken.fd

    def abandon(self, taken: "Taken") -> None:
        """Let ``taken`` go: the switch did not happen."""
        if taken.fd is not None:
            os.close(taken.fd)

    def release(self) -> None:
        """Let go of the conversation held (closing the file releases the lock)."""
        if self._fd is not None:
            os.close(self._fd)
        self._key = self._fd = None

    def _path(self, key: tuple) -> Path:
        digest = hashlib.sha256(json.dumps(list(key)).encode()).hexdigest()[:32]
        return self._folder / f"{digest}.lock"


@dataclass(frozen=True, slots=True)
class Taken:
    """A conversation held before the switch to it is done (``acquire``)."""

    key: tuple
    fd: int | None


def _holder(fd: int) -> int:
    try:
        return int(os.pread(fd, 32, 0).decode().strip() or UNKNOWN_HOLDER)
    except (OSError, ValueError):
        return UNKNOWN_HOLDER
