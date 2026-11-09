"""The conversation the CLI's turns run in (``ConversationCore``).

A conversation lives in one CLI process at a time: two processes on one
conversation lose a turn (``cli/conversation_lock.py``). ``ConversationCore``
keeps what makes the conversation the app is in: its id, whether a turn ran
in it since it was opened (``turned``: its memories are proposed on leaving
it, or at exit, only then), and the lock this process holds on it, at its
``Place``: the backend and the store it is kept in, the app and the user (the
same id in another backend or store is another conversation).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_cli.cli.conversation_lock import ConversationLock, Taken


@dataclass(frozen=True, slots=True)
class Place:
    """Where a stored conversation is kept: what its lock is keyed by, beside its id.

    Attributes:
        backend: The orchestrator that keeps it (``adk``, ``deepagents``).
        store: The session store (``sqlite``, ``postgres``, ``memory``).
        app_name: The app's name.
        user: The user it belongs to.
        locks: The folder its lock file is in (``<sessions_dir>/locks``).
    """

    backend: str
    store: str
    app_name: str
    user: str | None
    locks: Path

    @classmethod
    def of(cls, settings: Any) -> "Place":
        """The place ``settings`` keep conversations in, as the workflow reads them."""
        orchestrator = getattr(settings, "orchestrator", "")
        return cls(
            backend=getattr(orchestrator, "value", orchestrator),
            store=settings.session_store,
            app_name=settings.app_name,
            user=settings.default_user,
            locks=settings.sessions_dir / "locks",
        )

    @property
    def locked(self) -> bool:
        """Whether its conversations are locked: the memory store's are this
        process's alone."""
        return self.store != "memory"

    def key(self, session_id: str) -> tuple:
        """What names ``session_id``'s conversation for its lock."""
        return (self.backend, self.store, self.app_name, self.user, session_id)


@dataclass(frozen=True, slots=True)
class Hold:
    """A conversation's lock taken beside the one held, before the switch to
    it is done (``ConversationCore.acquire``): ``commit`` keeps it, ``abandon``
    lets it go.

    Attributes:
        session_id: The conversation's id.
        place: Where it is kept.
        taken: The lock's own hold; None where nothing is locked (the memory
            store).
    """

    session_id: str
    place: Place
    taken: Taken | None


class ConversationCore:
    """The conversation the app is in: its id, ``turned``, and its lock.

    Args:
        session_id: The conversation's id.
        place: Where it is kept; an app sets it before it holds anything.
    """

    def __init__(self, session_id: str | None = None, place: Place | None = None) -> None:
        self.session_id = session_id
        #: A turn ran in it since it was opened: its memories are proposed on
        #: leaving it, or at exit, only then.
        self.turned = False
        self.place = place
        self._lock: ConversationLock | None = None

    def acquire(self, session_id: str, place: Place | None = None) -> Hold | int:
        """Hold ``session_id``'s conversation beside the one held now.

        Args:
            session_id: The conversation to hold.
            place: Where it is kept; the current conversation's place by default.

        Returns:
            The hold to ``commit`` or ``abandon``, or the process id of
            another process that has it open (``UNKNOWN_HOLDER`` when unknown).
        """
        place = self._at(place)
        if not place.locked:
            return Hold(session_id, place, None)
        taken = self._lock_for(place).acquire(place.key(session_id))
        if isinstance(taken, int):
            return taken
        return Hold(session_id, place, taken)

    def commit(self, hold: Hold) -> None:
        """Keep ``hold`` alone: the conversation held before is let go."""
        if hold.taken is not None:
            self._lock_for(hold.place).commit(hold.taken)
        elif self._lock is not None:  # the memory store: nothing to hold there
            self._lock.release()

    def abandon(self, hold: Hold) -> None:
        """Let ``hold`` go: the switch to it did not happen."""
        if hold.taken is not None and self._lock is not None:
            self._lock.abandon(hold.taken)

    def hold(self, session_id: str, place: Place | None = None) -> int | None:
        """Hold ``session_id``'s conversation, letting go of the one held before.

        Returns:
            None once it is held; else the process id of the one holding it
            (0 when unknown), and the conversation held before stays held.
        """
        hold = self.acquire(session_id, place)
        if isinstance(hold, int):
            return hold
        self.commit(hold)
        return None

    def release(self) -> None:
        """Let go of the conversation held (at exit)."""
        if self._lock is not None:
            self._lock.release()

    def _at(self, place: Place | None) -> Place:
        place = place or self.place
        if place is None:
            raise RuntimeError("The conversation's place is not known yet")
        return place

    def _lock_for(self, place: Place) -> ConversationLock:
        """The lock, made on first use in its place's folder (one for the
        process: every place's folder is ``<sessions_dir>/locks``)."""
        if self._lock is None:
            self._lock = ConversationLock(place.locks)
        return self._lock
