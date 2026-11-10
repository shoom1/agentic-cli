"""The conversation the CLI's turns run in (``ConversationCore``).

A conversation lives in one CLI process at a time: two processes on one
conversation lose a turn (``cli/conversation_lock.py``). ``ConversationCore``
keeps what makes the conversation the app is in: its id, whether a turn ran
in it since it was opened (``turned``: its memories are proposed on leaving
it, or at exit, only then), and the lock this process holds on it, at its
``Place``: the backend and the store it is kept in, the app and the user (the
same id in another backend or store is another conversation).

Every change of conversation goes through one protocol, ``switch``: ``/new``
(on the running backend or another: a conversation never changes backend),
``/resume`` (on the backend that keeps the conversation, which runs then),
the conversation a run starts in (``--continue``, ``--resume``, a new one),
and the same id in another place, while the workflow is down and its next
start builds on another backend or store.
So the conversation the turns write to is always the one held and loaded,
whichever way it changed and wherever a Ctrl+C lands.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_cli.cli.conversation_lock import ConversationLock, Taken
from agentic_cli.logging import Loggers

logger = Loggers.cli()


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
    def of(cls, settings: Any, *, backend: str | None = None) -> "Place":
        """The place ``settings`` keep conversations in, as the workflow reads
        them, on ``backend`` (the settings' orchestrator by default)."""
        orchestrator = getattr(settings, "orchestrator", "")
        return cls(
            backend=backend or getattr(orchestrator, "value", orchestrator),
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


@dataclass(frozen=True, slots=True)
class SwitchParts:
    """What a switch acts on besides the conversation: the app's parts as they are.

    Attributes:
        workflow: The workflow manager when it is ready, else None (nothing
            is loaded then: the next start, or the next turn, takes the id).
        processor: The message processor, whose task list is forgotten
            (``reset()``).
        tracker: The usage tracker, zeroed (``reset()``), or only its context
            for the same conversation in another place (``reset_context()``:
            the session's totals stay); None for none.
        propose: Proposes the memories of a conversation left, by its id.
        status: Draws the status line again.
    """

    workflow: Any
    processor: Any
    tracker: Any
    propose: Callable[[str], Awaitable[None]]
    status: Callable[[], None]


@dataclass(frozen=True, slots=True)
class Switch:
    """How a switch ended.

    Attributes:
        made: The app is in the target conversation now, holding it.
        holder: Not made: the process id of another process that has the
            target open (0 when unknown).
        gone: A stored target the workflow no longer holds or lists (deleted
            since it was read): not made; or made, when there was nowhere
            else to stay (a conversation on another backend, which runs
            already), and it starts empty.
        loaded: Made: whether the workflow found it stored (its
            ``load_session``); None when it was not asked (not ready).
    """

    made: bool
    holder: int | None = None
    gone: bool = False
    loaded: bool | None = None


class ConversationCore:
    """The conversation the app is in: its id, ``turned``, and its lock.

    Args:
        session_id: The conversation's id.
        place: Where it is kept; an app sets it before it holds anything.
        opened_empty: Nothing of it was stored when it was opened: an id this
            run made (an id given to the app is not, until a load finds it
            empty).
        deferred: Its id was made by this run: nothing holds it until its
            first write (``hold_to_write``).
    """

    def __init__(
        self,
        session_id: str | None = None,
        place: Place | None = None,
        *,
        opened_empty: bool = False,
        deferred: bool = False,
    ) -> None:
        self.session_id = session_id
        #: A turn ran in it since it was opened: its memories are proposed on
        #: leaving it, or at exit, only then.
        self.turned = False
        #: Nothing of it was stored when it was opened (``fresh``).
        self.opened_empty = opened_empty
        #: A turn of it ran on a ready workflow, which stores it: the app says
        #: so (a turn the workflow was not up for stores nothing).
        self.written = False
        #: Its id was made by this run, and nothing holds it yet: no store
        #: lists it before its first write, so no other process can open it,
        #: and it is held then (``hold_to_write``). A run in which no turn runs
        #: makes no lock file (a line typed while the workflow is down takes
        #: the hold before the start its turn retries, which might write). An
        #: id the app was given is held from the start.
        self.deferred = deferred
        self.place = place
        self._lock: ConversationLock | None = None

    @property
    def fresh(self) -> bool:
        """Nothing of it is stored: it began in this run as a new one (or a
        load found it empty), and no turn of it ran on a ready workflow.

        The one condition under which it may run on another backend: a
        conversation never changes backend once anything of it is stored,
        since its history is in that backend's store alone.
        """
        return self.opened_empty and not self.written

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
        elif self._lock is not None:  # nothing to hold (the memory store, or not yet)
            self._lock.release()

    def hold_to_write(self) -> int | None:
        """Hold the conversation before its first write, when its id was made
        by this run (``deferred``); the caller holds the turn lock.

        Returns:
            None once it is held (or was, or nothing holds one here); else the
            process id of another process that has it open (a collision of
            random ids), and nothing changed: the write must not happen.
        """
        if not self.deferred or self.session_id is None:
            return None
        taken = self.acquire(self.session_id)
        if isinstance(taken, int):
            return taken
        self.commit(taken)
        self.deferred = False
        return None

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
        taken = self.acquire(session_id, place)
        if isinstance(taken, int):
            return taken
        self.commit(taken)
        if session_id == self.session_id:
            self.deferred = False
        return None

    def release(self) -> None:
        """Let go of the conversation held (at exit)."""
        if self._lock is not None:
            self._lock.release()

    async def switch(
        self,
        to: str,
        parts: SwitchParts,
        *,
        place: Place | None = None,
        stored: bool = False,
        propose: bool = True,
        hold: Hold | None = None,
        generated: bool = False,
    ) -> Switch:
        """Make ``to`` the conversation the turns run in: the one protocol.

        The caller holds the turn lock, so no turn, job resume or
        ``/compact`` writes to either conversation meanwhile. In order:

        1. ``to``'s lock is taken beside the one held (or ``hold`` is, taken
           by the caller): one another process has open is refused;
        2. the memories of the conversation left are proposed, when a turn
           ran in it since it was opened and ``propose`` is on; it then
           counts as unused, so a switch that fails now proposes none again;
        3. the workflow, when it is ready, loads ``to``: a ``stored`` one it
           finds empty and no longer lists is gone (deleted since it was
           read), and the switch is not made (made already, with nowhere
           else to stay, it starts empty: ``gone`` says so);
        4. the switch is made: the lock on ``to`` is kept and the one before
           let go, ``to`` is the conversation (unused), and the task list
           and the usage counters start over; else ``to``'s lock is let go;
        5. the status line is drawn again.

        A switch is made once the workflow has the conversation: until then a
        Ctrl+C or a failure leaves the app in the conversation it was in,
        holding it alone, and a workflow that took ``to`` is given that
        conversation back (shielded, so a second Ctrl+C cannot stop it
        half way). A switch with no other conversation to stay in is made
        before the load, whatever the load says: to the conversation the app
        is already in (the one a run starts in), or to another place (a new
        conversation, or a stored one, on another backend, which runs
        already: the one left is not there).

        Args:
            to: The conversation's id.
            parts: The app's parts it acts on.
            place: Where ``to`` is kept: the current place by default; another
                one for a conversation on another backend or store. The same
                conversation in another place keeps the session's usage
                totals and forgets only the context.
            stored: ``to`` must be in the store (``/resume``); a new
                conversation's load finds nothing, as it should.
            propose: Propose the memories of the conversation left (not when
                the caller proposed them already, while the backend that
                holds them ran).
            hold: ``to``'s lock, already taken (``acquire``); the switch keeps
                it or lets it go.
            generated: ``to`` is an id this run made (``/new``): no store
                lists it, so nothing is held for it until its first write
                (``deferred``); the one left is let go once the switch is
                made. The conversation the app is in stays as it is: a
                ``deferred`` one is not held for a switch to itself either.

        Returns:
            How it ended: made, refused (``holder``), or gone.

        Raises:
            Whatever the workflow's load raises, and whatever drawing the
            status line raises once the switch is made.
        """
        left = self.session_id
        if hold is None and (generated or (to == left and self.deferred)):
            hold = Hold(to, self._at(place), None)  # held from its first write
        elif hold is None:
            taken = self.acquire(to, place)
            if isinstance(taken, int):
                return Switch(made=False, holder=taken)
            hold = taken
        elsewhere = hold.place != self.place
        moved = elsewhere and to == left  # the same conversation, in another place
        workflow = parts.workflow
        made = told = gone = False
        loaded: bool | None = None
        try:
            if to == left or elsewhere:  # nothing else to stay in: made whatever the load says
                self._make(
                    hold, parts, moved=moved, opened_empty=not stored, deferred=generated
                )
                made = True
            elif propose and self.turned:
                await parts.propose(left)
                self.turned = False
            if workflow is not None:
                loaded = await workflow.load_session(to)
                told = True
                if stored and not loaded and await workflow.session_info(to) is None:
                    if not made:
                        return Switch(made=False, gone=True)
                    self.opened_empty = gone = True  # nowhere else to stay: it starts empty
                if loaded:  # it holds what an earlier run stored
                    self.opened_empty = False
                elif to == left and not stored:  # an id given to the app, found empty
                    self.opened_empty = True
            if not made:
                self._make(
                    hold, parts, moved=moved, opened_empty=not (stored or loaded),
                    deferred=generated,
                )
                made = True
        finally:
            if not made:
                self.abandon(hold)
                if told and left is not None:
                    await _give_back(workflow, left)
        parts.status()
        return Switch(made=True, loaded=loaded, gone=gone)

    def switch_now(self, to: str | None, parts: SwitchParts) -> int | None:
        """What ``switch`` does without an await, for a caller that cannot
        wait (``BaseCLIApp``'s deprecated ``session_id`` setter).

        ``to``'s lock is taken and the one before let go, ``to`` is the
        conversation, unused, its task list and usage counters start over,
        and the status line is drawn. The workflow does not load it (the next
        turn names it), and the memories of the conversation left are not
        proposed.

        None names no conversation: the turns then run in the one the
        workflow loaded last (``session_ref`` falls back to its own id). A
        ready workflow's (``parts.workflow.session_id``) is held, letting go
        of the one held before (an earlier setter may have moved it, which the
        workflow was not told); with none ready the hold stays. Nothing else
        changes: ``turned``, the task list and the usage counters stay.

        Returns:
            None once ``to`` is the conversation (or was); else the process
            id of another process that has it open, and nothing changed.
        """
        if to == self.session_id:
            return None
        if to is None:  # the turns write to the workflow's own: hold that one
            loaded = getattr(parts.workflow, "session_id", None)
            if isinstance(loaded, str):
                holder = self.hold(loaded)
                if holder is not None:
                    return holder
            self.session_id, self.deferred = None, False
            return None
        taken = self.acquire(to)
        if isinstance(taken, int):
            return taken
        self._make(taken, parts, moved=False, opened_empty=False, deferred=False)
        parts.status()
        return None

    def _make(
        self,
        hold: Hold,
        parts: SwitchParts,
        *,
        moved: bool,
        opened_empty: bool,
        deferred: bool,
    ) -> None:
        """The switch itself, all at once (no await): the lock, the id, and
        the conversation's task list and usage counters. The same
        conversation in another place keeps what it was."""
        same = hold.session_id == self.session_id
        self.commit(hold)
        self.place = hold.place
        self.session_id = hold.session_id
        self.turned = False
        if not same:
            self.opened_empty, self.written = opened_empty, False
            self.deferred = deferred
        _start_over(parts, moved=moved)

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


async def _give_back(workflow: Any, session_id: str) -> None:
    """Give ``session_id`` back to the workflow, after a load that adopted a
    conversation the switch then did not make.

    Shielded: a Ctrl+C stops the wait, not the load, so the workflow does not
    stay on a conversation the app is not in (``/status`` names the
    workflow's). Never raises: the turns name their conversation anyway.
    """
    async def give() -> None:
        try:
            await workflow.load_session(session_id)
        except Exception:  # noqa: BLE001 - a lookup; the next turn goes on
            logger.debug("session_readopt_failed", session_id=session_id, exc_info=True)

    await asyncio.shield(asyncio.ensure_future(give()))


def _start_over(parts: SwitchParts, *, moved: bool) -> None:
    """A conversation switched to starts without the task list and the usage
    counters of the one left (only the context when the same conversation
    ``moved`` to another place: the session's totals stay)."""
    parts.processor.reset()
    if parts.tracker is not None:
        if moved:
            parts.tracker.reset_context()
        else:
            parts.tracker.reset()
