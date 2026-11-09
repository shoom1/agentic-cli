"""The conversation the CLI's turns run in (``ConversationCore``).

A conversation lives in one CLI process at a time: two processes on one
conversation lose a turn (``cli/conversation_lock.py``). ``ConversationCore``
keeps what makes the conversation the app is in: its id, whether a turn ran
in it since it was opened (``turned``: its memories are proposed on leaving
it, or at exit, only then), and the lock this process holds on it, at its
``Place``: the backend and the store it is kept in, the app and the user (the
same id in another backend or store is another conversation).

Every change of conversation goes through one protocol, ``switch``: ``/new``,
``/resume``, the conversation a run starts in (``--continue``, ``--resume``, a
new one) and an orchestrator switch, which keeps the id in another backend.
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


@dataclass(frozen=True, slots=True)
class SwitchParts:
    """What a switch acts on besides the conversation: the app's parts as they are.

    Attributes:
        workflow: The workflow manager when it is ready, else None (nothing
            is loaded then: the next start, or the next turn, takes the id).
        processor: The message processor, whose task list is forgotten
            (``reset()``).
        tracker: The usage tracker, zeroed (``reset()``), or only its context
            after a backend switch (``reset_context()``: the session's totals
            stay); None for none.
        propose: Proposes the memories of a conversation left, by its id.
        status: Draws the status line again; told True after a backend switch.
    """

    workflow: Any
    processor: Any
    tracker: Any
    propose: Callable[[str], Awaitable[None]]
    status: Callable[[bool], None]


@dataclass(frozen=True, slots=True)
class Switch:
    """How a switch ended.

    Attributes:
        made: The app is in the target conversation now, holding it.
        holder: Not made: the process id of another process that has the
            target open (0 when unknown).
        gone: Not made: a stored target the workflow no longer holds or lists
            (deleted since it was read).
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
        taken = self.acquire(session_id, place)
        if isinstance(taken, int):
            return taken
        self.commit(taken)
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
           read), and the switch is not made;
        4. the switch is made: the lock on ``to`` is kept and the one before
           let go, ``to`` is the conversation (unused), and the task list
           and the usage counters start over; else ``to``'s lock is let go;
        5. the status line is drawn again.

        A switch is made once the workflow has the conversation: until then a
        Ctrl+C or a failure leaves the app in the conversation it was in,
        holding it alone, and a workflow that took ``to`` is given that
        conversation back (shielded, so a second Ctrl+C cannot stop it
        half way). A switch to the conversation the app is already in (the
        one a run starts in, or the same id in another backend after an
        orchestrator switch) has no other to stay in: it is made before the
        load, whatever the load says.

        Args:
            to: The conversation's id.
            parts: The app's parts it acts on.
            place: Where ``to`` is kept: the current place by default; another
                one after an orchestrator switch, whose usage counters keep
                the session's totals and forget only the context.
            stored: ``to`` must be in the store (``/resume``); a new
                conversation's load finds nothing, as it should.
            propose: Propose the memories of the conversation left (never
                after an orchestrator switch: the backend that holds them is
                gone).
            hold: ``to``'s lock, already taken (``acquire``); the switch keeps
                it or lets it go.

        Returns:
            How it ended: made, refused (``holder``), or gone.

        Raises:
            Whatever the workflow's load raises, and whatever drawing the
            status line raises once the switch is made.
        """
        left = self.session_id
        if hold is None:
            taken = self.acquire(to, place)
            if isinstance(taken, int):
                return Switch(made=False, holder=taken)
            hold = taken
        swapped = hold.place != self.place
        workflow = parts.workflow
        made = told = False
        loaded: bool | None = None
        try:
            if to == left:  # nothing else to stay in: held whatever the load says
                self._make(hold, parts, swapped=swapped)
                made = True
            elif propose and self.turned:
                await parts.propose(left)
                self.turned = False
            if workflow is not None:
                loaded = await workflow.load_session(to)
                told = True
                if stored and not loaded and await workflow.session_info(to) is None:
                    return Switch(made=False, gone=True)
            if not made:
                self._make(hold, parts, swapped=swapped)
                made = True
        finally:
            if not made:
                self.abandon(hold)
                if told and left is not None:
                    await _give_back(workflow, left)
        parts.status(swapped)
        return Switch(made=True, loaded=loaded)

    def switch_now(self, to: str | None, parts: SwitchParts) -> int | None:
        """What ``switch`` does without an await, for a caller that cannot
        wait (``BaseCLIApp``'s deprecated ``session_id`` setter).

        ``to``'s lock is taken and the one before let go, ``to`` is the
        conversation, unused, its task list and usage counters start over,
        and the status line is drawn. The workflow does not load it (the next
        turn names it), and the memories of the conversation left are not
        proposed.

        None names no conversation: the turns then run in the one the
        workflow loaded last (``session_ref`` falls back to its own id), which
        is the conversation held whenever the workflow ran at its switch.
        So nothing else changes: the lock, ``turned``, the task list and the
        usage counters stay.

        Returns:
            None once ``to`` is the conversation (or was); else the process
            id of another process that has it open, and nothing changed.
        """
        if to == self.session_id:
            return None
        if to is None:  # the workflow's own: the conversation held, as it was
            self.session_id = None
            return None
        taken = self.acquire(to)
        if isinstance(taken, int):
            return taken
        self._make(taken, parts, swapped=False)
        parts.status(False)
        return None

    def _make(self, hold: Hold, parts: SwitchParts, *, swapped: bool) -> None:
        """The switch itself, all at once (no await): the lock, the id, and
        the conversation's task list and usage counters."""
        self.commit(hold)
        self.place = hold.place
        self.session_id = hold.session_id
        self.turned = False
        _start_over(parts, swapped=swapped)

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


def _start_over(parts: SwitchParts, *, swapped: bool) -> None:
    """A conversation switched to starts without the task list and the usage
    counters of the one left (only the context after a backend switch: the
    session's totals stay)."""
    parts.processor.reset()
    if parts.tracker is not None:
        if swapped:
            parts.tracker.reset_context()
        else:
            parts.tracker.reset()
