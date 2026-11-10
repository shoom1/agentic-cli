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

What writes to the conversation runs here too, under the core's turn lock,
which every switch takes as well: a turn (``run_turn``) and ``/compact``
(``compact``), each held to write first (an id this run made is held from
its first write). ``BaseCLIApp`` forwards to them, on its parts as they are
(``TurnParts``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentic_cli.cli.conversation_lock import ConversationLock, Taken
from agentic_cli.cli.message_processor import CANCELLED_LINE, TurnResult, TurnStatus
from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from agentic_cli.workflow.context import CompactResult

logger = Loggers.cli()

#: What the CLI says when asked for a conversation it does not keep (``/resume``,
#: the picker, ``--resume``, ``--continue``, ``/sessions``): on the memory store
#: (``Place.kept``) it is in one conversation at a time.
NOT_KEPT = (
    "session_store is memory: conversations are not kept. "
    "Set it to sqlite to keep and resume them (applies at the next start)."
)


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

    @property
    def kept(self) -> bool:
        """Whether the CLI keeps its conversations once they are left. Not on
        the memory store, which writes none to disk: the app is in one
        conversation at a time there, ``/new`` drops the one it leaves, and
        nothing is listed or opened again (``NOT_KEPT``)."""
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
class TurnParts:
    """What a write to the conversation runs on besides it: the app's parts as they are.

    Attributes:
        controller: The workflow controller (``ensure_initialized``,
            ``is_ready``, ``workflow``).
        processor: The message processor, which runs the turn and draws it.
        ui: The session the turn is drawn in.
        settings: The app's settings.
        tracker: The usage tracker the turn's token counts go to; None for none.
    """

    controller: Any
    processor: Any
    ui: Any
    settings: Any
    tracker: Any


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
    """The conversation the app is in: its id, ``turned``, its lock, and its
    turn lock, under which what writes to it runs (a turn, ``/compact``).

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
        #: Serializes what writes to the conversation and every change of
        #: it: a turn, a job's notice, ``/compact``, ``/new`` and ``/resume``
        #: (not ``/settings``, nor the ``/resume`` picker, which waits on a
        #: dialog). thinking-prompt runs one input handler at a time, so it
        #: orders them against the startup resume (a task of its own) and an
        #: app's own tasks.
        self.turn_lock = asyncio.Lock()
        self._lock: ConversationLock | None = None
        #: The run said its conversation runs unheld (a lock that could not be
        #: taken): said once a run.
        self._said_unheld = False

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
                totals and forgets only the context; the same id ``stored``
                there is another conversation (a twin another backend
                keeps), which starts over.
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
        # The same conversation, in another place (one stored there is
        # another conversation: the same id another backend keeps).
        moved = elsewhere and to == left and not stored
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

    async def run_turn(self, message: str, parts: TurnParts) -> TurnResult | None:
        """Run a turn of the user's ``message`` in the conversation, under the turn lock.

        The line is echoed as its turn starts: one typed while the
        conversation ``--continue``/``--resume`` asked for opens shows under
        its replay; one Ctrl+C stops while it waits is shown and said
        cancelled (it never ran), not dropped unseen. The conversation is
        held to write first, then the message processor runs the turn
        (``MessageProcessor.process``) in it. It is ``turned`` from then on
        (its memories are proposed on leaving it), and ``written`` once the
        turn reached a ready workflow, which stores it (a turn the workflow
        was not up for stores nothing; one that raised or was cancelled
        counts when the workflow was ready: it may have begun).

        Args:
            message: What the user typed.
            parts: The app's parts it runs on.

        Returns:
            The turn's outcome; None when it did not run (another process has
            the conversation open, said).

        Raises:
            asyncio.CancelledError: Ctrl+C stopped it, as the processor
                says, or while it waited for the turn lock.
        """
        ui = parts.ui
        try:
            await self.turn_lock.acquire()
        except asyncio.CancelledError:
            ui.add_message("user", message)
            ui.add_message("system", CANCELLED_LINE)
            raise
        try:
            ui.add_message("user", message)
            if not self._held_to_write(ui):
                return None
            self.turned = True  # the conversation now holds this message
            reached = False  # the turn reached a ready workflow, which stores it
            try:
                result = await parts.processor.process(
                    message=message,
                    workflow_controller=parts.controller,
                    ui=ui,
                    settings=parts.settings,
                    usage_tracker=parts.tracker,
                    session_id=self.session_id,
                )
                reached = getattr(result, "status", None) is not TurnStatus.UNAVAILABLE
            except BaseException:
                reached = parts.controller.is_ready  # it may have begun
                raise
            finally:
                if reached:
                    self.written = True
            return result
        finally:
            self.turn_lock.release()

    async def compact(self, focus: str | None, parts: TurnParts) -> "CompactResult | None":
        """Summarize the conversation now (``/compact [focus]``), as a turn runs.

        Under the turn lock, held to write first: the workflow's ``compact``
        runs with a turn's events box, status line, token counts and Ctrl+C
        (``MessageProcessor.process_compact``), which says what it did, or
        why nothing was.

        Args:
            focus: What the summary should keep above all, if anything.
            parts: The app's parts it runs on.

        Returns:
            What the summary did; None when nothing was summarized (or the
            conversation is open in another process, said).
        """
        async with self.turn_lock:
            if not self._held_to_write(parts.ui):
                return None
            return await parts.processor.process_compact(
                focus,
                workflow_controller=parts.controller,
                ui=parts.ui,
                settings=parts.settings,
                usage_tracker=parts.tracker,
                session_id=self.session_id,
            )

    def _held_to_write(self, ui: Any) -> bool:
        """Hold the conversation before it is written, when its id was made by
        this run (``hold_to_write``); the caller holds the turn lock. False,
        said, when another process has it open (a collision of random ids):
        the write must not happen.

        A lock that cannot be taken (``<sessions_dir>/locks`` read-only, say,
        on a postgres store) does not stop the write: the conversation runs
        unheld, as an id given to the app does whose startup hold failed,
        and the run says so once.
        """
        try:
            holder = self.hold_to_write()
        except OSError as error:
            logger.debug(
                "conversation_hold_failed", session_id=self.session_id, exc_info=True
            )
            self.deferred = False  # written unheld: not tried again
            if not self._said_unheld:
                self._said_unheld = True
                reason = str(error) or type(error).__name__
                ui.add_warning(
                    f"Could not hold conversation '{self.session_id}' for this "
                    f"process: {reason}. Its turns run unheld, so another process "
                    "could open it too."
                )
            return True
        if holder is None:
            return True
        ui.add_error(
            f"Conversation '{self.session_id}' is open in another process ({holder})."
        )
        return False

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
        conversation, here or ``moved`` to another place, keeps what it was;
        the same id stored in another place is another conversation."""
        same = hold.session_id == self.session_id and (moved or hold.place == self.place)
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
