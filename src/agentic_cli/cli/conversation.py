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
which every switch takes as well: a turn (``run_turn``), ``/compact``
(``compact``), and a finished job's notice (``run_job_resume``), for each
job that waits for the conversation (``deliver_finished_jobs``, by
``cli/waiting_jobs.py``'s rule, which ``waiting_jobs`` names them by), each
held to write first (an id this run made is held from its first write).
``BaseCLIApp`` forwards to them, on its parts as they are (``TurnParts``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentic_cli.cli.conversation_lock import ConversationLock, Taken
from agentic_cli.cli.message_processor import (
    CANCELLED_LINE,
    TurnStatus,
    _says_so_when_stopped,
)
from agentic_cli.cli.waiting_jobs import (
    CanResume,
    Here,
    KeptElsewhere,
    opens_anywhere,
    told_here,
    waiting_here,
)
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
        kept_elsewhere: Whether another installed backend than the running
            one keeps a job's conversation (``BaseCLIApp._kept_elsewhere``),
            asked when the workflow's ``can_resume`` says no: such a job
            waits for it (``waiting_jobs.opens_anywhere``). None: the
            workflow's answer alone.
    """

    controller: Any
    processor: Any
    ui: Any
    settings: Any
    tracker: Any
    kept_elsewhere: KeptElsewhere | None = None


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
    turn lock, under which what writes to it runs (a turn, ``/compact``, a
    finished job's notice).

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
        #: (job, conversation) pairs whose "could not be resumed now" was
        #: said: the deliveries the CLI makes on its own say it once per
        #: conversation.
        self._said_not_told: set[tuple[str, str | None]] = set()

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

    async def run_turn(self, message: str, parts: TurnParts) -> bool:
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
            Whether the turn ran: False when another process has the
            conversation open (said), and nothing ran. The core says so, not
            the processor's result: an app's processor may return none.

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
                return False
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
            return True
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

    async def deliver_finished_jobs(self, parts: TurnParts) -> int:
        """Tell the agent of each finished, resume-flagged background job that waits here.

        Only the jobs that wait for this conversation
        (``cli/waiting_jobs.py``: its own, and those with no conversation or
        whose conversation is gone, which are told it is unavailable), as the
        status bar's ``↻N`` and the line an opened conversation says count
        them: another conversation's job waits until that conversation is
        opened, instead of answering in this one's transcript. Each notice is
        a turn of its own (``run_job_resume``, under the turn lock), so it
        never overlaps a user turn or another notice.

        Delivery follows the job's resume lifecycle: the job is *claimed*
        (pending → resuming) once the turn lock is held, the workflow found
        ready and the conversation looked up, and before the turn runs, so a
        crash cannot silently re-deliver it, and only recorded delivered once
        the turn actually completed — a failed or cancelled resume is
        recorded as such instead of being dropped. The check above is made
        again under the lock (``run_job_resume``): the conversation may have
        changed while the delivery waited for it.

        Args:
            parts: The app's parts it runs on.

        Returns:
            The number of jobs picked up: a resume turn run for it, or, for
            one that could not be told now (its conversation's lookup failed,
            before any claim), the failure said. Whether each one was
            delivered is recorded on the job and reported by ``/jobs``; a
            failed delivery is surfaced to the user by the turn itself.
        """
        controller = parts.controller
        if not controller.is_ready:
            return 0
        jm = getattr(controller.workflow, "job_manager", None)
        if jm is None:
            return 0

        attempted = 0
        # Ctrl+C stops these waits with no box open: they say so themselves.
        async with _says_so_when_stopped(parts.ui):
            records = await asyncio.to_thread(jm.awaiting_resume)
        for record in records:
            if not await self._told_here(record, parts):
                continue  # told when its conversation is opened
            if await self.run_job_resume(jm, record, parts) is not None:
                attempted += 1
        return attempted

    async def waiting_jobs(self, parts: TurnParts) -> list[Any]:
        """The finished jobs that wait for this conversation, in their order,
        by the rule the deliveries follow (``cli/waiting_jobs.py``): what the
        line an opened conversation says counts.

        The job manager is asked on a worker thread: it takes the jobs
        directory's lock, which another CLI of the app may hold.
        """
        workflow = parts.controller.workflow
        jm = getattr(workflow, "job_manager", None)
        if jm is None:
            return []
        return await waiting_here(
            await asyncio.to_thread(jm.awaiting_resume), self._here(parts.settings),
            _opens_anywhere(workflow, parts.kept_elsewhere),
        )

    async def run_job_resume(self, jm: Any, record: Any, parts: TurnParts) -> bool | None:
        """Claim one job and run its notice as a turn, and close out its transition.

        Under the turn lock the job is looked at again: another conversation
        may have become current while this waited for it (an app's own task
        that called ``resume_session`` or ``start_new_session`` took the lock
        first). A job that now waits for its conversation is not claimed, nor
        told here, where ``ctx`` and the task list are another conversation's:
        it stays pending until its conversation is opened. The conversation is
        held to write first: the notice is written to it. The claim is
        durable, so an interrupted delivery is recoverable and two
        coordinators cannot both deliver the same result.

        The job is claimed only once its result can reach the agent:
        ``process_resume`` calls ``claim`` after it found the workflow ready
        and looked the conversation up, before it says anything. A delivery
        stopped before then (the workflow not started, Ctrl+C or a failure
        during the lookup) claims nothing, and the job stays pending for a
        later one. A failure is said: each time the user asks (``/jobs
        resume``), and once per job and conversation when the CLI tells
        jobs on its own (``telling_on_its_own``: a turn's end, a
        conversation opened), not at every turn.

        Every exit path closes a claim, so no record can be left ``RESUMING``
        in this process: a normal outcome records delivered/failed, an
        exception records failed with the reason, and a cancellation records
        failed before re-raising (cancellation still propagates): the turn
        may have written the job's notice to the conversation already.

        The job manager is called on a worker thread (``_claim_delivery``,
        ``complete_resume``): it takes the jobs directory's lock, which
        another CLI of the app may hold. Each call is awaited before the next
        step, and one a second Ctrl+C stops waiting for still finishes.

        Args:
            jm: The JobManager.
            record: The job record.
            parts: The app's parts it runs on.

        Returns:
            True if the result was delivered to the agent; False if it was
            not (claimed and not delivered, or a failure before the claim,
            said); None if it was not claimed and nothing was said (it
            waits for its conversation, the workflow was not ready, the
            conversation is open in another process, or another coordinator
            has it).
        """
        claimed = False

        async def claim() -> bool:
            nonlocal claimed
            claimed = await _claim_delivery(jm, record.job_id)
            if claimed and getattr(record, "session_id", None) == self.session_id:
                self.turned = True  # the resume adds to this conversation
                # Defensive: a job of this conversation was submitted by a
                # turn, which stored it already (fresh was False before).
                self.written = True
            return claimed

        ui = parts.ui
        try:
            async with self.turn_lock:
                if not await self._told_here(record, parts):
                    return None
                if not self._held_to_write(ui):  # its notice is written here
                    return None
                result = await parts.processor.process_resume(
                    record=record,
                    workflow_controller=parts.controller,
                    ui=ui,
                    settings=parts.settings,
                    usage_tracker=parts.tracker,
                    claim=claim,
                )
        except asyncio.CancelledError:
            if claimed:
                logger.info("job_resume_cancelled", job_id=record.job_id)
                await asyncio.to_thread(
                    jm.complete_resume, record.job_id, delivered=False,
                    error="resume cancelled",
                )
            raise
        except Exception as exc:  # noqa: BLE001 - the claim must always close
            if not claimed:
                # Before the claim (the conversation's lookup, say): nothing
                # reached the agent, and a later delivery tells it.
                logger.warning(
                    "job_resume_unclaimed_failure", job_id=record.job_id, error=str(exc)
                )
                said = (record.job_id, self.session_id)
                if not (_TELLING_ON_ITS_OWN.get() and said in self._said_not_told):
                    ui.add_error(
                        f"Background job '{record.name}' could not be resumed now: {exc}"
                    )
                    self._said_not_told.add(said)
                return False
            await asyncio.to_thread(
                jm.complete_resume, record.job_id, delivered=False, error=str(exc)
            )
            logger.warning(
                "job_resume_raised", job_id=record.job_id, error=str(exc)
            )
            ui.add_error(
                f"Background job '{record.name}' could not be resumed: {exc}"
            )
            return False

        if not claimed:
            return None  # stopped before its result could reach the agent
        await asyncio.to_thread(
            jm.complete_resume, record.job_id, delivered=result.delivered,
            error=result.error,
        )
        if not result.delivered:
            logger.info(
                "job_resume_not_delivered",
                job_id=record.job_id,
                status=result.status.value,
                error=result.error,
            )
        return result.delivered

    def _here(self, settings: Any) -> Here:
        """The conversation, as ``cli/waiting_jobs.py`` names it: (user, session)."""
        return getattr(settings, "default_user", None), self.session_id

    async def _told_here(self, record: Any, parts: TurnParts) -> bool:
        """Whether ``record``, a job waiting for delivery, is told in this conversation.

        A Ctrl+C during the conversation's lookup says the request stopped.
        """
        workflow = parts.controller.workflow
        async with _says_so_when_stopped(parts.ui):
            return await told_here(
                record, self._here(parts.settings),
                _opens_anywhere(workflow, parts.kept_elsewhere),
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


#: Set while the CLI tells finished jobs on its own (at a turn's end, or a
#: conversation opened), not because the user asked (``/jobs resume``).
_TELLING_ON_ITS_OWN: ContextVar[bool] = ContextVar("telling_on_its_own", default=False)


@contextmanager
def telling_on_its_own() -> Iterator[None]:
    """The deliveries inside are the CLI's own (a turn's end, a conversation
    opened), not the user's ask (``/jobs resume``): a job that cannot be told
    now is said once per job and conversation (``run_job_resume``)."""
    token = _TELLING_ON_ITS_OWN.set(True)
    try:
        yield
    finally:
        _TELLING_ON_ITS_OWN.reset(token)


def _opens_anywhere(workflow: Any, kept_elsewhere: KeptElsewhere | None) -> CanResume | None:
    """Whether a job's conversation can be opened, on any installed backend
    (``waiting_jobs.opens_anywhere``): the running ``workflow``'s
    ``can_resume``, then ``kept_elsewhere``."""
    return opens_anywhere(getattr(workflow, "can_resume", None), kept_elsewhere)


async def _claim_delivery(jm: Any, job_id: str) -> bool:
    """Claim the job's delivery (``jm.begin_resume``) on a worker thread.

    The claim takes the jobs directory's lock, which another CLI of the app
    may hold, so it runs off the event loop. A Ctrl+C while it is taken stops
    the wait, not the claim, which its thread may still take: a claim taken
    then is handed back (``JobManager.release_resume``), never left
    ``RESUMING``, and the job waits for a later delivery, nothing having
    reached the agent.

    Returns:
        True if this process now delivers the job.
    """
    claim = asyncio.ensure_future(asyncio.to_thread(jm.begin_resume, job_id))
    try:
        return await asyncio.shield(claim)
    except asyncio.CancelledError:
        claim.add_done_callback(lambda done: _release_abandoned_claim(jm, job_id, done))
        raise


def _release_abandoned_claim(jm: Any, job_id: str, claim: "asyncio.Future[bool]") -> None:
    """Hand back, on a worker thread, a claim taken for a delivery Ctrl+C stopped."""
    if claim.cancelled() or claim.exception() is not None or not claim.result():
        return
    logger.info("job_resume_claim_released", job_id=job_id)

    def release() -> None:
        try:
            jm.release_resume(job_id)
        except Exception:  # noqa: BLE001 - startup recovery fails it once we are gone
            logger.warning("job_resume_release_failed", job_id=job_id, exc_info=True)

    asyncio.get_running_loop().run_in_executor(None, release)


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
