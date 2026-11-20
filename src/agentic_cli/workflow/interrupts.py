"""What a tool call a cancel or a crash left open is answered with.

Ctrl+C stops the whole turn: the agent loop, its model calls, its tool calls
and its sub-agents (a delegation is a call of its parent's). Every call the
model made then gets exactly one result, so the history stays valid for every
provider (Claude and OpenAI refuse a call without its result) and the model
knows what happened rather than running a side effect again blindly:

- a call that finished before the cancel keeps its real result, and the
  state it changed (``save_tasks``' tasks, a delegation's plan);
- a running call, or a delegation, is answered ``INTERRUPTED_BY_USER``;
- a call waiting at its permission prompt (both gates mark it) or at a
  question it asked (``BaseWorkflowManager.request_user_input``, by the
  active tool call) is ``NOT_RUN``;
- a cancel with no call open (during the first model call, or one after
  tool results) leaves ``MARKER`` as the main agent's reply, so the next
  turn answers the new message rather than resuming the old one;
- a call a crash left open (nothing ran on the way out) is answered
  ``INTERRUPTED`` on the next turn;
- a turn that fails (its one ``ERROR``) keeps the record of its calls that
  returned (``OpenCalls.failed``): at the next turn each of them that has no
  stored result is answered with the result it returned, and any other
  call left open ``INTERRUPTED``.

Background jobs a long-running tool started are not part of the turn: they
go on, and their results arrive later. A call whose job was submitted is
answered with the job's state (``job_started``), so the model knows it runs.
A job answers only a call of the run that submitted it
(``JobRecord.started_by``).

``OpenCalls`` is a manager's record of what became of its calls, kept by
conversation and, within it, by the instance whose run made each call (the
root's own, or a sub-agent's: ``workflow.sessions.get_active_run``), since a
provider that numbers its call ids per agent gives a sub-agent's call the
id of its parent's delegation. An instance runs one run at a time and its
next run (a follow-up) answers, before its message, what an earlier run left
open, so the record outlives the run. The backends answer from it (ADK as the
cancelled turn stops, through its ``abort_signal``, and a failed turn's
calls as that conversation's next turn starts; Deep Agents before the first
model call of that conversation's next turn, however many turns ran in
others since).

Every part of this pairs a result with its call by the call's id. A
provider that numbers its call ids per response (``call_0`` in every one)
reuses an id the agent's conversation already holds, so both backends give
such a call an id of its own as the model's answer arrives
(``fresh_call_id``), before it is stored or run: its result, the record and
every later request then name it alone.
"""

from __future__ import annotations

import asyncio
import enum
import re
import uuid
from collections.abc import Container
from contextlib import contextmanager
from typing import Any, Iterator

from agentic_cli.workflow.agents.runs import run_instance
from agentic_cli.workflow.service_registry import JOB_MANAGER, get_service
from agentic_cli.workflow.sessions import get_active_run, get_active_turn, instance_of

#: A call that was running when the user pressed Ctrl+C.
INTERRUPTED_BY_USER = (
    "Interrupted by the user before it finished; it may have partly run. "
    "Check before running it again."
)
#: A call left open by something else: a crash, or a cancel from before this rule.
INTERRUPTED = (
    "Interrupted before it finished (its turn or the app stopped); it may have "
    "partly run. Check before running it again."
)
#: A call the user cancelled at its permission prompt.
NOT_RUN = "Not run: the user cancelled it."
#: The main agent's reply to a turn cancelled with no call open.
MARKER = "(Interrupted by the user.)"


def interrupted(text: str) -> dict[str, Any]:
    """A call's result that says why it has none."""
    return {"success": False, "error": text}


#: The longest call id ``fresh_call_id`` makes: OpenAI's Chat Completions
#: takes none longer.
_CALL_ID_MAX = 40
#: How many hex digits ``fresh_call_id`` adds, after a ``-``.
_SUFFIX_DIGITS = 8
#: A character Claude takes in no call id, which ``fresh_call_id`` replaces.
_NOT_IN_A_CALL_ID = re.compile(r"[^A-Za-z0-9_-]")


def fresh_call_id(call_id: str, used: Container[str]) -> str:
    """An id for a call whose provider gave it ``call_id``, which its
    conversation already holds: ``call_id`` and a short random suffix
    (``call_0-3f9a1c2b``), none of ``used``.

    Only ``[A-Za-z0-9_-]``, at most ``_CALL_ID_MAX`` characters, which every
    provider takes: any other character of ``call_id`` becomes ``_``, and a
    long one is cut before the suffix.
    """
    stem = _NOT_IN_A_CALL_ID.sub("_", call_id)[: _CALL_ID_MAX - 1 - _SUFFIX_DIGITS]
    while True:
        candidate = f"{stem}-{uuid.uuid4().hex[:_SUFFIX_DIGITS]}"
        if candidate not in used:
            return candidate


def job_started(record: Any) -> dict[str, Any]:
    """The result of a call whose background job was submitted before the cancel."""
    return {
        "success": True,
        "job_id": record.job_id,
        "state": record.state.value,
        "note": "The job started before the turn stopped; its result arrives when it ends.",
    }


def job_for_call(call_id: str | None, *, started_by: str | None = None) -> Any | None:
    """The background job the call ``call_id`` submitted, if there is one.

    Only a job of the active turn's conversation (its session and user), and
    none outside a turn: the job store holds all the user's jobs, from every
    start of the app, and a provider that numbers its call ids (``call_0``,
    ...) repeats them in every conversation. And only a job ``started_by``
    started: the caller names the run whose call it is by its instance (None
    for the root's own call), as ``JobRecord.started_by`` records it, since
    such a provider repeats them across agents too, and a sub-agent's job
    must not answer its parent's call. A caller that cannot name the run (ADK's
    ``left_open``, for a sub-agent's call) asks for no job.

    Looked up among the jobs the manager holds (``JobManager.snapshot``), with
    no lock: it is asked on the event loop (ADK answers a stopped turn's
    calls in an async callback), where waiting for the jobs directory's lock
    froze the app. A job a call submitted is the manager's own, or was loaded
    when it started (a crash's), and what is matched never changes.
    """
    jobs = get_service(JOB_MANAGER)
    turn = get_active_turn()
    if jobs is None or turn is None or not call_id:
        return None
    return next((
        record for record in jobs.snapshot()
        if record.call_id == call_id
        and record.session_id == turn.session_id
        and record.user_id == turn.user_id
        # A record a test or an app built may not have the field: the root's.
        and getattr(record, "started_by", None) == started_by
    ), None)


def unrecorded(
    call_id: str | None, *, by_user: bool, started_by: str | None = None
) -> dict[str, Any]:
    """The answer for an open call no record holds: its job's state, else why it has none.

    ``OpenCalls.answer`` gives it for a call it holds nothing for, matching
    the job by the instance of the call's run (``started_by``); ADK's
    ``left_open`` answers the root's own calls through it (``started_by``
    None), and a sub-agent's call with no job, its run being numbered no
    more.
    """
    record = job_for_call(call_id, started_by=started_by)
    if record is not None:
        return job_started(record)
    return interrupted(INTERRUPTED_BY_USER if by_user else INTERRUPTED)


class _Active(enum.Enum):
    """The default of ``OpenCalls``' ``run=``: the run in progress."""

    RUN = "the run in progress"


#: ``run=`` left out: the run whose call is in progress
#: (``workflow.sessions.get_active_run``; the root's own calls are no run).
ACTIVE_RUN = _Active.RUN

#: A call's place in its conversation's record: the instance whose run made
#: it (None for the root's own), and its id.
_Mark = tuple[str | None, str]


def _instance(run: str | None | _Active) -> str | None:
    """The instance whose call ``run`` names, as a job's ``started_by`` records it.

    The run in progress by default (``instance_of``); a run named by its id
    is its instance's (``agents.runs.run_instance``); None is the root's own.
    """
    if run is ACTIVE_RUN:
        return instance_of(get_active_run())
    return run_instance(run) if run else None


class OpenCalls:
    """What became of a manager's tool calls, for answering those a cancel leaves open.

    A call is recorded ``finished`` with its result, or ``asking`` while it
    waits at its permission prompt; a conversation is recorded ``cancelled``
    when the user stops its turn. Calls are kept by conversation, the active
    turn's (``workflow.sessions.get_active_turn``), which ``answer`` reads:
    Deep Agents answers a stopped turn's calls at that conversation's next
    turn, after turns in others. ``answer`` forgets nothing: that turn may
    be stopped too before its answers are saved, and the one after answers
    them again. A turn that ends without a cancel has every call answered and
    saved, so ``clear`` forgets its conversation's calls; one that fails
    keeps those that finished (``failed``).

    Within its conversation a call is the instance's whose run made it
    (``run=``: by default the run in progress, ``get_active_run()``, or a
    run's id; None for the root's own calls), so a sub-agent's call never
    stands for its parent's: a provider that numbers its call ids per agent
    (``call_0``, ...) gives a sub-agent's first call the id of the
    delegation that started it. A sub-agent instance's next run reads what
    an earlier run of it recorded, and a stop of one of its runs is recorded
    for it (``cancelled(instance=)``): ``clear`` keeps its calls until its
    next run takes the stop.
    """

    def __init__(self) -> None:
        self._finished: dict[Any, dict[_Mark, Any]] = {}
        self._asking: dict[Any, set[_Mark]] = {}
        # The conversations whose turn the user stopped, and the (conversation,
        # instance) pairs whose run was stopped.
        self._cancelled: set[Any] = set()
        self._stopped: set[tuple[Any, str]] = set()

    def finished(
        self, call_id: str | None, result: Any, *, run: str | None | _Active = ACTIVE_RUN
    ) -> None:
        """Record that ``call_id`` of ``run`` returned ``result``."""
        if call_id:
            self._finished.setdefault(get_active_turn(), {})[(_instance(run), call_id)] = result

    def recorded(
        self, call_id: str | None, *, run: str | None | _Active = ACTIVE_RUN
    ) -> Any | None:
        """What ``finished`` recorded for ``call_id`` of ``run`` in the active
        turn's conversation, or None."""
        if not call_id:
            return None
        return self._finished.get(get_active_turn(), {}).get((_instance(run), call_id))

    @contextmanager
    def asking(
        self, call_id: str | None, *, run: str | None | _Active = ACTIVE_RUN
    ) -> Iterator[None]:
        """Mark ``call_id`` of ``run`` as waiting at its prompt while the block runs.

        A cancel leaves the mark for ``answer`` to read; any other end clears it.
        """
        if not call_id:
            yield
            return
        key = get_active_turn()
        mark = (_instance(run), call_id)
        self._asking.setdefault(key, set()).add(mark)
        try:
            yield
        except asyncio.CancelledError:
            raise
        except BaseException:
            self._unmark(key, mark)
            raise
        self._unmark(key, mark)

    def answer(
        self, call_id: str | None, *, by_user: bool, run: str | None | _Active = ACTIVE_RUN
    ) -> Any:
        """An open call's result: what it returned if it finished, else why it has none.

        The call is ``run``'s, in the active turn's conversation. It stays
        recorded until a turn of that conversation ends with every call
        answered (``clear``): the step that answers it may be stopped before
        it is saved, and a failed turn keeps it (``failed``) for the next
        turn to answer.

        Args:
            call_id: The call.
            by_user: Whether the user stopped its turn (else a crash did, or
                a cancel this process did not see).
            run: The run that made the call: by default the run in progress.
                A job answers the call only when that run's instance
                submitted it (``job_for_call``).
        """
        if call_id:
            key = get_active_turn()
            mark = (_instance(run), call_id)
            finished = self._finished.get(key, {})
            if mark in finished:
                return finished[mark]
            if mark in self._asking.get(key, ()):
                return interrupted(NOT_RUN)
        # Its job, if it submitted one: the job of the run it was given.
        return unrecorded(call_id, by_user=by_user, started_by=_instance(run))

    def cancelled(self, key: Any, *, instance: str | None = None) -> None:
        """The user stopped the turn of conversation ``key``, or, with ``instance``,
        a run of that instance of it."""
        if instance is None:
            self._cancelled.add(key)
        else:
            self._stopped.add((key, instance))

    def take_cancel(self, key: Any, *, instance: str | None = None) -> bool:
        """Whether the turn of ``key`` (or a run of its ``instance``) was stopped;
        asking forgets it."""
        held: set[Any] = self._cancelled if instance is None else self._stopped
        mark = key if instance is None else (key, instance)
        if mark in held:
            held.discard(mark)
            return True
        return False

    def clear(self, key: Any) -> None:
        """Forget conversation ``key``'s calls: its turn ended with every one answered.

        A sub-agent instance whose run was stopped keeps its calls until its
        next run takes the stop (``take_cancel(instance=)``): that run
        answers them.
        """
        kept = {instance for conversation, instance in self._stopped if conversation == key}
        for held in (self._finished, self._asking):
            marks = held.pop(key, None)
            if marks is None:
                continue
            if isinstance(marks, dict):
                marks = {m: v for m, v in marks.items() if m[0] in kept}
            else:
                marks = {m for m in marks if m[0] in kept}
            if marks:
                held[key] = marks

    def failed(self, key: Any) -> None:
        """The turn of conversation ``key`` failed: its next turn answers its calls.

        The finished ones keep their results; the others are answered
        ``INTERRUPTED``, one that waited at its prompt too, since the user
        did not cancel it.
        """
        self._asking.pop(key, None)

    def _unmark(self, key: Any, mark: _Mark) -> None:
        marks = self._asking.get(key)
        if marks is None:
            return
        marks.discard(mark)
        if not marks:
            del self._asking[key]
