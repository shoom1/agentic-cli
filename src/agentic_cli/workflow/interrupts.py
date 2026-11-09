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
  ``INTERRUPTED`` on the next turn.

Background jobs a long-running tool started are not part of the turn: they
go on, and their results arrive later. A call whose job was submitted is
answered with the job's state (``job_started``), so the model knows it runs.

``OpenCalls`` is a manager's record of what became of its calls, kept by
conversation: the backends answer from it (ADK as the cancelled turn stops,
through its ``abort_signal``; Deep Agents before that conversation's next
run's first model call, however many turns ran in others since).
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import Any, Iterator

from agentic_cli.workflow.service_registry import JOB_MANAGER, get_service
from agentic_cli.workflow.sessions import get_active_turn

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


def job_started(record: Any) -> dict[str, Any]:
    """The result of a call whose background job was submitted before the cancel."""
    return {
        "success": True,
        "job_id": record.job_id,
        "state": record.state.value,
        "note": "The job started before the turn stopped; its result arrives when it ends.",
    }


def job_for_call(call_id: str | None) -> Any | None:
    """The background job the call ``call_id`` submitted, if there is one.

    Only a job of the active turn's conversation (its session and user), and
    none outside a turn: the job store holds all the user's jobs, from every
    run, and a provider that numbers its call ids (``call_0``, ...) repeats
    them in every conversation.

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
    ), None)


class OpenCalls:
    """What became of a manager's tool calls, for answering those a cancel leaves open.

    A call is recorded ``finished`` with its result, or ``asking`` while it
    waits at its permission prompt; a conversation is recorded ``cancelled``
    when the user stops its turn. Calls are kept by conversation, the active
    turn's (``workflow.sessions.get_active_turn``), which ``answer`` reads:
    Deep Agents answers a stopped turn's calls at that conversation's next
    run, after turns in others. ``answer`` forgets nothing: that run may be
    stopped too before its answers are saved, and the next one answers them
    again. A turn that ends without a cancel has every call answered and
    saved, so ``clear`` forgets its conversation's calls; one that fails
    keeps those that finished (``failed``).
    """

    def __init__(self) -> None:
        self._finished: dict[Any, dict[str, Any]] = {}
        self._asking: dict[Any, set[str]] = {}
        self._cancelled: set[Any] = set()

    def finished(self, call_id: str | None, result: Any) -> None:
        if call_id:
            self._finished.setdefault(get_active_turn(), {})[call_id] = result

    @contextmanager
    def asking(self, call_id: str | None) -> Iterator[None]:
        """Mark ``call_id`` as waiting at its prompt while the block runs.

        A cancel leaves the mark for ``answer`` to read; any other end clears it.
        """
        if not call_id:
            yield
            return
        key = get_active_turn()
        self._asking.setdefault(key, set()).add(call_id)
        try:
            yield
        except asyncio.CancelledError:
            raise
        except BaseException:
            self._unmark(key, call_id)
            raise
        self._unmark(key, call_id)

    def answer(self, call_id: str | None, *, by_user: bool) -> Any:
        """An open call's result: what it returned if it finished, else why it has none.

        The call is the active turn's conversation's. It stays recorded until
        that conversation's turn ends (``clear``): the step that answers it
        may be stopped before it is saved.

        Args:
            call_id: The call.
            by_user: Whether the user stopped its turn (else a crash did, or
                a cancel this process did not see).
        """
        key = get_active_turn()
        finished = self._finished.get(key, {})
        if call_id in finished:
            return finished[call_id]
        if call_id in self._asking.get(key, ()):
            return interrupted(NOT_RUN)
        record = job_for_call(call_id)
        if record is not None:
            return job_started(record)
        return interrupted(INTERRUPTED_BY_USER if by_user else INTERRUPTED)

    def cancelled(self, key: Any) -> None:
        """The user stopped the turn of conversation ``key``."""
        self._cancelled.add(key)

    def take_cancel(self, key: Any) -> bool:
        """Whether the user stopped the last turn of ``key``; asking forgets it."""
        if key in self._cancelled:
            self._cancelled.discard(key)
            return True
        return False

    def clear(self, key: Any) -> None:
        """Forget conversation ``key``'s calls: its turn ended with every one answered."""
        self._finished.pop(key, None)
        self._asking.pop(key, None)

    def failed(self, key: Any) -> None:
        """The turn of conversation ``key`` failed: its next run answers its calls.

        The finished ones keep their results; the others are answered
        ``INTERRUPTED``, one that waited at its prompt too, since the user
        did not cancel it.
        """
        self._asking.pop(key, None)

    def _unmark(self, key: Any, call_id: str) -> None:
        marks = self._asking.get(key)
        if marks is None:
            return
        marks.discard(call_id)
        if not marks:
            del self._asking[key]
