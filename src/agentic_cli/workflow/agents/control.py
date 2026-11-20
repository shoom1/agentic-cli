"""One sub-agent run's control: its limits, a cancel of the run alone, and steering.

``Delegations`` (``agents/delegate.py``) makes a ``RunControl`` for each run
it starts, with the run's limits (its type's over the policy's), binds it to
the run's own task, and hands it to the backend's hook as
``AgentRun.control``; ``Delegations.control(run)`` finds a working run's.

- ``cancel(by=, error=)`` cancels the run's task: its parent's stop, the
  user's, or a limit's. The first cancel wins, and a run that has ended is
  not cancelled, nor one whose hook said its work has ended (``finish``)
  while it still reads back what the run did. The hook, handling the
  ``CancelledError`` in the run's task, asks ``own_cancel`` whether it is
  the run's own: then it stops the run as a stopped turn's run stops and
  returns what it has, and the run ends ``cancelled``; any other cancel
  (the turn's: Ctrl+C) goes on.
- The time limit (``timeout``) runs while the hook does (``timing``); past
  it the run is cancelled by its limit.
- The limits checked at the run's model calls: before each, ``model_call``
  counts it and says whether it is the last allowed (``max_model_calls``),
  whose request ends with ``STEP_LIMIT_NOTE``; after each, ``used`` counts
  its tokens, and the run's total (its own calls' tokens and what its other
  calls used, ``side_tokens``: its tools' side calls, its summaries, as its
  usage record counts them) is checked: an answer past ``max_tokens`` that
  asks for tools (the run would go on) has the run cancelled by its limit
  at that answer; one that asks for none ends the run as it would, its
  result. On the call that is the run's last (``final_call``) the backend
  drops the answer's tool calls, so the run ends with that answer: no call
  past a limit is made, and nothing the model asked for runs after it. A
  model that reports no usage counts none of its own.
- ``steer(message)`` gives the run a message, which its next model call
  (``take_steering``) adds after its last message, as a user's. A message
  stays unstored until its backend says the run's transcript holds it
  (``stored``); ``Delegations`` gives the unstored ones (``unstored``) to
  the instance's next run, so a message is never lost.
- ``stop_cause`` is what a call the run left at its permission prompt is
  told stopped the run: its parent, its time limit or its token limit
  (None for the user's stop).

All of it runs on the event loop; nothing here blocks.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from agentic_cli.workflow.agents.envelope import (
    CANCELLED_BY,
    TIME_LIMIT_ERROR,
    TOKEN_LIMIT_ERROR,
)

if TYPE_CHECKING:
    from agentic_cli.workflow.agents.policy import AgentLimits
    from agentic_cli.workflow.sessions import RunRef

#: What the run's last allowed model call is told, after its last message.
STEP_LIMIT_NOTE = (
    "You have reached the most model calls this task may make. Answer now with your best "
    "result from what you have; no tools are available for this answer."
)
#: Who cancels a run at one of its limits.
LIMIT = "limit"
#: What stopped a run, as a call it left at its prompt is told.
PARENT_CAUSE = "its parent"
TIME_LIMIT_CAUSE = "its time limit"
TOKEN_LIMIT_CAUSE = "its token limit"


class RunControl:
    """One sub-agent run's control (see the module docstring).

    Args:
        ref: The run.
        limits: Its limits: its type's over the policy's.
        side_tokens: What the run's other model calls have used so far (its
            tools' side calls, its summaries; not its own steps, which
            ``used`` counts), input plus output tokens, as its usage record
            counts them: counted with its own at each check. None: none.

    Attributes:
        ref: The run.
        limits: Its limits.
        cancelled: ``(by, error)`` once it is cancelled: who cancelled it
            (``parent``, ``user`` or ``limit``) and why, one line.
    """

    def __init__(
        self,
        ref: RunRef,
        limits: AgentLimits,
        *,
        side_tokens: Callable[[], int] | None = None,
    ) -> None:
        self.ref = ref
        self.limits = limits
        self._side_tokens = side_tokens
        self.cancelled: tuple[str, str] | None = None
        self._steering: list[str] = []
        # Messages taken by a model call whose backend has not said it stored them.
        self._taken: list[str] = []
        self._calls = 0
        self._tokens = 0
        self._final = False
        self._task: asyncio.Task[Any] | None = None
        # Whether cancel() cancelled the task, and own_cancel() has not yet
        # taken that cancel.
        self._cancelled_task = False
        # Whether the hook said the run's work has ended (finish()).
        self._finished = False

    # -- the run's task --------------------------------------------------------

    def bind(self, task: asyncio.Task[Any]) -> None:
        """Make ``task`` the run's: what ``cancel`` cancels.

        A cancel that came before (the run had no task yet) cancels it now.
        """
        self._task = task
        if self.cancelled is not None and not task.done():
            self._cancel_task()

    def cancel(self, *, by: str, error: str) -> None:
        """Cancel the run's task: ``by`` its parent, the user or a limit, for ``error``.

        Does nothing once the run is cancelled (the first cancel wins), its
        task has ended or its work has (``finish``), and cancels no task its
        turn is cancelling already (the run ends as the turn's).

        Raises:
            ValueError: ``by`` is not ``parent``, ``user`` or ``limit``.
        """
        if by not in CANCELLED_BY:
            raise ValueError(f"A run is cancelled by its parent, user or limit, not {by!r}.")
        if (
            self.cancelled is not None
            or self._finished
            or (self._task is not None and self._task.done())
        ):
            return
        self.cancelled = (by, error)
        if self._task is not None:
            self._cancel_task()

    def finish(self) -> None:
        """In the run's task: its work has ended, so no cancel stops it from now on.

        A hook calls it as the run's stream ends, before it reads back what
        the run did (ADK reads its child session's state): a limit or a stop
        that lands during that read would otherwise end the run
        ``cancelled`` and lose an answer it has made. A cancel the run's
        work already took (``own_cancel``) stays. The turn's cancel (Ctrl+C)
        is not the control's, and still stops the read.
        """
        self._finished = True

    def own_cancel(self) -> bool:
        """In the run's task, handling a ``CancelledError``: whether it is the run's own cancel.

        True for the cancel ``cancel`` made (asked once): the task's cancel
        request is withdrawn (``uncancel``), so the hook may stop the run
        and return what it has. False for any other cancel (the turn's):
        the hook lets it go on.
        """
        if not self._cancelled_task:
            return False
        self._cancelled_task = False
        task = asyncio.current_task()
        if task is not None and task is self._task:
            task.uncancel()
        return True

    @contextmanager
    def timing(self) -> Iterator[None]:
        """Run the block under the run's time limit: past it, the run is cancelled by its limit.

        The clock starts with the block, as the run starts to work, and
        stops with it.
        """
        timeout = self.limits.timeout
        if timeout is None:
            yield
            return
        handle = asyncio.get_running_loop().call_later(timeout, self._out_of_time)
        try:
            yield
        finally:
            handle.cancel()

    def _out_of_time(self) -> None:
        self.cancel(by=LIMIT, error=TIME_LIMIT_ERROR)

    def _cancel_task(self) -> None:
        assert self._task is not None
        if self._task.cancelling():
            # Its turn is unwinding it already (Ctrl+C): one more cancel
            # would cut its cleanup short, and the run is the turn's.
            return
        self._cancelled_task = True
        self._task.cancel()

    # -- the limits at its model calls -------------------------------------------

    def model_call(self) -> bool:
        """Count a model call the run is about to make; True when it is the run's last.

        The last allowed (``max_model_calls``), or any call after one the
        token limit made the last, should a backend make one. That call's
        request ends with ``STEP_LIMIT_NOTE``, and its answer is the run's
        last (``final_call``).
        """
        self._calls += 1
        self._final = self._final or self.at_step_limit
        return self._final

    def used(self, input_tokens: int, output_tokens: int, *, calls: bool = True) -> None:
        """Count what one of the run's model calls used, as ``workflow/usage.py`` counts it.

        The run's total is its own calls' tokens and what its other calls
        used (``side_tokens``). Past ``max_tokens``, an answer that makes
        tool calls (``calls``: the run would go on past it) is the run's
        last, cancelled by its limit: ``cancelled`` is set and the call is
        the run's last (``final_call``), so it ends with that answer rather
        than losing it. An answer that makes none ends the run anyway: it is
        its result, whole, and nothing is cancelled. Its tokens are counted
        either way.
        """
        self._tokens += input_tokens + output_tokens
        limit = self.limits.max_tokens
        if limit is None or self.cancelled is not None or not calls:
            return
        side = self._side_tokens() if self._side_tokens is not None else 0
        if self._tokens + side <= limit:
            return
        self.cancelled = (LIMIT, TOKEN_LIMIT_ERROR)
        self._final = True

    @property
    def at_step_limit(self) -> bool:
        """Whether the run has made its last allowed model call."""
        limit = self.limits.max_model_calls
        return limit is not None and self._calls >= limit

    @property
    def final_call(self) -> bool:
        """Whether the model call in progress is the run's last: its answer's tool calls are dropped."""
        return self._final

    # -- steering --------------------------------------------------------------

    def steer(self, message: str) -> None:
        """Give the run ``message``, for its next model call.

        Raises:
            ValueError: ``message`` is not text, or blank.
        """
        if not isinstance(message, str) or not message.strip():
            raise ValueError("A message steered to a run is text.")
        self._steering.append(message)

    def take_steering(self) -> list[str]:
        """The messages given since the run's last model call, oldest first; given once."""
        taken, self._steering = self._steering, []
        self._taken.extend(taken)
        return taken

    def stored(self, count: int | None = None) -> None:
        """The run's transcript now holds the first ``count`` messages taken (all, for None)."""
        del self._taken[:count]

    def unstored(self) -> list[str]:
        """The messages the run's transcript does not hold: taken and not yet stored, then never taken."""
        return [*self._taken, *self._steering]

    @property
    def stop_cause(self) -> str | None:
        """What stopped the run, for a call it left at its prompt; None while it works or for the user's stop."""
        if self.cancelled is None:
            return None
        by, error = self.cancelled
        if by == "parent":
            return PARENT_CAUSE
        if by == LIMIT:
            return TOKEN_LIMIT_CAUSE if error == TOKEN_LIMIT_ERROR else TIME_LIMIT_CAUSE
        return None
