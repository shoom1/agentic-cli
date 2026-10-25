"""Events reported while a turn's stream is blocked.

A turn's events come from its backend's stream, one item at a time. Some
work runs where the stream yields nothing: a summary call runs inside a
model call (ADK's ``run_async``, Deep Agents' model node), and can take a
minute at a large size. What it reports (``CONTEXT_SUMMARIZING`` as it
starts) must reach the user while it runs, not after.

``merged(source, live)`` iterates the source in one pump task, in lockstep
with the consumer, and yields every event ``live`` receives as soon as it
arrives. Code deep inside the turn reports through the context-local
reporter: ``reporting(live)`` sets it for the turn, ``report(event)`` emits
to it, from any thread.

    live = LiveEvents()
    with reporting(live):
        async for item in merged(stream, live):
            ...  # a stream item, or a WorkflowEvent someone reported
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable, Iterator
from contextvars import ContextVar
from typing import Any, TypeVar

from agentic_cli.logging import Loggers
from agentic_cli.workflow.events import WorkflowEvent

logger = Loggers.workflow()

T = TypeVar("T")

#: The name of the task that iterates a merged source.
PUMP_TASK = "live-events-pump"

#: What the queue carries: an event someone emitted, the source's next item,
#: the exception it raised, or its end.
_EVENT = "event"
_ITEM = "item"
_ERROR = "error"
_DONE = "done"

_reporter: ContextVar["LiveEvents | None"] = ContextVar(
    "agentic_cli_live_events", default=None
)


class LiveEvents:
    """One turn's channel for events that cannot wait for the stream.

    Made inside the turn's event loop, which it captures: ``emit`` may then
    be called from any thread, and never blocks.
    """

    def __init__(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        self._seen: set[str] = set()

    def emit(self, event: WorkflowEvent) -> None:
        """Hand ``event`` to whoever merges this channel, from any thread."""
        self._put(_EVENT, event)

    def first(self, key: str) -> bool:
        """True the first time ``key`` is asked for, on this channel.

        For a notice given once per turn (a summary that failed, requests
        trimmed to fit).
        """
        if key in self._seen:
            return False
        self._seen.add(key)
        return True

    def _put(self, kind: str, value: Any) -> None:
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            self._queue.put_nowait((kind, value))
            return
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, (kind, value))
        except RuntimeError:  # the turn's loop is closed: nobody is listening
            logger.debug("live_event_dropped", kind=kind)


@contextlib.contextmanager
def reporting(live: LiveEvents) -> Iterator[LiveEvents]:
    """Make ``live`` the reporter of the current context, and restore the last."""
    token = _reporter.set(live)
    try:
        yield live
    finally:
        _reporter.reset(token)


def report(event: WorkflowEvent) -> bool:
    """Emit ``event`` to the current context's reporter; False when there is none."""
    live = _reporter.get()
    if live is None:
        return False
    live.emit(event)
    return True


def current() -> LiveEvents | None:
    """The current context's reporter, if any."""
    return _reporter.get()


#: How long a turn cancelled twice still waits for its source to close.
STOP_GRACE = 5.0


async def merged(
    source: AsyncIterator[T], live: LiveEvents, *, on_stop: Callable[[], None] | None = None
) -> AsyncIterator[T | WorkflowEvent]:
    """``source``'s items, with the events ``live`` receives as they arrive.

    One pump task, created in the caller's context (so its context variables
    reach the source), pulls the source's next item only after the consumer
    took the previous one. Events emitted meanwhile are yielded as soon as
    they arrive, in the order emitted, before the item that follows them; an
    event emitted after the last item still comes. An exception from the
    source is raised to the consumer after the events emitted before it.

    When the consumer stops (done, closed or cancelled), the pump is told to
    stop, cancelled and awaited, and the source is closed in the pump's own
    task: a stream built on anyio or LangGraph's runner must be closed in the
    task that iterated it. A source that swallows the cancel and yields once
    more does not keep the pump pulling. ``on_stop`` is called when the
    consumer stops before the source ended, before the source is cancelled:
    ADK's runner seals what a stopped invocation left open only when its
    abort signal is set by then.
    """
    queue = live._queue
    ended = False
    pulled = asyncio.Event()
    pulled.set()
    stopped = asyncio.Event()
    pump = asyncio.create_task(_pump(source, live, pulled, stopped), name=PUMP_TASK)
    try:
        while True:
            kind, value = await queue.get()
            if kind == _EVENT:
                yield value
            elif kind == _ITEM:
                yield value
                pulled.set()
            elif kind == _ERROR:
                ended = True
                raise value
            else:  # _DONE
                ended = True
                break
        while not queue.empty():
            kind, value = queue.get_nowait()
            if kind == _EVENT:
                yield value
    finally:
        if not ended and on_stop is not None:
            on_stop()
        # Wake a pump waiting for the consumer, so that it stops instead of
        # pulling again, then cancel one still inside the source, and wait
        # for it: ADK seals a stopped run as its runner closes. A second
        # cancel (Ctrl+C pressed again) waits too, up to STOP_GRACE, so the
        # turn ends only once that is done, then raises.
        stopped.set()
        pulled.set()
        pump.cancel()
        again: BaseException | None = None
        deadline: float | None = None
        loop = asyncio.get_running_loop()
        while not pump.done():
            timeout = None if deadline is None else max(0.0, deadline - loop.time())
            try:
                await asyncio.wait([pump], timeout=timeout)
            except asyncio.CancelledError as cancel:
                again = cancel
                deadline = deadline or loop.time() + STOP_GRACE
                continue
            if not pump.done():  # past the grace: it ends on its own
                logger.debug("live_events_pump_left_running")
                break
        if pump.done() and not pump.cancelled() and pump.exception() is not None:
            logger.debug("live_events_pump_failed", error_type=type(pump.exception()).__name__)
        if again is not None:
            raise again


async def _pump(
    source: AsyncIterator[Any],
    live: LiveEvents,
    pulled: asyncio.Event,
    stopped: asyncio.Event,
) -> None:
    """Pull ``source`` one item at a time, as the consumer takes each, until it stops."""
    try:
        while True:
            await pulled.wait()
            if stopped.is_set():
                return
            pulled.clear()
            try:
                item = await anext(source)
            except StopAsyncIteration:
                live._put(_DONE, None)
                return
            except Exception as error:
                live._put(_ERROR, error)
                return
            except BaseException as error:
                # Raised within the source (a cancel of its own, say), not
                # because the consumer stopped: the consumer, still waiting,
                # must not wait forever.
                if not stopped.is_set():
                    live._put(_ERROR, error)
                raise
            live._put(_ITEM, item)
    finally:
        aclose = getattr(source, "aclose", None)
        if aclose is not None:
            await aclose()
