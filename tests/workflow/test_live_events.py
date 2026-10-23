"""Events reported while a turn's stream is blocked (``workflow/live_events.py``).

A summary runs inside a model call, so the turn's stream yields nothing
while it does; "Summarizing the earlier conversation…" has to show anyway.
``merged(source, live)`` pumps the source in one task, in lockstep with the
consumer, and yields each event ``live`` receives as soon as it arrives, in
the order emitted, before the item that follows it. Code deep inside the
turn reports through the context-local reporter ``reporting(live)`` sets.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars

import pytest

from agentic_cli.workflow import live_events
from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.live_events import LiveEvents, current, merged, report, reporting


def note(text: str) -> WorkflowEvent:
    return WorkflowEvent.text(text)


async def consume(stream) -> list:
    return [item async for item in stream]


def contents(items: list) -> list:
    """Each item, an event as its text."""
    return [item.content if isinstance(item, WorkflowEvent) else item for item in items]


class TestMerged:
    async def test_items_come_through_in_order(self):
        async def source():
            yield 1
            yield 2

        assert await consume(merged(source(), LiveEvents())) == [1, 2]

    async def test_an_event_emitted_while_the_source_is_blocked_comes_first(self):
        live = LiveEvents()
        released = asyncio.Event()
        seen: list = []

        async def source():
            yield "first"
            live.emit(note("summarizing"))
            await released.wait()  # the summary call: the stream yields nothing
            yield "second"

        stream = merged(source(), live)
        seen.append(await anext(stream))
        seen.append(await anext(stream))  # arrives while the source is still blocked
        released.set()
        seen += [item async for item in stream]

        assert contents(seen) == ["first", "summarizing", "second"]

    async def test_events_keep_the_order_they_were_emitted_in(self):
        live = LiveEvents()

        async def source():
            live.emit(note("a"))
            live.emit(note("b"))
            yield "item"
            live.emit(note("c"))

        assert contents(await consume(merged(source(), live))) == ["a", "b", "item", "c"]

    async def test_the_source_is_not_pulled_ahead_of_the_consumer(self):
        pulled: list[int] = []

        async def source():
            for n in range(3):
                pulled.append(n)
                yield n

        stream = merged(source(), LiveEvents())
        assert await anext(stream) == 0
        for _ in range(5):
            await asyncio.sleep(0)
        assert pulled == [0]  # the next item waits for the consumer

        assert await anext(stream) == 1
        assert pulled == [0, 1]
        await stream.aclose()

    async def test_an_exception_from_the_source_reaches_the_consumer(self):
        live = LiveEvents()

        async def source():
            yield 1
            live.emit(note("reported before the failure"))
            raise ValueError("boom")

        seen: list = []
        with pytest.raises(ValueError, match="boom"):
            async for item in merged(source(), live):
                seen.append(item)

        assert contents(seen) == [1, "reported before the failure"]

    async def test_a_cancelled_consumer_cancels_the_pump_and_closes_the_source(self):
        blocked = asyncio.Event()
        closed = asyncio.Event()

        async def source():
            try:
                yield 1
                blocked.set()
                await asyncio.Event().wait()  # never
                yield 2
            finally:
                closed.set()

        async def consumer():
            async with contextlib.aclosing(merged(source(), LiveEvents())) as stream:
                async for _ in stream:
                    pass

        task = asyncio.create_task(consumer())
        await blocked.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert closed.is_set()
        assert not [t for t in asyncio.all_tasks() if t.get_name() == live_events.PUMP_TASK]

    async def test_a_consumer_that_stops_early_closes_the_source_in_the_pump(self):
        closing_task: list = []

        async def source():
            try:
                yield 1
                yield 2
            finally:
                closing_task.append(asyncio.current_task().get_name())

        stream = merged(source(), LiveEvents())
        assert await anext(stream) == 1
        await stream.aclose()

        # Closed in the task that iterated it, as an anyio-based stream needs.
        assert closing_task == [live_events.PUMP_TASK]

    async def test_events_emitted_after_the_last_item_still_come(self):
        live = LiveEvents()

        async def source():
            yield "last"
            live.emit(note("after"))

        assert contents(await consume(merged(source(), live))) == ["last", "after"]

    async def test_an_event_from_another_thread(self):
        live = LiveEvents()

        async def source():
            await asyncio.to_thread(live.emit, note("from a thread"))
            yield "item"

        assert contents(await consume(merged(source(), live))) == ["from a thread", "item"]

    async def test_the_sources_context_is_the_callers(self):
        variable: contextvars.ContextVar[str] = contextvars.ContextVar("variable", default="unset")
        seen: list[str] = []

        async def source():
            seen.append(variable.get())
            yield "item"

        token = variable.set("the turn's")
        try:
            await consume(merged(source(), LiveEvents()))
        finally:
            variable.reset(token)

        assert seen == ["the turn's"]

    async def test_a_source_that_swallows_the_cancel_does_not_hang_the_consumer(self):
        closed = asyncio.Event()

        async def source():
            try:
                yield 1
                try:
                    await asyncio.Event().wait()  # never
                except asyncio.CancelledError:
                    pass  # a library that cleans up and yields what it had
                yield 2
                yield 3
            finally:
                closed.set()

        stream = merged(source(), LiveEvents())
        assert await anext(stream) == 1
        waiting = asyncio.ensure_future(anext(stream))
        await asyncio.sleep(0.01)
        waiting.cancel()

        # The pump stops instead of pulling 3, and the source is closed.
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(waiting), 2)
        await asyncio.wait_for(stream.aclose(), 2)
        assert closed.is_set()
        assert not [t for t in asyncio.all_tasks() if t.get_name() == live_events.PUMP_TASK]

    async def test_a_base_exception_raised_in_the_source_reaches_the_consumer(self):
        class Halt(BaseException):
            """Neither an Exception nor the consumer's own cancel."""

        async def source():
            yield 1
            raise Halt

        with pytest.raises(Halt):
            await asyncio.wait_for(consume(merged(source(), LiveEvents())), 2)


class TestReporting:
    async def test_without_a_reporter_nothing_is_reported(self):
        assert current() is None
        assert report(note("lost")) is False

    async def test_report_reaches_the_reporter_of_the_context(self):
        live = LiveEvents()

        async def source():
            assert report(note("reported")) is True
            yield "item"

        with reporting(live) as reporter:
            assert reporter is live
            assert current() is live
            items = await consume(merged(source(), live))
        assert current() is None

        assert contents(items) == ["reported", "item"]

    async def test_reporting_nests_and_restores(self):
        outer, inner = LiveEvents(), LiveEvents()

        with reporting(outer):
            with reporting(inner):
                assert current() is inner
            assert current() is outer

    async def test_first_is_true_once_per_key(self):
        live = LiveEvents()

        assert live.first("summary_failed") is True
        assert live.first("summary_failed") is False
        assert live.first("fit_guard") is True
        assert LiveEvents().first("summary_failed") is True  # per instance

