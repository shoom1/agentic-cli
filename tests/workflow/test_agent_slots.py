"""Working slots (``workflow/agents/slots.py``), alone.

A ``SlotPool`` lets ``size`` runs work at once; the others wait in a heap
ordered by priority (``RETURNING`` before ``NEW``), then arrival. A run that
waits on its children or the user lends its slot and takes it back after,
and every cancel leaves the pool as it found it: no waiter left in the heap,
no slot lost or counted twice. Driven here with plain asyncio tasks.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from agentic_cli.workflow.agents.slots import NEW, RETURNING, Slot, SlotPool
from tests.loop_guard import no_blocking_calls


async def until(condition: Callable[[], bool], *, within: float = 5.0) -> None:
    """Wait until ``condition()`` holds, letting the other tasks run."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + within
    while not condition():
        if loop.time() > deadline:
            raise AssertionError(f"not within {within} s")
        await asyncio.sleep(0)


def waiter(pool: SlotPool, name: str, served: list[str], priority: int = NEW) -> asyncio.Task[Slot]:
    """A task that acquires a slot at ``priority`` and writes ``name`` once it has one."""

    async def wait() -> Slot:
        slot = await pool.acquire(priority)
        served.append(name)
        return slot

    return asyncio.create_task(wait(), name=name)


class TestAcquireAndRelease:
    async def test_a_run_gets_a_free_slot_at_once_and_gives_it_back(self):
        pool = SlotPool(2)

        first, second = await pool.acquire(), await pool.acquire()

        assert (pool.free, pool.waiting, first.held, second.held) == (0, 0, True, True)
        pool.release(first)
        assert (pool.free, first.held) == (1, False)
        pool.release(first)  # given back once: a second release is nothing
        assert pool.free == 1

    async def test_waiters_are_served_by_priority_then_arrival(self):
        pool = SlotPool(1)
        holder = await pool.acquire()
        served: list[str] = []
        a = waiter(pool, "a", served)
        b = waiter(pool, "b", served)
        back = waiter(pool, "back", served, RETURNING)
        await until(lambda: pool.waiting == 3)

        for task in (back, a, b):
            pool.release(holder)
            holder = await task

        assert served == ["back", "a", "b"]
        assert (pool.free, pool.waiting) == (0, 0)

    def test_its_size_is_a_positive_int(self):
        for size in (0, -1, True, 1.5, "2"):
            with pytest.raises(ValueError, match="positive int"):
                SlotPool(size)

    async def test_a_slot_goes_back_to_its_own_pool(self):
        one, other = SlotPool(1), SlotPool(1)
        slot = await one.acquire()

        with pytest.raises(ValueError, match="pool it came from"):
            other.release(slot)


class TestCancellation:
    async def test_a_cancelled_waiter_leaves_the_heap_and_the_next_is_served(self):
        pool = SlotPool(1)
        holder = await pool.acquire()
        served: list[str] = []
        a, b = waiter(pool, "a", served), waiter(pool, "b", served)
        await until(lambda: pool.waiting == 2)

        a.cancel()
        with pytest.raises(asyncio.CancelledError):
            await a
        assert pool.waiting == 1
        pool.release(holder)
        await b

        assert served == ["b"]
        assert (pool.free, pool.waiting) == (0, 0)

    async def test_a_waiter_handed_a_slot_and_cancelled_in_the_same_tick_hands_it_on(self):
        pool = SlotPool(1)
        holder = await pool.acquire()
        served: list[str] = []
        a, b = waiter(pool, "a", served), waiter(pool, "b", served)
        await until(lambda: pool.waiting == 2)

        pool.release(holder)  # a's: handed over, a not yet resumed
        a.cancel()  # the same tick
        with pytest.raises(asyncio.CancelledError):
            await a
        slot = await b

        assert served == ["b"]
        pool.release(slot)
        assert (pool.free, pool.waiting) == (1, 0)

    async def test_a_cancelled_holder_releases_in_its_finally(self):
        pool = SlotPool(1)
        working = asyncio.Event()

        async def run() -> None:
            slot = await pool.acquire()
            try:
                working.set()
                await asyncio.Event().wait()
            finally:
                pool.release(slot)

        task = asyncio.create_task(run())
        await working.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert (pool.free, pool.waiting) == (1, 0)


class TestLending:
    async def test_a_lent_slot_works_for_another_run_and_is_taken_back_after(self):
        pool = SlotPool(1)
        parent = await pool.acquire()
        served: list[str] = []

        async with pool.lent(parent):
            assert (parent.lent, parent.held, pool.free) == (True, False, 1)
            child = await waiter(pool, "child", served)
            pool.release(child)

        assert (parent.held, pool.free, served) == (True, 0, ["child"])
        pool.release(parent)
        assert pool.free == 1

    async def test_a_returning_run_goes_ahead_of_runs_that_have_not_started(self):
        pool = SlotPool(1)
        parent = await pool.acquire()
        served: list[str] = []
        first = waiter(pool, "first", served)
        await until(lambda: pool.waiting == 1)

        async def lends() -> None:
            async with pool.lent(parent):
                await until(lambda: served == ["first"])
                waiter(pool, "second", served)
                await until(lambda: pool.waiting == 1)
            served.append("parent")

        lending = asyncio.create_task(lends())
        holding = await first  # the parent's lent slot
        await until(lambda: pool.waiting == 2)  # the parent waits to take it back, and "second"
        pool.release(holding)
        await lending
        pool.release(parent)
        await until(lambda: len(served) == 3)

        assert served == ["first", "parent", "second"]

    async def test_nested_lends_of_one_slot_count_once(self):
        pool = SlotPool(1)
        parent = await pool.acquire()

        async with pool.lent(parent):
            async with pool.lent(parent):
                assert pool.free == 1
            assert (parent.lent, pool.free) == (True, 1)  # still waiting on the other
        assert (parent.held, pool.free) == (True, 0)

    async def test_a_run_cancelled_while_its_slot_is_lent_does_not_take_it_back(self):
        pool = SlotPool(1)
        lending = asyncio.Event()

        async def run() -> None:
            slot = await pool.acquire()
            try:
                async with pool.lent(slot):
                    lending.set()
                    await asyncio.Event().wait()
            finally:
                pool.release(slot)

        task = asyncio.create_task(run())
        await lending.wait()
        other = await pool.acquire()  # the lent slot, at work for another run
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert (pool.free, pool.waiting) == (0, 0)
        pool.release(other)
        assert pool.free == 1

    async def test_a_run_cancelled_while_it_waits_to_take_its_slot_back_leaves_the_heap(self):
        pool = SlotPool(1)
        lending = asyncio.Event()

        async def run() -> None:
            slot = await pool.acquire()
            try:
                async with pool.lent(slot):
                    lending.set()
                    await asyncio.sleep(0)
            finally:
                pool.release(slot)

        task = asyncio.create_task(run())
        await lending.wait()
        other = await pool.acquire()  # holds the slot while the run wants it back
        await until(lambda: pool.waiting == 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert pool.waiting == 0
        pool.release(other)
        assert pool.free == 1

    async def test_an_error_while_lent_is_no_cancel_the_slot_comes_back(self):
        pool = SlotPool(1)
        slot = await pool.acquire()

        with pytest.raises(RuntimeError):
            async with pool.lent(slot):
                raise RuntimeError("no one to ask")

        assert (slot.held, pool.free) == (True, 0)

    async def test_a_wait_that_starts_while_the_slot_waits_to_come_back_lends_it_on(self):
        """Two waits of one run that do not overlap (two calls of one
        response): the second starts while the first waits for the slot to
        come back. The first ends without it, since the run waits again, and
        the slot comes back once, after the second."""
        pool = SlotPool(1)
        parent = await pool.acquire()
        first_over, second_over = asyncio.Event(), asyncio.Event()

        async def waits(over: asyncio.Event) -> None:
            async with pool.lent(parent):
                await over.wait()

        first = asyncio.create_task(waits(first_over))
        await until(lambda: parent.lent)
        other = await pool.acquire()  # works in the parent's lent slot
        first_over.set()
        await until(lambda: pool.waiting == 1)  # the first wants it back
        second = asyncio.create_task(waits(second_over))
        try:
            await until(first.done, within=1)
            assert (parent.lent, pool.waiting) == (True, 0)
            second_over.set()
            await until(lambda: pool.waiting == 1)  # the second wants it back
            pool.release(other)
            await until(second.done, within=1)
        finally:
            first.cancel()
            second.cancel()

        assert (parent.held, pool.free, pool.waiting) == (True, 0, 0)
        pool.release(parent)
        assert (pool.free, pool.waiting) == (1, 0)

    async def test_a_wait_over_at_once_as_the_slot_comes_back_leaves_no_wait_behind(self):
        """The run's first wait has been handed its slot back but has not yet
        gone on when another of its waits starts and ends at once (a prompt
        answered without waiting): the run holds its slot once, and nothing
        of it is left waiting in the heap."""
        pool = SlotPool(1)
        parent = await pool.acquire()
        first_over = asyncio.Event()

        async def waits() -> None:
            async with pool.lent(parent):
                await first_over.wait()

        async def over_at_once() -> None:
            async with pool.lent(parent):
                pass

        first = asyncio.create_task(waits())
        await until(lambda: parent.lent)
        other = await pool.acquire()  # works in the parent's lent slot
        first_over.set()
        await until(lambda: pool.waiting == 1)  # the first wants it back
        second = asyncio.create_task(over_at_once())  # runs before the first wakes
        pool.release(other)  # the first is handed the slot
        try:
            await asyncio.wait_for(asyncio.gather(first, second), 1)
        finally:
            first.cancel()
            second.cancel()

        assert (parent.held, pool.free, pool.waiting) == (True, 0, 0)
        pool.release(parent)
        assert (pool.free, pool.waiting) == (1, 0)

    async def test_a_slot_released_while_it_waits_to_come_back_is_not_taken(self):
        pool = SlotPool(1)
        slot = await pool.acquire()
        over = asyncio.Event()

        async def waits() -> None:
            async with pool.lent(slot):
                await over.wait()

        lending = asyncio.create_task(waits())
        await until(lambda: slot.lent)
        other = await pool.acquire()
        over.set()
        await until(lambda: pool.waiting == 1)
        pool.release(slot)  # its run ended meanwhile
        try:
            await until(lending.done, within=1)
        finally:
            lending.cancel()

        assert pool.waiting == 0
        pool.release(other)
        assert (pool.free, pool.waiting) == (1, 0)


class TestHandBack:
    async def test_a_child_hands_its_slot_to_its_parent_waiting_on_it_alone(self):
        pool = SlotPool(1)
        parent = await pool.acquire()
        served: list[str] = []

        async with pool.lent(parent):
            child = await pool.acquire()
            late = waiter(pool, "late", served)
            await until(lambda: pool.waiting == 1)
            pool.hand_back(child, to=parent)
            assert (parent.held, child.held) == (True, False)

        assert (parent.held, pool.free, pool.waiting, served) == (True, 0, 1, [])
        pool.release(parent)
        await late
        assert served == ["late"]

    async def test_to_a_parent_waiting_to_take_its_slot_back_ends_that_wait(self):
        """The parent's wait is over and it waits in the heap for its slot
        when its child hands it back: that take-back ends without a slot,
        and the parent holds the child's."""
        pool = SlotPool(1)
        parent = await pool.acquire()
        over = asyncio.Event()

        async def waits() -> None:
            async with pool.lent(parent):
                await over.wait()

        lending = asyncio.create_task(waits())
        await until(lambda: parent.lent)
        child = await pool.acquire()  # works in the parent's lent slot
        over.set()
        await until(lambda: pool.waiting == 1)  # the parent wants it back
        pool.hand_back(child, to=parent)
        try:
            await until(lending.done, within=1)
        finally:
            lending.cancel()

        assert (parent.held, child.held, pool.free, pool.waiting) == (True, False, 0, 0)
        pool.release(parent)
        assert (pool.free, pool.waiting) == (1, 0)

    async def test_to_a_parent_that_does_not_lend_it_is_a_release(self):
        pool = SlotPool(2)
        parent, child = await pool.acquire(), await pool.acquire()

        pool.hand_back(child, to=parent)

        assert (parent.held, child.held, pool.free) == (True, False, 1)

    async def test_a_parent_handed_its_slot_lends_it_again_for_a_new_wait(self):
        """A wait that starts before the lend its child ended has closed
        lends the slot again; the last lend to end takes it back."""
        pool = SlotPool(1)
        parent = await pool.acquire()

        async with pool.lent(parent):
            child = await pool.acquire()
            pool.hand_back(child, to=parent)
            assert (parent.held, pool.free) == (True, 0)
            async with pool.lent(parent):  # the user, asked meanwhile
                assert (parent.lent, pool.free) == (True, 1)
            assert (parent.lent, pool.free) == (True, 1)
        assert (parent.held, pool.free) == (True, 0)


async def test_no_path_blocks_the_loop():
    """Spec 13: every new async path once under the loop guard."""
    pool = SlotPool(1)
    with no_blocking_calls():
        holder = await pool.acquire()
        served: list[str] = []
        queued = waiter(pool, "queued", served)
        async with pool.lent(holder):
            pool.release(await queued)
        pool.release(holder)

    assert (pool.free, pool.waiting, served) == (1, 0, ["queued"])
