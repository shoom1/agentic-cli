"""Working slots: how many sub-agent runs work at once.

A conversation's sub-agent runs share a ``SlotPool`` of
``SubAgentPolicy.max_active`` slots (``agents/delegate.py::Delegations``). A
run takes one (``acquire``) before it starts to work and gives it back
(``release``) when it ends, however it ends. A run that finds every slot
held waits for one, in a heap ordered by priority, then arrival:
``RETURNING`` (a run taking its slot back) before ``NEW`` (a run that has
not started).

A run that waits on its child runs or on the user lends its slot meanwhile
(``async with pool.lent(slot):``): another run works with it, and the run
takes one back at ``RETURNING`` when the wait is over. A child that ends
while its parent waits on it alone hands its slot straight to the parent
(``hand_back``: whether the parent waits on anything else is its caller's
to know), ahead of every waiter. Everything holding a slot is working, so
nested delegations never deadlock, and the trees in progress finish before
new runs start.

A cancel never loses or leaks a slot: a waiter cancelled leaves the heap
(one handed a slot in the tick it was cancelled hands it to the next); a
holder releases in its ``finally``; a run cancelled while its slot is lent
does not take it back, and its release is then nothing.

Not ``asyncio.Semaphore``: it wakes its waiters in arrival order, with no
priority. All of it runs on the event loop; nothing here blocks.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

#: A run taking its slot back, after waiting on its children or the user.
RETURNING = 0
#: A run that has not started.
NEW = 1

_HELD = "held"
_LENT = "lent"
_RELEASED = "released"


class Slot:
    """One run's working slot, from a ``SlotPool``.

    Held while its run works; lent while its run waits on its children or
    the user (``SlotPool.lent``); released once the run has given it back.
    """

    __slots__ = ("_pool", "_state", "_lends")

    def __init__(self, pool: SlotPool) -> None:
        self._pool = pool
        self._state = _HELD
        # The lends in progress (``SlotPool.lent``, nested): the slot is lent
        # on the first and taken back after the last.
        self._lends = 0

    @property
    def held(self) -> bool:
        """Whether its run holds it: it works with it."""
        return self._state == _HELD

    @property
    def lent(self) -> bool:
        """Whether its run lends it, while it waits."""
        return self._state == _LENT


class SlotPool:
    """``size`` working slots, and the runs waiting for one (see the module docstring).

    Args:
        size: How many runs work at once.

    Raises:
        ValueError: ``size`` is not a positive int.
    """

    def __init__(self, size: int) -> None:
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError(f"A slot pool's size is a positive int, not {size!r}.")
        self.size = size
        self._free = size
        # One (priority, arrival, future) per waiting run, a heap: a slot
        # handed to it sets its future's result. Arrivals are unique, so two
        # entries never compare their futures.
        self._waiters: list[tuple[int, int, asyncio.Future[None]]] = []
        self._arrivals = itertools.count()

    @property
    def free(self) -> int:
        """How many slots no run holds (a run acquiring one waits only while this is 0)."""
        return self._free

    @property
    def waiting(self) -> int:
        """How many runs wait for a slot."""
        return len(self._waiters)

    async def acquire(self, priority: int = NEW) -> Slot:
        """A slot for a run: at once when one is free, else once one is handed to it.

        The waiters are served by ``priority`` (``RETURNING`` before ``NEW``),
        then in the order they came. A cancel while it waits takes it out of
        the heap; a slot handed to it in the same tick goes to the next.
        """
        await self._take(priority)
        return Slot(self)

    def release(self, slot: Slot) -> None:
        """Give ``slot`` back: to the first waiter, else it is free.

        Nothing for a slot released already, nor for a lent one (its run
        was cancelled while it waited: the slot went back as it was lent).

        Raises:
            ValueError: ``slot`` is another pool's.
        """
        self._own(slot)
        state, slot._state = slot._state, _RELEASED
        if state == _HELD:
            self._give()

    def hand_back(self, slot: Slot, to: Slot) -> None:
        """Release a child's ``slot`` at its end to ``to``, its parent's, which waits on it alone.

        A lent ``to`` is held again with the child's slot, ahead of every
        waiter: its lends then end without waiting, and one that starts
        before they have ended lends it again. A ``to`` that is not lent
        takes nothing: as ``release``. The caller knows that the parent
        waits on nothing else (no other child, not the user).

        Raises:
            ValueError: either slot is another pool's.
        """
        self._own(slot)
        self._own(to)
        if slot._state == _HELD and to._state == _LENT:
            slot._state = _RELEASED
            to._state = _HELD
            return
        self.release(slot)

    @asynccontextmanager
    async def lent(self, slot: Slot) -> AsyncIterator[None]:
        """Lend ``slot`` for the block, while its run waits on its children or the user.

        A lend of a held slot frees it for another run (of nested lends, the
        first); when the last ends, it is taken back at ``RETURNING`` (at
        once when a slot is free, else ahead of the runs that have not
        started), unless a child handed it back already. A lend that ends in
        a cancel takes nothing back: its run is being stopped, and its
        ``release`` is then nothing. An error is no cancel: the run goes on,
        with its slot back.

        Raises:
            ValueError: ``slot`` is another pool's.
        """
        self._own(slot)
        slot._lends += 1
        if slot._state == _HELD:
            slot._state = _LENT
            self._give()
        try:
            yield
        except asyncio.CancelledError:
            slot._lends -= 1
            if slot._lends == 0 and slot._state == _LENT:
                slot._state = _RELEASED
            raise
        except BaseException:
            slot._lends -= 1
            if slot._lends == 0:
                await self._take_back(slot)
            raise
        slot._lends -= 1
        if slot._lends == 0:
            await self._take_back(slot)

    async def _take_back(self, slot: Slot) -> None:
        """A lent ``slot`` held again, at ``RETURNING``; nothing for one handed back or released."""
        if slot._state != _LENT:
            return
        try:
            await self._take(RETURNING)
        except asyncio.CancelledError:
            slot._state = _RELEASED
            raise
        slot._state = _HELD

    async def _take(self, priority: int) -> None:
        """Count one slot taken: a free one, else wait in the heap until one is handed over."""
        if self._free > 0:
            self._free -= 1
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        entry = (priority, next(self._arrivals), future)
        heapq.heappush(self._waiters, entry)
        try:
            await future
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                # Handed a slot in the tick it was cancelled: the next one's.
                self._give()
            else:
                self._leave(entry)
            raise

    def _give(self) -> None:
        """A slot came back: hand it to the first waiter still waiting, else it is free."""
        while self._waiters:
            _, _, future = heapq.heappop(self._waiters)
            if not future.done():
                future.set_result(None)
                return
        self._free += 1

    def _leave(self, entry: tuple[int, int, asyncio.Future[None]]) -> None:
        """Take a cancelled waiter's ``entry`` out of the heap (``_give`` may have popped it)."""
        for index, waiting in enumerate(self._waiters):
            if waiting is entry:
                self._waiters.pop(index)
                heapq.heapify(self._waiters)
                return

    def _own(self, slot: Slot) -> None:
        if slot._pool is not self:
            raise ValueError("A slot goes back to the pool it came from.")
