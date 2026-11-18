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
takes one back at ``RETURNING`` when the wait is over (``on_wait`` is
called when that take-back has to wait for a slot). A child that ends
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
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

#: A run taking its slot back, after waiting on its children or the user.
RETURNING = 0
#: A run that has not started.
NEW = 1

_HELD = "held"
_LENT = "lent"
_RELEASED = "released"

#: A waiting run's entry in the heap: its priority, its arrival, and the
#: future a slot handed to it sets (True), or a withdrawn take-back (False).
_Waiter = tuple[int, int, "asyncio.Future[bool]"]


class Slot:
    """One run's working slot, from a ``SlotPool``.

    Held while its run works; lent while its run waits on its children or
    the user (``SlotPool.lent``); released once the run has given it back.
    """

    __slots__ = ("_pool", "_state", "_lends", "_back")

    def __init__(self, pool: SlotPool) -> None:
        self._pool = pool
        self._state = _HELD
        # The lends in progress (``SlotPool.lent``, nested): the slot is lent
        # on the first and taken back after the last.
        self._lends = 0
        # Its run's take-back waiting for a slot (``SlotPool._take_back``),
        # from when it joins the heap until its wait ends: for a moment it
        # may name one ``_give`` has handed a slot already (out of the heap,
        # its task not yet resumed), whose take-back then keeps that slot or
        # hands it on.
        self._back: _Waiter | None = None

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
        self._waiters: list[_Waiter] = []
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
        else:
            self._withdraw(slot)

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
            self._withdraw(to)
            return
        self.release(slot)

    @asynccontextmanager
    async def lent(
        self, slot: Slot, *, on_wait: Callable[[], None] | None = None
    ) -> AsyncIterator[None]:
        """Lend ``slot`` for the block, while its run waits on its children or the user.

        A lend of a held slot frees it for another run (of nested lends, the
        first); when the last ends, it is taken back at ``RETURNING`` (at
        once when a slot is free, else ahead of the runs that have not
        started), unless a child handed it back already. ``on_wait``, when
        given, is called as that take-back starts to wait for a slot (none
        is free): the run's wait is over, and it now waits for its slot. A
        lend that starts while another of the run's waits is taking the slot
        back keeps it lent: that wait ends without it, and this one takes it
        back. A lend that ends in a cancel takes nothing back: its run is
        being stopped, and its ``release`` is then nothing. So a cancel that
        does not stop the run gives the slot up for the rest of it: an app's
        tool that puts ``asyncio.wait_for`` or ``asyncio.timeout`` around a
        question (``request_user_input``) and goes on when the timeout fires
        leaves its run working without a slot (another run may have it, so
        more than ``size`` work at once), and the run says
        ``input_required`` until its next question is answered, when it says
        ``running`` again, still without a slot, or until it ends. An error
        is no cancel: the run goes on, with its slot back.

        Raises:
            ValueError: ``slot`` is another pool's.
        """
        self._own(slot)
        slot._lends += 1
        if slot._state == _HELD:
            slot._state = _LENT
            self._give()
        else:
            # Lent already: an earlier wait of the run that is taking it back
            # no longer needs it, since the run waits again; this lend takes
            # it back when it ends.
            self._withdraw(slot)
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
                await self._take_back(slot, on_wait)
            raise
        slot._lends -= 1
        if slot._lends == 0:
            await self._take_back(slot, on_wait)

    async def _take_back(self, slot: Slot, on_wait: Callable[[], None] | None = None) -> None:
        """A lent ``slot`` held again, at ``RETURNING``; nothing for one handed back or released.

        A take-back in the heap the slot no longer needs (a child handed it
        back, its run released it, or the run lent it again) is withdrawn
        and takes nothing. One handed a slot before it could be withdrawn
        gives that slot to the next waiter when the slot is lent again,
        handed back or released by the time it goes on; else it holds the
        slot, and a take-back queued meanwhile (a wait of the run that
        started and ended in between) is withdrawn. So a run's slot has at
        most one take-back waiting for a slot, and none once it is held.
        ``on_wait`` is called if it has to wait.
        """
        if slot._state != _LENT:
            return
        try:
            took = await self._take(RETURNING, slot, on_wait)
        except asyncio.CancelledError:
            if slot._state == _LENT and slot._lends == 0:
                slot._state = _RELEASED
            raise
        if not took:
            return
        if slot._state != _LENT or slot._lends > 0:
            self._give()
            return
        slot._state = _HELD
        # A take-back another wait of the run queued while this one was
        # being handed the slot: the slot is back, so it ends without one.
        self._withdraw(slot)

    async def _take(
        self,
        priority: int,
        slot: Slot | None = None,
        on_wait: Callable[[], None] | None = None,
    ) -> bool:
        """Count one slot taken: a free one, else wait in the heap until one is handed over.

        False when the wait is ``slot``'s take-back and was withdrawn
        (``_withdraw``): no slot was taken. ``on_wait`` is called when it
        has to wait, before its entry goes in the heap.
        """
        if self._free > 0:
            self._free -= 1
            return True
        if on_wait is not None:
            on_wait()
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        entry = (priority, next(self._arrivals), future)
        heapq.heappush(self._waiters, entry)
        if slot is not None:
            slot._back = entry
        try:
            return await future
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                if future.result():
                    # Handed a slot in the tick it was cancelled: the next one's.
                    self._give()
            else:
                self._leave(entry)
            raise
        finally:
            if slot is not None and slot._back is entry:
                slot._back = None

    def _give(self) -> None:
        """A slot came back: hand it to the first waiter still waiting, else it is free."""
        while self._waiters:
            _, _, future = heapq.heappop(self._waiters)
            if not future.done():
                future.set_result(True)
                return
        self._free += 1

    def _withdraw(self, slot: Slot) -> None:
        """End ``slot``'s take-back waiting in the heap, if any, without a slot."""
        entry, slot._back = slot._back, None
        if entry is None:
            return
        self._leave(entry)
        if not entry[2].done():
            entry[2].set_result(False)

    def _leave(self, entry: _Waiter) -> None:
        """Take a cancelled or withdrawn waiter's ``entry`` out of the heap.

        ``_give`` may have popped it already.
        """
        for index, waiting in enumerate(self._waiters):
            if waiting is entry:
                self._waiters.pop(index)
                heapq.heapify(self._waiters)
                return

    def _own(self, slot: Slot) -> None:
        if slot._pool is not self:
            raise ValueError("A slot goes back to the pool it came from.")
