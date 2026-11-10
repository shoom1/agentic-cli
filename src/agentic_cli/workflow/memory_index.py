"""The memory index the main agent's requests carry: one snapshot per conversation.

A main agent with memory tools gets an index of the saved memories (one line
each) first in its requests, so it need not search to know what was saved.
The index is taken once per conversation, the first time the conversation
asks for it, and never stored in it: it stays the same for the conversation's
life, so the prompt prefix stays cached. It is dropped where the prefix
changes anyway: ``/resume``, ``/compact`` and any summary (the next request
takes a new one).

:class:`MemoryIndex` is what a manager holds for that: the snapshots, and the
code that takes one (the memory store's ``index``, at the cap the root's room
allows). It reads four things of its manager, each as a callable, so it needs
neither the manager nor a backend: the backends take the manager's own
``_memory_index_now``, ``_memory_index_cached`` and ``_forget_memory_index``,
which forward here (ADK's ``MemoryIndexPlugin``, Deep Agents'
``MemoryIndexMiddleware`` and the summarizers' ``index=`` and ``on_written=``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from agentic_cli.logging import Loggers
from agentic_cli.workflow.sessions import SessionRef, get_active_turn

if TYPE_CHECKING:
    from agentic_cli.models import ProviderKind

logger = Loggers.workflow()

#: The share of the root's input budget the memory index may take at most.
MEMORY_INDEX_BUDGET_SHARE = 0.1
#: The least the memory index is given, whatever the budget.
MEMORY_INDEX_MIN_BYTES = 1_000


class MemoryIndex:
    """The memory index of each conversation, taken once and kept.

    Attributes:
        snapshots: The index each conversation took, by conversation. ``""``
            is a snapshot too (no memory tool, no store, or a store that
            failed to read): the conversation does not ask again.
    """

    def __init__(
        self,
        *,
        settings: Callable[[], Any],
        store: Callable[[], Any],
        tools: Callable[[], tuple[str, ...]],
        room: Callable[[], tuple[int, ProviderKind] | None],
        limit: Callable[[], int] | None = None,
    ) -> None:
        """Hold no snapshot yet.

        Each argument is asked when a snapshot is taken (or the cap is), not
        before: a manager's settings, services and resolved models change
        under it.

        Args:
            settings: The settings (``memory_index_enabled``,
                ``memory_index_max_bytes``).
            store: The memory store, or None when there is none.
            tools: The memory tools the root has, by name; none, no index.
            room: The root's input budget in tokens and the kind of its
                provider, or None when it has none (a backend that manages no
                context), which leaves the cap to the setting.
            limit: How many bytes a snapshot may take, asked when one is
                taken. :meth:`cap` unless given: a manager gives its own
                ``_memory_index_cap``, so a subclass that overrides that one
                is still the one asked.
        """
        self._settings = settings
        self._store = store
        self._tools = tools
        self._room = room
        self._limit = limit if limit is not None else self.cap
        self.snapshots: dict[SessionRef, str] = {}

    def now(self) -> str:
        """The index the turn's conversation carries, taken the first time it is
        asked for; ``""`` for none (and outside a turn).

        It reads the memory files: call it off the event loop.
        """
        ref = get_active_turn()
        if ref is None:
            return ""
        text = self.snapshots.get(ref)
        if text is None:
            text = self._render()
            self.snapshots[ref] = text
        return text

    def cached(self) -> str:
        """The turn's conversation's snapshot if one was taken, else ``""``; reads nothing."""
        ref = get_active_turn()
        return self.snapshots.get(ref, "") if ref is not None else ""

    def forget(self, ref: SessionRef | None = None) -> None:
        """Drop a conversation's snapshot (the turn's by default): its next request takes a new one."""
        ref = ref or get_active_turn()
        if ref is not None:
            self.snapshots.pop(ref, None)

    def clear(self) -> None:
        """Drop every conversation's snapshot."""
        self.snapshots.clear()

    def cap(self) -> int:
        """How many bytes the memory index may take: ``memory_index_max_bytes``, or,
        on a root whose budget it would crowd, a tenth of that budget.

        Nothing drops the index from a request (the fit guard keeps it, the
        summarization counts it), so it must not be the part of a small budget
        that leaves no room for the conversation. The tenth is the root's input
        budget in characters, as the fit guard estimates a request (the
        provider's characters per token, plus its margin), and at least
        ``MEMORY_INDEX_MIN_BYTES``. Without a budget (a backend that manages
        none), the setting alone.
        """
        from agentic_cli.workflow.context import ESTIMATE_MARGIN, chars_per_token

        cap = self._settings().memory_index_max_bytes
        room = self._room()
        if room is None:
            return cap
        budget, kind = room
        share = int(budget * MEMORY_INDEX_BUDGET_SHARE * chars_per_token(kind) / ESTIMATE_MARGIN)
        return max(MEMORY_INDEX_MIN_BYTES, min(cap, share))

    def _render(self) -> str:
        """The index as the store gives it now; ``""`` for none. Never raises."""
        store = self._store()
        tools = self._tools()
        if store is None or not tools or not getattr(self._settings(), "memory_index_enabled", True):
            return ""
        try:
            return store.index(max_bytes=self._limit(), tools=tools)
        except Exception as e:  # the turn goes on without it
            logger.warning("memory_index_unavailable", error=f"{type(e).__name__}: {e}")
            return ""
