"""Each conversation's sub-agent records, written to its store as they change.

A manager keeps its conversations' records (``AgentRuns``) in memory, and
beside each conversation in its store (``BaseWorkflowManager``'s store
hooks), so they outlive the process. ``RecordsWriter`` makes those writes:
one at a time per conversation, each taking the records as they are when it
starts (``export``), so changes made while one is under way are written
together by the next (the latest wins), and a burst of run events costs one
write or two. A write never fails a run or a turn: a failure is warned of
once per conversation and logged at debug after that. ``flush`` waits until
a conversation's records are written (a turn's end, a delete); its own
cancel leaves the write to finish.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from agentic_cli.constants import truncate
from agentic_cli.logging import Loggers
from agentic_cli.workflow.sessions import SessionRef

logger = Loggers.workflow()

#: The longest error text a failed write's log line carries.
_MAX_LOGGED_ERROR = 500


class RecordsWriter:
    """Writes each conversation's records to its store as they change.

    Args:
        export: A conversation's records as its store keeps them
            (``AgentRuns.export``), taken as each write starts.
        save: Writes them to the conversation's store (the backend's
            ``_save_agent_records``).
    """

    def __init__(
        self,
        *,
        export: Callable[[SessionRef], Mapping[str, Any]],
        save: Callable[[SessionRef, Mapping[str, Any]], Awaitable[None]],
    ) -> None:
        self._export = export
        self._save = save
        # The conversations whose records changed since their last write began.
        self._dirty: set[SessionRef] = set()
        # Each conversation's write under way.
        self._writing: dict[SessionRef, asyncio.Task[None]] = {}
        # The conversations a failed write was warned of.
        self._warned: set[SessionRef] = set()
        # The conversations whose latest write failed: what the store keeps
        # lacks what the manager holds.
        self._unsaved: set[SessionRef] = set()

    def changed(self, conversation: SessionRef) -> None:
        """``conversation``'s records changed: write them, after the write under way.

        Outside a running event loop nothing is written (there is no store
        to write to).
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._dirty.add(conversation)
        task = self._writing.get(conversation)
        if task is not None and not task.done():
            return
        self._writing[conversation] = loop.create_task(
            self._write(conversation), name=f"sub-agent records {conversation.session_id}"
        )

    def unsaved(self, conversation: SessionRef) -> bool:
        """Whether ``conversation``'s latest write failed (the store lacks its latest records)."""
        return conversation in self._unsaved

    async def flush(self, conversation: SessionRef) -> None:
        """Wait until ``conversation``'s records are written. Never raises.

        A cancel of the wait leaves the write to finish.
        """
        task = self._writing.get(conversation)
        if task is not None and not task.done():
            await asyncio.wait([task])

    async def flush_all(self) -> None:
        """Wait until every conversation's records are written. Never raises."""
        tasks = [task for task in self._writing.values() if not task.done()]
        if tasks:
            await asyncio.wait(tasks)

    async def discard(self, conversation: SessionRef) -> None:
        """Write nothing more of ``conversation`` (it is being deleted): what
        waits is dropped, and the write under way is waited for."""
        self._dirty.discard(conversation)
        await self.flush(conversation)
        self._warned.discard(conversation)
        self._unsaved.discard(conversation)

    async def _write(self, conversation: SessionRef) -> None:
        try:
            while conversation in self._dirty:
                self._dirty.discard(conversation)
                try:
                    await self._save(conversation, self._export(conversation))
                except Exception as error:  # noqa: BLE001 - a run is not its records' to fail
                    self._failed(conversation, error)
                else:
                    self._unsaved.discard(conversation)
        finally:
            # Nothing awaits between the last look and this: a change made
            # after it starts a write of its own.
            if self._writing.get(conversation) is asyncio.current_task():
                del self._writing[conversation]

    def _failed(self, conversation: SessionRef, error: Exception) -> None:
        self._unsaved.add(conversation)
        if conversation in self._warned:
            logger.debug(
                "subagent_records_not_saved", session_id=conversation.session_id, exc_info=error
            )
            return
        self._warned.add(conversation)
        logger.warning(
            "subagent_records_not_saved", session_id=conversation.session_id,
            error=truncate(str(error) or type(error).__name__, _MAX_LOGGED_ERROR),
        )
        logger.debug("subagent_records_not_saved", exc_info=error)
