"""The stored conversations the app can open, on every installed backend.

A conversation belongs to one backend for its whole life, and each backend
keeps its own store (no index joins them). The app lists, finds and deletes
conversations on all of them: the running workflow answers for its own
backend, as it always did (an app's own manager, the memory store), and each
other installed backend's ``SessionReader`` (``workflow/session_readers.py``)
for its own, opened for one look and closed after it. With no workflow
running (before the first start), every installed backend's reader answers.

Each row names its backend (``SessionInfo.backend``), and the rows of all
backends are merged newest first: an id two backends keep (left by the live
switch of backend an earlier version made) is two rows.

A backend whose conversations cannot be read (its extra not installed, its
store failing) is left out of the look, and ``notes`` says so, for a caller
that found nothing where the user asked for something.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from agentic_cli.logging import Loggers
from agentic_cli.workflow.session_readers import open_session_reader, reader_backends

if TYPE_CHECKING:
    from types import TracebackType

    from agentic_cli.workflow.session_readers import SessionReader
    from agentic_cli.workflow.sessions import SessionInfo

logger = Loggers.cli()

#: Where a row without a time sorts: last.
_NEVER = datetime.min.replace(tzinfo=timezone.utc)


class StoredConversations:
    """One look at the conversations stored on every installed backend.

    An async context manager: the readers it opens on entry are closed when
    it ends.

    Args:
        settings: The app's settings: where the stores are, the app and the
            user.
        store: The session store the run keeps (``sqlite``, ``postgres``,
            ``memory``).
        workflow: The running workflow manager, ready, or None (none runs
            yet: every installed backend is read through its reader).
        backend: The running workflow's backend; its conversations are the
            workflow's to answer for.
        only: Look at this backend's conversations alone.
    """

    def __init__(
        self,
        settings: Any,
        store: str,
        *,
        workflow: Any = None,
        backend: str | None = None,
        only: str | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._workflow = workflow if only in (None, backend) else None
        self._backend = backend if self._workflow is not None else None
        self._only = only
        self._readers: list["SessionReader"] = []
        #: Each backend whose conversations were not read, and why (a sentence).
        self.unread: dict[str, str] = {}

    async def __aenter__(self) -> "StoredConversations":
        try:
            for backend in reader_backends():
                if backend == self._backend or self._only not in (None, backend):
                    continue
                reader = await self._open(backend)
                if reader is not None:
                    self._readers.append(reader)
        except BaseException:
            await self.aclose()
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: "TracebackType | None",
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close every reader opened. Never raises."""
        readers, self._readers = self._readers, []
        for reader in readers:
            await reader.aclose()

    @property
    def backends(self) -> tuple[str, ...]:
        """The backends looked at: the running one first, then those read."""
        running = (self._backend,) if self._backend is not None else ()
        return running + tuple(reader.backend for reader in self._readers)

    async def rows(self, limit: int | None = None) -> list["SessionInfo"]:
        """Every backend's conversations, newest first, each naming its
        backend; ``limit`` keeps the newest this many.

        Raises:
            Exception: What the running workflow's listing raises.
        """
        rows: list["SessionInfo"] = []
        if self._lists_its_own():
            rows += self._named(await self._workflow.list_sessions(limit=limit))
        for reader in self._readers:
            rows += await self._read(reader, reader.list(limit), [])
        rows.sort(key=_when, reverse=True)  # stable: the running backend's first on a tie
        return rows if limit is None else rows[:limit]

    async def where(
        self, session_id: str, backend: str | None = None, *, running: bool = True
    ) -> list["SessionInfo"]:
        """The rows of ``session_id``: one per backend that keeps it, the
        running one's first, then the others newest first. Only
        ``backend``'s when one is named; not the running workflow's when
        ``running`` is False (the conversation the app is in is there).

        Raises:
            Exception: What the running workflow's ``session_info`` raises.
        """
        found: list["SessionInfo"] = []
        if running and self._lists_its_own() and backend in (None, self._backend):
            row = await self._workflow.session_info(session_id)
            if row is not None:
                found.append(self._named([row])[0])
        others: list["SessionInfo"] = []
        for reader in self._readers:
            if backend not in (None, reader.backend):
                continue
            row = await self._read(reader, reader.info(session_id), None)
            if row is not None:
                others.append(row)
        return found + sorted(others, key=_when, reverse=True)

    async def delete(self, session_id: str, backend: str) -> bool:
        """Delete ``session_id`` on ``backend``; False when it keeps none.

        Raises:
            ValueError: ``backend`` was not looked at.
            Exception: What the backend's delete raises.
        """
        if backend == self._backend and self._workflow is not None:
            return bool(await self._workflow.delete_session(session_id))
        for reader in self._readers:
            if reader.backend == backend:
                return await reader.delete(session_id)
        raise ValueError(f"The conversations on {backend} were not read.")

    def notes(self) -> list[str]:
        """What a caller that found nothing adds: the backends not looked at, and why."""
        return list(self.unread.values())

    def _lists_its_own(self) -> bool:
        return self._workflow is not None and getattr(self._workflow, "supports_sessions", True)

    def _named(self, rows: list["SessionInfo"]) -> list["SessionInfo"]:
        """The running workflow's rows, each naming its backend."""
        return [row if row.backend else replace(row, backend=self._backend) for row in rows]

    async def _open(self, backend: str) -> "SessionReader | None":
        """``backend``'s reader, or None (none to open, or it failed: ``unread``)."""
        try:
            return await open_session_reader(backend, self._settings, store=self._store)
        except ImportError:
            logger.debug("session_reader_not_installed", backend=backend, exc_info=True)
            self.unread[backend] = (
                f"The {backend} backend is not installed: its conversations were not looked at."
            )
        except Exception as error:  # noqa: BLE001 - one backend's store, not the look
            self._failed(backend, error)
        return None

    async def _read(self, reader: "SessionReader", read: Any, default: Any) -> Any:
        """What ``read`` (a reader's call) answers; ``default`` when it fails (``unread``)."""
        try:
            return await read
        except Exception as error:  # noqa: BLE001 - one backend's store, not the look
            self._failed(reader.backend, error)
            return default

    def _failed(self, backend: str, error: Exception) -> None:
        """``backend``'s store failed: its conversations are left out, said
        in the log (a warning: they are missing from what the user sees)."""
        logger.warning("session_reader_failed", backend=backend, error=str(error))
        logger.debug("session_reader_failed", backend=backend, exc_info=error)
        self.unread[backend] = _unreadable(backend, error)


def _when(row: "SessionInfo") -> datetime:
    """When ``row`` last changed, comparable: a time without a zone taken as UTC."""
    updated = row.updated
    if updated is None:
        return _NEVER
    return updated if updated.tzinfo is not None else updated.replace(tzinfo=timezone.utc)


def _unreadable(backend: str, error: BaseException) -> str:
    reason = (str(error) or type(error).__name__).split("\n")[0].rstrip(".")
    return f"Could not read the conversations on {backend}: {reason}."
