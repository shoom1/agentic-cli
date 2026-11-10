"""Read a backend's stored conversations without building its manager.

A conversation belongs to one backend for its whole life, and each backend
keeps its own store: ADK its session service on ``session_db_url``
(``workflow/adk/sessions.py``), Deep Agents a checkpointer and the session
index beside it (``workflow/deepagents/sessions.py``). To list, find or
delete a conversation of a backend that is not running, a caller (the CLI's
``/resume``, ``/sessions``, ``--continue``, ``--resume``) opens that
backend's ``SessionReader`` from the settings and the store: it reads with
the backend's own session code, so it answers what the manager's session
hooks answer, and it builds no agent, resolves no model, builds no service
and opens nothing but the store (no store at all where none was made yet).

This module sits beside the factory (``workflow/factory.py``), which builds a
backend's manager, for the same reasons: one entry per backend, imported only
when it is asked for (on a worker thread: a backend's import takes a while;
Deep Agents', which takes a second, not at all while its sqlite file is not
made), and no CLI dependency. A backend whose extra is not installed has no
reader;
neither has LangGraph (deprecated: its running manager reads its own), nor
the memory store, whose conversations live in the manager that made them.
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol

from agentic_cli.logging import Loggers
from agentic_cli.workflow.factory import check_backend
from agentic_cli.workflow.settings import OrchestratorType
from agentic_cli.workflow.sessions import SessionInfo, SessionRef

if TYPE_CHECKING:
    from types import TracebackType

logger = Loggers.workflow()

# Orchestrator → the "module:function" that opens its conversations from the
# settings and a store (``open_sessions(settings, store)``).
_OPENERS: dict[OrchestratorType, str] = {
    OrchestratorType.ADK: "agentic_cli.workflow.adk.sessions:open_sessions",
    OrchestratorType.DEEPAGENTS: "agentic_cli.workflow.deepagents.sessions:open_sessions",
}

# Orchestrator → the "module:function" that names its sqlite file from the
# settings, without importing the backend: a file not made holds nothing, so
# the backend is not imported to read it (Deep Agents' import takes a second).
# ADK, the core backend, is imported to run anyway.
_SQLITE_FILES: dict[OrchestratorType, str] = {
    OrchestratorType.DEEPAGENTS: "agentic_cli.workflow.checkpointers:deepagents_sqlite_location",
}


class StoredSessions(Protocol):
    """A backend's session code over its open store: what a reader reads
    (``ServiceSessions`` on ADK, ``CheckpointSessions`` on Deep Agents)."""

    async def list(
        self, app_name: str, user_id: str, *, limit: int | None = None
    ) -> list[SessionInfo]: ...

    async def info(self, ref: SessionRef) -> SessionInfo | None: ...

    async def exists(self, ref: SessionRef) -> bool: ...

    async def delete(self, ref: SessionRef) -> bool: ...


class SessionReader:
    """One backend's stored conversations of one app and user, read without
    its manager.

    Each row it returns names its backend (``SessionInfo.backend``). Close
    it when done (``aclose``, or ``async with``): it holds the store's
    connections. Opened by ``open_session_reader``.

    Args:
        backend: The backend whose conversations it reads (``adk``,
            ``deepagents``).
        sessions: The backend's session code over its open store; None when
            the store holds nothing yet (a sqlite file not made).
        app_name: The app whose conversations it reads.
        user_id: Their user.
        close: What closes the store's connections, if anything.
    """

    def __init__(
        self,
        backend: str,
        sessions: StoredSessions | None,
        *,
        app_name: str,
        user_id: str,
        close: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.backend = backend
        self._sessions = sessions
        self._app_name = app_name
        self._user_id = user_id
        self._close = close

    async def list(self, limit: int | None = None) -> list[SessionInfo]:
        """The stored conversations, newest first (``limit``: only the newest this many)."""
        if self._sessions is None:
            return []
        rows = await self._sessions.list(self._app_name, self._user_id, limit=limit)
        return [self._named(row) for row in rows]

    async def info(self, session_id: str) -> SessionInfo | None:
        """``session_id``'s row, or None when it is not stored."""
        if self._sessions is None:
            return None
        row = await self._sessions.info(self._ref(session_id))
        return None if row is None else self._named(row)

    async def exists(self, session_id: str, *, user_id: str | None = None) -> bool:
        """Whether ``session_id``'s conversation is stored (``user_id``'s:
        the reader's user by default)."""
        if self._sessions is None:
            return False
        ref = SessionRef(self._app_name, user_id or self._user_id, session_id)
        return await self._sessions.exists(ref)

    async def delete(self, session_id: str) -> bool:
        """Delete ``session_id``'s conversation; False if it was not stored.

        It deletes what the manager's ``delete_session`` deletes. Nothing is
        held here: a caller that may race another process holds the
        conversation first (the CLI's ``conversation_held``).
        """
        if self._sessions is None:
            return False
        return await self._sessions.delete(self._ref(session_id))

    async def aclose(self) -> None:
        """Close the store's connections. Idempotent; never raises."""
        close, self._close, self._sessions = self._close, None, None
        if close is None:
            return
        try:
            await close()
        except Exception as exc:  # noqa: BLE001 - a reader's close must not fail its caller
            logger.warning("session_reader_close_failed", backend=self.backend, error=str(exc))

    async def __aenter__(self) -> "SessionReader":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: "TracebackType | None",
    ) -> None:
        await self.aclose()

    def _ref(self, session_id: str) -> SessionRef:
        return SessionRef(self._app_name, self._user_id, session_id)

    def _named(self, row: SessionInfo) -> SessionInfo:
        return row if row.backend == self.backend else replace(row, backend=self.backend)


def reader_backends() -> tuple[str, ...]:
    """The backends that have a reader, installed or not (``adk``, ``deepagents``)."""
    return tuple(orchestrator.value for orchestrator in _OPENERS)


async def open_session_reader(
    backend: "str | OrchestratorType", settings: Any, *, store: str | None = None
) -> SessionReader | None:
    """Open ``backend``'s reader of the conversations ``settings`` keep.

    The app is ``settings.app_name`` and the user ``settings.default_user``,
    as the manager's own hooks default them. The backend's module is
    imported on a worker thread, the first time.

    Args:
        backend: The backend: an orchestrator's value.
        settings: The app's settings: where its stores are.
        store: The session store to read (``sqlite``, ``postgres``); the
            settings' ``session_store`` by default. A CLI run passes the one
            it started with.

    Returns:
        The reader; None where there is nothing to read without the
        manager: the memory store, or a backend without a reader
        (LangGraph). A sqlite store whose file is not made yet reads as
        empty, and Deep Agents' is not imported to read it.

    Raises:
        ValueError: ``backend`` is no orchestrator.
        ImportError: Its extra is not installed (and its store holds
            something); the message says how to install it.
        Exception: What opening the store raises.
    """
    orchestrator = OrchestratorType(backend)
    store = store or settings.session_store
    target = _OPENERS.get(orchestrator)
    if target is None or store == "memory":
        return None
    if store == "sqlite" and orchestrator in _SQLITE_FILES:
        sqlite_file = _function(_SQLITE_FILES[orchestrator])
        if not sqlite_file(settings).is_file():  # nothing stored: nothing to import
            return SessionReader(
                orchestrator.value, None,
                app_name=settings.app_name, user_id=settings.default_user,
            )
    # The backend first, as the factory imports it: a missing extra raises
    # its install hint.
    await asyncio.to_thread(check_backend, orchestrator)
    module_name, _, function_name = target.partition(":")
    module = await asyncio.to_thread(importlib.import_module, module_name)
    sessions, close = await getattr(module, function_name)(settings, store)
    logger.debug("session_reader_opened", backend=orchestrator.value, store=store,
                 empty=sessions is None)
    return SessionReader(
        orchestrator.value, sessions,
        app_name=settings.app_name, user_id=settings.default_user, close=close,
    )


def _function(target: str) -> Any:
    """The function a light ``"module:function"`` names."""
    module_name, _, function_name = target.partition(":")
    return getattr(importlib.import_module(module_name), function_name)
