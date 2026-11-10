"""A conversation's row on ADK, kept in its session state.

``list_sessions`` shows each stored conversation as a ``SessionInfo``: its
title (the first prompt) and how many turns the user ran in it. ADK's
listing returns every session's state but none of its events, so both are
kept in the state, under ``TITLE_KEY`` and ``TURNS_KEY``, saved with the
user's message of each turn (``process()`` passes ``turn_state`` to
``Runner.run_async(state_delta=)``). A job's resume saves nothing: nobody
typed it.

A conversation from before these keys has neither: its row is read from its
events (``row_from_events``), and nothing is written back, since a write
would change its last-update time and so its place in the list. Its next
turn saves both, counting the turns it already had.

Only what the user typed is a turn (``typed_prompts``): a text event of the
user's that is not a job's notice. ADK stores a summary (a compaction) as a
user event without content, a delegated request not at all, and a job's
result as a notice (``JOB_NOTICE_HEADER``), or, before the notice, a
function response; none of them is a typed prompt.

The session hooks read the session service through ``ServiceSessions``, the
manager's (``session_exists``, ``list_sessions``, ``session_info``,
``delete_session``) and its reader's alike: ``open_sessions`` opens the
service of a store from the settings, without the manager, for
``workflow/session_readers.py``.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentic_cli.logging import Loggers
from agentic_cli.tools.jobs.resume import JOB_NOTICE_HEADER
from agentic_cli.workflow.sessions import SessionInfo, SessionRef, session_title

if TYPE_CHECKING:
    from google.adk.events import Event
    from google.adk.sessions import BaseSessionService, Session

logger = Loggers.workflow()

#: The conversation's title, its first prompt (``session_title``).
TITLE_KEY = "agentic_cli.title"
#: How many turns the user ran in it.
TURNS_KEY = "agentic_cli.turns"


def typed_prompts(events: list["Event"]) -> list[str]:
    """The texts the user typed, oldest first."""
    prompts: list[str] = []
    for event in events:
        content = event.content
        if event.author != "user" or content is None or content.role != "user":
            continue
        text = " ".join(part.text for part in content.parts or () if part.text and not part.thought)
        if text.strip() and not text.startswith(JOB_NOTICE_HEADER):
            prompts.append(text)
    return prompts


def turn_state(session: "Session", message: str) -> dict[str, Any]:
    """What a user turn saves with its message: the turn count, and the title when there is none.

    A blank message saves nothing: it is no prompt, as ``typed_prompts``
    and Deep Agents count.

    Args:
        session: The conversation as it stands before the turn.
        message: What the user typed.
    """
    if not message.strip():
        return {}
    state = session.state or {}
    turns = state.get(TURNS_KEY)
    title = state.get(TITLE_KEY)
    typed = typed_prompts(session.events) if not isinstance(turns, int) or not title else []
    if not isinstance(turns, int):  # a conversation from before the keys
        turns = len(typed)
    delta: dict[str, Any] = {TURNS_KEY: turns + 1}
    if not title:
        delta[TITLE_KEY] = session_title(typed[0] if typed else message)
    return delta


def row_from_state(session: "Session") -> SessionInfo | None:
    """The row the session's state holds, or None when it holds none."""
    state = session.state or {}
    turns = state.get(TURNS_KEY)
    if not isinstance(turns, int):
        return None
    return SessionInfo(session.id, str(state.get(TITLE_KEY) or ""), turns, last_updated(session))


def row_from_events(session: "Session") -> SessionInfo:
    """The row of a conversation from before the keys, read from its events."""
    typed = typed_prompts(session.events)
    title = session_title(typed[0]) if typed else ""
    return SessionInfo(session.id, title, len(typed), last_updated(session))


def last_updated(session: "Session") -> datetime | None:
    """When the session last changed, aware (UTC); None when unknown."""
    stamp = session.last_update_time
    return datetime.fromtimestamp(stamp, tz=timezone.utc) if stamp else None


class ServiceSessions:
    """The session hooks, answered from an ADK session service.

    What the manager's hooks read and what its reader reads
    (``open_sessions``): one implementation, whoever opened the service.

    Args:
        service: The session service the conversations are in.
    """

    def __init__(self, service: "BaseSessionService") -> None:
        self._service = service

    async def exists(self, ref: SessionRef) -> bool:
        """Whether the store holds ``ref``'s session with any events.

        It reads the newest event alone (``num_recent_events=1``), not the
        whole conversation.
        """
        from google.adk.sessions.base_session_service import GetSessionConfig

        session = await self._service.get_session(
            app_name=ref.app_name, user_id=ref.user_id, session_id=ref.session_id,
            config=GetSessionConfig(num_recent_events=1),
        )
        return session is not None and bool(getattr(session, "events", None))

    async def info(self, ref: SessionRef) -> SessionInfo | None:
        """``ref``'s row, or None when it is not stored (or holds no event)."""
        session = await self._service.get_session(
            app_name=ref.app_name, user_id=ref.user_id, session_id=ref.session_id,
        )
        if session is None or not session.events:
            return None
        return row_from_state(session) or row_from_events(session)

    async def list(
        self,
        app_name: str,
        user_id: str,
        *,
        limit: int | None = None,
        row_of: Callable[[str], Awaitable[SessionInfo | None]] | None = None,
    ) -> list[SessionInfo]:
        """One app and user's stored conversations, newest first.

        Each row comes from its session's state, which ADK's listing returns
        without the events; a conversation from before those keys is read
        once, for the rows returned (``row_of``), and one that cannot be read
        is listed by its id. A session without events is no conversation and
        is left out.

        Args:
            app_name: The app.
            user_id: The user.
            limit: Only the newest this many (None: all of them).
            row_of: How a row the state does not hold is read, by session id:
                ``info`` by default (the manager reads it through its own
                ``session_info``).
        """
        if row_of is None:
            async def row_of(session_id: str) -> SessionInfo | None:
                return await self.info(SessionRef(app_name, user_id, session_id))

        resp = await self._service.list_sessions(app_name=app_name, user_id=user_id)
        listed = sorted(
            getattr(resp, "sessions", []),
            key=lambda s: s.last_update_time or 0,
            reverse=True,
        )
        rows: list[SessionInfo] = []
        for session in listed:
            if limit is not None and len(rows) >= limit:
                break
            row = row_from_state(session)
            if row is None:
                try:
                    row = await row_of(session.id)
                except Exception:  # noqa: BLE001 - one unreadable row is no unreadable store
                    logger.debug("session_row_unreadable", session_id=session.id, exc_info=True)
                    row = SessionInfo(session.id, updated=last_updated(session))
            if row is not None:
                rows.append(row)
        return rows

    async def delete(self, ref: SessionRef) -> bool:
        """Delete ``ref``'s session; False if there was none.

        ADK's own delete answers nothing, for an unknown id too, so the
        session is looked up first, without its events
        (``num_recent_events=0``).
        """
        from google.adk.sessions.base_session_service import GetSessionConfig

        stored = await self._service.get_session(
            app_name=ref.app_name, user_id=ref.user_id, session_id=ref.session_id,
            config=GetSessionConfig(num_recent_events=0),
        )
        if stored is None:
            return False
        await self._service.delete_session(
            app_name=ref.app_name, user_id=ref.user_id, session_id=ref.session_id,
        )
        return True


def session_db_url(settings: Any, store: str) -> str | None:
    """``settings.session_db_url()`` for ``store``: None for the memory store.

    Called as 0.6.4 documented it, with no argument, while ``store`` is the
    settings' own, so an app's override without a ``store`` parameter keeps
    working; with ``store=`` only when they differ (a ``/settings`` change of
    ``session_store`` in the run: the run keeps its store). An override that
    cannot take it then fails rather than name the other store.

    Raises:
        RuntimeError: The settings' ``session_db_url`` takes no store, and
            ``store`` is not theirs.
    """
    if store == settings.session_store:
        return settings.session_db_url()
    if not _takes_store(settings.session_db_url):
        raise RuntimeError(
            "This app's settings.session_db_url() takes no store, so this run "
            "cannot rebuild on its session store after session_store changed: "
            "restart the app."
        )
    return settings.session_db_url(store=store)


def make_session_service(settings: Any, store: str) -> "BaseSessionService":
    """The ADK session service of ``store``.

    ``memory`` → ephemeral ``InMemorySessionService``; ``sqlite``/``postgres``
    → native ``DatabaseSessionService`` (durable across restarts, full event
    fidelity). The sqlite file is made owner-only before the engine opens
    it, its directory created if missing (``private_sqlite_file``; one
    ``sqlite_uri`` names, and that exists, is left as it is).
    """
    db_url = session_db_url(settings, store)
    if db_url is None:
        from google.adk.sessions import InMemorySessionService

        logger.debug("using_in_memory_session_service")
        return InMemorySessionService()

    if db_url.startswith("sqlite"):
        from agentic_cli.workflow.checkpointers import private_sqlite_file

        private_sqlite_file(
            _sqlite_database(db_url),
            tighten_existing=not getattr(settings, "sqlite_uri", None),
        )

    from google.adk.sessions import DatabaseSessionService

    logger.info("using_database_session_service", store=store)
    return DatabaseSessionService(db_url=db_url)


#: What closes a store a reader opened.
Close = Callable[[], Awaitable[None]]


async def open_sessions(settings: Any, store: str) -> tuple[ServiceSessions | None, Close | None]:
    """Open ``store``'s ADK conversations from ``settings``, without a manager
    (``workflow/session_readers.py``).

    A store ADK has not written yet holds none of its conversations, and
    reading it writes nothing there: ``(None, None)`` for a sqlite file not
    made, and for a database without ADK's ``sessions`` table (a file
    ``sqlite_uri`` names that Deep Agents or a team uses, a Postgres
    database ADK never ran on), looked at read-only before ADK's session
    service creates any table. Otherwise the session service the manager
    would open (``make_session_service``), and what closes it.
    """
    db_url = session_db_url(settings, store)
    if db_url is None:
        raise ValueError("The memory store's conversations live in the manager that made them.")
    if db_url.startswith("sqlite") and not Path(_sqlite_database(db_url)).is_file():
        return None, None
    service = make_session_service(settings, store)
    try:
        ours = await _has_table(service, SESSIONS_TABLE)
    except BaseException:
        await service.close()
        raise
    if not ours:
        await service.close()
        return None, None
    return ServiceSessions(service), service.close


#: The table ADK's session service keeps a conversation's row in, in both of
#: its schemas: a database without it is not ADK's yet.
SESSIONS_TABLE = "sessions"


async def _has_table(service: Any, table: str) -> bool:
    """Whether the database ``service``'s engine names has ``table``: read
    only (no ``prepare_tables``), on any dialect."""
    from sqlalchemy import inspect as inspect_database

    async with service.db_engine.connect() as connection:
        return await connection.run_sync(
            lambda sync: inspect_database(sync).has_table(table)
        )


def _sqlite_database(db_url: str) -> str:
    """The file a sqlite URL names, as SQLAlchemy reads it: without a query
    (``?timeout=30``), which goes to sqlite3."""
    from sqlalchemy.engine import make_url

    return make_url(db_url).database or ""


def _takes_store(method: Any) -> bool:
    """Whether ``method`` (an app's ``session_db_url``, maybe overridden as
    0.6.4 documented it) takes ``store=``."""
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):  # no signature to read: try it
        return True
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        or (p.name == "store" and p.kind is not inspect.Parameter.POSITIONAL_ONLY)
        for p in parameters
    )
