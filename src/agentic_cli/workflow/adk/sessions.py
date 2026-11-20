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

A sub-agent instance's transcript is a child session in the same session
service: app ``child_app(app_name)`` (``<app>--agents``), session
``child_session_id(conversation, instance)`` (``<conversation>/<instance>``,
the conversation's ``%`` and ``/`` percent-encoded: ``a/b``'s are
``a%2Fb/<instance>``, so the first ``/`` of a child app's id always ends its
conversation, and an id of the CLI's, which holds neither, is as it was),
the conversation's user. Under its own app name it never lists as a
conversation. ADK keys a session by at most ``MAX_KEY`` characters, so a
longer id is refused before anything is stored. A conversation's children
are deleted with it (``delete_children``: ``ServiceSessions.delete``, the
manager's and its reader's), and each turn numbers its instances past the
ones they keep (``stored_instances``), so none is ever another run's.

A conversation's sub-agent records (``AgentRuns.export``: its instances,
its runs, its id counters) are one more session of the child app, named as
the conversation, encoded (``records_session_id``: no ``/``, where its
instances' ids hold one), its state holding them under ``RECORDS_KEY``:
made at the first write (``save_records``), each later write an event whose state delta
replaces them, since ADK takes no second writer on the conversation's own
session while a turn runs. No two conversations' records or transcripts
share an id. Its children are found by the instances the records name, a listing of the user's child sessions only for a
conversation without records. ``ServiceSessions.delete`` deletes the
transcripts first, then the records, then the conversation, so a delete cut
short leaves the conversation, by which the rest is found again.
"""


from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

from agentic_cli.logging import Loggers
from agentic_cli.tools.jobs.resume import JOB_NOTICE_HEADER
from agentic_cli.workflow.agents.runs import stored_instance_ids
from agentic_cli.workflow.sessions import SessionInfo, SessionRef, session_title

if TYPE_CHECKING:
    from google.adk.events import Event
    from google.adk.sessions import BaseSessionService, Session

logger = Loggers.workflow()

#: The conversation's title, its first prompt (``session_title``).
TITLE_KEY = "agentic_cli.title"
#: How many turns the user ran in it.
TURNS_KEY = "agentic_cli.turns"

#: The state key of a conversation's sub-agent records, in its records session.
RECORDS_KEY = "agentic_cli.agents"
#: The author of a records session's events (each a write of the records).
RECORDS_AUTHOR = "agentic_cli"

#: What a child session's app name adds to the conversation's app name.
CHILD_APP_SUFFIX = "--agents"
#: The longest app name or session id ADK stores (``sessions/schemas/shared.py``).
MAX_KEY = 128


def child_app(app_name: str) -> str:
    """The app the sub-agent instances of ``app_name``'s conversations are stored under."""
    return f"{app_name}{CHILD_APP_SUFFIX}"


def _encoded(conversation: str) -> str:
    """``conversation`` as the child app's ids hold it: its ``%`` and ``/``
    percent-encoded, so it holds no ``/`` (``urllib.parse.unquote`` reads it back)."""
    return conversation.replace("%", "%25").replace("/", "%2F")


def records_session_id(conversation: str) -> str:
    """The child app's session that holds the conversation ``conversation``'s records."""
    return _encoded(conversation)


def child_of(session_id: str) -> tuple[str, str] | None:
    """The conversation and the instance a child app's session id names
    (``child_session_id``, decoded); None for a records session or an id that names none."""
    encoded, slash, instance = session_id.partition("/")
    if not slash or not instance or "/" in instance:
        return None
    return unquote(encoded), instance


def child_session_id(conversation: str, instance: str) -> str:
    """The session that holds ``instance``'s transcript in the conversation ``conversation``.

    ``<conversation>/<instance>``, the conversation's ``%`` and ``/``
    percent-encoded.

    Raises:
        ValueError: The id is longer than ADK stores (``MAX_KEY``).
    """
    session_id = f"{_encoded(conversation)}/{instance}"
    if len(session_id) > MAX_KEY:
        raise ValueError(
            f"The session of sub-agent {instance} would be {len(session_id)} characters long; "
            f"ADK stores at most {MAX_KEY}."
        )
    return session_id


async def _children(
    service: "BaseSessionService", app_name: str, user_id: str, conversation: str
) -> list[str]:
    """The ids of ``conversation``'s child sessions, ``user_id``'s, in
    ``app_name``'s child app.

    Each is ``child_session_id(conversation, <instance>)``. A conversation
    with records has the instances they name (``stored_instance_ids``: one
    look at its records session); one without, every child session of the
    user whose id names the conversation (``child_of``, decoded), read with
    ADK's public listing (which returns each session without its events): a
    records session names none, and the conversation ``s/x``'s children
    (``s%2Fx/<instance>``) are not ``s``'s.
    """
    named = stored_instance_ids(
        await load_records(service, SessionRef(app_name, user_id, conversation))
    )
    if named is not None:
        prefix = f"{_encoded(conversation)}/"
        return [
            f"{prefix}{instance}" for instance in named
            if "/" not in instance and len(prefix) + len(instance) <= MAX_KEY
        ]
    listed = await service.list_sessions(app_name=child_app(app_name), user_id=user_id)
    return [
        session.id for session in getattr(listed, "sessions", None) or ()
        if (child_of(session.id) or ("",))[0] == conversation
    ]


async def load_records(service: "BaseSessionService", ref: SessionRef) -> Any | None:
    """What ``ref``'s records session holds under ``RECORDS_KEY``; None for none.

    Read without its events (``num_recent_events=0``).
    """
    from google.adk.sessions.base_session_service import GetSessionConfig

    session = await service.get_session(
        app_name=child_app(ref.app_name), user_id=ref.user_id,
        session_id=records_session_id(ref.session_id),
        config=GetSessionConfig(num_recent_events=0),
    )
    if session is None:
        return None
    return (session.state or {}).get(RECORDS_KEY)


async def save_records(
    service: "BaseSessionService", ref: SessionRef, data: Mapping[str, Any]
) -> None:
    """Keep ``data`` as ``ref``'s sub-agent records, replacing what its records session held.

    The session is made at the first write, with ``data`` as its state;
    each later write is one event of its own whose state delta replaces
    them (``append_event``: ADK keeps a session's state only through its
    events).
    """
    from google.adk.events import Event, EventActions
    from google.adk.sessions.base_session_service import GetSessionConfig

    app = child_app(ref.app_name)
    session_id = records_session_id(ref.session_id)
    session = await service.get_session(
        app_name=app, user_id=ref.user_id, session_id=session_id,
        config=GetSessionConfig(num_recent_events=0),
    )
    if session is None:
        await service.create_session(
            app_name=app, user_id=ref.user_id, session_id=session_id,
            state={RECORDS_KEY: dict(data)},
        )
        return
    await service.append_event(session, Event(
        author=RECORDS_AUTHOR, actions=EventActions(state_delta={RECORDS_KEY: dict(data)}),
    ))


async def delete_records(service: "BaseSessionService", ref: SessionRef) -> None:
    """Delete ``ref``'s records session, its events with it; none is nothing to do."""
    await service.delete_session(
        app_name=child_app(ref.app_name), user_id=ref.user_id,
        session_id=records_session_id(ref.session_id),
    )


async def delete_children(
    service: "BaseSessionService", app_name: str, user_id: str, conversation: str
) -> int:
    """Delete ``conversation``'s child sessions: its sub-agent instances' transcripts.

    Every session of ``app_name``'s child app (``child_app``) that is
    ``user_id``'s and ``conversation``'s (``child_session_id``);
    their events go with them (ADK's session store deletes a session's
    events with it). What a conversation's delete leaves of it otherwise.

    Args:
        service: The session service the conversation is in.
        app_name: The conversation's app (not the child app).
        user_id: The conversation's user.
        conversation: The conversation's id.

    Returns:
        How many it deleted.
    """
    children = await _children(service, app_name, user_id, conversation)
    for session_id in children:
        await service.delete_session(
            app_name=child_app(app_name), user_id=user_id, session_id=session_id
        )
    return len(children)


async def stored_instances(
    service: "BaseSessionService", app_name: str, user_id: str, conversation: str
) -> dict[str, int]:
    """The highest instance number of each agent type ``conversation``'s child sessions keep.

    Read from their ids (``child_session_id``: ``<conversation>/<type>-<n>``,
    ``AgentRuns``' instance ids, decoded with ``child_of``): what another
    manager ran there (before a restart, or in
    another process between two turns), which this one numbers past. An id
    that names no such instance is left out.

    Args:
        service: The session service the conversation is in.
        app_name: The conversation's app (not the child app).
        user_id: The conversation's user.
        conversation: The conversation's id.
    """
    counts: dict[str, int] = {}
    for session_id in await _children(service, app_name, user_id, conversation):
        named = child_of(session_id)
        if named is None:
            continue
        agent, _, number = named[1].rpartition("-")
        if agent and number.isascii() and number.isdigit():
            counts[agent] = max(counts.get(agent, 0), int(number))
    return counts


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
        """Delete ``ref``'s session, its sub-agents' transcripts and its records;
        False if there was no session.

        ADK's own delete answers nothing, for an unknown id too, so the
        session is looked up first, without its events
        (``num_recent_events=0``). Its child sessions (``delete_children``)
        go first, then its records (``delete_records``), then the session,
        whatever the look found: a conversation already gone may have left
        some, and a delete cut short leaves the conversation, by which the
        rest is found again. The answer is the conversation's.
        """
        from google.adk.sessions.base_session_service import GetSessionConfig

        stored = await self._service.get_session(
            app_name=ref.app_name, user_id=ref.user_id, session_id=ref.session_id,
            config=GetSessionConfig(num_recent_events=0),
        )
        await delete_children(self._service, ref.app_name, ref.user_id, ref.session_id)
        await delete_records(self._service, ref)
        if stored is not None:
            await self._service.delete_session(
                app_name=ref.app_name, user_id=ref.user_id, session_id=ref.session_id,
            )
        return stored is not None


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
