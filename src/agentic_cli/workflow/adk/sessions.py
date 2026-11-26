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
user's that is not a job's notice and not a notice to the root
(``NOTICE_METADATA``). ADK stores a summary (a compaction) as a user event
without content, a delegated request not at all, and a job's result as a
notice (``JOB_NOTICE_HEADER``), or, before the notice, a function response;
none of them is a typed prompt.

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
instances' ids hold one): made at the first write (``save_records``), each
later write an event, since ADK takes no second writer on the conversation's
own session while a turn runs. Its state holds the records split by key, so
a write's event holds what changed and the records grow in step with the
conversation, not with the square of it: ``RECORDS_KEY`` holds the header
(version, time, counters) and ``RECORDS_KEY.instance.<n>`` and
``RECORDS_KEY.run.<n>`` the n-th instance and run. Its events would still
grow without bound, so past ``RECORDS_REWRITE_AFTER`` writes the next one
rewrites the records whole into a second session (``records_session_ids``:
the conversation's id and that plus ``/``, alternating, ``_GENERATION_KEY``
telling the newer, which a read takes), then deletes the first. No two conversations'
records or transcripts share an id. Numbering past the stored instances
finds a conversation's children by the instances its records name: a first
run's session is made only once its start is in the records, so they name
every one, and a conversation without records has none. A listing of the
user's child sessions is only for records that do not read (another
version's), which this manager writes none of its runs to; a delete follows
the same rule. ``ServiceSessions.delete``
deletes the transcripts first, then the records, then the conversation, so a
delete cut short leaves the conversation, by which the rest is found again.
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
from agentic_cli.workflow.outside_content import OUTSIDE_CONTENT_KEY
from agentic_cli.workflow.sessions import SessionInfo, SessionRef, session_title

if TYPE_CHECKING:
    from google.adk.events import Event
    from google.adk.sessions import BaseSessionService, Session

logger = Loggers.workflow()

#: The conversation's title, its first prompt (``session_title``).
TITLE_KEY = "agentic_cli.title"
#: How many turns the user ran in it.
TURNS_KEY = "agentic_cli.turns"

#: The mark of a notice to the root's run, which the user did not type
#: (``tell_root``): passed as ``RunConfig.custom_metadata``, which ADK stamps on
#: each event of the run, the notice's own user event among them, and stores
#: with it.
NOTICE_METADATA = {"agentic_cli": {"kind": "notice"}}

#: The state key of a conversation's sub-agent records, in its records session.
RECORDS_KEY = "agentic_cli.agents"
#: The state key an instance's first run sets with its message (the run's id),
#: in its child session: the event that sets it is the instance's brief, which
#: the fit guard keeps on every later run (``fit_guard_plugin.brief_of``).
BRIEF_KEY = "agentic_cli.brief"
#: The author of a records session's events (each a write of the records).
RECORDS_AUTHOR = "agentic_cli"
#: The state keys of an instance and a run of the records (then their number).
_INSTANCE_KEY = f"{RECORDS_KEY}.instance."
_RUN_KEY = f"{RECORDS_KEY}.run."

#: How many writes are appended to one records session before the next write
#: rewrites the records whole into the other one (``save_records``).
RECORDS_REWRITE_AFTER = 100
#: The state key of a records session's generation: ``{"n": <int>, "writes":
#: <int>}``, the rewrites so far and the writes appended since the last.
_GENERATION_KEY = f"{RECORDS_KEY}.generation"

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
    """The child app's session that holds the conversation ``conversation``'s records.

    Raises:
        ValueError: The id is longer than ADK stores (``MAX_KEY``).
    """
    session_id = _encoded(conversation)
    if len(session_id) > MAX_KEY:
        raise ValueError(
            f"The session of a conversation's sub-agent records would be {len(session_id)} "
            f"characters long; ADK stores at most {MAX_KEY}."
        )
    return session_id


def records_session_ids(conversation: str) -> tuple[str, str | None]:
    """The two child-app sessions that alternate as ``conversation``'s records session.

    ``(records_session_id(conversation), that + "/")``: the records are
    rewritten whole into the other one past ``RECORDS_REWRITE_AFTER`` writes
    (``save_records``). The second is None when its id would be longer than
    ADK stores (``MAX_KEY``): that conversation keeps appending to its one.
    Neither names an instance (``child_of``).

    Raises:
        ValueError: The first id is longer than ADK stores, as ``records_session_id``.
    """
    first = records_session_id(conversation)
    second = f"{first}/"
    return first, second if len(second) <= MAX_KEY else None


def child_of(session_id: str) -> tuple[str, str] | None:
    """The conversation and the instance a child app's session id names
    (``child_session_id``, decoded); None for a records session (either of its
    two, ``records_session_ids``) or an id that names none."""
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


async def _named_children(
    service: "BaseSessionService", app_name: str, user_id: str, conversation: str
) -> list[str] | None:
    """The ids of the child sessions ``conversation``'s stored records name (one
    look at its records session).

    A first run's session is made only once its start is in the records, so
    they name every one: none when the conversation has no records. None
    only for records that do not read (another version's), which only a
    listing can answer for.
    """
    stored = await load_records(service, SessionRef(app_name, user_id, conversation))
    if stored is None:
        return []
    named = stored_instance_ids(stored)
    if named is None:
        return None
    prefix = f"{_encoded(conversation)}/"
    return [
        f"{prefix}{instance}" for instance in named
        if "/" not in instance and len(prefix) + len(instance) <= MAX_KEY
    ]


async def _listed_children(
    service: "BaseSessionService", app_name: str, user_id: str, conversation: str
) -> list[str]:
    """The ids of every child session of ``user_id`` in ``app_name``'s child app
    that names the conversation (``child_of``, decoded), read with ADK's public
    listing (which returns each session, the records sessions' state with it).
    A records session names none, and the conversation ``s/x``'s children
    (``s%2Fx/<instance>``) are not ``s``'s."""
    listed = await service.list_sessions(app_name=child_app(app_name), user_id=user_id)
    return [
        session.id for session in getattr(listed, "sessions", None) or ()
        if (child_of(session.id) or ("",))[0] == conversation
    ]


async def _children(
    service: "BaseSessionService", app_name: str, user_id: str, conversation: str
) -> list[str]:
    """The ids of ``conversation``'s child sessions, ``user_id``'s, to number past.

    Each is ``child_session_id(conversation, <instance>)``: the instances the
    conversation's records name, none without records, and a listing's only
    for records that do not read.
    """
    named = await _named_children(service, app_name, user_id, conversation)
    if named is not None:
        return named
    return await _listed_children(service, app_name, user_id, conversation)


async def _current_records(
    service: "BaseSessionService", ref: SessionRef
) -> tuple["Session", str, str | None] | None:
    """The records session of ``ref`` with the higher generation (``_GENERATION_KEY``),
    with the two ids (``records_session_ids``); None for none, or when no id fits.

    Read without events (``num_recent_events=0``). Both sessions exist only
    when a rewrite was cut short before it deleted the older one: the newer
    holds the whole records, and a session without the key is generation 0.
    """
    from google.adk.sessions.base_session_service import GetSessionConfig

    try:
        first, second = records_session_ids(ref.session_id)
    except ValueError:
        return None  # its records could never be stored: none are
    best: Session | None = None
    best_generation = -1
    for session_id in (first, second):
        if session_id is None:
            continue
        session = await service.get_session(
            app_name=child_app(ref.app_name), user_id=ref.user_id, session_id=session_id,
            config=GetSessionConfig(num_recent_events=0),
        )
        if session is None:
            continue
        generation = _generation(session.state or {})
        if generation["n"] > best_generation:
            best, best_generation = session, generation["n"]
    return None if best is None else (best, first, second)


def _generation(state: Mapping[str, Any]) -> dict[str, int]:
    """``state``'s generation: its rewrites so far (``n``) and the writes
    appended since (``writes``)."""
    held = state.get(_GENERATION_KEY)
    if not isinstance(held, Mapping):
        return {"n": 0, "writes": 0}
    n, writes = held.get("n"), held.get("writes")
    return {
        "n": n if type(n) is int else 0,
        "writes": writes if type(writes) is int else 0,
    }


async def load_records(service: "BaseSessionService", ref: SessionRef) -> Any | None:
    """``ref``'s records, put back together from its records session; None for none.

    The header under ``RECORDS_KEY`` with ``instances`` and ``runs`` from the
    keys that hold each (in order). Read without the session's events
    (``num_recent_events=0``), from the newer of the two sessions the records
    alternate between (``records_session_ids``).
    """
    found = await _current_records(service, ref)
    if found is None:
        return None
    state = found[0].state or {}
    header = state.get(RECORDS_KEY)
    if not isinstance(header, Mapping):
        return None
    return {
        **header,
        "instances": _held(state, _INSTANCE_KEY),
        "runs": _held(state, _RUN_KEY),
    }


def _held(state: Mapping[str, Any], prefix: str) -> list[Any]:
    """The values of ``state``'s keys under ``prefix``, in key order, but the removed (None)."""
    return [state[key] for key in sorted(state) if key.startswith(prefix) and state[key] is not None]


def _flat(data: Mapping[str, Any]) -> dict[str, Any]:
    """``data`` (records) as state: the header, then a key for each instance and each run."""
    flat: dict[str, Any] = {
        RECORDS_KEY: {k: v for k, v in data.items() if k not in ("instances", "runs")}
    }
    for prefix, name in ((_INSTANCE_KEY, "instances"), (_RUN_KEY, "runs")):
        for number, value in enumerate(data.get(name) or ()):
            flat[f"{prefix}{number:06d}"] = value
    return flat


async def save_records(
    service: "BaseSessionService", ref: SessionRef, data: Mapping[str, Any]
) -> None:
    """Keep ``data`` as ``ref``'s sub-agent records, replacing what its records session held.

    The session is made at the first write, with the records as its state
    (``_flat``); each later write is one event of its own whose state delta
    holds only the keys that differ from what the session holds (the header,
    the instances and runs that changed or came, a key the records no longer
    hold set to None), so a write costs what changed (``append_event``: ADK
    keeps a session's state only through its events). Its events would grow
    without bound, so once ``RECORDS_REWRITE_AFTER`` writes were appended the
    next one rewrites the records whole into the other of the two sessions
    (``records_session_ids``), one generation on (``_GENERATION_KEY``), and
    deletes the first: the older session is deleted before the new one is
    made (one a cut-short rewrite left), and the new one before the old, so
    a crash between leaves both and a read takes the newer.
    """
    from google.adk.events import Event, EventActions

    app = child_app(ref.app_name)
    first, second = records_session_ids(ref.session_id)
    flat = _flat(data)
    found = await _current_records(service, ref)
    if found is None:
        await service.create_session(
            app_name=app, user_id=ref.user_id, session_id=first,
            state={**flat, _GENERATION_KEY: {"n": 0, "writes": 0}},
        )
        return
    session = found[0]
    held = session.state or {}
    delta: dict[str, Any] = {key: value for key, value in flat.items() if held.get(key) != value}
    for key, value in held.items():
        if key.startswith((_INSTANCE_KEY, _RUN_KEY)) and key not in flat and value is not None:
            delta[key] = None
    if not delta:
        return
    generation = _generation(held)
    if generation["writes"] >= RECORDS_REWRITE_AFTER and second is not None:
        other = second if session.id == first else first
        await service.delete_session(app_name=app, user_id=ref.user_id, session_id=other)
        await service.create_session(
            app_name=app, user_id=ref.user_id, session_id=other,
            state={**flat, _GENERATION_KEY: {"n": generation["n"] + 1, "writes": 0}},
        )
        await service.delete_session(app_name=app, user_id=ref.user_id, session_id=session.id)
        return
    delta[_GENERATION_KEY] = {"n": generation["n"], "writes": generation["writes"] + 1}
    await service.append_event(session, Event(
        author=RECORDS_AUTHOR, actions=EventActions(state_delta=delta),
    ))


async def delete_records(service: "BaseSessionService", ref: SessionRef) -> None:
    """Delete ``ref``'s records sessions (both: ``records_session_ids``).

    Their events go with them; none is nothing to do.
    """
    try:
        ids = records_session_ids(ref.session_id)
    except ValueError:
        return  # its records could never be stored: none are
    for session_id in ids:
        if session_id is not None:
            await service.delete_session(
                app_name=child_app(ref.app_name), user_id=ref.user_id, session_id=session_id,
            )


async def delete_children(
    service: "BaseSessionService", app_name: str, user_id: str, conversation: str
) -> int:
    """Delete ``conversation``'s child sessions: its sub-agent instances' transcripts.

    Every session of ``app_name``'s child app (``child_app``) that is
    ``user_id``'s and ``conversation``'s (``child_session_id``): those its
    records name, which name every one (none without records), and, for
    records that do not read, those a listing finds; their events go with
    them (ADK's session store deletes a session's events with it). What a
    conversation's delete leaves of it otherwise.

    Args:
        service: The session service the conversation is in.
        app_name: The conversation's app (not the child app).
        user_id: The conversation's user.
        conversation: The conversation's id.

    Returns:
        How many it deleted: the instances' transcripts, not the records.
    """
    named = await _named_children(service, app_name, user_id, conversation)
    children = sorted(
        named if named is not None
        else await _listed_children(service, app_name, user_id, conversation)
    )
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


def is_notice(event: "Event") -> bool:
    """Whether ``event`` was stored by a notice to the root's run (``NOTICE_METADATA``)."""
    mark = (event.custom_metadata or {}).get("agentic_cli")
    return isinstance(mark, dict) and mark.get("kind") == "notice"


def typed_prompts(events: list["Event"]) -> list[str]:
    """The texts the user typed, oldest first."""
    prompts: list[str] = []
    for event in events:
        content = event.content
        if event.author != "user" or content is None or content.role != "user":
            continue
        if is_notice(event):
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


async def keep_conversation(
    service: "BaseSessionService", ref: SessionRef, objective: str, *, author: str
) -> None:
    """Store ``ref``'s conversation when it holds no event yet, titled from ``objective``.

    ADK lists a session only once it holds an event, and keeps its state only
    through events: the session is made when there is none, and one state-only
    event (no content: no request carries it) sets its title and no turns of
    the user's, the row ``turn_state`` would leave, and an unmarked
    outside-content record (a session with events and none is read as one
    from before 0.7.0 and marked). Its ``author`` is the root
    agent's name: it is only ever the first event of a new conversation, and
    ADK's router, which warns of an event from an agent the app lacks, picks
    the root for it as it would anyway. A session with events is left as it is.
    """
    from google.adk.events import Event, EventActions
    from google.adk.sessions.base_session_service import GetSessionConfig

    session = await service.get_session(
        app_name=ref.app_name, user_id=ref.user_id, session_id=ref.session_id,
        config=GetSessionConfig(num_recent_events=1),
    )
    if session is not None and session.events:
        return
    if session is None:
        session = await service.create_session(
            app_name=ref.app_name, user_id=ref.user_id, session_id=ref.session_id,
        )
    # The conversation is new: it holds no outside content (what a first turn records).
    delta: dict[str, Any] = {
        TURNS_KEY: 0, TITLE_KEY: session_title(objective), OUTSIDE_CONTENT_KEY: "",
    }
    await service.append_event(session, Event(
        author=author, actions=EventActions(state_delta=delta),
    ))


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
