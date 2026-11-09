"""Conversations on a LangGraph checkpointer, keyed by the whole ``SessionRef``.

A conversation is the root agent's thread. Its ``thread_id`` is the triple
itself, ``json.dumps([app_name, user_id, session_id])``: no two triples give
the same id (a joined key would let ``("a/b", "c")`` and ``("a", "b/c")``
share one), and ``json.loads`` reads it back. Every checkpoint of the turn
also carries the three values in its metadata (``run_config``), so a scan can
ask the saver for one app and user.

Sub-agents store no checkpoints (``agent_builder`` compiles them with
``checkpointer=False``). A thread an older version wrote may hold some, under
their task's namespace (``tools:<task-id>``): a scan keeps the root namespace
(``""``), and deleting the thread removes them with it.

A conversation's row (``SessionInfo``) is kept in an index, a LangGraph store
beside the checkpointer (``checkpointers.open_store``), under
``("agentic_cli", "sessions", <app>, <user>)``, keyed by the session id. The
manager writes it as each turn starts, the row the turn will leave (its input
counted), so a turn the process does not survive still lists its
conversation; and from the conversation's latest root checkpoint as each turn
ends, however it ends, and after ``/compact`` (``record``). A listing reads
one small row per conversation rather than every checkpoint of the user, each
of which holds the whole conversation. A database written before the index is
scanned once per app and user, on its first listing (the backfill), which then
leaves a mark under ``("agentic_cli", "session_backfill", <app>, <user>)``.

A finished job reports to the conversation in a ``HumanMessage`` of its own
(``job_notification``), marked in its ``response_metadata``, which no
provider adapter sends. The model reads it as the rest of the conversation;
``recent`` leaves it out, because the user did not write it, and so does a
conversation's row: its title is the first prompt the user typed, its turns
how many they typed.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.store.base import PutOp

from agentic_cli.logging import Loggers
from agentic_cli.tools.jobs.resume import job_notice
from agentic_cli.workflow.sessions import SessionInfo, SessionRef, session_title

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple
    from langgraph.store.base import BaseStore

logger = Loggers.workflow()

#: The namespace of the root agent's checkpoints.
_ROOT = ""

#: The index's namespaces (each then takes the app and the user): the rows,
#: and the backfill's marks.
_ROWS = ("agentic_cli", "sessions")
_BACKFILLS = ("agentic_cli", "session_backfill")

#: The key of an app and user's backfill mark.
_MARK_KEY = "done"

#: The format of a row (its ``v``).
_ROW_VERSION = 1

#: The backfill a mark records (its ``v``). Raising it scans every app and
#: user once more, on their next listing.
_BACKFILL_VERSION = 1

#: How many rows the first search of the index asks for.
_SEARCH_LIMIT = 500

#: Where a message the framework wrote is marked (``response_metadata``).
_MARK = "agentic_cli"

#: The mark's ``kind`` on a job notification.
JOB_RESULT = "job_result"


def thread_id(ref: SessionRef) -> str:
    """The checkpointer thread of ``ref``'s conversation."""
    return json.dumps([ref.app_name, ref.user_id, ref.session_id])


def ref_from_thread_id(value: str) -> SessionRef | None:
    """The conversation a thread id names, or None for one that names none."""
    try:
        parts = json.loads(value)
    except (TypeError, ValueError):
        return None
    if (
        isinstance(parts, list)
        and len(parts) == 3
        and all(isinstance(part, str) for part in parts)
    ):
        return SessionRef(*parts)
    return None


def run_config(ref: SessionRef, **extra: Any) -> dict[str, Any]:
    """The run config of a turn in ``ref``'s conversation.

    Args:
        ref: The conversation.
        **extra: More run config keys (``callbacks``, for one).
    """
    return {
        "configurable": {"thread_id": thread_id(ref)},
        "metadata": {
            "app_name": ref.app_name,
            "user_id": ref.user_id,
            "session_id": ref.session_id,
        },
        **extra,
    }


def job_notification(
    record: Any, result: Any = None, *, notice: str | None = None
) -> HumanMessage:
    """The message that tells the root agent ``record``'s job finished.

    Its text is ``job_notice``: ``JOB_NOTICE_HEADER``, ``JOB_NOTICE_ASK``,
    then the payload (``job_result_payload``) as JSON, as on ADK.

    Args:
        record: The finished ``JobRecord``.
        result: The job's result, or None.
        notice: The text, when it was built already (the base's
            ``resume_with_job_result`` builds it); ``result`` is then unused.
    """
    return HumanMessage(
        content=job_notice(record, result) if notice is None else notice,
        response_metadata={_MARK: {"kind": JOB_RESULT, "job_id": record.job_id}},
    )


def is_job_notification(message: BaseMessage) -> bool:
    """Whether ``message`` is a job notification (by its mark, never its text)."""
    mark = message.response_metadata.get(_MARK)
    return isinstance(mark, dict) and mark.get("kind") == JOB_RESULT


class CheckpointSessions:
    """The session hooks, answered from a checkpointer and the index beside it.

    Reads go to the saver and the store, not to a compiled graph, so they
    work whenever the checkpointer is open, also while the agents are being
    rebuilt. Listing and ``info`` read the index; ``exists`` and ``recent``
    read the conversation itself, from its latest root checkpoint.

    Args:
        saver: The checkpointer the root agent writes to.
        store: The store holding the conversations' rows (``open_store``).
        backfill: Whether the first listing of an app and user scans the
            saver for conversations written before the index. False for a
            saver opened with its store, which holds none (the in-memory one).
    """

    def __init__(
        self, saver: "BaseCheckpointSaver", store: "BaseStore", *, backfill: bool = True
    ) -> None:
        self._saver = saver
        self._store = store
        self._backfill = backfill

    async def exists(self, ref: SessionRef) -> bool:
        """Whether ``ref``'s root thread holds at least one message."""
        return bool(_messages(await self._root(ref)))

    async def info(self, ref: SessionRef) -> SessionInfo | None:
        """``ref``'s row, or None when its root thread holds no message.

        The index's row; a conversation it does not hold (written before it,
        or whose row could not be written) is read from its checkpoint.
        """
        namespace = _rows_of(ref.app_name, ref.user_id)
        item = await self._store.aget(namespace, ref.session_id)
        row = _from_value(ref.session_id, item.value) if item is not None else None
        if row is not None:
            return row
        checkpoint = await self._root(ref)
        return _row(ref.session_id, checkpoint) if _messages(checkpoint) else None

    async def record(self, ref: SessionRef, adding: Sequence[BaseMessage] = ()) -> None:
        """Write ``ref``'s row from its latest root checkpoint. Never raises.

        The manager calls it as each turn starts, with the turn's input as
        ``adding``: the row is then the one the turn will leave, those
        messages counted after the checkpoint's (a typed prompt is a turn, a
        job's notice none) and the time now. A process killed in the turn
        (SIGKILL, the terminal closed) runs no end of turn, and its
        conversation is still listed. The manager calls it again as each turn
        ends, however it ends, and after ``/compact``, without ``adding``,
        which corrects that row. A conversation without a message has none:
        a first turn stopped before its input was stored takes its row back.
        A failed write is logged at debug and costs nothing else: the
        conversation's next turn writes its row again.
        """
        namespace = _rows_of(ref.app_name, ref.user_id)
        try:
            item = await self._root(ref)
            if adding:
                messages = [*_messages(item), *adding]
                row = _row_of(ref.session_id, messages, datetime.now(timezone.utc))
            elif _messages(item):
                row = _row(ref.session_id, item)
            else:
                await self._store.adelete(namespace, ref.session_id)
                return
            await self._store.aput(namespace, ref.session_id, _value(row), index=False)
        except Exception:  # noqa: BLE001 - the turn is not the index's to fail
            logger.debug(
                "session_row_not_written", session_id=ref.session_id, exc_info=True
            )

    async def list(
        self, app_name: str, user_id: str, *, limit: int | None = None
    ) -> list[SessionInfo]:
        """The conversations of one app and one user, newest first.

        The index's rows, one per conversation. The first listing of an app
        and user whose conversations may predate the index fills it first,
        once (``_fill``).

        Args:
            app_name: The app.
            user_id: The user.
            limit: Only the newest this many (None: all of them).
        """
        if self._backfill and not await self._filled(app_name, user_id):
            rows = await self._fill(app_name, user_id)
        else:
            rows = await self._rows(app_name, user_id)
        listed = sorted(rows.values(), key=lambda row: row.updated, reverse=True)
        return listed if limit is None else listed[:limit]

    async def delete(self, ref: SessionRef) -> bool:
        """Delete ``ref``'s thread and its row; False if there was neither.

        The thread goes with any sub-agent checkpoints an older version
        stored in it.
        """
        namespace = _rows_of(ref.app_name, ref.user_id)
        stored = await self._root(ref) is not None
        indexed = await self._store.aget(namespace, ref.session_id) is not None
        if stored:
            await self._saver.adelete_thread(thread_id(ref))
        if indexed:
            await self._store.adelete(namespace, ref.session_id)
        return stored or indexed

    async def recent(self, ref: SessionRef, limit: int) -> list[dict]:
        """The last ``limit`` user and assistant texts of ``ref``'s conversation.

        Each is ``{"role": "user" | "assistant", "content": <text>}``, the
        message's text blocks joined; tool calls and results, reasoning,
        messages without text and job notifications are left out.
        """
        out: list[dict] = []
        for message in _messages(await self._root(ref)):
            if isinstance(message, HumanMessage):
                if is_job_notification(message):
                    continue
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                continue
            text = _text(message)
            if text:
                out.append({"role": role, "content": text})
        return out[-limit:] if limit > 0 else []

    async def outside_content(self, ref: SessionRef) -> str:
        """The conversation's mark in its newest root checkpoint (``""``: none)."""
        item = await self._root(ref)
        if item is None:
            return ""
        value = item.checkpoint["channel_values"].get("outside_content")
        return value if isinstance(value, str) else ""

    async def _root(self, ref: SessionRef) -> "CheckpointTuple | None":
        """The latest checkpoint of ``ref``'s root agent, or None."""
        return await self._saver.aget_tuple(
            {"configurable": {"thread_id": thread_id(ref), "checkpoint_ns": _ROOT}}
        )

    async def _rows(self, app_name: str, user_id: str) -> dict[str, SessionInfo]:
        """The index's rows of one app and user, by session id.

        One search returns them all; one that comes back full is asked again
        for more. Not pages: the stores order a search by their own write
        time, which a batch's rows share, so an offset could skip a row. The
        sqlite and Postgres stores match a namespace prefix with ``LIKE``,
        case-blind in sqlite and with ``_`` and ``%`` as wildcards, so a
        search returns other users' rows too: only this namespace's are kept.
        """
        namespace = _rows_of(app_name, user_id)
        limit = _SEARCH_LIMIT
        found = await self._store.asearch(namespace, limit=limit)
        while len(found) == limit:
            limit *= 4
            found = await self._store.asearch(namespace, limit=limit)
        rows: dict[str, SessionInfo] = {}
        for item in found:
            if tuple(item.namespace) != namespace:
                continue
            row = _from_value(item.key, item.value)
            if row is not None:
                rows[item.key] = row
        return rows

    async def _filled(self, app_name: str, user_id: str) -> bool:
        """Whether the app and user's backfill mark is there, and current."""
        mark = await self._store.aget(_marks_of(app_name, user_id), _MARK_KEY)
        version = mark.value.get("v") if mark is not None else None
        return isinstance(version, int) and version >= _BACKFILL_VERSION

    async def _fill(self, app_name: str, user_id: str) -> dict[str, SessionInfo]:
        """Index the app and user's conversations from their checkpoints, once.

        Scans the saver (``_scan``), writes the row of each conversation the
        index lacks or holds an older row of, and marks the app and user
        done, in one batch. The rows are read after the scan, so a newer one
        that another process's turn wrote meanwhile is kept. A failed write
        is logged at debug, and the next listing scans again.

        Returns:
            The rows, by session id: the index's, with the scan's written over.
        """
        scanned = await self._scan(app_name, user_id)
        rows = await self._rows(app_name, user_id)
        namespace = _rows_of(app_name, user_id)
        writes: list[PutOp] = []
        for session_id, row in scanned.items():
            held = rows.get(session_id)
            if held is None or held.updated < row.updated:
                rows[session_id] = row
                writes.append(PutOp(namespace, session_id, _value(row), index=False))
        mark = PutOp(
            _marks_of(app_name, user_id), _MARK_KEY, {"v": _BACKFILL_VERSION},
            index=False,
        )
        try:
            await self._store.abatch([*writes, mark])
        except Exception:  # noqa: BLE001 - the rows are read; the index waits
            logger.debug("session_index_not_filled", app_name=app_name, exc_info=True)
        else:
            logger.info("session_index_filled", app_name=app_name, rows=len(writes))
        return rows

    async def _scan(self, app_name: str, user_id: str) -> dict[str, SessionInfo]:
        """The rows of one app and user's conversations, from every checkpoint.

        Each comes from the conversation's latest root checkpoint: its time,
        and the prompts the user typed (``_row``). Threads whose id does not
        name this app and user are left out, whatever their metadata says,
        and so is a thread whose latest root checkpoint holds no message
        (``info`` and ``exists`` say it is not stored). Every checkpoint of
        the user is read: the backfill alone does this.
        """
        latest: dict[str, "CheckpointTuple"] = {}
        wanted = {"app_name": app_name, "user_id": user_id}
        async for item in self._saver.alist(None, filter=wanted):
            configurable = item.config["configurable"]
            if configurable.get("checkpoint_ns", _ROOT) != _ROOT:
                continue  # a sub-agent's checkpoint, from an older version
            ref = ref_from_thread_id(configurable["thread_id"])
            if ref is None or (ref.app_name, ref.user_id) != (app_name, user_id):
                continue
            seen = latest.get(ref.session_id)
            if seen is None or _updated(item) > _updated(seen):
                latest[ref.session_id] = item
        return {
            session_id: _row(session_id, item)
            for session_id, item in latest.items()
            if _messages(item)
        }


def _label(value: str) -> str:
    """``value`` as a store namespace label: JSON, its periods escaped.

    A label may be neither empty nor hold a period (the sqlite and Postgres
    stores join a namespace's labels with periods). A JSON string never is
    empty, and ``\\u002e`` reads back as the period it stands for; no JSON
    escape holds one.
    """
    return json.dumps(value).replace(".", "\\u002e")


def _rows_of(app_name: str, user_id: str) -> tuple[str, ...]:
    """The namespace of one app and user's rows."""
    return (*_ROWS, _label(app_name), _label(user_id))


def _marks_of(app_name: str, user_id: str) -> tuple[str, ...]:
    """The namespace of one app and user's backfill mark."""
    return (*_BACKFILLS, _label(app_name), _label(user_id))


def _value(row: SessionInfo) -> dict[str, Any]:
    """A row as the index stores it."""
    return {
        "title": row.title,
        "turns": row.turns,
        "updated": row.updated.isoformat(),
        "v": _ROW_VERSION,
    }


def _from_value(session_id: str, value: Any) -> SessionInfo | None:
    """The row a stored value holds, or None for one of another format."""
    if not isinstance(value, dict) or value.get("v") != _ROW_VERSION:
        return None
    fields = (value.get("title"), value.get("turns"), value.get("updated"))
    if [type(field) for field in fields] != [str, int, str]:
        return None
    title, turns, updated = fields
    try:
        return SessionInfo(session_id, title, turns, datetime.fromisoformat(updated))
    except ValueError:
        return None


def _messages(item: "CheckpointTuple | None") -> list[BaseMessage]:
    if item is None:
        return []
    return list(item.checkpoint["channel_values"].get("messages") or ())


def _updated(item: "CheckpointTuple") -> datetime:
    return datetime.fromisoformat(item.checkpoint["ts"])


def _row(session_id: str, item: "CheckpointTuple") -> SessionInfo:
    """A conversation's row from its latest root checkpoint."""
    return _row_of(session_id, _messages(item), _updated(item))


def _row_of(
    session_id: str, messages: Sequence[BaseMessage], updated: datetime
) -> SessionInfo:
    """A conversation's row: its first typed prompt, how many, and ``updated``."""
    typed = [
        text
        for message in messages
        if isinstance(message, HumanMessage) and not is_job_notification(message)
        if (text := _text(message))
    ]
    title = session_title(typed[0]) if typed else ""
    return SessionInfo(session_id, title, len(typed), updated)


def _text(message: BaseMessage) -> str:
    """A message's text blocks, joined (its reasoning and tool calls left out)."""
    return " ".join(
        block["text"]
        for block in message.content_blocks
        if block.get("type") == "text" and block.get("text")
    ).strip()
