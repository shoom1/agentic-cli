"""Conversations on a LangGraph checkpointer, keyed by the whole ``SessionRef``.

A conversation is the root agent's thread. Its ``thread_id`` is the triple
itself, ``json.dumps([app_name, user_id, session_id])``: no two triples give
the same id (a joined key would let ``("a/b", "c")`` and ``("a", "b/c")``
share one), and ``json.loads`` reads it back. Every checkpoint of the turn,
sub-agents' included, also carries the three values in its metadata
(``run_config``), so a listing can ask the store for one app and user.

Sub-agent checkpoints sit in the same thread under their task's namespace
(``tools:<task-id>``): a listing keeps the root namespace (``""``), and
deleting the thread removes them with it.

A finished job reports to the conversation in a ``HumanMessage`` of its own
(``job_notification``), marked in its ``response_metadata``, which no
provider adapter sends. The model reads it as the rest of the conversation;
``recent`` leaves it out, because the user did not write it.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agentic_cli.tools.jobs.resume import job_notice
from agentic_cli.workflow.sessions import SessionRef

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple

#: The namespace of the root agent's checkpoints.
_ROOT = ""

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


def job_notification(record: Any, result: Any) -> HumanMessage:
    """The message that tells the root agent ``record``'s job finished.

    Its text is ``job_notice``: ``JOB_NOTICE_HEADER``, then the payload ADK
    sends as the pending call's result (``job_result_payload``), as JSON.

    Args:
        record: The finished ``JobRecord``.
        result: The job's result, or None.
    """
    return HumanMessage(
        content=job_notice(record, result),
        response_metadata={_MARK: {"kind": JOB_RESULT, "job_id": record.job_id}},
    )


def is_job_notification(message: BaseMessage) -> bool:
    """Whether ``message`` is a job notification (by its mark, never its text)."""
    mark = message.response_metadata.get(_MARK)
    return isinstance(mark, dict) and mark.get("kind") == JOB_RESULT


class CheckpointSessions:
    """The session hooks, answered from a checkpointer.

    Reads go to the saver, not to a compiled graph, so they work whenever the
    checkpointer is open, also while the agents are being rebuilt.

    Args:
        saver: The checkpointer the root agent writes to.
    """

    def __init__(self, saver: "BaseCheckpointSaver") -> None:
        self._saver = saver

    async def exists(self, ref: SessionRef) -> bool:
        """Whether ``ref``'s root thread holds at least one message."""
        return bool(_messages(await self._root(ref)))

    async def list(self, app_name: str, user_id: str) -> list[dict]:
        """The conversations of one app and one user, newest first.

        Each is ``{"session_id", "last_update", "message_count"}``:
        ``last_update`` is the latest root checkpoint's time in epoch seconds,
        ``message_count`` the root conversation's length. Threads whose id
        does not name this app and user are left out, whatever their metadata
        says.
        """
        latest: dict[str, "CheckpointTuple"] = {}
        wanted = {"app_name": app_name, "user_id": user_id}
        async for item in self._saver.alist(None, filter=wanted):
            configurable = item.config["configurable"]
            if configurable.get("checkpoint_ns", _ROOT) != _ROOT:
                continue  # a sub-agent's checkpoint
            ref = ref_from_thread_id(configurable["thread_id"])
            if ref is None or (ref.app_name, ref.user_id) != (app_name, user_id):
                continue
            seen = latest.get(ref.session_id)
            if seen is None or _epoch(item) > _epoch(seen):
                latest[ref.session_id] = item
        sessions = [
            {
                "session_id": session_id,
                "last_update": _epoch(item),
                "message_count": len(_messages(item)),
            }
            for session_id, item in latest.items()
        ]
        sessions.sort(key=lambda entry: entry["last_update"], reverse=True)
        return sessions

    async def delete(self, ref: SessionRef) -> bool:
        """Delete ``ref``'s thread, sub-agents included; False if there is none."""
        if await self._root(ref) is None:
            return False
        await self._saver.adelete_thread(thread_id(ref))
        return True

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

    async def _root(self, ref: SessionRef) -> "CheckpointTuple | None":
        """The latest checkpoint of ``ref``'s root agent, or None."""
        return await self._saver.aget_tuple(
            {"configurable": {"thread_id": thread_id(ref), "checkpoint_ns": _ROOT}}
        )


def _messages(item: "CheckpointTuple | None") -> list[BaseMessage]:
    if item is None:
        return []
    return list(item.checkpoint["channel_values"].get("messages") or ())


def _epoch(item: "CheckpointTuple") -> float:
    return datetime.fromisoformat(item.checkpoint["ts"]).timestamp()


def _text(message: BaseMessage) -> str:
    """A message's text blocks, joined (its reasoning and tool calls left out)."""
    return " ".join(
        block["text"]
        for block in message.content_blocks
        if block.get("type") == "text" and block.get("text")
    ).strip()
