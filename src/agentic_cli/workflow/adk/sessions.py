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
result as a function response; none of them is text.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from agentic_cli.tools.jobs.resume import JOB_NOTICE_HEADER
from agentic_cli.workflow.sessions import SessionInfo, session_title

if TYPE_CHECKING:
    from google.adk.events import Event
    from google.adk.sessions import Session

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
    return SessionInfo(session.id, str(state.get(TITLE_KEY) or ""), turns, _updated(session))


def row_from_events(session: "Session") -> SessionInfo:
    """The row of a conversation from before the keys, read from its events."""
    typed = typed_prompts(session.events)
    title = session_title(typed[0]) if typed else ""
    return SessionInfo(session.id, title, len(typed), _updated(session))


def _updated(session: "Session") -> datetime | None:
    stamp = session.last_update_time
    return datetime.fromtimestamp(stamp, tz=timezone.utc) if stamp else None
