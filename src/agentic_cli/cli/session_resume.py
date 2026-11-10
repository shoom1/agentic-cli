"""What reopening a stored conversation shows: the picker, the line, the replay.

``BaseCLIApp.pick_session`` asks which conversation to open and
``BaseCLIApp.resume_session`` opens it; this module holds what they show:

- ``picker(rows, now, placement=, backends=)``: the dialog listing the
  conversations, newest first, one option each: how long ago, its turns, its
  backend (when the rows are not all the running one's), its title and its
  id. The parts that tell rows apart come first, since a terminal narrower
  than a row clips its end; the backend and the id keep every option unique.
  It returns, for each option, the id it stands for as a
  ``PickedConversation``: a ``str`` that names its backend too;
- ``resumed_line(info, now, backend=)``: "Resumed: <title> (<id>, 12 turns,
  last active 2h ago).", "Resumed on <backend>: …" on another backend;
- ``replay_lines(messages)`` and ``show_replay(session, lines)``: the last
  ``REPLAY_MESSAGES`` messages, the user's as typed and the main agent's as a
  turn shows them, each cut at ``REPLAY_CHARS``, after "… N earlier
  messages" when there are more. A message's content blocks are read for
  their text (a tool call is never replayed), and one without text is left
  out.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from thinking_prompt import ChoiceField, Dialog

if TYPE_CHECKING:
    from thinking_prompt import Placement, ThinkingPromptSession

    from agentic_cli.workflow.sessions import SessionInfo

#: The messages a resume replays.
REPLAY_MESSAGES = 10
#: The longest replayed message, in characters.
REPLAY_CHARS = 2_000
#: The conversations the picker lists.
PICK_LIMIT = 50
#: A conversation's title before its first prompt.
NO_PROMPT = "(no prompt yet)"


def ago(updated: datetime | None, now: datetime) -> str:
    """How long before ``now`` ``updated`` was: "just now", "5m ago", "2h ago", "3d ago".

    A time without a zone is taken as UTC.
    """
    if updated is None:
        return ""
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    minutes = int(max(0.0, (now - updated).total_seconds()) // 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes}m ago"
    if minutes < 24 * 60:
        return f"{minutes // 60}h ago"
    return f"{minutes // (24 * 60)}d ago"


def _turns(turns: int | None) -> str:
    if turns is None:
        return ""
    return "1 turn" if turns == 1 else f"{turns} turns"


def title_of(info: "SessionInfo") -> str:
    """A row's title: its first prompt, ``NO_PROMPT`` for one stored without
    any, or "" when the backend cannot tell (LangGraph knows the id alone)."""
    if info.title:
        return info.title
    return NO_PROMPT if info.turns is not None else ""


def resumed_line(info: "SessionInfo", now: datetime, *, backend: str | None = None) -> str:
    """What a resume says: "Resumed: <title> (<id>, 12 turns, last active 2h ago).".

    With ``backend`` (a conversation opened on another backend than the one
    that ran: it runs now) it says so: "Resumed on deepagents: …".
    """
    details = [info.session_id]
    if info.turns is not None:
        details.append(_turns(info.turns))
    when = ago(info.updated, now)
    if when:
        details.append(f"last active {when}")
    title = title_of(info)
    resumed = f"Resumed on {backend}" if backend else "Resumed"
    if not title:
        return f"{resumed}: {', '.join(details)}."
    return f"{resumed}: {title} ({', '.join(details)})."


def _text(content: Any) -> str:
    """A message's text: a string as it is, a list of content blocks' text joined."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block if isinstance(block, str) else block.get("text", "")
            for block in content
            if isinstance(block, str) or (isinstance(block, dict) and block.get("type") == "text")
        ]
        return " ".join(part for part in parts if part)
    return "" if content is None else str(content)


def replay_lines(messages: list[dict]) -> list[tuple[str, str]]:
    """What a resume shows of ``messages`` (``recent_messages`` dicts), in order.

    Each is ``(role, text)``: ``"system"`` for the earlier-messages line,
    ``"user"`` or ``"assistant"``.
    """
    texts = [
        ("user" if message.get("role") == "user" else "assistant", text)
        for message in messages
        if (text := _text(message.get("content")))
    ]
    shown = texts[-REPLAY_MESSAGES:]
    lines: list[tuple[str, str]] = []
    earlier = len(texts) - len(shown)
    if earlier:
        plural = "" if earlier == 1 else "s"
        lines.append(("system", f"… {earlier} earlier message{plural}"))
    for role, text in shown:
        if len(text) > REPLAY_CHARS:
            text = text[:REPLAY_CHARS] + "… (cut)"
        lines.append((role, text))
    return lines


def show_replay(session: "ThinkingPromptSession", lines: list[tuple[str, str]]) -> None:
    """Show ``replay_lines``: the main agent's as a turn shows them (markdown)."""
    for role, text in lines:
        if role == "assistant":
            session.add_response(text, markdown=True)
        else:
            session.add_message(role, text)


class PickedConversation(str):
    """A conversation's id as the picker returns it: a ``str``, whose
    ``backend`` names the backend that keeps it (None when the row did not
    say). ``BaseCLIApp.resume_session`` opens it there: an id two backends
    keep is two rows, and the one picked is the one opened."""

    backend: str | None

    def __new__(cls, session_id: str, backend: str | None = None) -> "PickedConversation":
        picked = super().__new__(cls, session_id)
        picked.backend = backend
        return picked


def picker(
    rows: list["SessionInfo"],
    now: datetime,
    *,
    placement: "Placement | None" = None,
    backends: bool = False,
) -> tuple[Dialog, dict[str, str]]:
    """The dialog that lists ``rows``, and the id each of its options stands for.

    Its only field is a choice list, so Enter on an option closes it with
    that option, and Esc with None. ``placement`` is where it is drawn
    (the ``dialog_placement`` setting now: the session's own is the one it
    started with); None for the session's. With ``backends`` each option
    names its row's backend (``SessionInfo.backend``), after its turns. Each
    id is a ``PickedConversation``, naming its row's backend.
    """
    ids: dict[str, str] = {}
    width = max((len(info.backend or "") for info in rows), default=0)
    for info in rows:
        when, turns = ago(info.updated, now).ljust(8), _turns(info.turns).ljust(9)
        title = title_of(info)
        middle = f" · {title}" if title else ""
        backend = f" · {(info.backend or '').ljust(width)}" if backends else ""
        label = f"{when} · {turns}{backend}{middle} · {info.session_id}"
        ids[label] = PickedConversation(info.session_id, info.backend)
    dialog = Dialog(
        "Resume a conversation",
        fields=[ChoiceField("session", "", options=list(ids))],
        placement=placement,
    )
    return dialog, ids
