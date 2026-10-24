"""Backend-neutral conversation identity.

Every durable session is addressed by the triple ``(app_name, user_id,
session_id)`` — the ADK session services key on exactly that, and any
replacement backend has to carry the same information. ``SessionRef`` makes
that identity explicit so a session created for one user cannot be looked up,
listed, or deleted as another user's by accident.

A stored conversation is listed as a ``SessionInfo`` row, its title made by
``session_title`` on every backend.

Two context-local markers say what is running now: the turn (its
``SessionRef``) and, inside it, the tool call (``ToolCallRef``).
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class SessionRef:
    """Identity of one conversation in a session store.

    Attributes:
        app_name: Namespace of the owning application.
        user_id: Owner of the conversation.
        session_id: Conversation identifier, unique within (app_name, user_id).
    """

    app_name: str
    user_id: str
    session_id: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.app_name}/{self.user_id}/{self.session_id}"


@dataclass(frozen=True, slots=True)
class SessionInfo:
    """One stored conversation, as a session list shows it.

    Attributes:
        session_id: The conversation's id.
        title: Its first prompt on one line (``session_title``); ``""`` when
            it has none.
        turns: How many turns the user ran in it (a job's resume is none);
            None when the backend cannot tell.
        updated: When it last changed; None when the backend cannot tell.
    """

    session_id: str
    title: str = ""
    turns: int | None = None
    updated: datetime | None = None


#: The longest title, in characters.
TITLE_MAX = 80


def session_title(text: str) -> str:
    """``text`` as a conversation's title: one line, at most ``TITLE_MAX`` characters.

    Control and invisible format characters are removed (a soft hyphen
    joins its word again), each run of whitespace becomes one space, and a
    longer text is cut with "…".
    """
    line = " ".join("".join(ch for ch in text if ch.isprintable() or ch.isspace()).split())
    if len(line) <= TITLE_MAX:
        return line
    return line[: TITLE_MAX - 1].rstrip() + "…"


# The conversation whose turn is currently executing. A ContextVar, not a
# manager attribute: one manager instance may drive several turns at once when
# it is embedded in a server (the CLI serializes turns, framework consumers do
# not), and concurrent tasks must never observe each other's identity. Each
# asyncio task gets its own copy of the context, so isolation is automatic.
_active_turn: ContextVar["SessionRef | None"] = ContextVar(
    "agentic_cli_active_turn", default=None
)


def set_active_turn(ref: "SessionRef | None") -> Token:
    """Mark ``ref`` as the turn running in this context.

    Returns:
        Token for :func:`reset_active_turn` — reset restores the *previous*
        value, so nested turns do not erase the outer one.
    """
    return _active_turn.set(ref)


def reset_active_turn(token: Token) -> None:
    """Restore the active turn recorded before the matching :func:`set_active_turn`."""
    _active_turn.reset(token)


def get_active_turn() -> "SessionRef | None":
    """The turn executing in this context, or None when idle."""
    return _active_turn.get()


@dataclass(frozen=True, slots=True)
class ToolCallRef:
    """One tool call of a turn, as the model made it.

    Attributes:
        call_id: The model's id for the call (its ``tool_call_id``).
        name: The tool the model called.
    """

    call_id: str
    name: str


# The tool call whose tool is running in this context. Set around each call
# (Deep Agents' ``ActiveToolCallMiddleware``) so ``JobManager.submit`` can link
# a job to the call that started it when the tool does not say which. A
# ContextVar for the same reason as the active turn: parallel calls, and a
# sub-agent's calls inside its coordinator's ``task`` call, each run in their
# own context and never see one another's.
_active_tool_call: ContextVar["ToolCallRef | None"] = ContextVar(
    "agentic_cli_active_tool_call", default=None
)


def set_active_tool_call(ref: "ToolCallRef | None") -> Token:
    """Mark ``ref`` as the tool call running in this context.

    Returns:
        Token for :func:`reset_active_tool_call`, which restores the call
        that was active before (a sub-agent's call nests inside ``task``).
    """
    return _active_tool_call.set(ref)


def reset_active_tool_call(token: Token) -> None:
    """Restore the tool call recorded before the matching :func:`set_active_tool_call`."""
    _active_tool_call.reset(token)


def get_active_tool_call() -> "ToolCallRef | None":
    """The tool call running in this context, or None outside one."""
    return _active_tool_call.get()
