"""Backend-neutral conversation identity.

Every durable session is addressed by the triple ``(app_name, user_id,
session_id)`` — the ADK session services key on exactly that, and any
replacement backend has to carry the same information. ``SessionRef`` makes
that identity explicit so a session created for one user cannot be looked up,
listed, or deleted as another user's by accident.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass


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
