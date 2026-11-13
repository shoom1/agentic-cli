"""Backend-neutral conversation identity.

Every durable session is addressed by the triple ``(app_name, user_id,
session_id)`` — the ADK session services key on exactly that, and any
replacement backend has to carry the same information. ``SessionRef`` makes
that identity explicit so a session created for one user cannot be looked up,
listed, or deleted as another user's by accident.

A stored conversation is listed as a ``SessionInfo`` row, its title made by
``session_title`` on every backend.

Three context-local markers say what is running now: the turn (its
``SessionRef``), inside it the sub-agent run (``RunRef``; none in the root's
own calls), and the tool call (``ToolCallRef``). A fourth, the turn's span,
outlives the turn in a task it started, and tells that task its turn has
ended.
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
        backend: The backend that keeps it (``adk``, ``deepagents``): a
            conversation belongs to one for its whole life. Named by a
            ``SessionReader`` (``workflow/session_readers.py``) and by the
            CLI's listings; None in a manager's own rows, which are all its
            backend's.
    """

    session_id: str
    title: str = ""
    turns: int | None = None
    updated: datetime | None = None
    backend: str | None = None


class ConversationChangedError(RuntimeError):
    """Another process changed the conversation while this turn ran.

    ADK refuses to append to a session changed in its store since it was
    read (``StaleSessionError``); the CLI holds a lock on the conversation it
    has open (``cli/conversation_lock.py``), so this comes from a writer that
    does not.

    Attributes:
        session_id: The conversation.
    """

    def __init__(self, session_id: str) -> None:
        super().__init__(
            f"Conversation '{session_id}' was changed by another process during this turn."
        )
        self.session_id = session_id


#: The longest title, in characters.
TITLE_MAX = 80


def session_title(text: str) -> str:
    """``text`` as a conversation's title: one line, at most ``TITLE_MAX`` characters.

    Control and invisible format characters are removed (a soft hyphen
    joins its word again; the zero-width joiners stay: they change the text,
    a Persian word or an emoji sequence, not hide it), each run of
    whitespace becomes one space, and a longer text is cut with "…".
    """
    line = " ".join("".join(ch for ch in text if _kept(ch)).split())
    if len(line) <= TITLE_MAX:
        return line
    return line[: TITLE_MAX - 1].rstrip() + "…"


def _kept(ch: str) -> bool:
    return ch.isprintable() or ch.isspace() or ch in _JOINERS


#: Zero-width non-joiner and joiner.
_JOINERS = "\u200c\u200d"


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


class _TurnScope:
    """One turn's span: open while the turn runs, ended as it ends.

    A task a turn starts copies the turn's context, this object with it; the
    active turn alone (the same ``SessionRef`` before and after) cannot tell
    such a task that the turn it came from is over, this can.
    """

    __slots__ = ("ended",)

    def __init__(self) -> None:
        self.ended = False


_turn_scope: ContextVar["_TurnScope | None"] = ContextVar(
    "agentic_cli_turn_scope", default=None
)


def open_turn_scope() -> Token:
    """Begin a turn's span in this context (``BaseWorkflowManager._workflow_context``).

    Returns:
        Token for :func:`close_turn_scope`.
    """
    return _turn_scope.set(_TurnScope())


def close_turn_scope(token: Token) -> None:
    """End the span ``token`` began, and restore the one before it.

    Every context copied from inside the span (a task the turn started that
    outlives it) now sees it ended: :func:`turn_has_ended`.
    """
    scope = _turn_scope.get()
    if scope is not None:
        scope.ended = True
    _turn_scope.reset(token)


def turn_has_ended() -> bool:
    """Whether this context belongs to a turn that has ended.

    True only in a context copied from inside a turn after that turn ended:
    an asyncio task the turn started that outlived it. False inside a turn,
    and outside any.
    """
    scope = _turn_scope.get()
    return scope is not None and scope.ended


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


@dataclass(frozen=True, slots=True)
class RunRef:
    """Which agent run an event, a call or a request belongs to.

    A sub-agent's run is named within its conversation
    (``workflow/agents/runs.py``): its instance ``<agent>-<n>`` and its run
    ``<instance>#<n>``, under its parent's run. The root's own work is no
    run (``root_run``): instance ``root``, ``run`` None.

    Attributes:
        agent: The agent's name (its ``AgentConfig.name``; an ADK agent's name).
        instance: ``root`` for the root; ``<agent>-<n>`` for a sub-agent's instance.
        run: ``<instance>#<n>``; None for the root's own work (and a depth-0
            agent's, one the root transfers to).
        parent_run: The parent's run id; None when the parent is the root.
        depth: 0 for the root, the parent's depth + 1.
    """

    agent: str
    instance: str
    run: str | None
    parent_run: str | None
    depth: int

    @property
    def is_root(self) -> bool:
        """Whether this is the root's own work: no sub-agent run."""
        return self.run is None


def root_run(agent: str) -> RunRef:
    """The root agent ``agent``'s own work: instance ``root``, no run, depth 0."""
    return RunRef(agent=agent, instance="root", run=None, parent_run=None, depth=0)


# The sub-agent run whose call is in progress in this context: set inside a
# sub-agent's run (ADK: as the agent starts, from its branch; Deep Agents:
# around its coordinator's ``task`` call), so its model calls, tool calls,
# permission checks and questions see it. None in the root's own calls. A
# ContextVar for the same reason as the active turn: parallel runs each run
# in their own context and never see one another's.
_active_run: ContextVar["RunRef | None"] = ContextVar("agentic_cli_active_run", default=None)


def set_active_run(ref: "RunRef | None") -> Token:
    """Mark ``ref`` as the sub-agent run in progress in this context.

    Returns:
        Token for :func:`reset_active_run`, which restores the run that was
        active before (a nested run's parent).
    """
    return _active_run.set(ref)


def reset_active_run(token: Token) -> None:
    """Restore the run recorded before the matching :func:`set_active_run`."""
    _active_run.reset(token)


def get_active_run() -> "RunRef | None":
    """The sub-agent run in progress in this context, or None (the root's own calls)."""
    return _active_run.get()


def instance_of(run: "RunRef | None") -> str | None:
    """The instance whose call ``run`` names: a sub-agent run's instance id,
    None for the root's own (no run, or ``root_run``).

    What a job a call of ``run`` starts records as its ``started_by``
    (``tools/jobs/manager.py``), and what a call's job is matched by
    (``workflow/interrupts.py::job_for_call``).
    """
    return None if run is None or run.is_root else run.instance
