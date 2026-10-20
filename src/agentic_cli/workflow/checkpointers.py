"""LangGraph checkpointers for the LangChain-based backends.

Opens the checkpointer a ``session_store`` kind names: an in-memory saver, or
an async sqlite/Postgres saver. The async savers come from async context
managers (``AsyncSqliteSaver``/``AsyncPostgresSaver.from_conn_string``) that
own their connection, so ``open_checkpointer`` returns the saver together with
that context manager. The caller closes it on shutdown
(``await cm.__aexit__(None, None, None)``); for ``memory`` and ``None`` it is
``None`` (nothing to close).

Each backend says where its sqlite file is. ``deepagents_sqlite_path`` puts
Deep Agents' file next to ADK's ``sessions.db`` and reads ``sqlite_uri`` the
way ADK does, so one ``sqlite_uri`` points both backends at one file: their
tables do not collide. LangGraph keeps its own location
(``workflow/langgraph/persistence/checkpointers.py``).

Example:
    saver, cm = await open_checkpointer("sqlite", sqlite_path=path)
    try:
        graph = builder.compile(checkpointer=saver)
        ...
    finally:
        if cm is not None:
            await cm.__aexit__(None, None, None)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from langgraph.checkpoint.base import BaseCheckpointSaver

logger = Loggers.workflow()

CheckpointerKind = Literal["memory", "sqlite", "postgres"]

# The URI forms ``sqlite_uri`` takes on ADK (``session_db_url`` turns the first
# into the second). What follows the prefix is the file, as SQLAlchemy reads it:
# ``sqlite:///rel.db`` is relative, ``sqlite:////abs/x.db`` absolute.
_SQLITE_URI_PREFIXES = ("sqlite:///", "sqlite+aiosqlite:///")


async def open_checkpointer(
    kind: CheckpointerKind | None,
    *,
    sqlite_path: str | Path | None = None,
    postgres_uri: str | None = None,
) -> tuple["BaseCheckpointSaver | None", "AbstractAsyncContextManager | None"]:
    """Open the checkpointer ``kind`` names.

    Args:
        kind: ``"memory"``, ``"sqlite"``, ``"postgres"``, or ``None`` for no
            checkpointer.
        sqlite_path: The sqlite database file (required for ``"sqlite"``).
        postgres_uri: The Postgres connection URI (required for ``"postgres"``).

    Returns:
        A ``(saver, context_manager)`` tuple. ``context_manager`` owns the
        saver's connection for sqlite and Postgres and must be closed on
        shutdown; it is ``None`` for ``memory`` and ``None``.

    Raises:
        ImportError: If the saver's package is not installed.
        ValueError: If ``kind`` is unknown or its location is missing.
    """
    if kind is None:
        logger.debug("checkpointer_disabled")
        return None, None

    if kind == "memory":
        return _memory_saver(), None

    if kind == "postgres":
        if not postgres_uri:
            raise ValueError(
                "PostgreSQL checkpointer requires 'postgres_uri' in settings. "
                "Set AGENTIC_POSTGRES_URI environment variable or add postgres_uri to settings."
            )
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except ImportError as e:
            raise ImportError(
                "PostgreSQL checkpointer requires langgraph-checkpoint-postgres. "
                "Install with: pip install langgraph-checkpoint-postgres"
            ) from e
        saver, cm = await _enter(AsyncPostgresSaver.from_conn_string(postgres_uri))
        logger.debug("postgres_checkpointer_created")
        return saver, cm

    if kind == "sqlite":
        if not sqlite_path:
            raise ValueError("The sqlite checkpointer requires sqlite_path.")
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ImportError as e:
            raise ImportError(
                "SQLite checkpointer requires langgraph-checkpoint-sqlite. "
                "Install with: pip install langgraph-checkpoint-sqlite"
            ) from e
        saver, cm = await _enter(AsyncSqliteSaver.from_conn_string(str(sqlite_path)))
        logger.debug("sqlite_checkpointer_created", path=str(sqlite_path))
        return saver, cm

    raise ValueError(
        f"Unknown checkpointer type: {kind}. "
        "Valid types: 'memory', 'postgres', 'sqlite', None"
    )


def _memory_saver() -> "BaseCheckpointSaver":
    """An in-memory saver (no connection to close)."""
    try:
        from langgraph.checkpoint.memory import InMemorySaver
    except ImportError as e:
        raise ImportError(
            "The memory checkpointer requires langgraph. Install with: pip install langgraph"
        ) from e
    logger.debug("memory_checkpointer_created")
    return InMemorySaver()


async def _enter(cm: Any) -> tuple["BaseCheckpointSaver", "AbstractAsyncContextManager"]:
    """Enter ``cm`` and create the saver's tables.

    If ``setup()`` fails, the connection just opened is closed before the
    error propagates: the caller never received ``cm``, so nothing else could
    close it.
    """
    saver = await cm.__aenter__()
    try:
        await saver.setup()
    except BaseException:
        try:
            await cm.__aexit__(None, None, None)
        except Exception:
            logger.debug("checkpointer_close_failed", exc_info=True)
        raise
    return saver, cm


def sqlite_file(value: str) -> str:
    """The database file a ``sqlite_uri`` value names.

    A ``sqlite:///…`` or ``sqlite+aiosqlite:///…`` URI becomes the path after
    the prefix, taken as written, as ADK's SQLAlchemy engine takes it. Any other
    value is a file path, returned as it is with ``~`` expanded.
    """
    for prefix in _SQLITE_URI_PREFIXES:
        if value.startswith(prefix):
            return value[len(prefix):]
    return os.path.expanduser(value)


def deepagents_sqlite_path(settings: Any) -> Path:
    """The Deep Agents backend's sqlite checkpoint file.

    ``sqlite_uri``'s file when it is set (the file ADK uses too), else
    ``{workspace}/sessions/checkpoints.db``, next to ADK's ``sessions.db``.
    The file's directory is created.
    """
    uri = getattr(settings, "sqlite_uri", None)
    path = Path(sqlite_file(uri)) if uri else settings.sessions_dir / "checkpoints.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
