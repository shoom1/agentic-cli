"""Checkpointer factory for LangGraph state persistence.

Supports memory, SQLite, and PostgreSQL backends.

The SQLite/PostgreSQL savers are async and are obtained from async context
managers (``AsyncSqliteSaver``/``AsyncPostgresSaver.from_conn_string``), so
``create_checkpointer`` is async and returns both the saver and the context
manager that owns its connection. The caller must close that context manager on
shutdown (``await cm.__aexit__(None, None, None)``). For ``memory``/``None`` the
context manager is ``None`` (nothing to close).

The previous implementation returned ``Saver.from_conn_string(...)`` directly —
in current ``langgraph-checkpoint-*`` that is a context manager, not a saver, so
``graph.compile(checkpointer=...)`` received the wrong object and broke. It also
used the sync savers, which cannot serve the async ``astream_events`` path.

Example:
    checkpointer, cm = await create_checkpointer("sqlite", settings)
    try:
        graph = builder.compile(checkpointer=checkpointer)
        ...
    finally:
        if cm is not None:
            await cm.__aexit__(None, None, None)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Any

from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings

logger = Loggers.workflow()

# Type alias for checkpointer types
CheckpointerType = Literal["memory", "postgres", "sqlite"] | None


async def create_checkpointer(
    checkpointer_type: CheckpointerType,
    settings: "BaseSettings",
) -> tuple[Any | None, Any | None]:
    """Create a checkpointer based on the specified type.

    Args:
        checkpointer_type: "memory", "postgres", "sqlite", or None.
        settings: Application settings containing connection URIs.

    Returns:
        A ``(checkpointer, context_manager)`` tuple. ``context_manager`` owns
        the saver's connection for the sqlite/postgres backends and must be
        closed on shutdown; it is ``None`` for ``memory``/``None``.

    Raises:
        ImportError: If required dependencies are not installed.
        ValueError: If required settings are missing for the checkpointer type.
    """
    if checkpointer_type is None:
        logger.debug("checkpointer_disabled")
        return None, None

    if checkpointer_type == "memory":
        return _create_memory_checkpointer(), None

    elif checkpointer_type == "postgres":
        postgres_uri = getattr(settings, "postgres_uri", None)
        if not postgres_uri:
            raise ValueError(
                "PostgreSQL checkpointer requires 'postgres_uri' in settings. "
                "Set AGENTIC_POSTGRES_URI environment variable or add postgres_uri to settings."
            )
        return await _create_postgres_checkpointer(postgres_uri)

    elif checkpointer_type == "sqlite":
        sqlite_uri = getattr(settings, "sqlite_uri", None)
        if not sqlite_uri:
            # Use default path in workspace
            sqlite_uri = str(settings.workspace_dir / "checkpoints.db")
            logger.debug("sqlite_uri_defaulted", path=sqlite_uri)
        return await _create_sqlite_checkpointer(sqlite_uri)

    else:
        raise ValueError(
            f"Unknown checkpointer type: {checkpointer_type}. "
            "Valid types: 'memory', 'postgres', 'sqlite', None"
        )


def _create_memory_checkpointer():
    """Create an in-memory checkpointer (no external connection)."""
    try:
        from langgraph.checkpoint.memory import MemorySaver

        logger.debug("memory_checkpointer_created")
        return MemorySaver()
    except ImportError as e:
        raise ImportError(
            "LangGraph dependencies not installed. "
            "Install with: pip install agentic-cli[langgraph]"
        ) from e


async def _create_sqlite_checkpointer(connection_string: str):
    """Create an async SQLite checkpointer and enter its connection context.

    Returns ``(saver, context_manager)``; close the context manager to release
    the connection.
    """
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ImportError as e:
        raise ImportError(
            "SQLite checkpointer requires langgraph-checkpoint-sqlite. "
            "Install with: pip install langgraph-checkpoint-sqlite"
        ) from e

    cm = AsyncSqliteSaver.from_conn_string(connection_string)
    saver = await cm.__aenter__()
    await saver.setup()
    logger.debug("sqlite_checkpointer_created", path=connection_string)
    return saver, cm


async def _create_postgres_checkpointer(connection_string: str):
    """Create an async PostgreSQL checkpointer and enter its connection context.

    Returns ``(saver, context_manager)``; close the context manager to release
    the connection.
    """
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError as e:
        raise ImportError(
            "PostgreSQL checkpointer requires langgraph-checkpoint-postgres. "
            "Install with: pip install langgraph-checkpoint-postgres"
        ) from e

    cm = AsyncPostgresSaver.from_conn_string(connection_string)
    saver = await cm.__aenter__()
    await saver.setup()
    logger.debug("postgres_checkpointer_created")
    return saver, cm
