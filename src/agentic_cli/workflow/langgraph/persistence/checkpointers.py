"""Checkpointer factory for LangGraph state persistence.

Provides factory function to create appropriate checkpointer based on
configuration. Supports memory, PostgreSQL, and SQLite backends.

Example:
    from agentic_cli.workflow.langgraph.persistence import create_checkpointer

    # Memory checkpointer (default, no persistence)
    checkpointer = create_checkpointer("memory", settings)

    # PostgreSQL checkpointer (persistent)
    checkpointer = create_checkpointer("postgres", settings)

    # SQLite checkpointer (file-based persistence)
    checkpointer = create_checkpointer("sqlite", settings)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Any

from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings

logger = Loggers.workflow()

# Type alias for checkpointer types
CheckpointerType = Literal["memory", "postgres", "sqlite"] | None


def create_checkpointer(
    checkpointer_type: CheckpointerType,
    settings: "BaseSettings",
) -> Any | None:
    """Create a checkpointer based on the specified type.

    Args:
        checkpointer_type: Type of checkpointer to create.
            - "memory": In-memory checkpointer (no persistence, fast)
            - "postgres": PostgreSQL-based checkpointer (requires langgraph-checkpoint-postgres)
            - "sqlite": SQLite-based checkpointer (requires langgraph-checkpoint-sqlite)
            - None: No checkpointing
        settings: Application settings containing connection URIs.

    Returns:
        Checkpointer instance, or None if checkpointing is disabled.

    Raises:
        ImportError: If required dependencies are not installed.
        ValueError: If required settings are missing for the checkpointer type.
    """
    if checkpointer_type is None:
        logger.debug("checkpointer_disabled")
        return None

    if checkpointer_type == "memory":
        return _create_memory_checkpointer()

    elif checkpointer_type == "postgres":
        postgres_uri = getattr(settings, "postgres_uri", None)
        if not postgres_uri:
            raise ValueError(
                "PostgreSQL checkpointer requires 'postgres_uri' in settings. "
                "Set AGENTIC_POSTGRES_URI environment variable or add postgres_uri to settings."
            )
        return _create_postgres_checkpointer(postgres_uri)

    elif checkpointer_type == "sqlite":
        sqlite_uri = getattr(settings, "sqlite_uri", None)
        if not sqlite_uri:
            # Use default path in workspace
            sqlite_uri = str(settings.workspace_dir / "checkpoints.db")
            logger.debug("sqlite_uri_defaulted", path=sqlite_uri)
        return _create_sqlite_checkpointer(sqlite_uri)

    else:
        raise ValueError(
            f"Unknown checkpointer type: {checkpointer_type}. "
            "Valid types: 'memory', 'postgres', 'sqlite', None"
        )


def _create_memory_checkpointer():
    """Create an in-memory checkpointer.

    Returns:
        MemorySaver instance.

    Raises:
        ImportError: If LangGraph is not installed.
    """
    try:
        from langgraph.checkpoint.memory import MemorySaver

        logger.debug("memory_checkpointer_created")
        return MemorySaver()
    except ImportError as e:
        raise ImportError(
            "LangGraph dependencies not installed. "
            "Install with: pip install agentic-cli[langgraph]"
        ) from e


def _create_postgres_checkpointer(connection_string: str):
    """Create a PostgreSQL checkpointer.

    Args:
        connection_string: PostgreSQL connection URI.

    Returns:
        PostgresSaver instance.

    Raises:
        ImportError: If langgraph-checkpoint-postgres is not installed.
    """
    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        logger.debug("postgres_checkpointer_creating", uri_masked="***")
        return PostgresSaver.from_conn_string(connection_string)
    except ImportError as e:
        raise ImportError(
            "PostgreSQL checkpointer requires langgraph-checkpoint-postgres. "
            "Install with: pip install langgraph-checkpoint-postgres"
        ) from e


def _create_sqlite_checkpointer(connection_string: str):
    """Create a SQLite checkpointer.

    Args:
        connection_string: SQLite connection URI or file path.

    Returns:
        SqliteSaver instance.

    Raises:
        ImportError: If langgraph-checkpoint-sqlite is not installed.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        logger.debug("sqlite_checkpointer_creating", path=connection_string)
        return SqliteSaver.from_conn_string(connection_string)
    except ImportError as e:
        raise ImportError(
            "SQLite checkpointer requires langgraph-checkpoint-sqlite. "
            "Install with: pip install langgraph-checkpoint-sqlite"
        ) from e
