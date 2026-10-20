"""Checkpointer factory for LangGraph state persistence.

Supports memory, SQLite, and PostgreSQL backends, opened by the shared
``agentic_cli.workflow.checkpointers.open_checkpointer``. This module keeps
only LangGraph's own sqlite location: ``sqlite_uri`` used as given (a file
name), else ``{workspace}/checkpoints.db``.

``create_checkpointer`` is async and returns both the saver and the context
manager that owns its connection. The caller must close that context manager
on shutdown (``await cm.__aexit__(None, None, None)``). For ``memory``/``None``
the context manager is ``None`` (nothing to close).

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

from typing import TYPE_CHECKING, Any

from agentic_cli.logging import Loggers
from agentic_cli.workflow.checkpointers import CheckpointerKind, open_checkpointer

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings

logger = Loggers.workflow()

# Type alias for checkpointer types
CheckpointerType = CheckpointerKind | None


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
    sqlite_path = None
    if checkpointer_type == "sqlite":
        sqlite_path = getattr(settings, "sqlite_uri", None)
        if not sqlite_path:
            # Use default path in workspace
            sqlite_path = str(settings.workspace_dir / "checkpoints.db")
            logger.debug("sqlite_uri_defaulted", path=sqlite_path)

    return await open_checkpointer(
        checkpointer_type,
        sqlite_path=sqlite_path,
        postgres_uri=getattr(settings, "postgres_uri", None),
    )
