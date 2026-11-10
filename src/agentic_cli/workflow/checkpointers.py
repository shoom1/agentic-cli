"""LangGraph checkpointers for the LangChain-based backends.

Opens the checkpointer a ``session_store`` kind names: an in-memory saver, or
an async sqlite/Postgres saver. The async savers come from async context
managers (``AsyncSqliteSaver``/``AsyncPostgresSaver.from_conn_string``) that
own their connection, so ``open_checkpointer`` returns the saver together with
that context manager. The caller closes it on shutdown
(``await cm.__aexit__(None, None, None)``); for ``memory`` and ``None`` it is
``None`` (nothing to close). ``open_store`` opens the LangGraph store of the
same kind beside it, the same way: Deep Agents keeps its session index there
(``workflow/deepagents/sessions.py``).

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

import asyncio
import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

from agentic_cli.file_utils import make_private_file, tighten_file
from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.store.base import BaseStore

logger = Loggers.workflow()

CheckpointerKind = Literal["memory", "sqlite", "postgres"]

# The URI forms ``sqlite_uri`` takes (ADK's ``session_db_url`` reads it through
# ``sqlite_file`` too). What follows the prefix is the file, as SQLAlchemy reads it:
# ``sqlite:///rel.db`` is relative, ``sqlite:////abs/x.db`` absolute.
_SQLITE_URI_PREFIXES = ("sqlite:///", "sqlite+aiosqlite:///")

# The files SQLite keeps beside a database: its rollback journal, and in WAL
# mode the write-ahead log and its shared-memory index.
_SQLITE_COMPANIONS = ("-journal", "-wal", "-shm")

# How many times a setup that lost a race is run again, and the pause before
# each run, in seconds: the other process's migrations take milliseconds, but
# a run that comes too soon can lose to them again.
_SETUP_RETRIES = 3
_SETUP_PAUSE = 0.1


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


async def open_store(
    kind: CheckpointerKind,
    *,
    sqlite_path: str | Path | None = None,
    postgres_uri: str | None = None,
) -> tuple["BaseStore", "AbstractAsyncContextManager | None"]:
    """Open the LangGraph store beside the checkpointer ``kind`` names.

    For sqlite it is the checkpointer's file, on a connection of its own: the
    saver's connection runs transactions it commits itself, the store's runs
    in autocommit. Its tables (``store``, ``store_migrations``) are neither
    the saver's nor ADK's, which may share the file. A setup that lost a
    race with another process migrating the same new file is run again
    (``_lost_migration_race``).

    Args:
        kind: ``"memory"``, ``"sqlite"`` or ``"postgres"``.
        sqlite_path: The sqlite database file (required for ``"sqlite"``).
        postgres_uri: The Postgres connection URI (required for ``"postgres"``).

    Returns:
        A ``(store, context_manager)`` tuple, as ``open_checkpointer``'s:
        ``context_manager`` owns the store's connection for sqlite and
        Postgres and must be closed on shutdown; it is ``None`` for
        ``memory``.

    Raises:
        ImportError: If the store's package is not installed.
        ValueError: If ``kind`` is unknown or its location is missing.
    """
    if kind == "memory":
        from langgraph.store.memory import InMemoryStore

        logger.debug("memory_store_created")
        return InMemoryStore(), None

    if kind == "postgres":
        if not postgres_uri:
            raise ValueError(
                "The PostgreSQL store requires 'postgres_uri' in settings. "
                "Set AGENTIC_POSTGRES_URI environment variable or add "
                "postgres_uri to settings."
            )
        try:
            from langgraph.store.postgres.aio import AsyncPostgresStore
        except ImportError as e:
            raise ImportError(
                "The PostgreSQL store requires langgraph-checkpoint-postgres. "
                "Install with: pip install langgraph-checkpoint-postgres"
            ) from e
        store, cm = await _enter(AsyncPostgresStore.from_conn_string(postgres_uri))
        logger.debug("postgres_store_created")
        return store, cm

    if kind == "sqlite":
        if not sqlite_path:
            raise ValueError("The sqlite store requires sqlite_path.")
        try:
            from langgraph.store.sqlite.aio import AsyncSqliteStore
        except ImportError as e:
            raise ImportError(
                "The SQLite store requires langgraph-checkpoint-sqlite. "
                "Install with: pip install langgraph-checkpoint-sqlite"
            ) from e
        store, cm = await _enter(
            AsyncSqliteStore.from_conn_string(str(sqlite_path)),
            again=_lost_migration_race,
        )
        logger.debug("sqlite_store_created", path=str(sqlite_path))
        return store, cm

    raise ValueError(
        f"Unknown store type: {kind}. Valid types: 'memory', 'postgres', 'sqlite'"
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


def _lost_migration_race(error: BaseException) -> bool:
    """Whether a sqlite store's setup failed as another process migrated the file.

    ``AsyncSqliteStore.setup`` reads the schema's version, then applies each
    later migration and records it, statement by statement in autocommit. Two
    processes opening a new file at once can both apply one: the second
    records a version already recorded (``UNIQUE constraint failed:
    store_migrations.v``) or adds a column already added (``duplicate column
    name``). Run again, its setup reads the version the other left.
    """
    return isinstance(error, sqlite3.IntegrityError) or (
        isinstance(error, sqlite3.OperationalError) and "duplicate column" in str(error)
    )


async def _enter(
    cm: Any, *, again: Callable[[BaseException], bool] | None = None
) -> tuple[Any, "AbstractAsyncContextManager"]:
    """Enter ``cm`` and create the tables of the saver or store it yields.

    If ``setup()`` fails, the connection just opened is closed before the
    error propagates: the caller never received ``cm``, so nothing else could
    close it.

    Args:
        cm: The context manager that opens the connection.
        again: Whether a setup that failed with this error is run again,
            after a pause (``_SETUP_RETRIES`` times at most).
    """
    saver = await cm.__aenter__()
    try:
        for retry in range(_SETUP_RETRIES + 1):
            try:
                await saver.setup()
                break
            except Exception as error:
                if again is None or retry == _SETUP_RETRIES or not again(error):
                    raise
                logger.debug("setup_run_again", error=str(error))
                await asyncio.sleep(_SETUP_PAUSE)
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


def private_sqlite_file(path: str | Path, *, tighten_existing: bool = True) -> None:
    """Make the sqlite database file ``path`` names owner-only before it is opened.

    The conversation store holds every word of every conversation, as the
    raw LLM log does, so it is 0600 in a directory created 0700
    (``file_utils.make_private_file``): its library would create it with the
    process's umask. ADK's ``sessions.db`` and Deep Agents' ``checkpoints.db``
    both come through here. ``:memory:`` names no file.

    Args:
        path: The database file.
        tighten_existing: Tighten a file that exists (one an older version
            made 0644): the backends' own location, ``sqlite_uri`` unset. A
            file at a location the user names (``sqlite_uri``) may be shared
            on purpose (0660 in a team folder), so there a missing file is
            created 0600 and an existing one is left as it is. The
            journal and WAL files SQLite keeps beside the database are
            tightened with it when they exist.
    """
    if str(path) in ("", ":memory:"):
        return
    make_private_file(Path(path), tighten=tighten_existing)
    if tighten_existing:
        # What a crash of an earlier build left beside it: SQLite writes the
        # new pages into them until a clean close.
        for suffix in _SQLITE_COMPANIONS:
            tighten_file(Path(f"{path}{suffix}"))


def deepagents_sqlite_path(settings: Any) -> Path:
    """The Deep Agents backend's sqlite checkpoint file.

    ``sqlite_uri``'s file when it is set (the file ADK uses too), else
    ``{workspace}/sessions/checkpoints.db``, next to ADK's ``sessions.db``.
    The file is made owner-only (``private_sqlite_file``; one ``sqlite_uri``
    names, and that exists, is left as it is), its directory created if
    missing.
    """
    path = deepagents_sqlite_location(settings)
    private_sqlite_file(path, tighten_existing=not getattr(settings, "sqlite_uri", None))
    return path


def deepagents_sqlite_location(settings: Any) -> Path:
    """Where ``deepagents_sqlite_path`` puts the file, without making it: a
    reader of the conversations makes no store that holds none."""
    uri = getattr(settings, "sqlite_uri", None)
    return Path(sqlite_file(uri)) if uri else settings.sessions_dir / "checkpoints.db"
