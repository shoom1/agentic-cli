"""Store factory for LangGraph long-term memory.

Provides factory function to create appropriate store based on
configuration. Stores provide semantic search and long-term memory
capabilities for LangGraph workflows.

Example:
    from agentic_cli.workflow.langgraph.persistence import create_store

    # Memory store (default, no persistence)
    store = create_store("memory", settings)

    # Memory store with embeddings for semantic search
    store = create_store("memory", settings, embeddings=my_embeddings)

    # PostgreSQL store (persistent with optional semantic search)
    store = create_store("postgres", settings, embeddings=my_embeddings)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Any

from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings

logger = Loggers.workflow()

# Type alias for store types
StoreType = Literal["memory", "postgres"] | None


def create_store(
    store_type: StoreType,
    settings: "BaseSettings",
    embeddings: Any | None = None,
) -> Any | None:
    """Create a store based on the specified type.

    Stores provide long-term memory and semantic search capabilities
    for LangGraph workflows. They complement checkpointers by storing
    persistent data across sessions.

    Args:
        store_type: Type of store to create.
            - "memory": In-memory store (no persistence, fast)
            - "postgres": PostgreSQL-based store (persistent)
            - None: No store
        settings: Application settings containing connection URIs.
        embeddings: Optional embeddings instance for semantic search.
            Should have `dims` attribute for embedding dimensions.

    Returns:
        Store instance, or None if store is disabled.

    Raises:
        ImportError: If required dependencies are not installed.
        ValueError: If required settings are missing for the store type.
    """
    if store_type is None:
        logger.debug("store_disabled")
        return None

    if store_type == "memory":
        return _create_memory_store(embeddings)

    elif store_type == "postgres":
        postgres_uri = getattr(settings, "postgres_uri", None)
        if not postgres_uri:
            raise ValueError(
                "PostgreSQL store requires 'postgres_uri' in settings. "
                "Set AGENTIC_POSTGRES_URI environment variable or add postgres_uri to settings."
            )
        return _create_postgres_store(postgres_uri, embeddings)

    else:
        raise ValueError(
            f"Unknown store type: {store_type}. "
            "Valid types: 'memory', 'postgres', None"
        )


def _create_memory_store(embeddings: Any | None = None):
    """Create an in-memory store.

    Args:
        embeddings: Optional embeddings for semantic search.

    Returns:
        InMemoryStore instance.

    Raises:
        ImportError: If LangGraph is not installed.
    """
    try:
        from langgraph.store.memory import InMemoryStore

        if embeddings:
            # Create store with semantic search index
            dims = getattr(embeddings, "dims", 384)  # Default for common models
            logger.debug("memory_store_with_embeddings", dims=dims)
            return InMemoryStore(
                index={
                    "embed": embeddings,
                    "dims": dims,
                }
            )

        logger.debug("memory_store_created")
        return InMemoryStore()

    except ImportError as e:
        raise ImportError(
            "LangGraph dependencies not installed. "
            "Install with: pip install agentic-cli[langgraph]"
        ) from e


def _create_postgres_store(
    connection_string: str,
    embeddings: Any | None = None,
):
    """Create a PostgreSQL store.

    Args:
        connection_string: PostgreSQL connection URI.
        embeddings: Optional embeddings for semantic search.

    Returns:
        PostgresStore instance.

    Raises:
        ImportError: If langgraph-checkpoint-postgres is not installed.
    """
    try:
        from langgraph.store.postgres import PostgresStore

        if embeddings:
            # Create store with semantic search index
            dims = getattr(embeddings, "dims", 384)
            logger.debug("postgres_store_with_embeddings", dims=dims)
            return PostgresStore.from_conn_string(
                connection_string,
                index={
                    "embed": embeddings,
                    "dims": dims,
                },
            )

        logger.debug("postgres_store_creating", uri_masked="***")
        return PostgresStore.from_conn_string(connection_string)

    except ImportError as e:
        raise ImportError(
            "PostgreSQL store requires langgraph-checkpoint-postgres. "
            "Install with: pip install langgraph-checkpoint-postgres"
        ) from e
