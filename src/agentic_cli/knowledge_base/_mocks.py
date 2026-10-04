"""Deprecated: ``MockEmbeddingService`` moved to ``agentic_cli.memory``;
``MockVectorStore`` moved to ``agentic_cli.memory.kb``. Removed in 0.7.0."""

from agentic_cli.memory._core.mock_embeddings import MockEmbeddingService  # noqa: F401
from agentic_cli.memory.kb._mock_vector_store import MockVectorStore  # noqa: F401
