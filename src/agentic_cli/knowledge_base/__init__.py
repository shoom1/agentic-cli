"""Deprecated: the knowledge base moved to ``agentic_cli.memory.kb``.

These modules keep the old import paths working until 0.7.0.
"""

import warnings

warnings.warn(
    "agentic_cli.knowledge_base is deprecated and will be removed in 0.7.0; import "
    "from agentic_cli.memory.kb (EmbeddingService from agentic_cli.memory, "
    "SearchSource from agentic_cli.tools.search_sources)",
    DeprecationWarning,
    stacklevel=2,
)

from agentic_cli.knowledge_base.bm25_index import create_bm25_index  # noqa: E402
from agentic_cli.knowledge_base.embeddings import EmbeddingService  # noqa: E402
from agentic_cli.knowledge_base.manager import KnowledgeBaseManager  # noqa: E402
from agentic_cli.knowledge_base.models import (  # noqa: E402
    Document,
    DocumentChunk,
    SearchResult,
    SourceType,
)
from agentic_cli.knowledge_base.sources import (  # noqa: E402
    PaperResult,
    SearchSource,
    SearchSourceResult,
    WebResult,
)
from agentic_cli.knowledge_base.vector_store import VectorStore  # noqa: E402

__all__ = [
    "KnowledgeBaseManager",
    "Document",
    "DocumentChunk",
    "PaperResult",
    "SearchResult",
    "SourceType",
    "WebResult",
    "EmbeddingService",
    "VectorStore",
    "SearchSource",
    "SearchSourceResult",
    "create_bm25_index",
]
