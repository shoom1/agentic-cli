"""Knowledge Base package for agentic CLI applications.

Provides semantic search over documents including academic papers,
internal documentation, and research notes.
"""

from agentic_cli.knowledge_base.embeddings import EmbeddingService
from agentic_cli.knowledge_base.manager import KnowledgeBaseManager
from agentic_cli.knowledge_base.models import (
    Document,
    DocumentChunk,
    PaperResult,
    SearchResult,
    SourceType,
    WebResult,
)
from agentic_cli.knowledge_base.vector_store import VectorStore
from agentic_cli.knowledge_base.sources import (
    SearchSource,
    SearchSourceResult,
)
from agentic_cli.knowledge_base.bm25_index import create_bm25_index

__all__ = [
    # Manager
    "KnowledgeBaseManager",
    # Models
    "Document",
    "DocumentChunk",
    "PaperResult",
    "SearchResult",
    "SourceType",
    "WebResult",
    # Services
    "EmbeddingService",
    "VectorStore",
    # Search Sources
    "SearchSource",
    "SearchSourceResult",
    # BM25
    "create_bm25_index",
]
