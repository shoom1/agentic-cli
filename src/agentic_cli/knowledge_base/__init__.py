"""Knowledge Base package for agentic CLI applications.

Provides semantic search over documents including academic papers,
internal documentation, and research notes.
"""

from agentic_cli.knowledge_base.embeddings import EmbeddingService, MockEmbeddingService
from agentic_cli.knowledge_base.manager import KnowledgeBaseManager
from agentic_cli.knowledge_base.models import (
    Document,
    DocumentChunk,
    PaperResult,
    SearchResult,
    SourceType,
    WebResult,
)
from agentic_cli.knowledge_base.vector_store import MockVectorStore, VectorStore
from agentic_cli.knowledge_base.sources import (
    SearchSource,
    SearchSourceResult,
    SearchSourceRegistry,
    ArxivSearchSource,
    WebSearchSource,
    get_search_registry,
    register_search_source,
)

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
    "MockEmbeddingService",
    "VectorStore",
    "MockVectorStore",
    # Search Sources
    "SearchSource",
    "SearchSourceResult",
    "SearchSourceRegistry",
    "ArxivSearchSource",
    "WebSearchSource",
    "get_search_registry",
    "register_search_source",
]
