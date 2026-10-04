"""The knowledge base: documents, hybrid search, sidecars and concept pages.

A component of ``agentic_cli.memory``. It imports nothing from the rest of
agentic-cli and never imports the memory store.
"""

from agentic_cli.memory.kb._mock_vector_store import MockVectorStore
from agentic_cli.memory.kb.bm25_index import create_bm25_index
from agentic_cli.memory.kb.concepts import ConceptStore
from agentic_cli.memory.kb.manager import (
    BackfillAlreadyRunning,
    KnowledgeBaseManager,
    Summarizer,
    matches_document_filters,
)
from agentic_cli.memory.kb.models import Document, DocumentChunk, SearchResult, SourceType
from agentic_cli.memory.kb.vector_store import VectorStore

__all__ = [
    "BackfillAlreadyRunning",
    "ConceptStore",
    "Document",
    "DocumentChunk",
    "KnowledgeBaseManager",
    "MockVectorStore",
    "SearchResult",
    "SourceType",
    "Summarizer",
    "VectorStore",
    "create_bm25_index",
    "matches_document_filters",
]
