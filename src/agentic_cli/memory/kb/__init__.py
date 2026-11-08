"""The knowledge base: documents, hybrid search, sidecars and concept pages.

A component of ``agentic_cli.memory``. It imports nothing from the rest of
agentic-cli and never imports the memory store.
"""

from agentic_cli.memory.kb.concepts import ConceptStore
from agentic_cli.memory.kb.convert import ConvertedDocument, UnsupportedDocument, convert_document
from agentic_cli.memory.kb.knowledge_base import (
    BackfillResult,
    ConversionFailed,
    DocumentNotFound,
    EmptyDocument,
    KnowledgeBase,
    ScopedDocument,
)
from agentic_cli.memory.kb.manager import (
    BackfillAlreadyRunning,
    KnowledgeBaseBusy,
    KnowledgeBaseManager,
    Summarizer,
    UnsafeKnowledgeBase,
)
from agentic_cli.memory.kb.models import Document, DocumentChunk, SearchResult, SourceType

__all__ = [
    "BackfillAlreadyRunning",
    "BackfillResult",
    "ConceptStore",
    "ConvertedDocument",
    "ConversionFailed",
    "Document",
    "DocumentChunk",
    "DocumentNotFound",
    "EmptyDocument",
    "KnowledgeBase",
    "KnowledgeBaseBusy",
    "KnowledgeBaseManager",
    "ScopedDocument",
    "SearchResult",
    "SourceType",
    "Summarizer",
    "UnsafeKnowledgeBase",
    "UnsupportedDocument",
    "convert_document",
]
