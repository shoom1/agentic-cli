"""Data models for the knowledge base.

Defines the core data structures for documents, chunks, and search results.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class SourceType(Enum):
    """Type of document source."""

    ARXIV = "arxiv"
    SSRN = "ssrn"
    WEB = "web"
    INTERNAL = "internal"
    USER = "user"


@dataclass
class DocumentChunk:
    """A chunk of a document for embedding and retrieval.

    Documents are split into smaller chunks for more granular
    semantic search and to fit within embedding model limits.
    """

    id: str
    document_id: str
    content: str
    chunk_index: int
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        document_id: str,
        content: str,
        chunk_index: int,
        metadata: dict[str, Any] | None = None,
    ) -> DocumentChunk:
        """Create a new chunk with generated ID."""
        return cls(
            id=str(uuid.uuid4()),
            document_id=document_id,
            content=content,
            chunk_index=chunk_index,
            embedding=None,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
            # Embedding not serialized (stored separately in vector index)
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentChunk:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            document_id=data["document_id"],
            content=data["content"],
            chunk_index=data["chunk_index"],
            embedding=None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class Document:
    """A document in the knowledge base.

    Represents a complete document with metadata and chunked content.
    """

    id: str
    title: str
    content: str
    source_type: SourceType
    source_url: str | None = None
    file_path: Path | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    chunks: list[DocumentChunk] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        title: str,
        content: str,
        source_type: SourceType,
        source_url: str | None = None,
        file_path: Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Create a new document with generated ID."""
        now = datetime.now()
        return cls(
            id=str(uuid.uuid4()),
            title=title,
            content=content,
            source_type=source_type,
            source_url=source_url,
            file_path=file_path,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
            chunks=[],
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "source_type": self.source_type.value,
            "source_url": self.source_url,
            "file_path": str(self.file_path) if self.file_path else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            title=data["title"],
            content=data["content"],
            source_type=SourceType(data["source_type"]),
            source_url=data.get("source_url"),
            file_path=Path(data["file_path"]) if data.get("file_path") else None,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
            chunks=[DocumentChunk.from_dict(c) for c in data.get("chunks", [])],
        )


@dataclass
class SearchResult:
    """Result from a knowledge base search.

    Contains the matched document, specific chunk, relevance score,
    and a highlighted excerpt.
    """

    document: Document
    chunk: DocumentChunk
    score: float
    highlight: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "document_id": self.document.id,
            "document_title": self.document.title,
            "source_type": self.document.source_type.value,
            "source_url": self.document.source_url,
            "chunk_id": self.chunk.id,
            "chunk_content": self.chunk.content,
            "score": self.score,
            "highlight": self.highlight,
            "metadata": {
                **self.document.metadata,
                **self.chunk.metadata,
            },
        }


@dataclass
class PaperResult:
    """Academic paper search result from ArXiv or SSRN."""

    title: str
    authors: list[str]
    abstract: str
    url: str
    published_date: str
    source: str  # "arxiv" or "ssrn"
    categories: list[str] = field(default_factory=list)
    pdf_url: str | None = None
    arxiv_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "url": self.url,
            "published_date": self.published_date,
            "source": self.source,
            "categories": self.categories,
            "pdf_url": self.pdf_url,
            "arxiv_id": self.arxiv_id,
        }


@dataclass
class WebResult:
    """Web search result."""

    title: str
    url: str
    snippet: str
    domain: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "domain": self.domain,
        }
