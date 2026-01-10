"""Knowledge Base Manager.

Main interface for knowledge base operations including document ingestion,
search, and management.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentic_cli.knowledge_base.embeddings import EmbeddingService, MockEmbeddingService
from agentic_cli.knowledge_base.models import (
    Document,
    DocumentChunk,
    SearchResult,
    SourceType,
)
from agentic_cli.knowledge_base.vector_store import MockVectorStore, VectorStore

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings


class KnowledgeBaseManager:
    """Main interface for knowledge base operations.

    Provides document ingestion, semantic search, and document management.
    """

    def __init__(
        self,
        settings: "BaseSettings | None" = None,
        use_mock: bool = False,
    ) -> None:
        """Initialize the knowledge base manager.

        Args:
            settings: Application settings. Required for paths configuration.
            use_mock: If True, use mock services (for testing without ML models).
        """
        self._settings = settings
        self._use_mock = use_mock

        # Get paths from settings or use defaults
        if settings:
            self.kb_dir = settings.knowledge_base_dir
            self.documents_dir = settings.knowledge_base_documents_dir
            self.embeddings_dir = settings.knowledge_base_embeddings_dir
            embedding_model = settings.embedding_model
            batch_size = settings.embedding_batch_size
        else:
            # Fallback defaults for standalone use
            self.kb_dir = Path.home() / ".agentic" / "knowledge_base"
            self.documents_dir = self.kb_dir / "documents"
            self.embeddings_dir = self.kb_dir / "embeddings"
            embedding_model = "all-MiniLM-L6-v2"
            batch_size = 32

        self.metadata_path = self.kb_dir / "metadata.json"

        # Ensure directories exist
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Initialize services
        # Fall back to mock mode if real embedding service can't be loaded
        # (e.g., sentence_transformers not installed)
        if use_mock:
            self._use_mock = True
            self._embedding_service: EmbeddingService | MockEmbeddingService = (
                MockEmbeddingService(
                    model_name=embedding_model,
                    batch_size=batch_size,
                )
            )
            self._vector_store: VectorStore | MockVectorStore = MockVectorStore(
                index_path=self.embeddings_dir / "index.mock",
                embedding_dim=self._embedding_service.embedding_dim,
            )
        else:
            try:
                self._embedding_service = EmbeddingService(
                    model_name=embedding_model,
                    batch_size=batch_size,
                )
                self._vector_store = VectorStore(
                    index_path=self.embeddings_dir / "index.faiss",
                    embedding_dim=self._embedding_service.embedding_dim,
                )
            except ImportError:
                # Fall back to mock services if dependencies not available
                self._use_mock = True
                self._embedding_service = MockEmbeddingService(
                    model_name=embedding_model,
                    batch_size=batch_size,
                )
                self._vector_store = MockVectorStore(
                    index_path=self.embeddings_dir / "index.mock",
                    embedding_dim=self._embedding_service.embedding_dim,
                )

        # Load document metadata
        self._documents: dict[str, Document] = {}
        self._chunks: dict[str, DocumentChunk] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load document metadata from disk."""
        if not self.metadata_path.exists():
            return

        try:
            data = json.loads(self.metadata_path.read_text())
            for doc_data in data.get("documents", []):
                doc = Document.from_dict(doc_data)
                self._documents[doc.id] = doc
                for chunk in doc.chunks:
                    self._chunks[chunk.id] = chunk
        except (json.JSONDecodeError, KeyError) as e:
            # Log error but continue with empty state
            print(f"Warning: Failed to load metadata: {e}")

    def _save_metadata(self) -> None:
        """Save document metadata to disk."""
        data = {
            "documents": [doc.to_dict() for doc in self._documents.values()],
            "updated_at": datetime.now().isoformat(),
        }
        self.metadata_path.write_text(json.dumps(data, indent=2))

    def ingest_document(
        self,
        content: str,
        title: str,
        source_type: SourceType,
        source_url: str | None = None,
        metadata: dict[str, Any] | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> Document:
        """Ingest a new document into the knowledge base.

        Args:
            content: Document content.
            title: Document title.
            source_type: Type of source (ARXIV, SSRN, WEB, etc.).
            source_url: Optional URL of the source.
            metadata: Optional additional metadata.
            chunk_size: Size of chunks for embedding.
            chunk_overlap: Overlap between chunks.

        Returns:
            The created Document object.
        """
        # Create document
        doc = Document.create(
            title=title,
            content=content,
            source_type=source_type,
            source_url=source_url,
            metadata=metadata,
        )

        # Chunk the document
        chunk_texts = self._embedding_service.chunk_document(
            content, chunk_size, chunk_overlap
        )

        # Create chunks
        for i, chunk_text in enumerate(chunk_texts):
            chunk = DocumentChunk.create(
                document_id=doc.id,
                content=chunk_text,
                chunk_index=i,
                metadata={"title": title},
            )
            doc.chunks.append(chunk)
            self._chunks[chunk.id] = chunk

        # Generate embeddings
        if doc.chunks:
            chunk_texts = [c.content for c in doc.chunks]
            embeddings = self._embedding_service.embed_batch(chunk_texts)

            # Update chunks with embeddings
            for chunk, embedding in zip(doc.chunks, embeddings):
                chunk.embedding = embedding

            # Add to vector store
            chunk_ids = [c.id for c in doc.chunks]
            self._vector_store.add_embeddings(chunk_ids, embeddings)

        # Store document
        self._documents[doc.id] = doc

        # Persist
        self._save_metadata()
        self._vector_store.save()

        return doc

    def search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Search the knowledge base using semantic similarity.

        Args:
            query: Natural language search query.
            filters: Optional filters (source_type, date_from, date_to).
            top_k: Maximum number of results.

        Returns:
            Dict with results, timing information, and metadata.
        """
        start_time = time.time()

        # Generate query embedding
        embed_start = time.time()
        query_embedding = self._embedding_service.embed_text(query)
        embed_time = (time.time() - embed_start) * 1000

        # Search vector store
        search_start = time.time()
        raw_results = self._vector_store.search(query_embedding, top_k * 2)
        search_time = (time.time() - search_start) * 1000

        # Build results with filtering
        results: list[SearchResult] = []
        for chunk_id, score in raw_results:
            if len(results) >= top_k:
                break

            chunk = self._chunks.get(chunk_id)
            if not chunk:
                continue

            doc = self._documents.get(chunk.document_id)
            if not doc:
                continue

            # Apply filters
            if filters:
                if not self._matches_filters(doc, filters):
                    continue

            # Create highlight (first 200 chars)
            highlight = (
                chunk.content[:200] + "..."
                if len(chunk.content) > 200
                else chunk.content
            )

            results.append(
                SearchResult(
                    document=doc,
                    chunk=chunk,
                    score=score,
                    highlight=highlight,
                )
            )

        total_time = (time.time() - start_time) * 1000

        return {
            "results": [r.to_dict() for r in results],
            "total_matches": len(results),
            "query_embedding_time_ms": round(embed_time, 2),
            "search_time_ms": round(search_time, 2),
            "total_time_ms": round(total_time, 2),
        }

    def _matches_filters(self, doc: Document, filters: dict[str, Any]) -> bool:
        """Check if document matches the given filters."""
        # Source type filter
        if "source_type" in filters:
            expected = filters["source_type"]
            if isinstance(expected, str):
                expected = SourceType(expected)
            if doc.source_type != expected:
                return False

        # Date range filter
        if "date_from" in filters:
            date_from = datetime.fromisoformat(filters["date_from"])
            if doc.created_at < date_from:
                return False

        if "date_to" in filters:
            date_to = datetime.fromisoformat(filters["date_to"])
            if doc.created_at > date_to:
                return False

        # Keywords filter
        if "keywords" in filters:
            keywords = filters["keywords"]
            content_lower = doc.content.lower()
            for keyword in keywords:
                if keyword.lower() not in content_lower:
                    return False

        return True

    def get_document(self, doc_id: str) -> Document | None:
        """Retrieve a specific document by ID.

        Args:
            doc_id: Document ID.

        Returns:
            Document if found, None otherwise.
        """
        return self._documents.get(doc_id)

    def list_documents(
        self,
        source_type: SourceType | None = None,
        limit: int = 100,
    ) -> list[Document]:
        """List documents with optional filtering.

        Args:
            source_type: Optional source type filter.
            limit: Maximum number of documents to return.

        Returns:
            List of documents.
        """
        docs = list(self._documents.values())

        if source_type:
            docs = [d for d in docs if d.source_type == source_type]

        # Sort by updated_at descending
        docs.sort(key=lambda d: d.updated_at, reverse=True)

        return docs[:limit]

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document from the knowledge base.

        Args:
            doc_id: Document ID.

        Returns:
            True if deleted, False if not found.
        """
        doc = self._documents.get(doc_id)
        if not doc:
            return False

        # Remove chunks from vector store
        chunk_ids = [c.id for c in doc.chunks]
        self._vector_store.remove_embeddings(chunk_ids)

        # Remove from memory
        for chunk_id in chunk_ids:
            self._chunks.pop(chunk_id, None)
        del self._documents[doc_id]

        # Persist
        self._save_metadata()
        self._vector_store.save()

        return True

    def get_stats(self) -> dict[str, Any]:
        """Get knowledge base statistics.

        Returns:
            Dict with document count, chunk count, and other stats.
        """
        source_counts: dict[str, int] = {}
        for doc in self._documents.values():
            source = doc.source_type.value
            source_counts[source] = source_counts.get(source, 0) + 1

        embedding_model = "unknown"
        if self._settings:
            embedding_model = self._settings.embedding_model

        return {
            "document_count": len(self._documents),
            "chunk_count": len(self._chunks),
            "vector_count": self._vector_store.size,
            "source_counts": source_counts,
            "embedding_model": embedding_model,
            "embedding_dim": self._embedding_service.embedding_dim,
        }

    def clear(self) -> None:
        """Clear all documents from the knowledge base."""
        self._documents = {}
        self._chunks = {}
        self._vector_store.clear()
        self._save_metadata()
        self._vector_store.save()
