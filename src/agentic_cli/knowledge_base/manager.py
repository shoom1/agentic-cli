"""Knowledge Base Manager.

Main interface for knowledge base operations including document ingestion,
search, and management.

Persistence format (v2):
    metadata.json    - Document headers and chunk metadata (no content)
    documents/{id}.json - Per-document content and chunk texts
    embeddings/      - FAISS index and mappings

Legacy format (v1): All content stored in metadata.json.
Auto-migrated to v2 on first load.
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
from agentic_cli.constants import truncate
from agentic_cli.logging import Loggers
from agentic_cli.persistence._utils import atomic_write_json

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings

logger = Loggers.knowledge_base()

# Current persistence format version
_FORMAT_VERSION = 3


def matches_document_filters(doc: Document, filters: dict[str, Any]) -> bool:
    """Check if a document matches the given filters.

    Standalone function for reuse outside the manager.

    Args:
        doc: Document to check.
        filters: Filter dict with optional keys:
            - source_type: SourceType or string
            - date_from: ISO date string (inclusive lower bound)
            - date_to: ISO date string (inclusive upper bound)
            - keywords: list of strings (all must be present, case-insensitive)

    Returns:
        True if document matches all filters.
    """
    if "source_type" in filters:
        expected = filters["source_type"]
        if isinstance(expected, str):
            expected = SourceType(expected)
        if doc.source_type != expected:
            return False

    if "date_from" in filters:
        date_from = datetime.fromisoformat(filters["date_from"])
        if doc.created_at < date_from:
            return False

    if "date_to" in filters:
        date_to = datetime.fromisoformat(filters["date_to"])
        if doc.created_at > date_to:
            return False

    if "keywords" in filters:
        content_lower = doc.content.lower()
        for keyword in filters["keywords"]:
            if keyword.lower() not in content_lower:
                return False

    return True


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
        self.files_dir = self.kb_dir / "files"

        # Ensure directories exist
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.files_dir.mkdir(parents=True, exist_ok=True)

        # Initialize services (fall back to mock if real deps unavailable)
        self._embedding_service, self._vector_store = self._create_services(
            embedding_model, batch_size
        )

        # Load document metadata
        self._documents: dict[str, Document] = {}
        self._chunks: dict[str, DocumentChunk] = {}
        self._load_metadata()


    def _create_services(
        self, embedding_model: str, batch_size: int
    ) -> tuple["EmbeddingService | MockEmbeddingService", "VectorStore | MockVectorStore"]:
        """Create embedding service and vector store.

        Tries real implementations first; falls back to mocks if
        dependencies (sentence_transformers, faiss) are unavailable.
        """
        if not self._use_mock:
            try:
                emb = EmbeddingService(
                    model_name=embedding_model,
                    batch_size=batch_size,
                )
                vs = VectorStore(
                    index_path=self.embeddings_dir / "index.faiss",
                    embedding_dim=emb.embedding_dim,
                )
                return emb, vs
            except ImportError:
                self._use_mock = True

        emb = MockEmbeddingService(
            model_name=embedding_model,
            batch_size=batch_size,
        )
        vs = MockVectorStore(
            index_path=self.embeddings_dir / "index.mock",
            embedding_dim=emb.embedding_dim,
        )
        return emb, vs

    # ------------------------------------------------------------------
    # Persistence — v2/v3 format (content in per-document files)
    # ------------------------------------------------------------------

    def _document_content_path(self, doc_id: str) -> Path:
        """Get the path for a document's content file."""
        return self.documents_dir / f"{doc_id}.json"

    def _save_document_content(self, doc: Document) -> None:
        """Save document content to its individual file."""
        chunk_contents = {c.id: c.content for c in doc.chunks}
        data = {
            "content": doc.content,
            "chunks": chunk_contents,
        }
        atomic_write_json(self._document_content_path(doc.id), data)

    def _load_document_content(self, doc_id: str) -> dict[str, Any] | None:
        """Load document content from its individual file.

        Returns:
            Dict with 'content' and 'chunks' keys, or None if not found.
        """
        path = self._document_content_path(doc_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "failed_to_load_document_content",
                doc_id=doc_id,
                error=str(e),
            )
            return None

    def _delete_document_content(self, doc_id: str) -> None:
        """Delete a document's content file."""
        path = self._document_content_path(doc_id)
        if path.exists():
            path.unlink()

    def _load_metadata(self) -> None:
        """Load document metadata from disk.

        Handles v1 (content in metadata.json), v2 (content in per-document
        files, no summary), and v3 (content in per-document files, with
        summary) formats. Auto-migrates older formats to v3.
        """
        if not self.metadata_path.exists():
            return

        try:
            data = json.loads(self.metadata_path.read_text())
            version = data.get("version", 1)

            for doc_data in data.get("documents", []):
                if version >= 2:
                    # v2/v3: Load header from index, content from per-doc file
                    doc = Document.from_dict(doc_data)
                    content_data = self._load_document_content(doc.id)
                    if content_data:
                        doc.content = content_data["content"]
                        chunk_contents = content_data.get("chunks", {})
                        for chunk in doc.chunks:
                            if chunk.id in chunk_contents:
                                chunk.content = chunk_contents[chunk.id]
                else:
                    # v1: Everything is in metadata.json
                    doc = Document.from_dict(doc_data)

                self._documents[doc.id] = doc
                for chunk in doc.chunks:
                    self._chunks[chunk.id] = chunk

            # Migrate older formats → v3
            if version < _FORMAT_VERSION and self._documents:
                logger.info(
                    "migrating_metadata_format",
                    from_version=version,
                    to_version=_FORMAT_VERSION,
                    document_count=len(self._documents),
                )
                if version < 2:
                    for doc in self._documents.values():
                        self._save_document_content(doc)
                self._save_metadata()

        except (json.JSONDecodeError, KeyError) as e:
            # Log error but continue with empty state
            logger.warning(
                "failed_to_load_metadata",
                error=str(e),
                path=str(self.metadata_path),
            )

    def _save_metadata(self) -> None:
        """Save document metadata index to disk (without content)."""
        doc_headers = []
        for doc in self._documents.values():
            header = doc.to_dict()
            # Strip content from index — stored in per-document files
            header.pop("content", None)
            for chunk_header in header.get("chunks", []):
                chunk_header.pop("content", None)
            doc_headers.append(header)

        data = {
            "version": _FORMAT_VERSION,
            "documents": doc_headers,
            "updated_at": datetime.now().isoformat(),
        }
        atomic_write_json(self.metadata_path, data)

    # ------------------------------------------------------------------
    # File storage
    # ------------------------------------------------------------------

    def store_file(self, doc_id: str, file_bytes: bytes, extension: str) -> Path:
        """Store a binary file (e.g. PDF) for a document.

        Args:
            doc_id: Document ID.
            file_bytes: Raw file bytes.
            extension: File extension including dot (e.g. ".pdf").

        Returns:
            Relative path within files_dir.
        """
        self.files_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{doc_id}{extension}"
        dest = self.files_dir / filename
        dest.write_bytes(file_bytes)
        return Path(filename)

    def get_file_path(self, doc_id: str) -> Path | None:
        """Get the absolute file path for a document's stored file.

        Args:
            doc_id: Document ID.

        Returns:
            Absolute Path if file exists, None otherwise.
        """
        doc = self._documents.get(doc_id)
        if doc and doc.file_path:
            abs_path = self.files_dir / doc.file_path
            if abs_path.exists():
                return abs_path
        return None

    @staticmethod
    def extract_text_from_pdf(file_path: Path) -> str:
        """Extract text from a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Extracted text, or empty string on failure.
        """
        try:
            import pypdf
            reader = pypdf.PdfReader(str(file_path))
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n\n".join(pages)
        except ImportError:
            logger.warning("pypdf_not_installed")
            return ""
        except Exception as e:
            logger.warning("pdf_text_extraction_failed", error=str(e))
            return ""

    def _generate_summary(self, content: str) -> str:
        """Generate a summary for document content.

        Uses the LLM summarizer from workflow context if available,
        otherwise falls back to the first ~500 chars of content.

        Args:
            content: Full document text.

        Returns:
            Summary string (~500 chars).
        """
        try:
            from agentic_cli.workflow.context import get_context_llm_summarizer
            summarizer = get_context_llm_summarizer()
            if summarizer is not None:
                summary = summarizer.summarize(content, max_length=500)
                if summary:
                    return summary
        except Exception:
            pass

        # Fallback: first ~500 chars
        return truncate(content, 500)

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------

    def ingest_document(
        self,
        content: str,
        title: str,
        source_type: SourceType,
        source_url: str | None = None,
        metadata: dict[str, Any] | None = None,
        file_bytes: bytes | None = None,
        file_extension: str = ".pdf",
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
            file_bytes: Optional raw file bytes to store.
            file_extension: Extension for stored file (default ".pdf").
            chunk_size: Size of chunks for embedding.
            chunk_overlap: Overlap between chunks.

        Returns:
            The created Document object.
        """
        # Generate summary
        summary = self._generate_summary(content) if content else ""

        # Create document
        doc = Document.create(
            title=title,
            content=content,
            source_type=source_type,
            summary=summary,
            source_url=source_url,
            metadata=metadata,
        )

        # Store file if provided
        if file_bytes is not None:
            rel_path = self.store_file(doc.id, file_bytes, file_extension)
            doc.file_path = rel_path

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

        # Persist (content in per-doc file, headers in metadata index)
        self._save_document_content(doc)
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

            highlight = truncate(chunk.content)

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
        return matches_document_filters(doc, filters)

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
        self._delete_document_content(doc_id)
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
        # Delete per-document content files
        for doc_id in list(self._documents):
            self._delete_document_content(doc_id)

        self._documents = {}
        self._chunks = {}
        self._vector_store.clear()
        self._save_metadata()
        self._vector_store.save()
