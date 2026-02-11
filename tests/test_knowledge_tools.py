"""Tests for knowledge_tools and KnowledgeBaseManager persistence.

Covers:
- E1: Error handling in ingest_document
- E2: success key in search_knowledge_base
- E4: ContextVar-based KB manager lifecycle
- E5: Content separation and v1→v2→v3 migration
- New tools: read_document, list_documents, open_document
- File storage and summary generation
- Paper store migration
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from agentic_cli.knowledge_base.manager import (
    KnowledgeBaseManager,
    _FORMAT_VERSION,
)
from agentic_cli.knowledge_base.models import Document, DocumentChunk, SourceType


# ============================================================================
# Helper to create a test KB manager
# ============================================================================


def _make_kb(tmp_path):
    """Create a KB manager with mock services in a temp dir."""
    from agentic_cli.knowledge_base.embeddings import MockEmbeddingService
    from agentic_cli.knowledge_base.vector_store import MockVectorStore

    manager = KnowledgeBaseManager.__new__(KnowledgeBaseManager)
    manager._settings = None
    manager._use_mock = True
    manager.kb_dir = tmp_path / "kb"
    manager.documents_dir = manager.kb_dir / "documents"
    manager.embeddings_dir = manager.kb_dir / "embeddings"
    manager.files_dir = manager.kb_dir / "files"
    manager.metadata_path = manager.kb_dir / "metadata.json"
    manager.kb_dir.mkdir(parents=True, exist_ok=True)
    manager.documents_dir.mkdir(parents=True, exist_ok=True)
    manager.embeddings_dir.mkdir(parents=True, exist_ok=True)
    manager.files_dir.mkdir(parents=True, exist_ok=True)
    manager._embedding_service = MockEmbeddingService()
    manager._vector_store = MockVectorStore(
        index_path=manager.embeddings_dir / "index.mock",
        embedding_dim=384,
    )
    manager._documents = {}
    manager._chunks = {}
    return manager


# ============================================================================
# KnowledgeBaseManager persistence tests (E5)
# ============================================================================


class TestKBManagerPersistenceV2:
    """Tests for v2/v3 persistence format (content in per-document files)."""

    @pytest.fixture
    def kb(self, tmp_path):
        return _make_kb(tmp_path)

    def test_ingest_creates_per_doc_file(self, kb):
        """Ingesting a document creates a per-document content file."""
        doc = kb.ingest_document(
            content="Hello world. This is a test.",
            title="Test Doc",
            source_type=SourceType.USER,
        )

        content_path = kb.documents_dir / f"{doc.id}.json"
        assert content_path.exists()

        data = json.loads(content_path.read_text())
        assert data["content"] == "Hello world. This is a test."
        assert len(data["chunks"]) > 0

    def test_metadata_index_has_no_content(self, kb):
        """Metadata index should not contain document or chunk content."""
        kb.ingest_document(
            content="Some test content for the knowledge base.",
            title="Test",
            source_type=SourceType.USER,
        )

        data = json.loads(kb.metadata_path.read_text())
        assert data["version"] == _FORMAT_VERSION

        for doc_data in data["documents"]:
            assert "content" not in doc_data
            for chunk_data in doc_data.get("chunks", []):
                assert "content" not in chunk_data

    def test_metadata_index_has_structure(self, kb):
        """Metadata index should contain document structure (id, title, etc)."""
        doc = kb.ingest_document(
            content="Test content.",
            title="My Document",
            source_type=SourceType.ARXIV,
            source_url="https://arxiv.org/abs/1234",
        )

        data = json.loads(kb.metadata_path.read_text())
        assert len(data["documents"]) == 1

        header = data["documents"][0]
        assert header["id"] == doc.id
        assert header["title"] == "My Document"
        assert header["source_type"] == "arxiv"
        assert header["source_url"] == "https://arxiv.org/abs/1234"
        assert "created_at" in header
        assert len(header["chunks"]) > 0

    def test_metadata_index_has_summary(self, kb):
        """Metadata index should include summary field."""
        doc = kb.ingest_document(
            content="Test content for summary generation.",
            title="Summary Doc",
            source_type=SourceType.USER,
        )

        data = json.loads(kb.metadata_path.read_text())
        header = data["documents"][0]
        assert "summary" in header
        assert doc.summary  # Should have a summary

    def test_load_roundtrip(self, kb):
        """Documents survive save/load cycle with v3 format."""
        doc = kb.ingest_document(
            content="Roundtrip test content. With multiple sentences.",
            title="Roundtrip Doc",
            source_type=SourceType.WEB,
            source_url="https://example.com",
        )
        original_content = doc.content
        original_chunk_count = len(doc.chunks)
        original_summary = doc.summary

        # Create fresh manager pointing at same directory
        kb2 = _make_kb(kb.kb_dir.parent)
        kb2.kb_dir = kb.kb_dir
        kb2.documents_dir = kb.documents_dir
        kb2.embeddings_dir = kb.embeddings_dir
        kb2.files_dir = kb.files_dir
        kb2.metadata_path = kb.metadata_path
        kb2._load_metadata()

        loaded_doc = kb2.get_document(doc.id)
        assert loaded_doc is not None
        assert loaded_doc.content == original_content
        assert loaded_doc.title == "Roundtrip Doc"
        assert loaded_doc.summary == original_summary
        assert len(loaded_doc.chunks) == original_chunk_count
        for chunk in loaded_doc.chunks:
            assert chunk.content, f"Chunk {chunk.id} has no content"

    def test_delete_removes_content_file(self, kb):
        """Deleting a document removes its per-document content file."""
        doc = kb.ingest_document(
            content="Delete me.",
            title="To Delete",
            source_type=SourceType.USER,
        )
        content_path = kb.documents_dir / f"{doc.id}.json"
        assert content_path.exists()

        kb.delete_document(doc.id)
        assert not content_path.exists()

    def test_clear_removes_all_content_files(self, kb):
        """Clearing KB removes all per-document content files."""
        kb.ingest_document(content="Doc 1.", title="D1", source_type=SourceType.USER)
        kb.ingest_document(content="Doc 2.", title="D2", source_type=SourceType.USER)

        doc_files = list(kb.documents_dir.glob("*.json"))
        assert len(doc_files) == 2

        kb.clear()

        doc_files = list(kb.documents_dir.glob("*.json"))
        assert len(doc_files) == 0


class TestKBManagerMigrationV1ToV2:
    """Tests for automatic migration from v1 to v3 format."""

    @pytest.fixture
    def kb_dir(self, tmp_path):
        kb = tmp_path / "kb"
        kb.mkdir()
        (kb / "documents").mkdir()
        (kb / "embeddings").mkdir()
        (kb / "files").mkdir()
        return kb

    def _make_v1_metadata(self, kb_dir):
        metadata = {
            "documents": [
                {
                    "id": "doc-v1-001",
                    "title": "V1 Document",
                    "content": "This is old-format content.",
                    "source_type": "user",
                    "source_url": None,
                    "file_path": None,
                    "created_at": "2024-06-01T12:00:00",
                    "updated_at": "2024-06-01T12:00:00",
                    "metadata": {},
                    "chunks": [
                        {
                            "id": "chunk-v1-001",
                            "document_id": "doc-v1-001",
                            "content": "This is old-format content.",
                            "chunk_index": 0,
                            "metadata": {"title": "V1 Document"},
                        }
                    ],
                }
            ],
            "updated_at": "2024-06-01T12:00:00",
        }
        (kb_dir / "metadata.json").write_text(json.dumps(metadata))
        return metadata

    def _make_kb_manager(self, kb_dir):
        from agentic_cli.knowledge_base.embeddings import MockEmbeddingService
        from agentic_cli.knowledge_base.vector_store import MockVectorStore

        manager = KnowledgeBaseManager.__new__(KnowledgeBaseManager)
        manager._settings = None
        manager._use_mock = True
        manager.kb_dir = kb_dir
        manager.documents_dir = kb_dir / "documents"
        manager.embeddings_dir = kb_dir / "embeddings"
        manager.files_dir = kb_dir / "files"
        manager.metadata_path = kb_dir / "metadata.json"
        manager._embedding_service = MockEmbeddingService()
        manager._vector_store = MockVectorStore(
            index_path=manager.embeddings_dir / "index.mock",
            embedding_dim=384,
        )
        manager._documents = {}
        manager._chunks = {}
        return manager

    def test_v1_loaded_correctly(self, kb_dir):
        """V1 format loads documents with content."""
        self._make_v1_metadata(kb_dir)
        kb = self._make_kb_manager(kb_dir)
        kb._load_metadata()

        doc = kb.get_document("doc-v1-001")
        assert doc is not None
        assert doc.title == "V1 Document"
        assert doc.content == "This is old-format content."
        assert len(doc.chunks) == 1
        assert doc.chunks[0].content == "This is old-format content."

    def test_v1_auto_migrated_to_v3(self, kb_dir):
        """V1 format is auto-migrated to v3 on load."""
        self._make_v1_metadata(kb_dir)
        kb = self._make_kb_manager(kb_dir)
        kb._load_metadata()

        # Per-document file should be created
        content_path = kb.documents_dir / "doc-v1-001.json"
        assert content_path.exists()

        content_data = json.loads(content_path.read_text())
        assert content_data["content"] == "This is old-format content."
        assert "chunk-v1-001" in content_data["chunks"]

        # Metadata index should be rewritten as v3
        meta_data = json.loads(kb.metadata_path.read_text())
        assert meta_data["version"] == _FORMAT_VERSION
        for doc_data in meta_data["documents"]:
            assert "content" not in doc_data
            for chunk_data in doc_data.get("chunks", []):
                assert "content" not in chunk_data

    def test_v2_not_re_migrated(self, kb_dir):
        """V2 format is loaded and upgraded to v3 (metadata rewrite only)."""
        metadata = {
            "version": 2,
            "documents": [
                {
                    "id": "doc-v2-001",
                    "title": "V2 Document",
                    "source_type": "user",
                    "source_url": None,
                    "file_path": None,
                    "created_at": "2024-06-01T12:00:00",
                    "updated_at": "2024-06-01T12:00:00",
                    "metadata": {},
                    "chunks": [
                        {
                            "id": "chunk-v2-001",
                            "document_id": "doc-v2-001",
                            "chunk_index": 0,
                            "metadata": {"title": "V2 Document"},
                        }
                    ],
                }
            ],
            "updated_at": "2024-06-01T12:00:00",
        }
        (kb_dir / "metadata.json").write_text(json.dumps(metadata))

        content_data = {
            "content": "V2 content here.",
            "chunks": {"chunk-v2-001": "V2 content here."},
        }
        (kb_dir / "documents" / "doc-v2-001.json").write_text(json.dumps(content_data))

        kb = self._make_kb_manager(kb_dir)
        kb._load_metadata()

        doc = kb.get_document("doc-v2-001")
        assert doc is not None
        assert doc.content == "V2 content here."
        assert doc.chunks[0].content == "V2 content here."


# ============================================================================
# File storage tests
# ============================================================================


class TestKBFileStorage:
    """Tests for file storage in KnowledgeBaseManager."""

    @pytest.fixture
    def kb(self, tmp_path):
        return _make_kb(tmp_path)

    def test_store_file(self, kb):
        """Test storing a file for a document."""
        file_bytes = b"%PDF-1.4 fake content"
        rel_path = kb.store_file("doc-123", file_bytes, ".pdf")

        assert rel_path == Path("doc-123.pdf")
        stored = kb.files_dir / "doc-123.pdf"
        assert stored.exists()
        assert stored.read_bytes() == file_bytes

    def test_get_file_path(self, kb):
        """Test getting file path for a stored document."""
        doc = kb.ingest_document(
            content="Test content",
            title="Test",
            source_type=SourceType.USER,
            file_bytes=b"fake pdf",
            file_extension=".pdf",
        )

        file_path = kb.get_file_path(doc.id)
        assert file_path is not None
        assert file_path.exists()
        assert file_path.read_bytes() == b"fake pdf"

    def test_get_file_path_no_file(self, kb):
        """Test getting file path for document without file."""
        doc = kb.ingest_document(
            content="Text only",
            title="No File",
            source_type=SourceType.USER,
        )

        file_path = kb.get_file_path(doc.id)
        assert file_path is None

    def test_ingest_with_file_bytes(self, kb):
        """Test ingesting document with file bytes stores the file."""
        doc = kb.ingest_document(
            content="Extracted text from PDF.",
            title="PDF Document",
            source_type=SourceType.ARXIV,
            file_bytes=b"%PDF-1.4 content",
            file_extension=".pdf",
        )

        assert doc.file_path is not None
        assert (kb.files_dir / str(doc.file_path)).exists()


# ============================================================================
# Summary generation tests
# ============================================================================


class TestKBSummaryGeneration:
    """Tests for summary generation in KnowledgeBaseManager."""

    @pytest.fixture
    def kb(self, tmp_path):
        return _make_kb(tmp_path)

    def test_summary_generated_on_ingest(self, kb):
        """Summary should be generated when ingesting a document."""
        doc = kb.ingest_document(
            content="This is a test document with some content that should be summarized.",
            title="Summary Test",
            source_type=SourceType.USER,
        )

        assert doc.summary != ""
        assert len(doc.summary) <= 503  # 500 + "..."

    def test_summary_fallback_to_truncation(self, kb):
        """Without LLM summarizer, summary falls back to truncated content."""
        content = "Short content."
        doc = kb.ingest_document(
            content=content,
            title="Short Doc",
            source_type=SourceType.USER,
        )

        assert doc.summary == content


# ============================================================================
# Paper store migration tests
# ============================================================================


# ============================================================================
# Tool error handling tests (E1, E2)
# ============================================================================


class TestSearchKnowledgeBaseTool:
    """Tests for search_knowledge_base error handling and success key."""

    def test_invalid_json_filters_returns_error(self):
        """Invalid JSON in filters returns error dict."""
        from agentic_cli.tools.knowledge_tools import search_knowledge_base

        result = search_knowledge_base("query", filters="not json")
        assert result["success"] is False

    def test_search_returns_success_key(self):
        """Search results include success=True when context is set."""
        from agentic_cli.workflow.context import set_context_kb_manager
        from agentic_cli.tools.knowledge_tools import search_knowledge_base

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            token = set_context_kb_manager(kb)
            try:
                result = search_knowledge_base("test query")
                assert result["success"] is True
                assert "results" in result
                assert "total_matches" in result
            finally:
                token.var.reset(token)

    def test_no_context_returns_error(self):
        """search_knowledge_base returns error when KB context is not set."""
        from agentic_cli.workflow.context import set_context_kb_manager
        from agentic_cli.tools.knowledge_tools import search_knowledge_base

        token = set_context_kb_manager(None)
        try:
            result = search_knowledge_base("query")
            assert result["success"] is False
            assert "error" in result
        finally:
            token.var.reset(token)


class TestIngestDocumentTool:
    """Tests for ingest_document error handling."""

    @pytest.mark.asyncio
    async def test_invalid_source_type_returns_error(self):
        """Invalid source_type returns error dict."""
        from agentic_cli.workflow.context import set_context_kb_manager
        from agentic_cli.tools.knowledge_tools import ingest_document

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            token = set_context_kb_manager(kb)
            try:
                result = await ingest_document(
                    content="Test",
                    title="Test",
                    source_type="invalid_type",
                )
                assert result["success"] is False
                assert "Invalid source_type" in result["error"]
            finally:
                token.var.reset(token)

    @pytest.mark.asyncio
    async def test_valid_ingest_returns_success(self):
        """Valid ingestion returns success=True with document info."""
        from agentic_cli.workflow.context import set_context_kb_manager
        from agentic_cli.tools.knowledge_tools import ingest_document

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            token = set_context_kb_manager(kb)
            try:
                result = await ingest_document(
                    content="Test content for ingestion.",
                    title="Test Document",
                    source_type="user",
                )
                assert result["success"] is True
                assert "document_id" in result
                assert result["title"] == "Test Document"
                assert result["chunks_created"] > 0
                assert "summary" in result
            finally:
                token.var.reset(token)

    @pytest.mark.asyncio
    async def test_no_context_returns_error(self):
        """ingest_document returns error when KB context is not set."""
        from agentic_cli.workflow.context import set_context_kb_manager
        from agentic_cli.tools.knowledge_tools import ingest_document

        token = set_context_kb_manager(None)
        try:
            result = await ingest_document(
                content="Test",
                title="Test",
            )
            assert result["success"] is False
            assert "error" in result
        finally:
            token.var.reset(token)

    @pytest.mark.asyncio
    async def test_no_content_or_file_returns_error(self):
        """Providing neither content nor url_or_path returns error."""
        from agentic_cli.workflow.context import set_context_kb_manager
        from agentic_cli.tools.knowledge_tools import ingest_document

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            token = set_context_kb_manager(kb)
            try:
                result = await ingest_document(title="Empty")
                assert result["success"] is False
                assert "No content" in result["error"]
            finally:
                token.var.reset(token)

    @pytest.mark.asyncio
    async def test_ingest_local_file(self):
        """Ingesting a local PDF copies and extracts text."""
        from agentic_cli.workflow.context import set_context_kb_manager
        from agentic_cli.tools.knowledge_tools import ingest_document

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            token = set_context_kb_manager(kb)
            try:
                # Create a fake PDF file
                pdf_path = Path(tmp) / "test.pdf"
                pdf_path.write_bytes(b"%PDF-1.4 fake content")

                result = await ingest_document(
                    url_or_path=str(pdf_path),
                    title="Local PDF",
                )
                assert result["success"] is True
                assert result["title"] == "Local PDF"
            finally:
                token.var.reset(token)

    @pytest.mark.asyncio
    async def test_ingest_local_file_not_found(self):
        """Ingesting a nonexistent file returns error."""
        from agentic_cli.workflow.context import set_context_kb_manager
        from agentic_cli.tools.knowledge_tools import ingest_document

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            token = set_context_kb_manager(kb)
            try:
                result = await ingest_document(
                    url_or_path="/nonexistent/file.pdf",
                )
                assert result["success"] is False
                assert "not found" in result["error"].lower()
            finally:
                token.var.reset(token)


# ============================================================================
# read_document tests
# ============================================================================


class TestReadDocument:
    """Tests for read_document tool."""

    def test_read_by_id(self):
        """Read document by exact ID."""
        from agentic_cli.workflow.context import set_context_kb_manager
        from agentic_cli.tools.knowledge_tools import read_document

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            doc = kb.ingest_document(
                content="Full document content here.",
                title="Readable Doc",
                source_type=SourceType.USER,
            )
            token = set_context_kb_manager(kb)
            try:
                result = read_document(doc.id)
                assert result["success"] is True
                assert result["content"] == "Full document content here."
                assert result["title"] == "Readable Doc"
                assert result["truncated"] is False
            finally:
                token.var.reset(token)

    def test_read_by_title(self):
        """Read document by title substring."""
        from agentic_cli.workflow.context import set_context_kb_manager
        from agentic_cli.tools.knowledge_tools import read_document

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            kb.ingest_document(
                content="Content here.",
                title="Unique Title ABC",
                source_type=SourceType.USER,
            )
            token = set_context_kb_manager(kb)
            try:
                result = read_document("Unique Title")
                assert result["success"] is True
                assert result["title"] == "Unique Title ABC"
            finally:
                token.var.reset(token)

    def test_read_not_found(self):
        """Read nonexistent document returns error."""
        from agentic_cli.workflow.context import set_context_kb_manager
        from agentic_cli.tools.knowledge_tools import read_document

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            token = set_context_kb_manager(kb)
            try:
                result = read_document("nonexistent")
                assert result["success"] is False
                assert "not found" in result["error"].lower()
            finally:
                token.var.reset(token)

    def test_read_truncation(self):
        """Long content is truncated to max_chars."""
        from agentic_cli.workflow.context import set_context_kb_manager
        from agentic_cli.tools.knowledge_tools import read_document

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            long_content = "x" * 50000
            doc = kb.ingest_document(
                content=long_content,
                title="Long Doc",
                source_type=SourceType.USER,
            )
            token = set_context_kb_manager(kb)
            try:
                result = read_document(doc.id, max_chars=1000)
                assert result["success"] is True
                assert len(result["content"]) == 1000
                assert result["truncated"] is True
            finally:
                token.var.reset(token)


# ============================================================================
# list_documents tests
# ============================================================================


class TestListDocuments:
    """Tests for list_documents tool."""

    def test_list_empty(self):
        """List returns empty when no documents."""
        from agentic_cli.workflow.context import set_context_kb_manager
        from agentic_cli.tools.knowledge_tools import list_documents

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            token = set_context_kb_manager(kb)
            try:
                result = list_documents()
                assert result["success"] is True
                assert result["count"] == 0
                assert result["documents"] == []
            finally:
                token.var.reset(token)

    def test_list_with_documents(self):
        """List returns documents with summaries."""
        from agentic_cli.workflow.context import set_context_kb_manager
        from agentic_cli.tools.knowledge_tools import list_documents

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            kb.ingest_document(
                content="Document A content.",
                title="Document A",
                source_type=SourceType.ARXIV,
                metadata={"authors": ["Alice"]},
            )
            kb.ingest_document(
                content="Document B content.",
                title="Document B",
                source_type=SourceType.USER,
            )
            token = set_context_kb_manager(kb)
            try:
                result = list_documents()
                assert result["success"] is True
                assert result["count"] == 2
                # Check document fields
                doc_a = next(d for d in result["documents"] if d["title"] == "Document A")
                assert doc_a["source_type"] == "arxiv"
                assert "summary" in doc_a
            finally:
                token.var.reset(token)

    def test_list_filter_source_type(self):
        """List filters by source type."""
        from agentic_cli.workflow.context import set_context_kb_manager
        from agentic_cli.tools.knowledge_tools import list_documents

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            kb.ingest_document(content="A.", title="A", source_type=SourceType.ARXIV)
            kb.ingest_document(content="B.", title="B", source_type=SourceType.USER)
            token = set_context_kb_manager(kb)
            try:
                result = list_documents(source_type="arxiv")
                assert result["count"] == 1
                assert result["documents"][0]["source_type"] == "arxiv"
            finally:
                token.var.reset(token)

    def test_list_filter_query(self):
        """List filters by title query."""
        from agentic_cli.workflow.context import set_context_kb_manager
        from agentic_cli.tools.knowledge_tools import list_documents

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            kb.ingest_document(content="A.", title="Attention Paper", source_type=SourceType.USER)
            kb.ingest_document(content="B.", title="BERT Paper", source_type=SourceType.USER)
            token = set_context_kb_manager(kb)
            try:
                result = list_documents(query="attention")
                assert result["count"] == 1
                assert result["documents"][0]["title"] == "Attention Paper"
            finally:
                token.var.reset(token)


# ============================================================================
# open_document tests
# ============================================================================


class TestOpenDocument:
    """Tests for open_document tool."""

    def test_open_not_found(self):
        """Open nonexistent document returns error."""
        from agentic_cli.workflow.context import set_context_kb_manager
        from agentic_cli.tools.knowledge_tools import open_document

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            token = set_context_kb_manager(kb)
            try:
                result = open_document("nonexistent")
                assert result["success"] is False
                assert "not found" in result["error"].lower()
            finally:
                token.var.reset(token)

    def test_open_no_file(self):
        """Open document without file returns error."""
        from agentic_cli.workflow.context import set_context_kb_manager
        from agentic_cli.tools.knowledge_tools import open_document

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            doc = kb.ingest_document(
                content="Text only.",
                title="No File Doc",
                source_type=SourceType.USER,
            )
            token = set_context_kb_manager(kb)
            try:
                result = open_document(doc.id)
                assert result["success"] is False
                assert "no file" in result["error"].lower()
            finally:
                token.var.reset(token)

    def test_open_success(self):
        """Open document with file calls system viewer."""
        from agentic_cli.workflow.context import set_context_kb_manager
        from agentic_cli.tools.knowledge_tools import open_document

        with tempfile.TemporaryDirectory() as tmp:
            kb = _make_kb(Path(tmp))
            doc = kb.ingest_document(
                content="PDF content.",
                title="PDF Doc",
                source_type=SourceType.USER,
                file_bytes=b"%PDF-1.4 content",
                file_extension=".pdf",
            )
            token = set_context_kb_manager(kb)
            try:
                with patch("agentic_cli.tools.knowledge_tools.subprocess.Popen") as mock_popen:
                    result = open_document(doc.id)
                assert result["success"] is True
                assert result["title"] == "PDF Doc"
                mock_popen.assert_called_once()
            finally:
                token.var.reset(token)


# ============================================================================
# ContextVar lifecycle tests (E4)
# ============================================================================


class TestKBManagerContextVar:
    """Tests for KB manager ContextVar integration."""

    def test_context_accessors_exist(self):
        from agentic_cli.workflow.context import (
            set_context_kb_manager,
            get_context_kb_manager,
        )
        assert callable(set_context_kb_manager)
        assert callable(get_context_kb_manager)

    def test_context_default_is_none(self):
        from agentic_cli.workflow.context import (
            set_context_kb_manager,
            get_context_kb_manager,
        )
        token = set_context_kb_manager(None)
        try:
            assert get_context_kb_manager() is None
        finally:
            token.var.reset(token)

    def test_context_roundtrip(self):
        from agentic_cli.workflow.context import (
            set_context_kb_manager,
            get_context_kb_manager,
        )
        sentinel = object()
        token = set_context_kb_manager(sentinel)
        try:
            assert get_context_kb_manager() is sentinel
        finally:
            token.var.reset(token)

    def test_kb_manager_in_manager_requirement(self):
        from agentic_cli.tools import ManagerRequirement
        from agentic_cli.tools.knowledge_tools import search_knowledge_base
        assert hasattr(search_knowledge_base, "requires")
        assert "kb_manager" in search_knowledge_base.requires

    def test_base_manager_has_kb_manager_slot(self):
        from agentic_cli.workflow.base_manager import BaseWorkflowManager
        assert hasattr(BaseWorkflowManager, "kb_manager")

    def test_base_manager_has_user_kb_manager_slot(self):
        from agentic_cli.workflow.base_manager import BaseWorkflowManager
        assert hasattr(BaseWorkflowManager, "user_kb_manager")

    def test_user_kb_manager_context_accessors_exist(self):
        from agentic_cli.workflow.context import (
            set_context_user_kb_manager,
            get_context_user_kb_manager,
        )
        assert callable(set_context_user_kb_manager)
        assert callable(get_context_user_kb_manager)


# ============================================================================
# Two-tier knowledge base tests
# ============================================================================


class TestTwoTierKnowledgeBase:
    """Tests for two-tier (project + user) knowledge base access."""

    @pytest.fixture
    def project_kb(self, tmp_path):
        """Create a project-scoped KB with a document."""
        kb = _make_kb(tmp_path / "project")
        kb.ingest_document(
            content="Project document about transformers.",
            title="Project Transformer Paper",
            source_type=SourceType.ARXIV,
        )
        return kb

    @pytest.fixture
    def user_kb(self, tmp_path):
        """Create a user-scoped KB with a different document."""
        kb = _make_kb(tmp_path / "user")
        kb.ingest_document(
            content="User curated document about attention mechanisms.",
            title="User Attention Paper",
            source_type=SourceType.USER,
        )
        return kb

    def _set_both_contexts(self, project_kb, user_kb):
        """Set both KB context vars and return tokens for cleanup."""
        from agentic_cli.workflow.context import (
            set_context_kb_manager,
            set_context_user_kb_manager,
        )
        t1 = set_context_kb_manager(project_kb)
        t2 = set_context_user_kb_manager(user_kb)
        return t1, t2

    def _reset_tokens(self, *tokens):
        for token in tokens:
            token.var.reset(token)

    def test_search_merges_both_kbs(self, project_kb, user_kb):
        """Search returns results from both KBs with scope tags."""
        from agentic_cli.tools.knowledge_tools import search_knowledge_base

        t1, t2 = self._set_both_contexts(project_kb, user_kb)
        try:
            result = search_knowledge_base("attention transformers")
            assert result["success"] is True
            scopes = {r.get("scope") for r in result["results"]}
            assert "project" in scopes or "user" in scopes
            # Both KBs should contribute results
            assert result["total_matches"] >= 1
        finally:
            self._reset_tokens(t1, t2)

    @pytest.mark.asyncio
    async def test_ingest_writes_to_project_only(self, project_kb, user_kb):
        """Ingestion only writes to project KB, not user KB."""
        from agentic_cli.tools.knowledge_tools import ingest_document

        t1, t2 = self._set_both_contexts(project_kb, user_kb)
        try:
            user_count_before = len(user_kb._documents)
            project_count_before = len(project_kb._documents)

            result = await ingest_document(
                content="New ingested document.",
                title="New Doc",
                source_type="user",
            )
            assert result["success"] is True

            # Project KB gains a document
            assert len(project_kb._documents) == project_count_before + 1
            # User KB unchanged
            assert len(user_kb._documents) == user_count_before
        finally:
            self._reset_tokens(t1, t2)

    def test_list_documents_shows_scope(self, project_kb, user_kb):
        """list_documents includes scope indicator for both tiers."""
        from agentic_cli.tools.knowledge_tools import list_documents

        t1, t2 = self._set_both_contexts(project_kb, user_kb)
        try:
            result = list_documents()
            assert result["success"] is True
            assert result["count"] == 2

            scopes = {d["scope"] for d in result["documents"]}
            assert scopes == {"project", "user"}
        finally:
            self._reset_tokens(t1, t2)

    def test_read_document_project_first(self, tmp_path):
        """When both KBs have a matching doc, project is returned first."""
        from agentic_cli.tools.knowledge_tools import read_document

        pkb = _make_kb(tmp_path / "proj")
        ukb = _make_kb(tmp_path / "usr")
        pkb.ingest_document(
            content="Project version of the doc.",
            title="Shared Title",
            source_type=SourceType.USER,
        )
        ukb.ingest_document(
            content="User version of the doc.",
            title="Shared Title",
            source_type=SourceType.USER,
        )

        t1, t2 = self._set_both_contexts(pkb, ukb)
        try:
            result = read_document("Shared Title")
            assert result["success"] is True
            assert result["content"] == "Project version of the doc."
        finally:
            self._reset_tokens(t1, t2)

    def test_read_document_falls_back_to_user(self, project_kb, user_kb):
        """Doc only in user KB is still found via fallback."""
        from agentic_cli.tools.knowledge_tools import read_document

        t1, t2 = self._set_both_contexts(project_kb, user_kb)
        try:
            result = read_document("User Attention Paper")
            assert result["success"] is True
            assert result["title"] == "User Attention Paper"
            assert "attention" in result["content"].lower()
        finally:
            self._reset_tokens(t1, t2)

    def test_same_dir_single_instance(self, tmp_path):
        """When project and user paths resolve to same dir, one manager is reused."""
        from agentic_cli.tools.knowledge_tools import search_knowledge_base

        kb = _make_kb(tmp_path / "shared")
        kb.ingest_document(
            content="Shared document.",
            title="Shared Doc",
            source_type=SourceType.USER,
        )

        # Set same KB for both
        from agentic_cli.workflow.context import (
            set_context_kb_manager,
            set_context_user_kb_manager,
        )
        t1 = set_context_kb_manager(kb)
        t2 = set_context_user_kb_manager(kb)
        try:
            result = search_knowledge_base("shared")
            assert result["success"] is True
            # Should not have duplicates — user KB is same instance, so merge is skipped
            doc_ids = [r["document_id"] for r in result["results"]]
            assert len(doc_ids) == len(set(doc_ids))
        finally:
            t1.var.reset(t1)
            t2.var.reset(t2)

    def test_user_kb_search_failure_non_fatal(self, project_kb):
        """Project results returned even if user KB search raises."""
        from agentic_cli.tools.knowledge_tools import search_knowledge_base
        from agentic_cli.workflow.context import (
            set_context_kb_manager,
            set_context_user_kb_manager,
        )

        # Create a mock user KB that raises on search
        failing_kb = MagicMock()
        failing_kb.search.side_effect = RuntimeError("User KB broken")

        t1 = set_context_kb_manager(project_kb)
        t2 = set_context_user_kb_manager(failing_kb)
        try:
            result = search_knowledge_base("transformers")
            assert result["success"] is True
            # Project results should still be present
            assert result["total_matches"] >= 1
        finally:
            t1.var.reset(t1)
            t2.var.reset(t2)
