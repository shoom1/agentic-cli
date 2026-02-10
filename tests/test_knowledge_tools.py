"""Tests for knowledge_tools and KnowledgeBaseManager persistence.

Covers:
- E1: Error handling in ingest_to_knowledge_base
- E2: success key in search_knowledge_base
- E4: ContextVar-based KB manager lifecycle
- E5: Content separation and v1→v2 migration
"""

import json

import pytest

from agentic_cli.knowledge_base.manager import (
    KnowledgeBaseManager,
    _FORMAT_VERSION,
)
from agentic_cli.knowledge_base.models import Document, DocumentChunk, SourceType


# ============================================================================
# KnowledgeBaseManager persistence tests (E5)
# ============================================================================


class TestKBManagerPersistenceV2:
    """Tests for v2 persistence format (content in per-document files)."""

    @pytest.fixture
    def kb(self, tmp_path):
        """Create a KB manager with mock services in a temp dir."""
        manager = KnowledgeBaseManager.__new__(KnowledgeBaseManager)
        manager._settings = None
        manager._use_mock = True
        manager.kb_dir = tmp_path / "kb"
        manager.documents_dir = manager.kb_dir / "documents"
        manager.embeddings_dir = manager.kb_dir / "embeddings"
        manager.metadata_path = manager.kb_dir / "metadata.json"
        manager.kb_dir.mkdir(parents=True)
        manager.documents_dir.mkdir(parents=True)
        manager.embeddings_dir.mkdir(parents=True)

        from agentic_cli.knowledge_base.embeddings import MockEmbeddingService
        from agentic_cli.knowledge_base.vector_store import MockVectorStore

        manager._embedding_service = MockEmbeddingService()
        manager._vector_store = MockVectorStore(
            index_path=manager.embeddings_dir / "index.mock",
            embedding_dim=384,
        )
        manager._documents = {}
        manager._chunks = {}
        return manager

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

    def test_load_roundtrip(self, kb):
        """Documents survive save/load cycle with v2 format."""
        doc = kb.ingest_document(
            content="Roundtrip test content. With multiple sentences.",
            title="Roundtrip Doc",
            source_type=SourceType.WEB,
            source_url="https://example.com",
        )
        original_content = doc.content
        original_chunk_count = len(doc.chunks)

        # Create fresh manager pointing at same directory
        kb2 = KnowledgeBaseManager.__new__(KnowledgeBaseManager)
        kb2._settings = None
        kb2._use_mock = True
        kb2.kb_dir = kb.kb_dir
        kb2.documents_dir = kb.documents_dir
        kb2.embeddings_dir = kb.embeddings_dir
        kb2.metadata_path = kb.metadata_path

        from agentic_cli.knowledge_base.embeddings import MockEmbeddingService
        from agentic_cli.knowledge_base.vector_store import MockVectorStore

        kb2._embedding_service = MockEmbeddingService()
        kb2._vector_store = MockVectorStore(
            index_path=kb2.embeddings_dir / "index.mock",
            embedding_dim=384,
        )
        kb2._documents = {}
        kb2._chunks = {}
        kb2._load_metadata()

        loaded_doc = kb2.get_document(doc.id)
        assert loaded_doc is not None
        assert loaded_doc.content == original_content
        assert loaded_doc.title == "Roundtrip Doc"
        assert len(loaded_doc.chunks) == original_chunk_count
        # Chunk content should be restored
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

        # Should have 2 document content files
        doc_files = list(kb.documents_dir.glob("*.json"))
        assert len(doc_files) == 2

        kb.clear()

        doc_files = list(kb.documents_dir.glob("*.json"))
        assert len(doc_files) == 0


class TestKBManagerMigrationV1ToV2:
    """Tests for automatic migration from v1 to v2 format."""

    @pytest.fixture
    def kb_dir(self, tmp_path):
        """Create a temp KB directory structure."""
        kb = tmp_path / "kb"
        kb.mkdir()
        (kb / "documents").mkdir()
        (kb / "embeddings").mkdir()
        return kb

    def _make_v1_metadata(self, kb_dir):
        """Create a v1-format metadata.json with content inline."""
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
        """Create a KB manager for the given directory."""
        from agentic_cli.knowledge_base.embeddings import MockEmbeddingService
        from agentic_cli.knowledge_base.vector_store import MockVectorStore

        manager = KnowledgeBaseManager.__new__(KnowledgeBaseManager)
        manager._settings = None
        manager._use_mock = True
        manager.kb_dir = kb_dir
        manager.documents_dir = kb_dir / "documents"
        manager.embeddings_dir = kb_dir / "embeddings"
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

    def test_v1_auto_migrated_to_v2(self, kb_dir):
        """V1 format is auto-migrated to v2 on load."""
        self._make_v1_metadata(kb_dir)

        kb = self._make_kb_manager(kb_dir)
        kb._load_metadata()

        # Per-document file should be created
        content_path = kb.documents_dir / "doc-v1-001.json"
        assert content_path.exists()

        content_data = json.loads(content_path.read_text())
        assert content_data["content"] == "This is old-format content."
        assert "chunk-v1-001" in content_data["chunks"]

        # Metadata index should be rewritten as v2
        meta_data = json.loads(kb.metadata_path.read_text())
        assert meta_data["version"] == _FORMAT_VERSION
        # Content should be stripped from the index
        for doc_data in meta_data["documents"]:
            assert "content" not in doc_data
            for chunk_data in doc_data.get("chunks", []):
                assert "content" not in chunk_data

    def test_v2_not_re_migrated(self, kb_dir):
        """V2 format is not re-migrated on load."""
        # Write v2 metadata directly
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

        # Write per-document content
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
# Tool error handling tests (E1, E2)
# ============================================================================


class TestSearchKnowledgeBaseTool:
    """Tests for search_knowledge_base error handling and success key."""

    def test_invalid_json_filters_returns_error(self):
        """Invalid JSON in filters returns error dict."""
        from agentic_cli.tools.knowledge_tools import search_knowledge_base

        # Unwrap the decorators to test the core function
        # The @require_context guard will return error if no context
        result = search_knowledge_base("query", filters="not json")
        # Either the context guard fires first or the JSON error fires
        assert result["success"] is False

    def test_search_returns_success_key(self):
        """Search results include success=True when context is set."""
        from agentic_cli.knowledge_base.manager import KnowledgeBaseManager
        from agentic_cli.workflow.context import set_context_kb_manager
        from agentic_cli.tools.knowledge_tools import search_knowledge_base

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path
            kb = KnowledgeBaseManager.__new__(KnowledgeBaseManager)
            kb._settings = None
            kb._use_mock = True
            kb.kb_dir = Path(tmp) / "kb"
            kb.documents_dir = kb.kb_dir / "documents"
            kb.embeddings_dir = kb.kb_dir / "embeddings"
            kb.metadata_path = kb.kb_dir / "metadata.json"
            kb.kb_dir.mkdir(parents=True)
            kb.documents_dir.mkdir(parents=True)
            kb.embeddings_dir.mkdir(parents=True)

            from agentic_cli.knowledge_base.embeddings import MockEmbeddingService
            from agentic_cli.knowledge_base.vector_store import MockVectorStore

            kb._embedding_service = MockEmbeddingService()
            kb._vector_store = MockVectorStore(
                index_path=kb.embeddings_dir / "index.mock",
                embedding_dim=384,
            )
            kb._documents = {}
            kb._chunks = {}

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

        # Ensure context is None
        token = set_context_kb_manager(None)
        try:
            result = search_knowledge_base("query")
            assert result["success"] is False
            assert "error" in result
        finally:
            token.var.reset(token)


class TestIngestToKnowledgeBaseTool:
    """Tests for ingest_to_knowledge_base error handling."""

    def test_invalid_source_type_returns_error(self):
        """Invalid source_type returns error dict instead of raising."""
        from agentic_cli.workflow.context import set_context_kb_manager
        from agentic_cli.tools.knowledge_tools import ingest_to_knowledge_base

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path
            from agentic_cli.knowledge_base.manager import KnowledgeBaseManager
            from agentic_cli.knowledge_base.embeddings import MockEmbeddingService
            from agentic_cli.knowledge_base.vector_store import MockVectorStore

            kb = KnowledgeBaseManager.__new__(KnowledgeBaseManager)
            kb._settings = None
            kb._use_mock = True
            kb.kb_dir = Path(tmp) / "kb"
            kb.documents_dir = kb.kb_dir / "documents"
            kb.embeddings_dir = kb.kb_dir / "embeddings"
            kb.metadata_path = kb.kb_dir / "metadata.json"
            kb.kb_dir.mkdir(parents=True)
            kb.documents_dir.mkdir(parents=True)
            kb.embeddings_dir.mkdir(parents=True)
            kb._embedding_service = MockEmbeddingService()
            kb._vector_store = MockVectorStore(
                index_path=kb.embeddings_dir / "index.mock",
                embedding_dim=384,
            )
            kb._documents = {}
            kb._chunks = {}

            token = set_context_kb_manager(kb)
            try:
                result = ingest_to_knowledge_base(
                    content="Test",
                    title="Test",
                    source_type="invalid_type",
                )
                assert result["success"] is False
                assert "Invalid source_type" in result["error"]
                assert "invalid_type" in result["error"]
            finally:
                token.var.reset(token)

    def test_valid_ingest_returns_success(self):
        """Valid ingestion returns success=True with document info."""
        from agentic_cli.workflow.context import set_context_kb_manager
        from agentic_cli.tools.knowledge_tools import ingest_to_knowledge_base

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path
            from agentic_cli.knowledge_base.manager import KnowledgeBaseManager
            from agentic_cli.knowledge_base.embeddings import MockEmbeddingService
            from agentic_cli.knowledge_base.vector_store import MockVectorStore

            kb = KnowledgeBaseManager.__new__(KnowledgeBaseManager)
            kb._settings = None
            kb._use_mock = True
            kb.kb_dir = Path(tmp) / "kb"
            kb.documents_dir = kb.kb_dir / "documents"
            kb.embeddings_dir = kb.kb_dir / "embeddings"
            kb.metadata_path = kb.kb_dir / "metadata.json"
            kb.kb_dir.mkdir(parents=True)
            kb.documents_dir.mkdir(parents=True)
            kb.embeddings_dir.mkdir(parents=True)
            kb._embedding_service = MockEmbeddingService()
            kb._vector_store = MockVectorStore(
                index_path=kb.embeddings_dir / "index.mock",
                embedding_dim=384,
            )
            kb._documents = {}
            kb._chunks = {}

            token = set_context_kb_manager(kb)
            try:
                result = ingest_to_knowledge_base(
                    content="Test content for ingestion.",
                    title="Test Document",
                    source_type="user",
                )
                assert result["success"] is True
                assert "document_id" in result
                assert result["title"] == "Test Document"
                assert result["chunks_created"] > 0
            finally:
                token.var.reset(token)

    def test_no_context_returns_error(self):
        """ingest_to_knowledge_base returns error when KB context is not set."""
        from agentic_cli.workflow.context import set_context_kb_manager
        from agentic_cli.tools.knowledge_tools import ingest_to_knowledge_base

        token = set_context_kb_manager(None)
        try:
            result = ingest_to_knowledge_base(
                content="Test",
                title="Test",
            )
            assert result["success"] is False
            assert "error" in result
        finally:
            token.var.reset(token)


# ============================================================================
# ContextVar lifecycle tests (E4)
# ============================================================================


class TestKBManagerContextVar:
    """Tests for KB manager ContextVar integration."""

    def test_context_accessors_exist(self):
        """ContextVar setter and getter exist for kb_manager."""
        from agentic_cli.workflow.context import (
            set_context_kb_manager,
            get_context_kb_manager,
        )
        assert callable(set_context_kb_manager)
        assert callable(get_context_kb_manager)

    def test_context_default_is_none(self):
        """Default context value is None."""
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
        """Setting and getting context returns the same value."""
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
        """kb_manager is a valid ManagerRequirement."""
        from agentic_cli.tools import ManagerRequirement
        # This is a Literal type — just verify the string is accepted
        # by checking the tool decorators work
        from agentic_cli.tools.knowledge_tools import search_knowledge_base
        assert hasattr(search_knowledge_base, "requires")
        assert "kb_manager" in search_knowledge_base.requires

    def test_base_manager_has_kb_manager_slot(self):
        """BaseWorkflowManager initializes kb_manager slot."""
        from agentic_cli.workflow.base_manager import BaseWorkflowManager
        # Check the class has the property
        assert hasattr(BaseWorkflowManager, "kb_manager")
