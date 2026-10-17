"""Tests for how the knowledge base stores documents: the v3 layout, migration
from older layouts, original files, and the summary written on ingest."""

import json
from pathlib import Path

import pytest

from agentic_cli.memory.kb import KnowledgeBaseManager, SourceType
from agentic_cli.memory.kb.manager import _FORMAT_VERSION


# ============================================================================
# KnowledgeBaseManager persistence tests (E5)
# ============================================================================


class TestKBManagerPersistenceV2:
    """Tests for v2/v3 persistence format (content in per-document files)."""

    @pytest.fixture
    def kb(self, tmp_path):
        return KnowledgeBaseManager(tmp_path / "kb", use_mock=True)

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
        kb2 = KnowledgeBaseManager(kb.kb_dir, use_mock=True)

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

    def test_v1_loaded_correctly(self, kb_dir):
        """V1 format loads documents with content."""
        self._make_v1_metadata(kb_dir)
        kb = KnowledgeBaseManager(kb_dir, use_mock=True)

        doc = kb.get_document("doc-v1-001")
        assert doc is not None
        assert doc.title == "V1 Document"
        assert doc.content == "This is old-format content."
        assert len(doc.chunks) == 1
        assert doc.chunks[0].content == "This is old-format content."

    def test_v1_auto_migrated_to_v3(self, kb_dir):
        """V1 format is auto-migrated to v3 on load."""
        self._make_v1_metadata(kb_dir)
        kb = KnowledgeBaseManager(kb_dir, use_mock=True)

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

        kb = KnowledgeBaseManager(kb_dir, use_mock=True)

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
        return KnowledgeBaseManager(tmp_path / "kb", use_mock=True)

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
# Summary on ingest
# ============================================================================


class TestKBSummaryGeneration:
    """Tests for summary generation in KnowledgeBaseManager."""

    @pytest.fixture
    def kb(self, tmp_path):
        return KnowledgeBaseManager(tmp_path / "kb", use_mock=True)

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
