"""Tests for how the knowledge base stores documents: the database, original
files, and the summary written on ingest. (Folders from older versions:
test_kb_import_06.py.)"""

from pathlib import Path

import pytest

from agentic_cli.memory.kb import KnowledgeBaseManager, SourceType
from tests.memory.kb_storage import read_rows


# ============================================================================
# KnowledgeBaseManager persistence tests (E5)
# ============================================================================


class TestKBManagerStorage:
    """The database holds each document: its header, its text and its chunks."""

    @pytest.fixture
    def kb(self, tmp_path):
        return KnowledgeBaseManager(tmp_path / "kb", use_mock=True)

    def test_ingest_stores_the_text_and_chunks_in_the_database(self, kb):
        doc = kb.ingest_document(
            content="Hello world. This is a test.",
            title="Test Doc",
            source_type=SourceType.USER,
        )

        assert read_rows(kb.kb_dir, "SELECT content FROM documents WHERE id = ?", [doc.id]) == [
            ("Hello world. This is a test.",)
        ]
        assert read_rows(kb.kb_dir, "SELECT count(*) FROM chunks WHERE document_id = ?", [doc.id])[0][0] > 0
        assert not (kb.documents_dir / f"{doc.id}.json").exists()

    def test_a_listing_carries_no_text_or_chunks(self, kb):
        """The listing reads headers only: no session keeps every text in memory."""
        kb.ingest_document(
            content="Some test content for the knowledge base.",
            title="Test",
            source_type=SourceType.USER,
        )

        [listed] = kb.list_documents()
        assert listed.content == "" and listed.chunks == []
        assert listed.chunk_count == 1

    def test_the_header_is_stored(self, kb):
        doc = kb.ingest_document(
            content="Test content.",
            title="My Document",
            source_type=SourceType.ARXIV,
            source_url="https://arxiv.org/abs/1234",
        )

        [header] = read_rows(
            kb.kb_dir, "SELECT id, title, source_type, source_url, created_at FROM documents"
        )
        assert header[:4] == (doc.id, "My Document", "arxiv", "https://arxiv.org/abs/1234")
        assert header[4] == doc.created_at.isoformat()

    def test_the_summary_is_stored(self, kb):
        doc = kb.ingest_document(
            content="Test content for summary generation.",
            title="Summary Doc",
            source_type=SourceType.USER,
        )

        assert doc.summary
        assert read_rows(kb.kb_dir, "SELECT summary FROM documents") == [(doc.summary,)]

    def test_load_roundtrip(self, kb):
        """Documents survive a reopen."""
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

    def test_delete_removes_the_rows(self, kb):
        doc = kb.ingest_document(
            content="Delete me.",
            title="To Delete",
            source_type=SourceType.USER,
        )

        kb.delete_document(doc.id)

        assert read_rows(kb.kb_dir, "SELECT count(*) FROM documents") == [(0,)]
        assert read_rows(kb.kb_dir, "SELECT count(*) FROM chunks") == [(0,)]

    def test_clear_removes_every_row(self, kb):
        kb.ingest_document(content="Doc 1.", title="D1", source_type=SourceType.USER)
        kb.ingest_document(content="Doc 2.", title="D2", source_type=SourceType.USER)

        kb.clear()

        assert read_rows(kb.kb_dir, "SELECT count(*) FROM documents") == [(0,)]
        assert read_rows(kb.kb_dir, "SELECT count(*) FROM chunks") == [(0,)]


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
