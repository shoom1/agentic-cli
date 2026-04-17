"""Tests for index.md auto-maintenance."""

import pytest

from agentic_cli.knowledge_base.models import SourceType
from tests.test_knowledge_tools import _make_kb


class TestIndexMdRebuild:
    @pytest.fixture
    def kb(self, tmp_path):
        return _make_kb(tmp_path)

    def test_index_md_created_on_first_ingest(self, kb):
        doc = kb.ingest_document(
            content="body",
            title="First Doc",
            source_type=SourceType.USER,
        )
        index_path = kb.kb_dir / "index.md"
        assert index_path.exists()
        text = index_path.read_text()
        assert "# Knowledge Base Index" in text
        assert "1 documents" in text
        assert "## user (1)" in text
        assert "First Doc" in text

    def test_index_md_groups_by_source_type(self, kb):
        kb.ingest_document(content="a", title="Arxiv One", source_type=SourceType.ARXIV)
        kb.ingest_document(content="b", title="Web One", source_type=SourceType.WEB)
        kb.ingest_document(content="c", title="Arxiv Two", source_type=SourceType.ARXIV)
        text = (kb.kb_dir / "index.md").read_text()
        assert "## arxiv (2)" in text
        assert "## web (1)" in text
        assert "Arxiv One" in text
        assert "Arxiv Two" in text
        assert "Web One" in text

    def test_index_md_removes_entry_on_delete(self, kb):
        kb.ingest_document(content="a", title="Keep Me", source_type=SourceType.USER)
        d2 = kb.ingest_document(content="b", title="Delete Me", source_type=SourceType.USER)
        kb.delete_document(d2.id)
        text = (kb.kb_dir / "index.md").read_text()
        assert "Keep Me" in text
        assert "Delete Me" not in text
        assert "1 documents" in text


class TestIndexMdLoadTimeMaterialization:
    """Verify the legacy-KB code path: load_metadata materializes index.md
    when documents exist but the file is absent."""

    def test_load_materializes_missing_index_md(self, tmp_path):
        from agentic_cli.knowledge_base.manager import KnowledgeBaseManager

        # First manager: ingest, confirm index.md exists, then delete it
        kb = _make_kb(tmp_path)
        kb.ingest_document(
            content="legacy body",
            title="Legacy Doc",
            source_type=SourceType.USER,
        )
        index_path = kb.kb_dir / "index.md"
        assert index_path.exists()
        index_path.unlink()
        assert not index_path.exists()

        # Second manager pointing at the same dir: load should re-create it
        kb2 = KnowledgeBaseManager.__new__(KnowledgeBaseManager)
        kb2._lock = __import__("threading").Lock()
        kb2._settings = None
        kb2._use_mock = True
        kb2.kb_dir = kb.kb_dir
        kb2.documents_dir = kb.documents_dir
        kb2.embeddings_dir = kb.embeddings_dir
        kb2.files_dir = kb.files_dir
        kb2.metadata_path = kb.metadata_path
        kb2._embedding_service = kb._embedding_service
        kb2._vector_store = kb._vector_store
        kb2._documents = {}
        kb2._chunks = {}
        kb2._load_metadata()

        assert index_path.exists()
        text = index_path.read_text()
        assert "Legacy Doc" in text
        assert "1 documents" in text
