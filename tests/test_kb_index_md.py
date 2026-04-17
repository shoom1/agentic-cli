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
        d1 = kb.ingest_document(content="a", title="Keep Me", source_type=SourceType.USER)
        d2 = kb.ingest_document(content="b", title="Delete Me", source_type=SourceType.USER)
        kb.delete_document(d2.id)
        text = (kb.kb_dir / "index.md").read_text()
        assert "Keep Me" in text
        assert "Delete Me" not in text
        assert "1 documents" in text
