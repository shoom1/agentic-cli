"""Tests for ingest_log.md."""

import pytest

from agentic_cli.knowledge_base.models import SourceType
from tests.test_knowledge_tools import _make_kb


class TestIngestLog:
    @pytest.fixture
    def kb(self, tmp_path):
        return _make_kb(tmp_path)

    def test_log_appends_one_line_per_ingest(self, kb):
        kb.ingest_document(content="a", title="One", source_type=SourceType.USER)
        kb.ingest_document(content="b", title="Two", source_type=SourceType.WEB)
        log = (kb.kb_dir / "ingest_log.md").read_text()
        lines = [ln for ln in log.splitlines() if ln.startswith("- ")]
        assert len(lines) == 2
        assert "ingest" in lines[0]
        assert "user" in lines[0]
        assert "One" in lines[0]
        assert "ingest" in lines[1]
        assert "web" in lines[1]
        assert "Two" in lines[1]

    def test_log_appends_delete_line(self, kb):
        d = kb.ingest_document(content="a", title="Doomed", source_type=SourceType.USER)
        kb.delete_document(d.id)
        log = (kb.kb_dir / "ingest_log.md").read_text()
        lines = [ln for ln in log.splitlines() if ln.startswith("- ")]
        assert len(lines) == 2
        assert "delete" in lines[1]

    def test_log_never_rewritten(self, kb):
        kb.ingest_document(content="a", title="One", source_type=SourceType.USER)
        log_path = kb.kb_dir / "ingest_log.md"
        first = log_path.read_text()
        kb.ingest_document(content="b", title="Two", source_type=SourceType.USER)
        second = log_path.read_text()
        assert second.startswith(first)

    def test_log_includes_arxiv_id_when_present(self, kb):
        kb.ingest_document(
            content="abstract",
            title="Attention Is All You Need",
            source_type=SourceType.ARXIV,
            metadata={"arxiv_id": "1706.03762"},
        )
        log = (kb.kb_dir / "ingest_log.md").read_text()
        line = next(ln for ln in log.splitlines() if ln.startswith("- "))
        assert "1706.03762" in line
        assert "arxiv" in line
        assert "Attention Is All You Need" in line
