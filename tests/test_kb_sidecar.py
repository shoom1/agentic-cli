"""Tests for the knowledge-base tools' use of sidecars (kb_ingest_*, kb_read).

The knowledge base's own sidecar behavior is tested in
``tests/memory/test_kb_sidecar.py``.
"""

import pytest

from agentic_cli.memory.kb.models import SourceType
from tests.test_knowledge_tools import _make_kb


class TestSidecarWrittenOnIngest:
    @pytest.fixture
    def kb(self, tmp_path):
        return _make_kb(tmp_path)

    async def test_ingest_writes_sidecar_with_llm_payload(self, kb):
        from agentic_cli.tools.knowledge_tools import _ingest_text_with_kb

        class FakeSummarizer:
            async def summarize(self, content: str, prompt: str) -> str:
                return (
                    "SUMMARY: A short summary.\n"
                    "CLAIMS:\n- Claim one.\n- Claim two.\n"
                    "ENTITIES:\nModels: Foo\n"
                )

        kb._summarizer = FakeSummarizer()
        result = await _ingest_text_with_kb(
            kb, content="body", title="Test", source_type="user",
        )

        assert result["success"] is True
        sidecar_path = kb.documents_dir / f"{result['document_id']}.md"
        assert sidecar_path.exists()
        text = sidecar_path.read_text()
        assert "## Summary" in text
        assert "A short summary." in text
        assert "## Key Claims" in text
        assert "- Claim one." in text
        assert "## Key Entities" in text
        assert "Models: Foo" in text

    async def test_empty_summary_in_payload_falls_back_to_truncation(self, kb):
        """If the LLM returns CLAIMS but no SUMMARY:, doc.summary should
        fall back to truncated content rather than being persisted as ''."""
        from agentic_cli.tools.knowledge_tools import _ingest_text_with_kb

        class ClaimsOnlySummarizer:
            async def summarize(self, content: str, prompt: str) -> str:
                return "CLAIMS:\n- one\n- two\n"  # no SUMMARY: line

        kb._summarizer = ClaimsOnlySummarizer()
        result = await _ingest_text_with_kb(
            kb, content="raw body for fallback", title="X", source_type="user",
        )

        assert result["success"] is True
        doc = kb.get_document(result["document_id"])
        # Summary should be the truncated body, not empty string
        assert doc.summary == "raw body for fallback"

