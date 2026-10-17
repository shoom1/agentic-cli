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


class TestKbReadLazySidecar:
    @pytest.fixture
    def kb(self, tmp_path):
        return _make_kb(tmp_path)

    async def test_kb_read_default_returns_sidecar_payload(self, kb):
        from agentic_cli.memory.kb.models import SourceType
        from agentic_cli.tools.knowledge_tools import _read_document_from_kbs

        doc = kb.ingest_document(
            content="raw body", title="X", source_type=SourceType.USER,
        )
        result = await _read_document_from_kbs(kb, None, doc.id)
        assert result["success"] is True
        assert result["full"] is False
        assert "summary" in result
        # Raw text should NOT be in the default response
        assert "content" not in result or not result.get("content")

    async def test_kb_read_full_returns_raw_text(self, kb):
        from agentic_cli.memory.kb.models import SourceType
        from agentic_cli.tools.knowledge_tools import _read_document_from_kbs

        doc = kb.ingest_document(
            content="raw body content", title="X", source_type=SourceType.USER,
        )
        result = await _read_document_from_kbs(kb, None, doc.id, full=True)
        assert result["full"] is True
        assert result["content"] == "raw body content"

    async def test_kb_read_lazily_generates_missing_sidecar(self, kb):
        from agentic_cli.memory.kb.models import SourceType
        from agentic_cli.tools.knowledge_tools import _read_document_from_kbs

        doc = kb.ingest_document(
            content="body", title="Y", source_type=SourceType.USER,
        )
        # Simulate legacy doc: remove sidecar
        kb.sidecar_path(doc.id).unlink()
        assert not kb.sidecar_path(doc.id).exists()

        result = await _read_document_from_kbs(kb, None, doc.id)
        assert result["success"] is True
        # Sidecar should have been generated
        assert kb.sidecar_path(doc.id).exists()

    async def test_kb_read_concurrent_reads_serialize(self, kb):
        """Two concurrent first-reads on the same doc must not double-LLM."""
        import asyncio
        from agentic_cli.memory.kb.models import SourceType
        from agentic_cli.tools.knowledge_tools import _read_document_from_kbs

        call_count = {"n": 0}

        class CountingSummarizer:
            async def summarize(self, content, prompt):
                call_count["n"] += 1
                await asyncio.sleep(0.05)  # simulate latency
                return "SUMMARY: lazy summary."

        doc = kb.ingest_document(content="body", title="Z", source_type=SourceType.USER)
        kb.sidecar_path(doc.id).unlink()

        kb._summarizer = CountingSummarizer()
        await asyncio.gather(
            _read_document_from_kbs(kb, None, doc.id),
            _read_document_from_kbs(kb, None, doc.id),
        )

        assert call_count["n"] == 1, "lock did not serialize concurrent first-reads"

    async def test_kb_read_lazy_gen_extracts_pdf_when_content_empty(self, kb, tmp_path):
        """Lazy sidecar gen for a legacy PDF doc (empty doc.content) should
        extract text from the file rather than caching a useless empty sidecar."""
        from agentic_cli.memory.kb.models import SourceType
        from agentic_cli.tools.knowledge_tools import _read_document_from_kbs
        from unittest.mock import patch

        # Ingest a PDF doc with content; remove the sidecar; null out doc.content
        # to simulate a legacy doc that didn't preserve text.
        fake_pdf_bytes = b"%PDF-1.4 fake"
        doc = kb.ingest_document(
            content="will-be-cleared",
            title="Legacy PDF",
            source_type=SourceType.USER,
            file_bytes=fake_pdf_bytes,
            file_extension=".pdf",
        )
        kb.sidecar_path(doc.id).unlink()
        kb._documents[doc.id].content = ""  # legacy: no in-memory content

        captured = {}

        class CapturingSummarizer:
            async def summarize(self, content, prompt):
                captured["content"] = content
                return "SUMMARY: extracted from pdf."

        kb._summarizer = CapturingSummarizer()
        with patch(
            "agentic_cli.tools.pdf_utils.extract_pdf_text",
            return_value="EXTRACTED PDF TEXT",
        ):
            result = await _read_document_from_kbs(kb, None, doc.id)

        assert result["success"] is True
        assert kb.sidecar_path(doc.id).exists()
        # The PDF text should have been the input to the LLM
        assert "EXTRACTED PDF TEXT" in captured["content"]

    async def test_kb_read_lazy_path_uses_accessor(self, kb):
        """The lazy sidecar generation path in knowledge_tools must go
        through the public accessor, not reach into `_sidecar_locks`
        directly.
        """
        from agentic_cli.tools.knowledge_tools import _read_document_from_kbs
        from unittest.mock import patch

        doc = kb.ingest_document(
            content="body", title="X", source_type=SourceType.USER,
        )
        kb.sidecar_path(doc.id).unlink()

        original = kb.get_or_create_sidecar_lock
        seen: list[str] = []

        def spy(doc_id: str):
            seen.append(doc_id)
            return original(doc_id)

        with patch.object(kb, "get_or_create_sidecar_lock", side_effect=spy):
            await _read_document_from_kbs(kb, None, doc.id)

        assert doc.id in seen


class TestDeleteCleansLockDict:
    @pytest.fixture
    def kb(self, tmp_path):
        return _make_kb(tmp_path)

    async def test_delete_removes_lock_entry(self, kb):
        from agentic_cli.tools.knowledge_tools import _read_document_from_kbs

        d = kb.ingest_document(content="body", title="X", source_type=SourceType.USER)
        # Force a lazy-read to populate the lock dict
        kb.sidecar_path(d.id).unlink()
        await _read_document_from_kbs(kb, None, d.id)
        assert d.id in kb._sidecar_locks

        kb.delete_document(d.id)
        assert d.id not in kb._sidecar_locks
