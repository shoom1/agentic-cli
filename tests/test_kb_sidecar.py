"""Tests for sidecar formatting helpers."""

from datetime import datetime
from pathlib import Path

from agentic_cli.knowledge_base.models import Document, SourceType


def _make_doc(**overrides):
    base = dict(
        title="Attention Is All You Need",
        content="full body text",
        source_type=SourceType.ARXIV,
        summary="Self-attention is sufficient for sequence transduction.",
        source_url="https://arxiv.org/abs/1706.03762",
        metadata={"authors": ["Vaswani", "Shazeer"], "arxiv_id": "1706.03762"},
    )
    base.update(overrides)
    doc = Document.create(**base)
    doc.created_at = datetime(2026, 4, 16, 10, 23, 0)
    doc.updated_at = doc.created_at
    return doc


class TestRenderSidecarMarkdown:
    def test_renders_yaml_frontmatter_then_body(self):
        from agentic_cli.knowledge_base.sidecar import render_sidecar_markdown

        doc = _make_doc()
        payload = {
            "summary": "Self-attention is sufficient for sequence transduction.",
            "claims": ["Self-attention alone suffices.", "28.4 BLEU on WMT14 En-De."],
            "entities": {
                "Models": ["Transformer"],
                "Datasets": ["WMT14"],
            },
        }

        md = render_sidecar_markdown(doc, payload)

        assert md.startswith("---\n")
        assert "id: " + doc.id in md
        assert "title: Attention Is All You Need" in md
        assert "source_type: arxiv" in md
        assert "arxiv_id: 1706.03762" in md
        assert "ingested_at: 2026-04-16T10:23:00" in md
        # Body sections
        assert "## Summary" in md
        assert "Self-attention is sufficient" in md
        assert "## Key Claims" in md
        assert "- Self-attention alone suffices." in md
        assert "## Key Entities" in md
        assert "Models: Transformer" in md

    def test_summary_only_when_payload_missing_claims_and_entities(self):
        from agentic_cli.knowledge_base.sidecar import render_sidecar_markdown

        doc = _make_doc()
        md = render_sidecar_markdown(doc, {"summary": doc.summary, "claims": [], "entities": {}})

        assert "## Summary" in md
        assert doc.summary in md
        assert "## Key Claims" not in md
        assert "## Key Entities" not in md


class TestParseSidecarFrontmatter:
    def test_round_trips_with_render(self):
        from agentic_cli.knowledge_base.sidecar import (
            render_sidecar_markdown,
            parse_sidecar_frontmatter,
        )

        doc = _make_doc()
        md = render_sidecar_markdown(doc, {"summary": doc.summary, "claims": [], "entities": {}})

        fm = parse_sidecar_frontmatter(md)
        assert fm["id"] == doc.id
        assert fm["title"] == "Attention Is All You Need"
        assert fm["source_type"] == "arxiv"

    def test_returns_empty_dict_when_no_frontmatter(self):
        from agentic_cli.knowledge_base.sidecar import parse_sidecar_frontmatter
        assert parse_sidecar_frontmatter("# Just a heading\n\nbody") == {}


import pytest

from tests.test_knowledge_tools import _make_kb


class TestGenerateSidecarPayload:
    @pytest.fixture
    def kb(self, tmp_path):
        return _make_kb(tmp_path)

    async def test_returns_structured_payload_from_llm(self, kb):
        from agentic_cli.workflow.service_registry import (
            set_service_registry,
            LLM_SUMMARIZER,
        )

        class FakeSummarizer:
            async def summarize(self, content: str, prompt: str) -> str:
                return (
                    "SUMMARY: Self-attention suffices.\n"
                    "CLAIMS:\n"
                    "- Self-attention alone is enough.\n"
                    "- 28.4 BLEU on WMT14.\n"
                    "ENTITIES:\n"
                    "Models: Transformer\n"
                    "Datasets: WMT14\n"
                )

        token = set_service_registry({LLM_SUMMARIZER: FakeSummarizer()})
        try:
            payload = await kb.generate_sidecar_payload(
                "Long body text here.",
                title="Attention Is All You Need",
            )
        finally:
            token.var.reset(token)

        assert payload["summary"] == "Self-attention suffices."
        assert "Self-attention alone is enough." in payload["claims"]
        assert "28.4 BLEU on WMT14." in payload["claims"]
        assert payload["entities"]["Models"] == ["Transformer"]
        assert payload["entities"]["Datasets"] == ["WMT14"]

    async def test_falls_back_when_no_summarizer(self, kb):
        payload = await kb.generate_sidecar_payload("body text", title="t")
        assert payload["summary"] == "body text"
        assert payload["claims"] == []
        assert payload["entities"] == {}

    async def test_falls_back_when_summarizer_errors(self, kb):
        from agentic_cli.workflow.service_registry import (
            set_service_registry,
            LLM_SUMMARIZER,
        )

        class BoomSummarizer:
            async def summarize(self, content: str, prompt: str) -> str:
                raise RuntimeError("boom")

        token = set_service_registry({LLM_SUMMARIZER: BoomSummarizer()})
        try:
            payload = await kb.generate_sidecar_payload("body text", title="t")
        finally:
            token.var.reset(token)
        assert payload["summary"] == "body text"
        assert payload["claims"] == []
        assert payload["entities"] == {}

    async def test_parser_tolerates_whitespace_and_bullet_prefixes(self, kb):
        from agentic_cli.workflow.service_registry import (
            set_service_registry,
            LLM_SUMMARIZER,
        )

        class WonkySummarizer:
            async def summarize(self, content: str, prompt: str) -> str:
                # Leading whitespace on headers; bullet-prefixed entity lines.
                return (
                    "  SUMMARY: ok.\n"
                    "  CLAIMS:\n"
                    "  - one\n"
                    "  ENTITIES:\n"
                    "- Models: A, B\n"
                )

        token = set_service_registry({LLM_SUMMARIZER: WonkySummarizer()})
        try:
            payload = await kb.generate_sidecar_payload("body", title="t")
        finally:
            token.var.reset(token)

        assert payload["summary"] == "ok."
        assert payload["claims"] == ["one"]
        assert payload["entities"] == {"Models": ["A", "B"]}


class TestSidecarWrittenOnIngest:
    @pytest.fixture
    def kb(self, tmp_path):
        return _make_kb(tmp_path)

    async def test_ingest_writes_sidecar_with_llm_payload(self, kb):
        from agentic_cli.workflow.service_registry import (
            set_service_registry,
            LLM_SUMMARIZER,
        )
        from agentic_cli.tools.knowledge_tools import _ingest_document_with_kb

        class FakeSummarizer:
            async def summarize(self, content: str, prompt: str) -> str:
                return (
                    "SUMMARY: A short summary.\n"
                    "CLAIMS:\n- Claim one.\n- Claim two.\n"
                    "ENTITIES:\nModels: Foo\n"
                )

        token = set_service_registry({LLM_SUMMARIZER: FakeSummarizer()})
        try:
            result = await _ingest_document_with_kb(
                kb, content="body", title="Test", source_type="user",
            )
        finally:
            token.var.reset(token)

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

    def test_direct_ingest_writes_sidecar_with_truncation_fallback(self, kb):
        # Sync direct path: no summarizer, no payload — sidecar still written
        # with summary-only content.
        doc = kb.ingest_document(
            content="raw body content",
            title="Direct Test",
            source_type=SourceType.USER,
        )
        sidecar_path = kb.documents_dir / f"{doc.id}.md"
        assert sidecar_path.exists()
        assert "## Summary" in sidecar_path.read_text()

    async def test_empty_summary_in_payload_falls_back_to_truncation(self, kb):
        """If the LLM returns CLAIMS but no SUMMARY:, doc.summary should
        fall back to truncated content rather than being persisted as ''."""
        from agentic_cli.workflow.service_registry import (
            set_service_registry,
            LLM_SUMMARIZER,
        )
        from agentic_cli.tools.knowledge_tools import _ingest_document_with_kb

        class ClaimsOnlySummarizer:
            async def summarize(self, content: str, prompt: str) -> str:
                return "CLAIMS:\n- one\n- two\n"  # no SUMMARY: line

        token = set_service_registry({LLM_SUMMARIZER: ClaimsOnlySummarizer()})
        try:
            result = await _ingest_document_with_kb(
                kb, content="raw body for fallback", title="X", source_type="user",
            )
        finally:
            token.var.reset(token)

        assert result["success"] is True
        doc = kb.get_document(result["document_id"])
        # Summary should be the truncated body, not empty string
        assert doc.summary == "raw body for fallback"


class TestBackfillSidecars:
    @pytest.fixture
    def kb(self, tmp_path):
        return _make_kb(tmp_path)

    def test_backfill_writes_missing_sidecars(self, kb):
        d1 = kb.ingest_document(content="a", title="One", source_type=SourceType.USER)
        d2 = kb.ingest_document(content="b", title="Two", source_type=SourceType.USER)
        kb._sidecar_path(d1.id).unlink()
        kb._sidecar_path(d2.id).unlink()

        import asyncio
        n = asyncio.run(kb.backfill_sidecars())

        assert n == 2
        assert kb._sidecar_path(d1.id).exists()
        assert kb._sidecar_path(d2.id).exists()

    def test_backfill_skips_existing(self, kb):
        d1 = kb.ingest_document(content="a", title="Has Sidecar", source_type=SourceType.USER)
        # Mtime sentinel
        sidecar = kb._sidecar_path(d1.id)
        original = sidecar.read_text()

        import asyncio
        n = asyncio.run(kb.backfill_sidecars())

        assert n == 0
        assert sidecar.read_text() == original

    async def test_backfill_serializes_with_per_doc_lock(self, kb):
        """Two concurrent backfills must not double-LLM the same doc."""
        import asyncio
        from agentic_cli.workflow.service_registry import (
            set_service_registry,
            LLM_SUMMARIZER,
        )

        d1 = kb.ingest_document(content="body one", title="One", source_type=SourceType.USER)
        kb._sidecar_path(d1.id).unlink()

        call_count = {"n": 0}

        class CountingSummarizer:
            async def summarize(self, content, prompt):
                call_count["n"] += 1
                await asyncio.sleep(0.05)
                return "SUMMARY: ok."

        token = set_service_registry({LLM_SUMMARIZER: CountingSummarizer()})
        try:
            results = await asyncio.gather(
                kb.backfill_sidecars(),
                kb.backfill_sidecars(),
            )
        finally:
            token.var.reset(token)

        # One backfill writes 1; the other sees the file already exists
        # (either via the outer .exists() check or the inner re-check) and
        # writes 0. Sum must be exactly 1.
        assert sum(results) == 1
        assert call_count["n"] == 1
        assert kb._sidecar_path(d1.id).exists()


class TestKbReadLazySidecar:
    @pytest.fixture
    def kb(self, tmp_path):
        return _make_kb(tmp_path)

    async def test_kb_read_default_returns_sidecar_payload(self, kb):
        from agentic_cli.knowledge_base.models import SourceType
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
        from agentic_cli.knowledge_base.models import SourceType
        from agentic_cli.tools.knowledge_tools import _read_document_from_kbs

        doc = kb.ingest_document(
            content="raw body content", title="X", source_type=SourceType.USER,
        )
        result = await _read_document_from_kbs(kb, None, doc.id, full=True)
        assert result["full"] is True
        assert result["content"] == "raw body content"

    async def test_kb_read_lazily_generates_missing_sidecar(self, kb):
        from agentic_cli.knowledge_base.models import SourceType
        from agentic_cli.tools.knowledge_tools import _read_document_from_kbs

        doc = kb.ingest_document(
            content="body", title="Y", source_type=SourceType.USER,
        )
        # Simulate legacy doc: remove sidecar
        kb._sidecar_path(doc.id).unlink()
        assert not kb._sidecar_path(doc.id).exists()

        result = await _read_document_from_kbs(kb, None, doc.id)
        assert result["success"] is True
        # Sidecar should have been generated
        assert kb._sidecar_path(doc.id).exists()

    async def test_kb_read_concurrent_reads_serialize(self, kb):
        """Two concurrent first-reads on the same doc must not double-LLM."""
        import asyncio
        from agentic_cli.knowledge_base.models import SourceType
        from agentic_cli.workflow.service_registry import (
            set_service_registry, LLM_SUMMARIZER,
        )
        from agentic_cli.tools.knowledge_tools import _read_document_from_kbs

        call_count = {"n": 0}

        class CountingSummarizer:
            async def summarize(self, content, prompt):
                call_count["n"] += 1
                await asyncio.sleep(0.05)  # simulate latency
                return "SUMMARY: lazy summary."

        doc = kb.ingest_document(content="body", title="Z", source_type=SourceType.USER)
        kb._sidecar_path(doc.id).unlink()

        token = set_service_registry({LLM_SUMMARIZER: CountingSummarizer()})
        try:
            await asyncio.gather(
                _read_document_from_kbs(kb, None, doc.id),
                _read_document_from_kbs(kb, None, doc.id),
            )
        finally:
            token.var.reset(token)

        assert call_count["n"] == 1, "lock did not serialize concurrent first-reads"
