"""Tool-layer tests for kb_write_concept and kb_search_concepts."""

import pytest


class TestKbWriteConcept:
    @pytest.fixture
    def kb(self, tmp_path):
        from tests.test_knowledge_tools import _make_kb
        return _make_kb(tmp_path)

    async def test_write_creates_concept_citing_existing_doc(self, kb):
        from agentic_cli.knowledge_base.models import SourceType
        from agentic_cli.tools.knowledge_tools import _write_concept_with_kb

        d = kb.ingest_document(
            content="body", title="Paper", source_type=SourceType.ARXIV,
        )
        result = await _write_concept_with_kb(
            kb, None,
            title="Diffusion Models",
            body="Synthesis of what the KB knows.",
            sources=[d.id],
        )

        assert result["success"] is True
        assert result["action"] == "created"
        assert result["slug"] == "diffusion-models"
        concept_path = kb.kb_dir / "concepts" / "diffusion-models.md"
        assert concept_path.exists()

    async def test_write_drops_invalid_source_ids(self, kb):
        from agentic_cli.knowledge_base.models import SourceType
        from agentic_cli.tools.knowledge_tools import _write_concept_with_kb

        d = kb.ingest_document(
            content="body", title="Paper", source_type=SourceType.ARXIV,
        )
        result = await _write_concept_with_kb(
            kb, None,
            title="X",
            body="b",
            sources=[d.id, "not-a-real-id", "also-fake"],
        )

        assert result["success"] is True
        assert result["invalid_sources"] == ["not-a-real-id", "also-fake"]

    async def test_write_fails_when_all_sources_invalid(self, kb):
        from agentic_cli.tools.knowledge_tools import _write_concept_with_kb

        result = await _write_concept_with_kb(
            kb, None,
            title="X", body="b", sources=["fake-1", "fake-2"],
        )

        assert result["success"] is False
        assert "at least one valid source" in result["error"]

    async def test_write_overwrites_explicit_slug(self, kb):
        from agentic_cli.knowledge_base.models import SourceType
        from agentic_cli.tools.knowledge_tools import _write_concept_with_kb

        d = kb.ingest_document(content="body", title="P", source_type=SourceType.ARXIV)
        await _write_concept_with_kb(
            kb, None, title="T", body="v1", sources=[d.id], slug="x",
        )
        result = await _write_concept_with_kb(
            kb, None, title="T", body="v2", sources=[d.id], slug="x",
        )
        assert result["action"] == "updated"

    async def test_write_empty_sources_fails(self, kb):
        from agentic_cli.tools.knowledge_tools import _write_concept_with_kb
        result = await _write_concept_with_kb(
            kb, None, title="X", body="b", sources=[],
        )
        assert result["success"] is False
