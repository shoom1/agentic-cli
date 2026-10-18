"""End-to-end smoke test for the KB layered-hybrid flow.

Exercises, through the tools: ingest → sidecar written → read via kb_read
(both modes) → write a concept citing the doc → search concepts → find it.
"""

import pytest

from agentic_cli.memory.kb import KnowledgeBase
from agentic_cli.workflow.service_registry import set_service_registry


class TestKbResearchFlow:
    @pytest.fixture
    def kb(self, tmp_path):
        from tests.test_knowledge_tools import _make_kb
        return _make_kb(tmp_path)

    async def test_ingest_then_sidecar_then_concept_then_search(self, kb):
        from agentic_cli.tools.knowledge_tools import (
            kb_ingest_text,
            kb_read,
            kb_search_concepts,
            kb_write_concept,
        )

        class FakeSummarizer:
            async def summarize(self, content: str, prompt: str) -> str:
                return (
                    "SUMMARY:\n"
                    "A brief synthesis of the ingested paper about "
                    "diffusion-based chip placement.\n"
                    "\n"
                    "CLAIMS:\n"
                    "- Diffusion models can generate valid placements.\n"
                    "- Achieves 28 percent wirelength reduction.\n"
                    "ENTITIES:\n"
                    "Models: DiffPlace\n"
                    "Datasets: ICCAD 2015\n"
                )

        kb._summarizer = FakeSummarizer()
        token = set_service_registry({"knowledge_base": KnowledgeBase({"project": kb})})
        try:
            # Ingest a document — sidecar written eagerly
            ingest_result = await kb_ingest_text(
                content="A paper on diffusion models for chip placement.",
                title="DiffPlace",
                source_type="arxiv",
            )
            doc_id = ingest_result["document_id"]

            # Sidecar file present on disk
            assert (kb.documents_dir / f"{doc_id}.md").exists()

            # index.md and ingest_log.md materialized
            assert (kb.kb_dir / "index.md").exists()
            assert (kb.kb_dir / "ingest_log.md").exists()

            # kb_read default returns the sidecar payload
            read_default = await kb_read(doc_id)
            assert read_default["full"] is False
            assert "sidecar" in read_default
            assert "DiffPlace" in read_default["sidecar"]

            # kb_read full=True returns raw text
            read_full = await kb_read(doc_id, full=True)
            assert read_full["full"] is True
            assert "diffusion models" in read_full["content"]

            # Write a concept page citing this doc
            concept_result = await kb_write_concept(
                title="Diffusion-Based Placement",
                body="Agent synthesis tying this paper to the broader topic.",
                sources=[doc_id],
            )
            assert concept_result["success"] is True
            concept_slug = concept_result["slug"]
            assert (kb.kb_dir / "concepts" / f"{concept_slug}.md").exists()

            # Search concepts finds it
            search_result = await kb_search_concepts(query="diffusion")
            assert search_result["success"] is True
            assert search_result["count"] == 1
            assert search_result["concepts"][0]["slug"] == concept_slug
        finally:
            token.var.reset(token)
