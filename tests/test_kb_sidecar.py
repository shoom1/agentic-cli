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
