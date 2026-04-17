"""Tests for the ConceptStore — pure file ops for concept pages."""

from datetime import datetime

import pytest


class TestSlugFromTitle:
    def test_basic_title(self):
        from agentic_cli.knowledge_base.concepts import slug_from_title
        assert slug_from_title("Diffusion Models") == "diffusion-models"

    def test_strips_punctuation(self):
        from agentic_cli.knowledge_base.concepts import slug_from_title
        assert slug_from_title("What is RAG?") == "what-is-rag"

    def test_collapses_whitespace(self):
        from agentic_cli.knowledge_base.concepts import slug_from_title
        assert slug_from_title("   Multi\tWord\n  Title  ") == "multi-word-title"

    def test_strips_non_ascii(self):
        from agentic_cli.knowledge_base.concepts import slug_from_title
        assert slug_from_title("Café résumé naïve") == "cafe-resume-naive"

    def test_truncates_at_80_chars(self):
        from agentic_cli.knowledge_base.concepts import slug_from_title
        long = "word " * 50
        slug = slug_from_title(long)
        assert len(slug) <= 80
        # Must not end with a trailing dash from mid-word truncation
        assert not slug.endswith("-")

    def test_empty_title_returns_untitled(self):
        from agentic_cli.knowledge_base.concepts import slug_from_title
        assert slug_from_title("") == "untitled"
        assert slug_from_title("!!!") == "untitled"


class TestRenderConceptMarkdown:
    def test_renders_frontmatter_and_body(self):
        from agentic_cli.knowledge_base.concepts import render_concept_markdown

        md = render_concept_markdown(
            slug="diffusion-models",
            title="Diffusion Models",
            body="## What it is\nSome synthesis text.\n",
            sources=["doc-id-a", "doc-id-b"],
            created_at=datetime(2026, 4, 17, 10, 0, 0),
            updated_at=datetime(2026, 4, 17, 10, 5, 0),
        )

        assert md.startswith("---\n")
        assert "slug: diffusion-models" in md
        assert "title: Diffusion Models" in md
        assert "created_at: 2026-04-17T10:00:00" in md
        assert "updated_at: 2026-04-17T10:05:00" in md
        assert "sources: [doc-id-a, doc-id-b]" in md
        # Body follows frontmatter
        assert "## What it is" in md
        assert "Some synthesis text." in md


class TestParseConceptMarkdown:
    def test_round_trips(self):
        from agentic_cli.knowledge_base.concepts import (
            render_concept_markdown,
            parse_concept_markdown,
        )

        md = render_concept_markdown(
            slug="diffusion-models",
            title="Diffusion Models",
            body="Body text here.\n\nSecond paragraph.\n",
            sources=["a", "b"],
            created_at=datetime(2026, 4, 17, 10, 0, 0),
            updated_at=datetime(2026, 4, 17, 10, 5, 0),
        )

        parsed = parse_concept_markdown(md)
        assert parsed["slug"] == "diffusion-models"
        assert parsed["title"] == "Diffusion Models"
        assert parsed["sources"] == ["a", "b"]
        assert "Body text here." in parsed["body"]
        assert "Second paragraph." in parsed["body"]

    def test_empty_sources(self):
        from agentic_cli.knowledge_base.concepts import (
            render_concept_markdown,
            parse_concept_markdown,
        )

        md = render_concept_markdown(
            slug="x", title="X", body="body", sources=[],
            created_at=datetime(2026, 4, 17, 10, 0, 0),
            updated_at=datetime(2026, 4, 17, 10, 0, 0),
        )
        parsed = parse_concept_markdown(md)
        assert parsed["sources"] == []

    def test_missing_frontmatter_returns_none(self):
        from agentic_cli.knowledge_base.concepts import parse_concept_markdown
        assert parse_concept_markdown("# Just a heading\n\nbody") is None
