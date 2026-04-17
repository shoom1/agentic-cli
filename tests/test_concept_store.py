"""Tests for the ConceptStore — pure file ops for concept pages."""

import time
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


class TestConceptStoreWrite:
    def test_write_creates_new_concept(self, tmp_path):
        from agentic_cli.knowledge_base.concepts import ConceptStore

        store = ConceptStore(tmp_path / "concepts")
        result = store.write(
            title="Diffusion Models",
            body="Synthesis text.",
            sources=["doc-a"],
            slug="diffusion-models",
        )

        assert result["action"] == "created"
        assert result["slug"] == "diffusion-models"
        assert result["path"].endswith("diffusion-models.md")
        assert result["invalid_sources"] == []

        on_disk = (tmp_path / "concepts" / "diffusion-models.md").read_text()
        assert "slug: diffusion-models" in on_disk
        assert "Synthesis text." in on_disk
        assert "sources: [doc-a]" in on_disk

    def test_write_creates_directory_if_missing(self, tmp_path):
        from agentic_cli.knowledge_base.concepts import ConceptStore

        base = tmp_path / "does-not-exist-yet" / "concepts"
        store = ConceptStore(base)
        store.write(title="X", body="b", sources=["a"], slug="x")
        assert (base / "x.md").exists()

    def test_failure_returns_uniform_dict_shape(self, tmp_path):
        """Failure path must include slug/path/action keys for safe access."""
        from agentic_cli.knowledge_base.concepts import ConceptStore

        store = ConceptStore(tmp_path / "concepts")
        result = store.write(title="X", body="b", sources=[])

        assert result["success"] is False
        assert result["slug"] == ""
        assert result["path"] == ""
        assert result["action"] == "failed"
        assert "error" in result


class TestConceptStoreSlugCollisions:
    def test_auto_slug_collision_appends_suffix(self, tmp_path):
        from agentic_cli.knowledge_base.concepts import ConceptStore

        store = ConceptStore(tmp_path / "concepts")
        r1 = store.write(title="Attention", body="b", sources=["a"])
        r2 = store.write(title="Attention", body="b", sources=["a"])
        r3 = store.write(title="Attention", body="b", sources=["a"])

        assert r1["slug"] == "attention"
        assert r2["slug"] == "attention-2"
        assert r3["slug"] == "attention-3"
        assert r1["action"] == "created"
        assert r2["action"] == "created"
        assert r3["action"] == "created"
        # Three distinct files on disk
        dir_ = tmp_path / "concepts"
        assert (dir_ / "attention.md").exists()
        assert (dir_ / "attention-2.md").exists()
        assert (dir_ / "attention-3.md").exists()

    def test_explicit_slug_existing_overwrites_not_collides(self, tmp_path):
        from agentic_cli.knowledge_base.concepts import ConceptStore

        store = ConceptStore(tmp_path / "concepts")
        store.write(title="A", body="first body", sources=["a"], slug="foo")
        result = store.write(
            title="A v2", body="second body", sources=["b"], slug="foo",
        )

        assert result["action"] == "updated"
        assert result["slug"] == "foo"
        # Only one file — no -2 suffix
        dir_ = tmp_path / "concepts"
        assert (dir_ / "foo.md").exists()
        assert not (dir_ / "foo-2.md").exists()


class TestConceptStoreSourcesValidation:
    def test_empty_sources_fails(self, tmp_path):
        from agentic_cli.knowledge_base.concepts import ConceptStore

        store = ConceptStore(tmp_path / "concepts")
        result = store.write(title="X", body="b", sources=[])

        assert result["success"] is False
        assert "at least one valid source" in result["error"]
        assert not (tmp_path / "concepts" / "x.md").exists()

    def test_invalid_ids_dropped_with_warning(self, tmp_path):
        from agentic_cli.knowledge_base.concepts import ConceptStore

        store = ConceptStore(tmp_path / "concepts")
        known = {"good-id-1", "good-id-2"}
        result = store.write(
            title="X",
            body="b",
            sources=["good-id-1", "bad-id", "good-id-2", "also-bad"],
            valid_ids_check=lambda i: i in known,
        )

        assert result["success"] is True
        assert result["invalid_sources"] == ["bad-id", "also-bad"]
        read = store.read(result["slug"])
        assert read["sources"] == ["good-id-1", "good-id-2"]

    def test_all_invalid_fails(self, tmp_path):
        from agentic_cli.knowledge_base.concepts import ConceptStore

        store = ConceptStore(tmp_path / "concepts")
        result = store.write(
            title="X", body="b", sources=["bad1", "bad2"],
            valid_ids_check=lambda i: False,
        )

        assert result["success"] is False
        assert result["invalid_sources"] == ["bad1", "bad2"]

    def test_duplicate_sources_deduplicated(self, tmp_path):
        from agentic_cli.knowledge_base.concepts import ConceptStore

        store = ConceptStore(tmp_path / "concepts")
        result = store.write(
            title="X", body="b", sources=["a", "a", "b", "a"],
        )

        assert result["success"] is True
        read = store.read(result["slug"])
        assert read["sources"] == ["a", "b"]


class TestConceptStoreOverwriteMerge:
    def test_overwrite_merges_sources_union(self, tmp_path):
        from agentic_cli.knowledge_base.concepts import ConceptStore

        store = ConceptStore(tmp_path / "concepts")
        store.write(title="X", body="v1", sources=["a", "b"], slug="x")
        store.write(title="X", body="v2", sources=["b", "c"], slug="x")

        read = store.read("x")
        # Union, preserving order of first appearance
        assert read["sources"] == ["a", "b", "c"]
        # Body replaced, not merged
        assert "v2" in read["body"]
        assert "v1" not in read["body"]

    def test_overwrite_preserves_created_at_bumps_updated_at(self, tmp_path):
        from agentic_cli.knowledge_base.concepts import ConceptStore

        store = ConceptStore(tmp_path / "concepts")
        store.write(title="X", body="v1", sources=["a"], slug="x")
        first = store.read("x")
        time.sleep(0.01)  # ensure tick
        store.write(title="X", body="v2", sources=["a"], slug="x")
        second = store.read("x")

        assert second["created_at"] == first["created_at"]
        assert second["updated_at"] > first["updated_at"]
