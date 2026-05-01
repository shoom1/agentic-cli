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

    def test_write_timestamps_are_utc_with_z_suffix(self, tmp_path):
        """Concept timestamps must use UTC with trailing Z (matches Phase 1
        sidecar/ingest_log format; avoids timezone drift across machines)."""
        from agentic_cli.knowledge_base.concepts import ConceptStore

        store = ConceptStore(tmp_path / "concepts")
        store.write(title="X", body="b", sources=["a"], slug="x")

        on_disk = (tmp_path / "concepts" / "x.md").read_text()
        # Line format: "created_at: 2026-04-17T10:09:23.391871Z"
        import re
        m = re.search(r"created_at: (\S+)", on_disk)
        assert m is not None
        assert m.group(1).endswith("Z")
        assert "+00:00" not in m.group(1)

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

    @pytest.mark.parametrize(
        "bad_slug",
        [
            "../escape",
            "../../etc/passwd",
            "/abs/path",
            "sub/dir",
            "back\\slash",
            "..",
            ".",
            "WithUpper",
            "trailing-dash-",
            "-leading-dash",
            "spaces here",
            "x" * 81,  # exceeds MAX_SLUG_LENGTH
        ],
    )
    def test_explicit_slug_traversal_is_rejected(self, tmp_path, bad_slug):
        """Explicit slugs that could escape base_dir or otherwise look unsafe
        must be rejected — agents writing concept pages should not be able to
        clobber files outside the concepts directory."""
        from agentic_cli.knowledge_base.concepts import ConceptStore

        base = tmp_path / "concepts"
        store = ConceptStore(base)
        result = store.write(
            title="X", body="b", sources=["doc-a"], slug=bad_slug,
        )

        assert result["success"] is False
        assert result["action"] == "failed"
        assert "slug" in result["error"].lower()
        # Make sure nothing got written outside the concepts directory.
        for p in tmp_path.rglob("*"):
            if p.is_file():
                assert base in p.parents


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


class TestConceptStoreList:
    def test_list_empty_dir(self, tmp_path):
        from agentic_cli.knowledge_base.concepts import ConceptStore
        store = ConceptStore(tmp_path / "concepts")
        assert store.list() == []

    def test_list_returns_summaries_sorted_by_updated_desc(self, tmp_path):
        from agentic_cli.knowledge_base.concepts import ConceptStore
        import time

        store = ConceptStore(tmp_path / "concepts")
        store.write(title="First", body="b", sources=["a"], slug="first")
        time.sleep(0.01)
        store.write(title="Second", body="b", sources=["a"], slug="second")
        time.sleep(0.01)
        store.write(title="Third", body="b", sources=["a"], slug="third")

        items = store.list()
        assert [it["slug"] for it in items] == ["third", "second", "first"]
        for it in items:
            assert "title" in it
            assert "updated_at" in it
            assert "sources" in it

    def test_list_ignores_non_md_files(self, tmp_path):
        from agentic_cli.knowledge_base.concepts import ConceptStore

        store = ConceptStore(tmp_path / "concepts")
        store.write(title="One", body="b", sources=["a"], slug="one")
        # Drop a non-concept file alongside
        (tmp_path / "concepts" / "notes.txt").write_text("not a concept")
        (tmp_path / "concepts" / "index.md").write_text("# Index\n")

        items = store.list()
        # "index.md" has no frontmatter → skipped; notes.txt is not .md
        assert len(items) == 1
        assert items[0]["slug"] == "one"


class TestConceptStoreSearch:
    def test_search_empty_store_returns_empty(self, tmp_path):
        from agentic_cli.knowledge_base.concepts import ConceptStore
        store = ConceptStore(tmp_path / "concepts")
        assert store.search("anything") == []

    def test_search_matches_title(self, tmp_path):
        from agentic_cli.knowledge_base.concepts import ConceptStore

        store = ConceptStore(tmp_path / "concepts")
        store.write(title="Diffusion Models", body="Unrelated body.", sources=["a"])
        store.write(title="Something Else", body="No match here.", sources=["a"])

        hits = store.search("diffusion")
        assert len(hits) == 1
        assert hits[0]["slug"] == "diffusion-models"
        assert "Diffusion" in hits[0]["snippet"]

    def test_search_matches_body(self, tmp_path):
        from agentic_cli.knowledge_base.concepts import ConceptStore

        store = ConceptStore(tmp_path / "concepts")
        store.write(
            title="Generic",
            body="Body discusses transformer attention mechanisms.",
            sources=["a"],
        )
        hits = store.search("attention")
        assert len(hits) == 1
        assert "attention" in hits[0]["snippet"].lower()

    def test_title_hits_rank_above_body_hits(self, tmp_path):
        from agentic_cli.knowledge_base.concepts import ConceptStore

        store = ConceptStore(tmp_path / "concepts")
        store.write(
            title="About Animals",
            body="The word transformer appears in the body.",
            sources=["a"],
            slug="animals",
        )
        store.write(
            title="Transformer Models",
            body="Body has nothing relevant.",
            sources=["a"],
            slug="transformers",
        )

        hits = store.search("transformer")
        assert len(hits) == 2
        # Title match ranked above body match
        assert hits[0]["slug"] == "transformers"
        assert hits[1]["slug"] == "animals"

    def test_search_is_case_insensitive(self, tmp_path):
        from agentic_cli.knowledge_base.concepts import ConceptStore

        store = ConceptStore(tmp_path / "concepts")
        store.write(title="FooBar", body="Body text", sources=["a"])
        assert len(store.search("foobar")) == 1
        assert len(store.search("FOOBAR")) == 1

    def test_search_snippet_centered_on_match(self, tmp_path):
        from agentic_cli.knowledge_base.concepts import ConceptStore

        store = ConceptStore(tmp_path / "concepts")
        long_body = "x" * 300 + " needle " + "y" * 300
        store.write(title="Generic", body=long_body, sources=["a"])

        hits = store.search("needle")
        assert len(hits) == 1
        snippet = hits[0]["snippet"]
        assert "needle" in snippet
        # Snippet should be bounded — not the entire 600-char body
        assert len(snippet) <= 350  # ±150 + the word + some leeway

    def test_search_limit_respected(self, tmp_path):
        from agentic_cli.knowledge_base.concepts import ConceptStore

        store = ConceptStore(tmp_path / "concepts")
        for i in range(5):
            store.write(
                title=f"match {i}", body="b", sources=["a"], slug=f"s{i}",
            )
        hits = store.search("match", limit=3)
        assert len(hits) == 3

    def test_search_whitespace_only_query_returns_empty(self, tmp_path):
        from agentic_cli.knowledge_base.concepts import ConceptStore

        store = ConceptStore(tmp_path / "concepts")
        store.write(title="Has a space in body", body="body text", sources=["a"])
        assert store.search("   ") == []
        assert store.search("\t\n ") == []


class TestManagerConceptsProperty:
    def test_manager_exposes_concepts_pointing_at_kb_dir(self, tmp_path):
        from tests.test_knowledge_tools import _make_kb

        kb = _make_kb(tmp_path)
        store = kb.concepts
        assert store.base_dir == kb.kb_dir / "concepts"
        # Idempotent — returns the same instance
        assert kb.concepts is store
