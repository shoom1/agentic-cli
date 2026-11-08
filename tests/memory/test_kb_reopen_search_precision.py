"""Regression: a reopened mock knowledge base must not resurrect vector noise.

MockEmbeddingService's "embeddings" are the MD5 hash of the chunk text —
meaningless noise, not real semantic vectors. The mock vector store once
auto-loaded its saved index on construction, so the noise vectors came back
too and got fused into results via Reciprocal Rank Fusion, pulling in
documents the query never mentioned. A keyword-only knowledge base now
stores no vectors at all, and searches by keyword (BM25) alone.
"""

from __future__ import annotations

from agentic_cli.memory.kb import KnowledgeBaseManager, SourceType

_TOPICS = {
    "Otters": (
        "Otters are playful aquatic mammals that live in rivers and coastal "
        "waters. Otters use tools to crack open shellfish. A group of otters "
        "is called a raft. Sea otters wrap themselves in kelp to avoid "
        "drifting out to sea while they sleep."
    ),
    "Volcanoes": (
        "Volcanoes form where magma rises through the crust to the surface. "
        "Volcanoes can be active, dormant, or extinct depending on their "
        "eruption history. Shield volcanoes have gentle slopes built from "
        "fluid lava flows over many eruptions."
    ),
    "Violins": (
        "Violins are string instruments played with a bow across four "
        "strings. A violin's body shape affects its tone and resonance. "
        "Violinists tune their violins to standard pitch before an "
        "orchestra rehearsal begins."
    ),
}


def test_reopening_a_mock_kb_search_returns_only_the_matching_document(tmp_path):
    kb_dir = tmp_path / "kb"

    for title, content in _TOPICS.items():
        session = KnowledgeBaseManager(kb_dir, use_mock=True)
        session.ingest_document(content=content, title=title, source_type=SourceType.USER)

    # A new manager over the same directory: the only path most installs
    # without the `kb` extra (no sentence-transformers) ever take.
    reopened = KnowledgeBaseManager(kb_dir, use_mock=True)

    hits = reopened.search("volcanoes", top_k=10)["results"]

    assert [hit["document_title"] for hit in hits] == ["Volcanoes"]
