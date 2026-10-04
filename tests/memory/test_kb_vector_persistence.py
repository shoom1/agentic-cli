"""KnowledgeBaseManager-level regression coverage for the MockVectorStore
auto-load fix (see test_mock_vector_store_persistence.py for the store-level
tests). A manager with ``use_mock=True`` is the only path most installs
without the ``kb`` extra (no sentence-transformers / faiss) ever exercise.
"""

from __future__ import annotations

from agentic_cli.memory.kb import KnowledgeBaseManager, SourceType


def test_reopening_a_mock_knowledge_base_keeps_every_sessions_vectors(tmp_path):
    kb_dir = tmp_path / "kb"

    first = KnowledgeBaseManager(kb_dir, use_mock=True)
    doc_a = first.ingest_document(
        content="Otters play in rivers.", title="A", source_type=SourceType.USER
    )

    # A second manager, opened on the same directory, must not start from an
    # empty vector store — ingesting here must not wipe out A's vectors.
    second = KnowledgeBaseManager(kb_dir, use_mock=True)
    doc_b = second.ingest_document(
        content="Beavers build dams.", title="B", source_type=SourceType.USER
    )

    third = KnowledgeBaseManager(kb_dir, use_mock=True)

    assert third.get_stats()["vector_count"] == len(doc_a.chunks) + len(doc_b.chunks)

    # Query identical to A's content: the mock embedding service hashes text
    # verbatim, so an exact match is the closest vector.
    hits = third.search("Otters play in rivers.", top_k=1)["results"]
    assert [hit["document_title"] for hit in hits] == ["A"]


def test_a_manager_opens_despite_a_corrupt_mock_index(tmp_path):
    kb_dir = tmp_path / "kb"
    embeddings_dir = kb_dir / "embeddings"
    embeddings_dir.mkdir(parents=True)
    (embeddings_dir / "index.mock").write_text("not json")

    kb = KnowledgeBaseManager(kb_dir, use_mock=True)

    assert kb.get_stats()["vector_count"] == 0
