"""A search finds every document that matches, however many chunks another has.

``KnowledgeBaseManager.search`` ranked ``top_k * 2`` chunks, then kept one hit
per document and dropped those the filters refuse. A document with many
matching chunks filled that list on its own, so a filtered search for another
document came back empty (with ``top_k=2``; ``top_k=100`` found it), and an
unfiltered one returned fewer documents than matched. The search now widens
the ranked list until ``top_k`` documents pass, or every chunk is ranked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_cli.memory import EmbeddingService
from agentic_cli.memory.kb import KnowledgeBaseManager, SourceType
from tests.memory.fake_embeddings import FakeEmbeddingService
from tests.memory.kb_extra import require_kb_module

# Forty paragraphs about lanterns, each its own chunk.
_LONG = "\n\n".join(
    f"Paragraph {i}: the lantern festival, lantern after lantern on the river." * 12
    for i in range(40)
)


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    """No test here may load a sentence-transformers model."""
    monkeypatch.setattr(EmbeddingService, "is_available", staticmethod(lambda: False))


def _kb(folder: Path, embedding_service=None) -> KnowledgeBaseManager:
    if embedding_service is None:
        kb = KnowledgeBaseManager(folder, use_mock=True)
    else:
        kb = KnowledgeBaseManager(folder, embedding_service=embedding_service)
    kb.ingest_document(content=_LONG, title="Festival", source_type=SourceType.USER)
    kb.ingest_document(content="A short note on a paper lantern.", title="The note",
                       source_type=SourceType.WEB)
    assert len(kb._chunks) > 20
    return kb


def _titles(result: dict) -> list[str]:
    return [r["document_title"] for r in result["results"]]


def test_a_filtered_search_finds_the_document_the_filter_keeps(tmp_path):
    kb = _kb(tmp_path)

    result = kb.search("lantern", filters={"source_type": "web"}, top_k=2)

    assert _titles(result) == ["The note"]


def test_an_unfiltered_search_returns_each_matching_document(tmp_path):
    kb = _kb(tmp_path)

    assert sorted(_titles(kb.search("lantern", top_k=2))) == ["Festival", "The note"]


def test_a_search_for_more_documents_than_match_returns_those_that_do(tmp_path):
    kb = _kb(tmp_path)

    assert sorted(_titles(kb.search("lantern", top_k=50))) == ["Festival", "The note"]
    assert kb.search("lantern", filters={"source_type": "arxiv"}, top_k=2)["results"] == []


def test_semantic_and_keyword_search_widen_alike(tmp_path):
    require_kb_module("faiss")
    kb = _kb(tmp_path, FakeEmbeddingService(embedding_dim=8))

    result = kb.search("lantern", filters={"source_type": "web"}, top_k=2)

    assert _titles(result) == ["The note"]
