"""Knowledge-base search works without the ``kb`` extra's BM25 libraries.

``create_bm25_index`` returned ``BM25sIndex`` whenever that class could be
imported, and it always could: bm25s itself is imported only when the index is
first searched. So without the ``kb`` extra every ``kb_search`` raised
``ModuleNotFoundError: bm25s``. The factory now checks that the library
imports and falls back to rank_bm25, then to the built-in index.

Each backend keeps its keyword index in its own file, so a knowledge base
opened with a different backend than the one that saved it found no index and
matched no keywords. It is now re-indexed from its chunks when it is opened.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agentic_cli.knowledge_base._bm25_backends import RankBM25Index
from agentic_cli.knowledge_base._mock_bm25 import MockBM25Index
from agentic_cli.knowledge_base._mocks import MockEmbeddingService, MockVectorStore
from agentic_cli.knowledge_base.bm25_index import create_bm25_index
from agentic_cli.knowledge_base.manager import KnowledgeBaseManager
from agentic_cli.knowledge_base.models import SourceType


@pytest.fixture
def without_bm25s(monkeypatch):
    # A None entry in sys.modules makes ``import bm25s`` raise ImportError.
    monkeypatch.setitem(sys.modules, "bm25s", None)


@pytest.fixture
def without_bm25_libs(monkeypatch, without_bm25s):
    monkeypatch.setitem(sys.modules, "rank_bm25", None)


def _keyword_hits(index) -> list[str]:
    index.add_documents(["c1", "c2"], ["alpha beta", "gamma delta"])
    return [chunk_id for chunk_id, _ in index.search("gamma")]


def test_without_bm25s_rank_bm25_is_used(without_bm25s):
    pytest.importorskip("rank_bm25")

    index = create_bm25_index()

    assert isinstance(index, RankBM25Index)
    assert _keyword_hits(index) == ["c2"]


def test_without_either_library_the_builtin_index_is_used(without_bm25_libs):
    index = create_bm25_index()

    assert isinstance(index, MockBM25Index)
    assert _keyword_hits(index) == ["c2"]


class _KeywordOnlyStore(MockVectorStore):
    """Semantic search finds nothing, so a hit can only come from the keyword index."""

    def search(self, query_embedding, top_k=10):
        return []


def _open_kb(base_dir: Path) -> KnowledgeBaseManager:
    return KnowledgeBaseManager(
        base_dir=base_dir,
        embedding_service=MockEmbeddingService(),
        vector_store=_KeywordOnlyStore(index_path=base_dir / "embeddings" / "index.mock"),
    )


def _titles(kb: KnowledgeBaseManager, query: str) -> list[str]:
    return [r["document_title"] for r in kb.search(query)["results"]]


def test_kb_search_without_bm25_libs(tmp_path, without_bm25_libs):
    kb = _open_kb(tmp_path)
    kb.ingest_document("how to moor a zeppelin", "Airships", SourceType.USER)

    assert _titles(kb, "zeppelin") == ["Airships"]


def test_a_kb_indexed_by_another_backend_is_reindexed_on_open(
    tmp_path, without_bm25_libs
):
    kb = _open_kb(tmp_path)
    kb.ingest_document("how to moor a zeppelin", "Airships", SourceType.USER)
    # Another backend keeps its index under its own file name, so the backend
    # opening the knowledge base next finds no index of its own.
    for saved in (tmp_path / "embeddings").glob("bm25*.json"):
        saved.unlink()

    reopened = _open_kb(tmp_path)

    assert _titles(reopened, "zeppelin") == ["Airships"]
