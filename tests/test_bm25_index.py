"""Tests for BM25 index."""

import pytest
from pathlib import Path


class TestMockBM25Index:
    """Tests using mock BM25 (no bm25s dependency needed)."""

    def test_add_and_search(self, tmp_path):
        from agentic_cli.knowledge_base._mock_bm25 import MockBM25Index

        index = MockBM25Index()
        index.add_documents(
            ["chunk1", "chunk2", "chunk3"],
            ["python programming language", "java programming language", "the weather is sunny"],
        )
        results = index.search("python programming", top_k=2)
        assert len(results) <= 2
        assert results[0][0] == "chunk1"
        assert results[0][1] > 0

    def test_remove_documents(self, tmp_path):
        from agentic_cli.knowledge_base._mock_bm25 import MockBM25Index

        index = MockBM25Index()
        index.add_documents(["c1", "c2"], ["hello world", "goodbye world"])
        index.remove_documents(["c1"])
        results = index.search("hello", top_k=10)
        chunk_ids = [cid for cid, _ in results]
        assert "c1" not in chunk_ids

    def test_save_and_load(self, tmp_path):
        from agentic_cli.knowledge_base._mock_bm25 import MockBM25Index

        index = MockBM25Index()
        index.add_documents(["c1"], ["test document content"])
        index.save(tmp_path)
        index2 = MockBM25Index()
        index2.load(tmp_path)
        results = index2.search("test document", top_k=5)
        assert len(results) == 1
        assert results[0][0] == "c1"

    def test_empty_search(self, tmp_path):
        from agentic_cli.knowledge_base._mock_bm25 import MockBM25Index

        index = MockBM25Index()
        results = index.search("anything", top_k=5)
        assert results == []

    def test_size_property(self, tmp_path):
        from agentic_cli.knowledge_base._mock_bm25 import MockBM25Index

        index = MockBM25Index()
        assert index.size == 0
        index.add_documents(["c1", "c2"], ["doc one", "doc two"])
        assert index.size == 2

    def test_rebuild(self, tmp_path):
        from agentic_cli.knowledge_base._mock_bm25 import MockBM25Index

        index = MockBM25Index()
        index.rebuild(["c1", "c2"], ["new doc one", "new doc two"])
        assert index.size == 2
        results = index.search("new doc one", top_k=5)
        assert results[0][0] == "c1"


def _rank_bm25_cls():
    pytest.importorskip("rank_bm25")
    from agentic_cli.knowledge_base._bm25_backends import RankBM25Index
    return RankBM25Index


def _bm25s_cls():
    pytest.importorskip("bm25s")
    from agentic_cli.knowledge_base._bm25_backends import BM25sIndex
    return BM25sIndex


@pytest.fixture(params=["rank_bm25", "bm25s"])
def bm25_cls(request):
    if request.param == "rank_bm25":
        return _rank_bm25_cls()
    return _bm25s_cls()


class TestBM25BackendContract:
    """Contract tests applied to every real BM25 backend.

    Same behavior as MockBM25Index — any backend that wants to plug into
    create_bm25_index() must satisfy all of these.
    """

    def test_add_and_search(self, tmp_path, bm25_cls):
        index = bm25_cls()
        index.add_documents(
            ["chunk1", "chunk2", "chunk3"],
            [
                "python programming language",
                "java programming language",
                "the weather is sunny",
            ],
        )
        results = index.search("python programming", top_k=2)
        assert len(results) <= 2
        assert results[0][0] == "chunk1"
        assert results[0][1] > 0

    def test_search_returns_chunk_ids_and_scores(self, tmp_path, bm25_cls):
        index = bm25_cls()
        index.add_documents(["a", "b"], ["alpha beta", "gamma delta"])
        results = index.search("alpha", top_k=5)
        assert results, "expected at least one result for matching query"
        for chunk_id, score in results:
            assert isinstance(chunk_id, str)
            assert isinstance(score, float)

    def test_remove_documents(self, tmp_path, bm25_cls):
        index = bm25_cls()
        index.add_documents(["c1", "c2"], ["hello world", "goodbye world"])
        index.remove_documents(["c1"])
        assert index.size == 1
        results = index.search("hello", top_k=10)
        chunk_ids = [cid for cid, _ in results]
        assert "c1" not in chunk_ids

    def test_save_and_load(self, tmp_path, bm25_cls):
        index = bm25_cls()
        index.add_documents(
            ["c1", "c2"],
            ["test document content", "unrelated fruit basket"],
        )
        index.save(tmp_path)
        index2 = bm25_cls()
        index2.load(tmp_path)
        assert index2.size == 2
        results = index2.search("test document", top_k=5)
        assert results
        assert results[0][0] == "c1"

    def test_empty_search(self, tmp_path, bm25_cls):
        index = bm25_cls()
        results = index.search("anything", top_k=5)
        assert results == []

    def test_size_property(self, tmp_path, bm25_cls):
        index = bm25_cls()
        assert index.size == 0
        index.add_documents(["c1", "c2"], ["doc one", "doc two"])
        assert index.size == 2

    def test_rebuild(self, tmp_path, bm25_cls):
        index = bm25_cls()
        index.add_documents(["old"], ["stale content"])
        index.rebuild(["c1", "c2"], ["new doc one", "new doc two"])
        assert index.size == 2
        results = index.search("new doc one", top_k=5)
        assert results[0][0] == "c1"

    def test_load_missing_is_noop(self, tmp_path, bm25_cls):
        index = bm25_cls()
        # Loading from an empty dir should not raise; just leaves index empty.
        index.load(tmp_path)
        assert index.size == 0


class TestCreateBM25IndexPicksRealBackend:
    """When a real BM25 library is installed, create_bm25_index() must return it,
    not the mock."""

    def test_prefers_bm25s_when_available(self):
        pytest.importorskip("bm25s")
        from agentic_cli.knowledge_base.bm25_index import create_bm25_index
        from agentic_cli.knowledge_base._bm25_backends import BM25sIndex

        index = create_bm25_index()
        assert isinstance(index, BM25sIndex), (
            f"expected BM25sIndex, got {type(index).__name__}"
        )

    def test_use_mock_flag_returns_mock(self):
        from agentic_cli.knowledge_base.bm25_index import create_bm25_index
        from agentic_cli.knowledge_base._mock_bm25 import MockBM25Index

        index = create_bm25_index(use_mock=True)
        assert isinstance(index, MockBM25Index)
