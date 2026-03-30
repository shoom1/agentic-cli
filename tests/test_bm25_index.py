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
