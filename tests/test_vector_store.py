"""Tests for VectorStore and MockVectorStore."""

import json
from pathlib import Path

import numpy as np
import pytest

from agentic_cli.knowledge_base.vector_store import MockVectorStore


class TestMockVectorStore:
    """Tests for MockVectorStore."""

    def test_init(self, tmp_path: Path):
        """Test empty store initialization."""
        store = MockVectorStore(index_path=tmp_path / "index.mock")
        assert store.size == 0

    def test_add_embeddings(self, tmp_path: Path):
        """Test adding embeddings."""
        store = MockVectorStore(index_path=tmp_path / "index.mock", embedding_dim=4)
        store.add_embeddings(
            ["chunk-1", "chunk-2"],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        )
        assert store.size == 2

    def test_search_empty(self, tmp_path: Path):
        """Test searching empty store."""
        store = MockVectorStore(index_path=tmp_path / "index.mock", embedding_dim=4)
        results = store.search([1.0, 0.0, 0.0, 0.0])
        assert results == []

    def test_search_returns_results(self, tmp_path: Path):
        """Test search returns ranked results."""
        store = MockVectorStore(index_path=tmp_path / "index.mock", embedding_dim=4)
        store.add_embeddings(
            ["chunk-a", "chunk-b"],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        )

        results = store.search([1.0, 0.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        # chunk-a should score highest (exact match)
        assert results[0][0] == "chunk-a"
        assert results[0][1] > results[1][1]

    def test_remove_embeddings(self, tmp_path: Path):
        """Test removing embeddings."""
        store = MockVectorStore(index_path=tmp_path / "index.mock", embedding_dim=4)
        store.add_embeddings(
            ["chunk-1", "chunk-2"],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        )

        removed = store.remove_embeddings(["chunk-1"])
        assert removed == 1
        assert store.size == 1

        # chunk-1 should no longer be searchable
        results = store.search([1.0, 0.0, 0.0, 0.0])
        chunk_ids = [r[0] for r in results]
        assert "chunk-1" not in chunk_ids

    def test_remove_nonexistent(self, tmp_path: Path):
        """Test removing non-existent chunk returns 0."""
        store = MockVectorStore(index_path=tmp_path / "index.mock")
        assert store.remove_embeddings(["nonexistent"]) == 0

    def test_clear(self, tmp_path: Path):
        """Test clearing the store."""
        store = MockVectorStore(index_path=tmp_path / "index.mock", embedding_dim=4)
        store.add_embeddings(["chunk-1"], [[1.0, 0.0, 0.0, 0.0]])
        assert store.size == 1

        store.clear()
        assert store.size == 0

    def test_save_and_load(self, tmp_path: Path):
        """Test persistence roundtrip."""
        index_path = tmp_path / "index.mock"
        store = MockVectorStore(index_path=index_path, embedding_dim=4)
        store.add_embeddings(
            ["chunk-1", "chunk-2"],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        )
        store.save()

        # Load into new store
        store2 = MockVectorStore(index_path=index_path, embedding_dim=4)
        store2.load()
        assert store2.size == 2

        results = store2.search([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert results[0][0] == "chunk-1"

    def test_rebuild_noop(self, tmp_path: Path):
        """Test rebuild is a no-op on mock store."""
        store = MockVectorStore(index_path=tmp_path / "index.mock", embedding_dim=4)
        store.add_embeddings(["chunk-1"], [[1.0, 0.0, 0.0, 0.0]])
        store.rebuild()  # Should not raise
        assert store.size == 1


class TestVectorStoreWithFAISS:
    """Tests for VectorStore (requires FAISS)."""

    @pytest.fixture(autouse=True)
    def _require_faiss(self):
        pytest.importorskip("faiss")

    @pytest.fixture
    def store(self, tmp_path: Path):
        from agentic_cli.knowledge_base.vector_store import VectorStore
        return VectorStore(index_path=tmp_path / "index.faiss", embedding_dim=4)

    def test_add_and_search(self, store):
        """Test adding vectors and searching."""
        store.add_embeddings(
            ["chunk-a", "chunk-b"],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        )
        assert store.size == 2

        results = store.search([1.0, 0.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0][0] == "chunk-a"

    def test_remove_and_rebuild_frees_memory(self, store):
        """Test that remove + rebuild compacts the index."""
        store.add_embeddings(
            ["chunk-a", "chunk-b", "chunk-c"],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        )
        assert store.size == 3

        removed = store.remove_embeddings(["chunk-b"])
        assert removed == 1
        # After rebuild (auto-called), index should be compacted
        assert store.size == 2

        # Remaining chunks should still be searchable
        results = store.search([1.0, 0.0, 0.0, 0.0], top_k=5)
        chunk_ids = [r[0] for r in results]
        assert "chunk-a" in chunk_ids
        assert "chunk-c" in chunk_ids
        assert "chunk-b" not in chunk_ids

    def test_rebuild_empty_after_removing_all(self, store):
        """Test rebuild when all vectors are removed."""
        store.add_embeddings(
            ["chunk-a"],
            [[1.0, 0.0, 0.0, 0.0]],
        )
        store.remove_embeddings(["chunk-a"])
        assert store.size == 0

    def test_save_load_roundtrip(self, store, tmp_path: Path):
        """Test persistence roundtrip with FAISS."""
        store.add_embeddings(
            ["chunk-a", "chunk-b"],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        )
        store.save()

        from agentic_cli.knowledge_base.vector_store import VectorStore
        store2 = VectorStore(index_path=tmp_path / "index.faiss", embedding_dim=4)
        assert store2.size == 2

        results = store2.search([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert results[0][0] == "chunk-a"
