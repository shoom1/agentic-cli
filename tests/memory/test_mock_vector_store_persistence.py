"""MockVectorStore auto-loads its saved index on construction.

Before this, a manager that reopened a persisted mock knowledge base saw an
empty vector store — the real, FAISS-backed VectorStore already loads on
construction; the mock never did. The next ingest/delete/clear then
overwrote ``index.mock`` with only that session's vectors, silently losing
every vector written by an earlier session.
"""

from __future__ import annotations

import pytest

from agentic_cli.memory.kb._mock_vector_store import MockVectorStore


def test_a_new_store_over_the_same_path_loads_its_saved_vectors(tmp_path):
    index_path = tmp_path / "index.mock"
    first = MockVectorStore(index_path=index_path, embedding_dim=4)
    first.add_embeddings(
        ["chunk-1", "chunk-2"],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    )
    first.save()

    reopened = MockVectorStore(index_path=index_path, embedding_dim=4)

    assert reopened.size == 2


@pytest.mark.parametrize(
    "garbage",
    [
        "not json at all",
        "",
        "{}",
        "[1, 2, 3]",
    ],
)
def test_a_corrupt_index_file_is_ignored_not_raised(tmp_path, garbage):
    index_path = tmp_path / "index.mock"
    index_path.write_text(garbage)

    store = MockVectorStore(index_path=index_path, embedding_dim=4)

    assert store.size == 0
