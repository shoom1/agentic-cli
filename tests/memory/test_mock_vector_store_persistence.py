"""MockVectorStore.load() tolerates a corrupt or foreign-shaped index file.

MockVectorStore does NOT auto-load its saved index on construction (see
``_mock_vector_store.py`` for why: in production it only ever pairs with
MockEmbeddingService, whose "embeddings" are MD5 noise with no semantic
content, so auto-loading them back in degrades search once an index exists
to load). A caller that does want the saved vectors calls ``load()``
explicitly, and that call must survive a corrupt file rather than raising.
"""

from __future__ import annotations

import pytest

from agentic_cli.memory.kb._mock_vector_store import MockVectorStore


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
    store.load()

    assert store.size == 0
