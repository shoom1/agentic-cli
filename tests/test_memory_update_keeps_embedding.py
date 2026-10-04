"""Changing only a memory's tags keeps it findable by semantic search.

``MemoryStore.update`` cleared the memory's embedding on every update but
computed a new one only when the content changed, so after a tags-only update
(``update_memory(item_id, tags=[...])``) the memory had no embedding and
semantic search skipped it until the next restart re-embedded it. The
embedding depends only on the content, so it is now replaced only when the
content changes.
"""

from __future__ import annotations

import pytest

from agentic_cli.memory._core.mock_embeddings import MockEmbeddingService
from agentic_cli.memory import MemoryStore
from agentic_cli.tools.memory_tools import _update_memory_with_store


@pytest.fixture
def store(mock_context):
    return MemoryStore(
        mock_context.settings.workspace_dir / "memory", embedding_service=MockEmbeddingService()
    )


def _found(store: MemoryStore, query: str) -> list[str]:
    return [item.content for item in store.search(query, limit=1)]


def test_a_tags_only_update_keeps_the_memory_searchable(store):
    item_id = store.store("User prefers markdown output", tags=["style"])
    store.store("Project deadline is Friday")

    assert _update_memory_with_store(store, item_id, None, ["preference"])["success"]

    assert _found(store, "User prefers markdown output") == ["User prefers markdown output"]


def test_a_content_update_is_searchable_by_the_new_content(store):
    item_id = store.store("User prefers markdown output")
    store.store("Project deadline is Friday")

    store.update(item_id, content="User prefers plain text output")

    assert _found(store, "User prefers plain text output") == [
        "User prefers plain text output"
    ]
