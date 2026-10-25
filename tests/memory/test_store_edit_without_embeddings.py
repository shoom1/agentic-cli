"""A memory edited where no embedding service runs is embedded again later.

A store without a service (no ``kb`` extra, say) clears an edited memory's
vector, which was of its old text. When that was the only vector, the store
wrote no embeddings file, so the old one kept it, and the next store with a
service paired the new text with the old text's vector: a search for the new
text missed it, and one for the old text found it.
"""

from __future__ import annotations

import json

import pytest

from agentic_cli.memory import MemoryStore
from tests.memory.fake_embeddings import FakeEmbeddingService


def test_an_edit_without_embeddings_is_embedded_again_on_the_next_open(tmp_path):
    service = FakeEmbeddingService("model-a", 8)
    memory_id = MemoryStore(tmp_path, embedding_service=service).store("The mascot is an otter")

    assert MemoryStore(tmp_path).update(memory_id, content="The mascot is a heron")

    assert memory_id not in json.loads((tmp_path / "memories_embeddings.json").read_text())
    reopened = MemoryStore(tmp_path, embedding_service=service)
    assert reopened._items[memory_id].embedding == service.embed_text("The mascot is a heron")


def test_an_edit_whose_embedding_fails_changes_nothing(tmp_path):
    """The new text and its vector change together: a failed embedding (a
    network error, say) left the new text with the old text's vector, which
    the next save wrote."""

    class Failing(FakeEmbeddingService):
        fail = False

        def embed_text(self, text):
            if self.fail:
                raise RuntimeError("embedding service unavailable")
            return super().embed_text(text)

    service = Failing("model-a", 8)
    store = MemoryStore(tmp_path, embedding_service=service)
    memory_id = store.store("The mascot is an otter")
    service.fail = True

    with pytest.raises(RuntimeError):
        store.update(memory_id, content="The mascot is a heron")

    service.fail = False
    store.store("User prefers dark mode")  # a later save
    item = MemoryStore(tmp_path, embedding_service=service)._items[memory_id]
    assert (item.content, item.embedding) == (
        "The mascot is an otter", service.embed_text("The mascot is an otter")
    )


def test_a_store_without_embeddings_writes_no_embeddings_file(tmp_path):
    MemoryStore(tmp_path).store("The mascot is an otter")

    assert not (tmp_path / "memories_embeddings.json").exists()
