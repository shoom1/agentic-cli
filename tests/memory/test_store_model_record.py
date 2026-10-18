"""The memory store records its embedding model and re-embeds after a change.

It loaded vectors another model made and compared them with the new model's
query vectors. It now writes ``embedding_model.json`` beside
``memories_embeddings.json`` (whose format is unchanged); on open, a
mismatch drops the stored vectors and embeds every memory again. A store
from 0.6.3 (no record) keeps its vectors when they have the model's
dimension. Without an embedding service it leaves its files alone.
"""

from __future__ import annotations

import json

from agentic_cli.memory import MemoryStore
from tests.memory.fake_embeddings import FakeEmbeddingService


def _record(path) -> dict:
    return json.loads((path / "embedding_model.json").read_text())


def _vectors(path) -> dict:
    return json.loads((path / "memories_embeddings.json").read_text())


def _seed(path, service) -> None:
    store = MemoryStore(path, embedding_service=service)
    store.store("The team mascot is an otter")
    store.store("User prefers dark mode")


def test_the_store_records_its_model(tmp_path):
    _seed(tmp_path, FakeEmbeddingService("model-a", 8))

    assert _record(tmp_path) == {"model_name": "model-a", "embedding_dim": 8}


def test_a_matching_record_keeps_the_vectors(tmp_path):
    _seed(tmp_path, FakeEmbeddingService("model-a", 8))
    service = FakeEmbeddingService("model-a", 8)

    MemoryStore(tmp_path, embedding_service=service)

    assert service.calls == 0


def test_another_model_re_embeds_every_memory(tmp_path):
    _seed(tmp_path, FakeEmbeddingService("model-a", 8))
    service = FakeEmbeddingService("model-b", 8)

    store = MemoryStore(tmp_path, embedding_service=service)

    assert service.calls == 2
    assert sorted(_vectors(tmp_path).values()) == sorted(
        service.vector(item.content) for item in store._items.values()
    )
    assert _record(tmp_path) == {"model_name": "model-b", "embedding_dim": 8}


def test_a_0_6_3_store_of_the_same_dimension_is_adopted(tmp_path):
    _seed(tmp_path, FakeEmbeddingService("model-a", 8))
    (tmp_path / "embedding_model.json").unlink()
    service = FakeEmbeddingService("model-a", 8)

    MemoryStore(tmp_path, embedding_service=service)

    assert service.calls == 0
    assert _record(tmp_path) == {"model_name": "model-a", "embedding_dim": 8}


def test_a_0_6_3_store_of_another_dimension_is_re_embedded(tmp_path):
    _seed(tmp_path, FakeEmbeddingService("model-a", 8))
    (tmp_path / "embedding_model.json").unlink()
    service = FakeEmbeddingService("model-a", 16)

    MemoryStore(tmp_path, embedding_service=service)

    assert service.calls == 2
    assert {len(v) for v in _vectors(tmp_path).values()} == {16}


def test_the_embeddings_file_keeps_its_format(tmp_path):
    _seed(tmp_path, FakeEmbeddingService("model-a", 8))
    MemoryStore(tmp_path, embedding_service=FakeEmbeddingService("model-b", 8))

    vectors = _vectors(tmp_path)
    assert isinstance(vectors, dict)
    assert all(isinstance(v, list) and all(isinstance(x, float) for x in v) for v in vectors.values())


def test_without_a_service_the_files_are_left_alone(tmp_path):
    _seed(tmp_path, FakeEmbeddingService("model-a", 8))
    record = (tmp_path / "embedding_model.json").read_bytes()
    vectors = _vectors(tmp_path)

    store = MemoryStore(tmp_path)
    found = [item.content for item in store.search("otter")]

    assert found == ["The team mascot is an otter"]
    assert (tmp_path / "embedding_model.json").read_bytes() == record
    assert _vectors(tmp_path) == vectors


def test_a_service_that_does_not_name_its_model_keeps_no_record(tmp_path):
    class Bare:
        def embed_text(self, text):
            return [1.0, 0.0]

        def embed_batch(self, texts):
            return [[1.0, 0.0] for _ in texts]

    store = MemoryStore(tmp_path, embedding_service=Bare())
    store.store("a fact")

    assert not (tmp_path / "embedding_model.json").exists()
    assert [item.content for item in store.search("a fact")] == ["a fact"]
