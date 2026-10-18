"""A knowledge base records its embedding model and re-embeds after a change.

A FAISS index built by one model was loaded by any other, and an index of
another dimension broke every search. A knowledge base now writes
``embeddings/model.json`` (model name and dimension) whenever it saves its
real index, and on open re-embeds every chunk from its stored text when the
record or the index does not match the model. A 0.6.3 index with no record
is adopted when its dimension matches and it covers the stored chunks. The
tests use a fake service with a fixed dimension; the kb-extra CI job runs
them against real FAISS.
"""

from __future__ import annotations

import json
import shutil

import pytest

from agentic_cli.memory import EmbeddingService, MockEmbeddingService
from agentic_cli.memory.kb import KnowledgeBaseManager, SourceType
from agentic_cli.memory.kb import manager as manager_module
from tests.memory.fake_embeddings import FakeEmbeddingService
from tests.memory.kb_extra import require_kb_module


class _Log:
    """Records the events the knowledge base logs."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def _record(self, event, **kwargs) -> None:
        self.events.append(event)

    debug = info = warning = _record


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    monkeypatch.setattr(EmbeddingService, "is_available", staticmethod(lambda: False))


@pytest.fixture
def log(monkeypatch) -> _Log:
    recorder = _Log()
    monkeypatch.setattr(manager_module, "logger", recorder)
    return recorder


@pytest.fixture
def faiss():
    return require_kb_module("faiss")


@pytest.fixture
def kb_dir(tmp_path):
    return tmp_path / "kb"


def _open(kb_dir, service) -> KnowledgeBaseManager:
    return KnowledgeBaseManager(kb_dir, embedding_service=service)


def _seed(kb: KnowledgeBaseManager, n: int = 3) -> None:
    for i in range(n):
        kb.ingest_document(
            content=f"Note {i} about basalt and lava.", title=f"Note {i}", source_type=SourceType.USER
        )


def _record(kb_dir) -> dict:
    return json.loads((kb_dir / "embeddings" / "model.json").read_text())


def _files(kb_dir) -> dict[str, bytes]:
    embeddings = kb_dir / "embeddings"
    return {p.name: p.read_bytes() for p in sorted(embeddings.iterdir()) if p.name.startswith(("index.", "model."))}


def test_saving_the_index_records_the_model(kb_dir, faiss):
    _seed(_open(kb_dir, FakeEmbeddingService("model-a", 8)), 1)

    assert _record(kb_dir) == {"model_name": "model-a", "embedding_dim": 8}


def test_a_matching_record_loads_the_index_without_embedding(kb_dir, faiss):
    _seed(_open(kb_dir, FakeEmbeddingService("model-a", 8)))
    service = FakeEmbeddingService("model-a", 8)

    kb = _open(kb_dir, service)

    assert service.calls == 0
    assert kb._vector_store.size == 3


@pytest.mark.parametrize(("model", "dim"), [("model-b", 8), ("model-a", 16), ("model-b", 16)])
def test_a_record_of_another_model_re_embeds_every_chunk(kb_dir, faiss, model, dim):
    _seed(_open(kb_dir, FakeEmbeddingService("model-a", 8)))
    service = FakeEmbeddingService(model, dim)

    kb = _open(kb_dir, service)

    assert service.calls == 3
    assert kb._vector_store.size == 3
    assert faiss.read_index(str(kb_dir / "embeddings" / "index.faiss")).d == dim
    assert _record(kb_dir) == {"model_name": model, "embedding_dim": dim}
    assert not (kb_dir / "embeddings" / "index.rebuild.faiss").exists()
    assert not (kb_dir / "embeddings" / "index.rebuild.mappings.json").exists()
    before = service.calls
    assert kb.search("basalt lava")["results"]
    assert service.calls == before + 1  # semantic search runs on the new index


def test_a_0_6_3_index_of_the_same_dimension_is_adopted(kb_dir, faiss, log):
    _seed(_open(kb_dir, FakeEmbeddingService("model-a", 8)))
    (kb_dir / "embeddings" / "model.json").unlink()  # as 0.6.3 left it
    service = FakeEmbeddingService("model-a", 8)

    _open(kb_dir, service)

    assert service.calls == 0
    assert _record(kb_dir) == {"model_name": "model-a", "embedding_dim": 8}
    assert "kb_index_adopted" in log.events


def test_a_0_6_3_index_of_another_dimension_is_re_embedded(kb_dir, faiss):
    _seed(_open(kb_dir, FakeEmbeddingService("model-a", 8)))
    (kb_dir / "embeddings" / "model.json").unlink()
    service = FakeEmbeddingService("model-b", 16)

    kb = _open(kb_dir, service)

    assert service.calls == 3
    assert kb._vector_store.embedding_dim == 16


def test_an_index_that_does_not_cover_the_chunks_is_not_adopted(kb_dir, faiss):
    """An interrupted swap can leave mappings that name other chunks."""
    kb = _open(kb_dir, FakeEmbeddingService("model-a", 8))
    _seed(kb, 3)
    mappings = kb_dir / "embeddings" / "index.mappings.json"
    three = mappings.read_bytes()
    kb.ingest_document(content="A fourth note.", title="Note 3", source_type=SourceType.USER)
    mappings.write_bytes(three)
    (kb_dir / "embeddings" / "model.json").unlink()
    service = FakeEmbeddingService("model-a", 8)

    _open(kb_dir, service)

    assert service.calls == 4


def test_chunks_without_a_real_index_are_embedded(kb_dir, faiss):
    """The kb extra was installed after the knowledge base was built."""
    _seed(KnowledgeBaseManager(kb_dir, use_mock=True))
    service = FakeEmbeddingService("model-a", 8)

    kb = _open(kb_dir, service)

    assert service.calls == 3
    assert kb._vector_store.size == 3
    assert _record(kb_dir) == {"model_name": "model-a", "embedding_dim": 8}


def test_deleting_the_index_forces_a_rebuild(kb_dir, faiss):
    _seed(_open(kb_dir, FakeEmbeddingService("model-a", 8)))
    (kb_dir / "embeddings" / "index.faiss").unlink()
    service = FakeEmbeddingService("model-a", 8)

    kb = _open(kb_dir, service)

    assert service.calls == 3
    assert kb._vector_store.size == 3


def test_a_record_that_matches_while_the_index_has_another_dimension_re_embeds(kb_dir, faiss):
    _seed(_open(kb_dir, FakeEmbeddingService("model-a", 8)))
    (kb_dir / "embeddings" / "model.json").write_text('{"model_name": "model-b", "embedding_dim": 16}')
    service = FakeEmbeddingService("model-b", 16)

    kb = _open(kb_dir, service)

    assert service.calls == 3
    assert kb._vector_store.embedding_dim == 16


def test_every_save_rewrites_the_record(kb_dir, faiss):
    """Another session with another model saved last: the record follows it."""
    first = _open(kb_dir, FakeEmbeddingService("model-a", 8))
    _seed(first, 1)
    _open(kb_dir, FakeEmbeddingService("model-b", 16))  # re-embeds, records model-b
    first.ingest_document(content="Saved by the first session.", title="Late", source_type=SourceType.USER)

    assert _record(kb_dir) == {"model_name": "model-a", "embedding_dim": 8}
    assert faiss.read_index(str(kb_dir / "embeddings" / "index.faiss")).d == 8
    service = FakeEmbeddingService("model-b", 16)
    reopened = _open(kb_dir, service)  # the record says model-a: re-embed, no crash
    assert reopened._vector_store.embedding_dim == 16


@pytest.mark.parametrize(
    "content",
    ["not json", "[]", '{"model_name": 7, "embedding_dim": "8"}', '{"model_name": "x", "embedding_dim": true}', "x" * 10_000],
)
def test_a_planted_record_never_stops_the_knowledge_base_from_opening(kb_dir, faiss, content):
    _seed(_open(kb_dir, FakeEmbeddingService("model-a", 8)))
    (kb_dir / "embeddings" / "model.json").write_text(content)
    service = FakeEmbeddingService("model-a", 8)

    kb = _open(kb_dir, service)

    assert service.calls == 0  # no record: the matching index is adopted
    assert kb._vector_store.size == 3
    assert _record(kb_dir) == {"model_name": "model-a", "embedding_dim": 8}


def test_a_failed_re_embedding_keeps_the_old_index_and_searches_by_keyword(kb_dir, faiss, log):
    _seed(_open(kb_dir, FakeEmbeddingService("model-a", 8)))
    before = _files(kb_dir)

    kb = _open(kb_dir, FakeEmbeddingService("model-b", 8, fail_after=0))

    assert _files(kb_dir) == before
    assert isinstance(kb._embedding_service, MockEmbeddingService)
    assert kb.search("Note 1")["results"][0]["document_title"] == "Note 1"
    assert "kb_reembed_failed" in log.events
    retry = FakeEmbeddingService("model-b", 8)
    _open(kb_dir, retry)
    assert retry.calls == 3  # the next open tries again


def test_a_failure_on_a_later_batch_also_keeps_the_old_index(kb_dir, faiss, monkeypatch):
    _seed(_open(kb_dir, FakeEmbeddingService("model-a", 8)))
    before = _files(kb_dir)
    monkeypatch.setattr(KnowledgeBaseManager, "_REEMBED_BATCH", 1)

    _open(kb_dir, FakeEmbeddingService("model-b", 8, fail_after=1))

    assert _files(kb_dir) == before


def test_mock_mode_leaves_the_real_index_and_record_alone(kb_dir):
    """No FAISS needed: mock mode never reads index.faiss."""
    embeddings = kb_dir / "embeddings"
    embeddings.mkdir(parents=True)
    (embeddings / "index.faiss").write_bytes(b"a real index")
    (embeddings / "model.json").write_text('{"model_name": "model-a", "embedding_dim": 8}')
    before = _files(kb_dir)

    kb = KnowledgeBaseManager(kb_dir, use_mock=True)
    _seed(kb, 1)

    assert {name: data for name, data in _files(kb_dir).items() if name != "index.mock"} == before
    assert kb.search("basalt")["results"]


def test_a_copy_of_a_knowledge_base_reopens_with_its_record(kb_dir, faiss, tmp_path):
    _seed(_open(kb_dir, FakeEmbeddingService("model-a", 8)))
    copy = tmp_path / "copy"
    shutil.copytree(kb_dir, copy)
    service = FakeEmbeddingService("model-a", 8)

    _open(copy, service)

    assert service.calls == 0
