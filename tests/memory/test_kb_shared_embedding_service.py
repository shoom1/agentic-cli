"""A knowledge base uses the embedding service it is given.

``KnowledgeBaseManager(embedding_service=...)`` without ``vector_store``
silently built a service of its own, so both knowledge bases and the memory
store each loaded the model. The manager now builds its vector store at the
given service's dimension. Without FAISS, or with ``use_mock=True``, it falls
back to the mock store and keyword search, as before.
"""

from __future__ import annotations

import sys

import pytest

from agentic_cli.memory import EmbeddingService, MockEmbeddingService
from agentic_cli.memory.kb import KnowledgeBaseManager, MockVectorStore, SourceType, VectorStore
from tests.memory.fake_embeddings import FakeEmbeddingService
from tests.memory.kb_extra import require_kb_module


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    """No test here may load a sentence-transformers model."""
    monkeypatch.setattr(EmbeddingService, "is_available", staticmethod(lambda: False))


def _add_basalt(kb: KnowledgeBaseManager) -> None:
    kb.ingest_document(
        content="Basalt columns form as thick lava cools slowly.", title="Basalt", source_type=SourceType.USER
    )


def test_a_given_service_sizes_the_vector_store(tmp_path):
    require_kb_module("faiss")
    service = FakeEmbeddingService(embedding_dim=8)

    kb = KnowledgeBaseManager(tmp_path / "kb", embedding_service=service)

    assert kb._embedding_service is service
    assert isinstance(kb._vector_store, VectorStore)
    assert kb._vector_store.embedding_dim == 8


def test_the_given_service_embeds_and_searches(tmp_path):
    require_kb_module("faiss")
    service = FakeEmbeddingService(embedding_dim=8)
    kb = KnowledgeBaseManager(tmp_path / "kb", embedding_service=service)
    _add_basalt(kb)
    before = service.calls

    result = kb.search("basalt")

    assert service.calls == before + 1  # the query was embedded: semantic search ran
    assert [r["document_title"] for r in result["results"]] == ["Basalt"]


def test_two_knowledge_bases_share_one_service(tmp_path):
    require_kb_module("faiss")
    service = FakeEmbeddingService(embedding_dim=8)

    first = KnowledgeBaseManager(tmp_path / "first", embedding_service=service)
    second = KnowledgeBaseManager(tmp_path / "second", embedding_service=service)

    assert first._embedding_service is second._embedding_service is service


def test_a_mock_service_is_kept_with_a_mock_store(tmp_path):
    service = MockEmbeddingService(embedding_dim=16)

    kb = KnowledgeBaseManager(tmp_path / "kb", embedding_service=service)

    assert kb._embedding_service is service
    assert isinstance(kb._vector_store, MockVectorStore)
    assert kb._vector_store.embedding_dim == 16


def test_without_faiss_the_knowledge_base_searches_by_keyword(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "faiss", None)
    service = FakeEmbeddingService(embedding_dim=8)
    kb = KnowledgeBaseManager(tmp_path / "kb", embedding_service=service)
    _add_basalt(kb)

    result = kb.search("basalt")

    assert isinstance(kb._vector_store, MockVectorStore)
    assert service.calls == 0
    assert [r["document_title"] for r in result["results"]] == ["Basalt"]


def test_use_mock_ignores_the_given_service(tmp_path):
    service = FakeEmbeddingService(embedding_dim=8)

    kb = KnowledgeBaseManager(tmp_path / "kb", use_mock=True, embedding_service=service)

    assert isinstance(kb._embedding_service, MockEmbeddingService)
    assert service.calls == 0


def test_a_service_and_a_store_given_together_are_used_as_before(tmp_path):
    service = MockEmbeddingService()
    store = MockVectorStore(index_path=tmp_path / "kb" / "embeddings" / "index.mock")

    kb = KnowledgeBaseManager(tmp_path / "kb", embedding_service=service, vector_store=store)

    assert kb._embedding_service is service
    assert kb._vector_store is store
