"""A knowledge base uses the embedding service it is given.

``KnowledgeBaseManager(embedding_service=...)`` without ``vector_store``
silently built a service of its own, so both knowledge bases and the memory
store each loaded the model. The manager now uses the given service, and the
database records its model and dimension. Any service that is not a
``MockEmbeddingService`` searches by meaning (FAISS is no longer needed);
with ``use_mock=True``, or a mock service, the knowledge base searches by
keyword.
"""

from __future__ import annotations

import sys

import pytest

from agentic_cli.memory import EmbeddingService, MockEmbeddingService
from agentic_cli.memory.kb import KnowledgeBaseManager, SourceType
from tests.memory.fake_embeddings import FakeEmbeddingService
from tests.memory.kb_storage import read_rows


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    """No test here may load a sentence-transformers model."""
    monkeypatch.setattr(EmbeddingService, "is_available", staticmethod(lambda: False))


def _add_basalt(kb: KnowledgeBaseManager) -> None:
    kb.ingest_document(
        content="Basalt columns form as thick lava cools slowly.", title="Basalt", source_type=SourceType.USER
    )


def test_a_given_service_sizes_the_stored_vectors(tmp_path):
    service = FakeEmbeddingService(embedding_dim=8)

    kb = KnowledgeBaseManager(tmp_path / "kb", embedding_service=service)
    _add_basalt(kb)

    assert kb._service is service
    assert kb.get_stats()["embedding_dim"] == 8
    info = dict(read_rows(tmp_path / "kb", "SELECT key, value FROM info"))
    assert (info["embedding_model"], info["embedding_dim"]) == ("fake-model", "8")
    assert read_rows(tmp_path / "kb", "SELECT length(vector) FROM chunks") == [(8 * 4,)]


def test_the_given_service_embeds_and_searches(tmp_path):
    service = FakeEmbeddingService(embedding_dim=8)
    kb = KnowledgeBaseManager(tmp_path / "kb", embedding_service=service)
    _add_basalt(kb)
    before = service.calls

    result = kb.search("basalt")

    assert service.calls == before + 1  # the query was embedded: semantic search ran
    assert [r["document_title"] for r in result["results"]] == ["Basalt"]


def test_two_knowledge_bases_share_one_service(tmp_path):
    service = FakeEmbeddingService(embedding_dim=8)

    first = KnowledgeBaseManager(tmp_path / "first", embedding_service=service)
    second = KnowledgeBaseManager(tmp_path / "second", embedding_service=service)

    assert first._service is second._service is service


def test_a_mock_service_is_kept_and_searches_by_keyword(tmp_path):
    service = MockEmbeddingService(embedding_dim=16)

    kb = KnowledgeBaseManager(tmp_path / "kb", embedding_service=service)
    _add_basalt(kb)

    assert kb._service is service
    assert kb.get_stats()["embedding_dim"] == 16
    assert kb.get_stats()["vector_count"] == 0
    assert [r["document_title"] for r in kb.search("basalt")["results"]] == ["Basalt"]


def test_without_faiss_a_given_service_still_searches_by_meaning(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "faiss", None)
    service = FakeEmbeddingService(embedding_dim=8)
    kb = KnowledgeBaseManager(tmp_path / "kb", embedding_service=service)
    _add_basalt(kb)
    before = service.calls

    result = kb.search("basalt")

    assert service.calls == before + 1
    assert [r["document_title"] for r in result["results"]] == ["Basalt"]


def test_use_mock_ignores_the_given_service(tmp_path):
    service = FakeEmbeddingService(embedding_dim=8)

    kb = KnowledgeBaseManager(tmp_path / "kb", use_mock=True, embedding_service=service)

    assert isinstance(kb._service, MockEmbeddingService)
    assert service.calls == 0
