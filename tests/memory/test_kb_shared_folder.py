"""Knowledge bases open on one folder keep each other's changes.

Every CLI session of an application opens the same knowledge bases (the
project's and the user's). Each kept the documents it read when it opened and
rewrote ``metadata.json``, the vector index and the keyword index from them
on every change, so whichever session wrote last dropped what the others had
added since: with A and B open, a document ingested through A, then one
through B, left only B's. Each read or change now holds the folder's lock and
reads again first what another process wrote.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from agentic_cli.memory import EmbeddingService
from agentic_cli.memory.kb import KnowledgeBaseManager, SourceType
from tests.memory.fake_embeddings import FakeEmbeddingService
from tests.memory.kb_extra import require_kb_module


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    """No test here may load a sentence-transformers model."""
    monkeypatch.setattr(EmbeddingService, "is_available", staticmethod(lambda: False))


def _kb(folder: Path, embedding_service=None) -> KnowledgeBaseManager:
    if embedding_service is None:
        return KnowledgeBaseManager(folder, use_mock=True)
    return KnowledgeBaseManager(folder, embedding_service=embedding_service)


def _ingest(kb: KnowledgeBaseManager, title: str, text: str):
    return kb.ingest_document(content=text, title=title, source_type=SourceType.USER)


def _titles(folder: Path) -> list[str]:
    return sorted(d.title for d in _kb(folder).list_documents(limit=None))


def test_documents_ingested_through_two_open_knowledge_bases_are_both_kept(tmp_path):
    a, b = _kb(tmp_path), _kb(tmp_path)

    _ingest(a, "Otters", "Otters nest in burrows by the river.")
    _ingest(b, "Herons", "Herons wade in the shallows at dawn.")

    assert _titles(tmp_path) == ["Herons", "Otters"]
    reopened = _kb(tmp_path)
    assert [r["document_title"] for r in reopened.search("otters")["results"]] == ["Otters"]
    assert [r["document_title"] for r in reopened.search("herons")["results"]] == ["Herons"]


def test_a_knowledge_base_finds_a_document_another_ingested_after_it_opened(tmp_path):
    a, b = _kb(tmp_path), _kb(tmp_path)

    doc = _ingest(a, "Otters", "Otters nest in burrows by the river.")

    assert [r["document_title"] for r in b.search("otters")["results"]] == ["Otters"]
    assert b.get_document(doc.id) is not None
    assert b.find_document("Otters") is not None
    assert b.get_stats()["document_count"] == 1


def test_a_delete_through_one_survives_an_ingest_through_the_other(tmp_path):
    a, b = _kb(tmp_path), _kb(tmp_path)
    doc = _ingest(a, "Otters", "Otters nest in burrows by the river.")

    assert b.delete_document(doc.id)
    _ingest(a, "Herons", "Herons wade in the shallows at dawn.")

    assert _titles(tmp_path) == ["Herons"]
    assert _kb(tmp_path).search("otters")["results"] == []


def test_a_sidecar_is_written_for_a_document_still_there(tmp_path):
    a, b = _kb(tmp_path), _kb(tmp_path)
    doc = _ingest(a, "Otters", "Otters nest in burrows by the river.")
    _ingest(b, "Herons", "Herons wade in the shallows at dawn.")  # A reads again

    assert a.write_sidecar_if_present(doc, {"summary": "otters", "claims": [], "entities": {}})
    assert a.sidecar_path(doc.id).exists()


def test_no_sidecar_is_written_for_a_document_another_deleted(tmp_path):
    a, b = _kb(tmp_path), _kb(tmp_path)
    doc = _ingest(a, "Otters", "Otters nest in burrows by the river.")
    b.delete_document(doc.id)

    assert not a.write_sidecar_if_present(doc, {"summary": "otters", "claims": [], "entities": {}})
    assert not a.sidecar_path(doc.id).exists()


def test_the_vector_index_keeps_both_knowledge_bases_vectors(tmp_path):
    require_kb_module("faiss")
    a = _kb(tmp_path, embedding_service=FakeEmbeddingService(embedding_dim=8))
    b = _kb(tmp_path, embedding_service=FakeEmbeddingService(embedding_dim=8))

    _ingest(a, "Otters", "Otters nest in burrows by the river.")
    _ingest(b, "Herons", "Herons wade in the shallows at dawn.")

    reopened = _kb(tmp_path, embedding_service=FakeEmbeddingService(embedding_dim=8))
    assert reopened.get_stats()["vector_count"] == reopened.get_stats()["chunk_count"] == 2


_WRITER = textwrap.dedent('''
    import sys
    from agentic_cli.memory.kb import KnowledgeBaseManager, SourceType
    kb = KnowledgeBaseManager(sys.argv[1], use_mock=True)
    for n in range(int(sys.argv[3])):
        kb.ingest_document(content=f"Note {n} from {sys.argv[2]}.", title=f"{sys.argv[2]}-{n}",
                           source_type=SourceType.USER)
''')


def test_processes_ingesting_at_once_lose_nothing(tmp_path):
    writers, each = 3, 6
    procs = [
        subprocess.Popen([sys.executable, "-c", _WRITER, str(tmp_path), f"w{w}", str(each)])
        for w in range(writers)
    ]
    assert [p.wait(timeout=180) for p in procs] == [0] * writers

    assert len(_titles(tmp_path)) == writers * each
    stats = _kb(tmp_path).get_stats()
    assert stats["chunk_count"] == writers * each
