"""Deleting from the knowledge base removes the data from disk.

``delete_document`` left the document's stored file (``files/{id}.pdf``) and
``clear`` left every stored file plus the keyword index, whose JSON holds the
text of every chunk. ``clear`` also kept that index in memory, so the next
ingest wrote the cleared documents' text back to disk.

The stored file's name comes from ``metadata.json``, which a project can ship,
so deletion removes only a file directly inside the files directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_cli.memory.kb._bm25_backends import BM25sIndex, RankBM25Index
from agentic_cli.memory.kb._mock_bm25 import MockBM25Index
from agentic_cli.memory._core.mock_embeddings import MockEmbeddingService
from agentic_cli.memory.kb._mock_vector_store import MockVectorStore
from agentic_cli.memory.kb.bm25_index import INDEX_FILES
from agentic_cli.memory.kb.manager import KnowledgeBaseManager
from agentic_cli.memory.kb.models import SourceType

MARKER = "quixotic"  # appears only in the deleted document's text


def _open_kb(kb_dir: Path) -> KnowledgeBaseManager:
    return KnowledgeBaseManager(
        base_dir=kb_dir,
        embedding_service=MockEmbeddingService(),
        vector_store=MockVectorStore(index_path=kb_dir / "embeddings" / "index.mock"),
    )


def _ingest(kb: KnowledgeBaseManager, text: str, title: str):
    return kb.ingest_document(
        content=text,
        title=title,
        source_type=SourceType.USER,
        file_bytes=text.encode(),
        file_extension=".txt",
    )


def _files_mentioning(root: Path, marker: str) -> list[str]:
    return sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and marker.encode() in path.read_bytes()
    )


def test_delete_document_leaves_none_of_its_text(tmp_path):
    kb = _open_kb(tmp_path)
    doomed = _ingest(kb, f"a {MARKER} plan for airships", "Plan")
    kept = _ingest(kb, "a sensible plan for balloons", "Balloons")

    assert kb.delete_document(doomed.id) is True

    assert _files_mentioning(tmp_path, MARKER) == []
    assert kb.get_file_path(kept.id).read_text() == "a sensible plan for balloons"


def test_clear_leaves_none_of_the_text(tmp_path):
    kb = _open_kb(tmp_path)
    _ingest(kb, f"a {MARKER} plan for airships", "Plan")

    kb.clear()

    assert _files_mentioning(tmp_path, MARKER) == []
    assert list(kb.files_dir.iterdir()) == []


def test_an_ingest_after_clear_does_not_bring_the_text_back(tmp_path):
    kb = _open_kb(tmp_path)
    _ingest(kb, f"a {MARKER} plan for airships", "Plan")
    kb.clear()

    _ingest(kb, "a sensible plan for balloons", "Balloons")

    assert _files_mentioning(tmp_path, MARKER) == []


def test_delete_removes_only_a_file_inside_the_files_directory(tmp_path):
    kb_dir = tmp_path / "kb"
    kb = _open_kb(kb_dir)
    doc = _ingest(kb, "a plan for airships", "Plan")
    neighbour = tmp_path / "neighbour.txt"
    neighbour.write_text("not the knowledge base's to delete")
    # metadata.json can arrive with the project, so its file name is untrusted.
    metadata = json.loads(kb.metadata_path.read_text())
    entries = metadata["documents"]
    entry = entries[doc.id] if isinstance(entries, dict) else next(
        e for e in entries if e["id"] == doc.id
    )
    entry["file_path"] = "../../neighbour.txt"
    kb.metadata_path.write_text(json.dumps(metadata))

    assert _open_kb(kb_dir).delete_document(doc.id) is True

    assert neighbour.read_text() == "not the knowledge base's to delete"


def _use_backend(monkeypatch, backend_cls):
    monkeypatch.setattr(
        "agentic_cli.memory.kb.bm25_index.create_bm25_index",
        lambda use_mock=False: backend_cls(),
    )


def test_an_index_saved_by_another_backend_does_not_outlive_its_documents(
    tmp_path, monkeypatch
):
    """Each backend keeps its own index file. When a knowledge base is opened
    with another backend (a BM25 library installed or removed since), the old
    file must not stay behind holding the text of documents deleted later."""
    _use_backend(monkeypatch, BM25sIndex)
    doomed = _ingest(_open_kb(tmp_path), f"a {MARKER} plan for airships", "Plan")
    _use_backend(monkeypatch, MockBM25Index)
    kb = _open_kb(tmp_path)

    kb.delete_document(doomed.id)

    assert _files_mentioning(tmp_path, MARKER) == []


def test_clear_removes_an_index_saved_by_another_backend(tmp_path, monkeypatch):
    _use_backend(monkeypatch, MockBM25Index)
    kb = _open_kb(tmp_path)
    doc = _ingest(kb, f"a {MARKER} plan for airships", "Plan")
    stale = BM25sIndex()  # an earlier release indexed with another backend
    stale.add_documents([c.id for c in doc.chunks], [c.content for c in doc.chunks])
    stale.save(kb.embeddings_dir)

    kb.clear()

    assert _files_mentioning(tmp_path, MARKER) == []


@pytest.mark.parametrize("backend_cls", [BM25sIndex, RankBM25Index, MockBM25Index])
def test_every_backend_saves_under_a_known_index_file(tmp_path, backend_cls):
    index = backend_cls()
    index.add_documents(["c1"], ["text"])

    index.save(tmp_path)

    (saved,) = tmp_path.iterdir()
    assert saved.name in INDEX_FILES
