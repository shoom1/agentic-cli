"""Deleting from the knowledge base removes the data from disk.

``delete_document`` left the document's stored file (``files/{id}.pdf``) and
``clear`` left every stored file plus the keyword index, whose JSON holds the
text of every chunk. ``clear`` also kept that index in memory, so the next
ingest wrote the cleared documents' text back to disk.

The knowledge base keeps its text in ``kb.sqlite``, where SQLite keeps deleted
rows in free pages, in the write-ahead log and in the keyword index's old
segments: the scans below read every file, ``kb.sqlite-wal`` included.

The stored file's name comes from the database, which a project can ship, so
deletion removes only a file directly inside the files directory.
"""

from __future__ import annotations

from pathlib import Path

from agentic_cli.memory.kb.manager import KnowledgeBaseManager
from agentic_cli.memory.kb.models import SourceType
from tests.memory.fake_embeddings import FakeEmbeddingService
from tests.memory.kb_storage import raw_write

MARKER = "quixotic"  # appears only in the deleted document's text


def _open_kb(kb_dir: Path) -> KnowledgeBaseManager:
    return KnowledgeBaseManager(kb_dir, use_mock=True)


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
    # The database can arrive with the project, so its file name is untrusted.
    raw_write(kb_dir, ("UPDATE documents SET file_name = '../../neighbour.txt' WHERE id = ?", [doc.id]))

    assert _open_kb(kb_dir).delete_document(doc.id) is True

    assert neighbour.read_text() == "not the knowledge base's to delete"


def test_deleted_text_leaves_the_keyword_index_too(tmp_path):
    kb = KnowledgeBaseManager(tmp_path, embedding_service=FakeEmbeddingService())
    doomed = _ingest(kb, f"a {MARKER} plan for airships " * 40, "Plan")
    _ingest(kb, "a sensible plan for balloons", "Balloons")
    assert kb.search(MARKER)["results"][0]["document_id"] == doomed.id

    kb.delete_document(doomed.id)

    assert _files_mentioning(tmp_path, MARKER) == []
    assert kb.search(MARKER)["results"][0]["document_title"] == "Balloons"  # found by meaning only
