"""A 0.6 knowledge base folder is imported once into kb.sqlite (review K1, memory-kb F2).

0.6 read metadata.json in one ``try``: a header that did not parse stopped
loading there, and the next save rewrote metadata.json from what had loaded,
so every document after it was lost. The import reads each header on its
own and moves the old files into pre-0.7.0/.
"""

from __future__ import annotations

import json

import pytest

from agentic_cli.memory import EmbeddingService
from agentic_cli.memory.kb import KnowledgeBaseManager, SourceType
from tests.memory.kb_storage import A, B, C, D, E, LogRecorder, header_06, read_rows, write_06


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    monkeypatch.setattr(EmbeddingService, "is_available", staticmethod(lambda: False))


def _titles(kb) -> list[str]:
    return sorted(doc.title for doc in kb.list_documents(limit=None))


def test_one_bad_header_costs_only_itself(tmp_path):
    no_updated_at = header_06(E, "Broken")
    del no_updated_at["updated_at"]
    write_06(tmp_path, [(A, "Otters", "Otters nest."), (B, "Herons", "Herons wade."), (C, "Kingfishers", "Kingfishers dive.")],
             extra_headers=[no_updated_at])

    kb = KnowledgeBaseManager(tmp_path, use_mock=True)
    assert _titles(kb) == ["Herons", "Kingfishers", "Otters"]
    kb.ingest_document(content="Beavers build dams.", title="Beavers", source_type=SourceType.USER)

    assert _titles(KnowledgeBaseManager(tmp_path, use_mock=True)) == ["Beavers", "Herons", "Kingfishers", "Otters"]


def test_the_old_files_move_aside_and_embeddings_go(tmp_path):
    write_06(tmp_path, [(A, "Otters", "Otters nest.")])

    KnowledgeBaseManager(tmp_path, use_mock=True)

    aside = tmp_path / "pre-0.7.0"
    assert json.loads((aside / "metadata.json").read_text())["documents"][0]["id"] == A
    assert (aside / "documents" / f"{A}.json").exists()
    assert not (tmp_path / "metadata.json").exists() and not (tmp_path / "embeddings").exists()
    assert (tmp_path / "documents" / f"{A}.md").exists()  # the sidecar stays
    assert (tmp_path / ".lock").read_bytes() == b""  # kept: another process may wait on it


def test_a_skipped_document_s_files_are_kept_aside(tmp_path):
    broken = header_06(E, "Broken", source_type="podcast", file_path=f"{E}.pdf")
    write_06(tmp_path, [(A, "Otters", "Otters nest.")], extra_headers=[broken])
    (tmp_path / "files" / f"{E}.pdf").write_bytes(b"%PDF-1.4 the only copy")
    (tmp_path / "documents" / f"{E}.md").write_text("# Broken\n")

    KnowledgeBaseManager(tmp_path, use_mock=True)
    KnowledgeBaseManager(tmp_path, use_mock=True)  # the next open's sweep must not find them either

    assert (tmp_path / "pre-0.7.0" / "files" / f"{E}.pdf").read_bytes() == b"%PDF-1.4 the only copy"
    assert (tmp_path / "pre-0.7.0" / "documents" / f"{E}.md").exists()
    assert not (tmp_path / "files" / f"{E}.pdf").exists()


def test_documents_are_chunked_again(tmp_path):
    write_06(tmp_path, [(A, "Выдры", "Выдры живут у рек и строят норы в берегах. " * 250)])

    kb = KnowledgeBaseManager(tmp_path, use_mock=True)

    chunks = kb.get_document(A).chunks
    assert len(chunks) > 15 and max(len(c.content) for c in chunks) <= 562


def test_v1_and_v2_folders_import(tmp_path):
    v1 = tmp_path / "v1"
    v1.mkdir()
    (v1 / "metadata.json").write_text(json.dumps({"documents": [
        {**header_06(A, "Otters"), "content": "Otters nest.",
         "chunks": [{"id": "c0", "document_id": A, "chunk_index": 0, "content": "Otters nest."}]},
    ]}))
    v2 = tmp_path / "v2"
    write_06(v2, [(B, "Herons", "Herons wade.")])
    data = json.loads((v2 / "metadata.json").read_text())
    data["version"] = 2
    del data["documents"][0]["summary"]
    (v2 / "metadata.json").write_text(json.dumps(data))

    assert [d.title for d in KnowledgeBaseManager(v1, use_mock=True).list_documents()] == ["Otters"]
    assert [d.title for d in KnowledgeBaseManager(v2, use_mock=True).list_documents()] == ["Herons"]


def test_a_header_naming_a_file_outside_files_imports_without_it(tmp_path):
    write_06(tmp_path, [], extra_headers=[header_06(A, "Otters", file_path="../../elsewhere.pdf")], position=0)
    (tmp_path / "documents" / f"{A}.json").write_text(json.dumps({"content": "Otters nest.", "chunks": {}}))

    kb = KnowledgeBaseManager(tmp_path, use_mock=True)

    assert kb.get_document(A).file_path is None


def test_a_document_whose_text_file_is_missing_is_imported_without_chunks(tmp_path):
    write_06(tmp_path, [(A, "Otters", "Otters nest.")])
    (tmp_path / "documents" / f"{A}.json").unlink()

    kb = KnowledgeBaseManager(tmp_path, use_mock=True)

    assert kb.get_document(A).chunk_count == 0


def test_an_import_interrupted_before_the_rename_runs_again(tmp_path):
    write_06(tmp_path, [(A, "Otters", "Otters nest.")])
    (tmp_path / "kb.sqlite.importing").write_bytes(b"half written")

    assert [d.title for d in KnowledgeBaseManager(tmp_path, use_mock=True).list_documents()] == ["Otters"]
    assert not (tmp_path / "kb.sqlite.importing").exists()


def test_a_cleanup_interrupted_after_the_rename_is_finished_by_the_next_open(tmp_path, monkeypatch):
    from agentic_cli.memory.kb import _import_06

    write_06(tmp_path, [(A, "Otters", "Otters nest.")])
    real = _import_06._move_aside

    def fail(*args, **kwargs):
        raise OSError("interrupted")

    monkeypatch.setattr(_import_06, "_move_aside", fail)
    with pytest.raises(OSError):
        KnowledgeBaseManager(tmp_path, use_mock=True)
    monkeypatch.setattr(_import_06, "_move_aside", real)

    kb = KnowledgeBaseManager(tmp_path, use_mock=True)

    assert [d.title for d in kb.list_documents()] == ["Otters"]
    assert (tmp_path / "pre-0.7.0" / "metadata.json").exists()


def test_a_metadata_json_an_old_version_writes_later_is_moved_aside_with_a_warning(tmp_path, monkeypatch):
    from agentic_cli.memory.kb import _import_06
    from tests.memory.kb_storage import LogRecorder

    log = LogRecorder().install(monkeypatch, _import_06)
    write_06(tmp_path, [(A, "Otters", "Otters nest.")])
    KnowledgeBaseManager(tmp_path, use_mock=True)
    (tmp_path / "metadata.json").write_text('{"version": 3, "documents": []}')

    KnowledgeBaseManager(tmp_path, use_mock=True)

    assert not (tmp_path / "metadata.json").exists()
    assert list((tmp_path / "pre-0.7.0").glob("after-import-*/metadata.json"))
    assert len(log.warnings("kb_legacy_files_ignored")) == 1


def test_two_processes_import_a_folder_once(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    write_06(tmp_path, [(A, "Otters", "Otters nest."), (B, "Herons", "Herons wade.")])
    child = (
        "import sys; sys.path.insert(0, sys.argv[2])\n"
        "from agentic_cli.memory import EmbeddingService\n"
        "EmbeddingService.is_available = staticmethod(lambda: False)\n"
        "from agentic_cli.memory.kb import KnowledgeBaseManager\n"
        "KnowledgeBaseManager(sys.argv[1], use_mock=True)\n"
    )
    repo = str(Path(__file__).resolve().parents[2])
    processes = [subprocess.Popen([sys.executable, "-c", child, str(tmp_path), repo]) for _ in range(2)]
    assert [p.wait(60) for p in processes] == [0, 0]

    assert sorted(d.title for d in KnowledgeBaseManager(tmp_path, use_mock=True).list_documents()) == ["Herons", "Otters"]


def test_the_first_open_with_a_model_embeds_the_imported_chunks(tmp_path):
    from tests.memory.fake_embeddings import FakeEmbeddingService
    from tests.memory.kb_storage import read_rows

    write_06(tmp_path, [(A, "Otters", "Otters nest in burrows. " * 40)])
    KnowledgeBaseManager(tmp_path, use_mock=True)
    assert read_rows(tmp_path, "SELECT count(*), count(vector) FROM chunks")[0][0] > 0
    assert read_rows(tmp_path, "SELECT count(*) FROM chunks WHERE vector IS NOT NULL") == [(0,)]

    KnowledgeBaseManager(tmp_path, embedding_service=FakeEmbeddingService())

    assert read_rows(tmp_path, "SELECT count(*) FROM chunks WHERE vector IS NULL") == [(0,)]


def test_a_pre_0_7_0_folder_that_links_elsewhere_is_refused(tmp_path):
    from agentic_cli.memory.kb import UnsafeKnowledgeBase

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    folder = tmp_path / "kb"
    write_06(folder, [(A, "Otters", "Otters nest.")])
    (folder / "pre-0.7.0").symlink_to(elsewhere)

    with pytest.raises(UnsafeKnowledgeBase):
        KnowledgeBaseManager(folder, use_mock=True)
    assert list(elsewhere.iterdir()) == []


def test_the_imported_database_is_in_wal_mode_with_no_leftovers(tmp_path):
    """Built in WAL mode and closed before the rename: sessions opening it at
    once need not switch it, and none of its companions is left behind."""
    from agentic_cli.memory.kb import _import_06

    write_06(tmp_path, [(A, "Otters", "Otters nest.")])

    _import_06.run(tmp_path)

    assert (tmp_path / "kb.sqlite").read_bytes()[18:20] == b"\x02\x02"  # WAL
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("kb.sqlite")) == ["kb.sqlite"]


def test_a_v1_header_whose_fields_do_not_parse_costs_only_itself(tmp_path, monkeypatch):
    from agentic_cli.memory.kb import _import_06

    log = LogRecorder().install(monkeypatch, _import_06)
    chunks = [{"id": "c0", "document_id": B, "chunk_index": 0, "content": "Herons wade."},
              {"id": "c1", "document_id": B, "chunk_index": "1", "content": "At dawn."}]
    (tmp_path / "metadata.json").write_text(json.dumps({"documents": [
        {**header_06(A, "Otters"), "content": 5},
        {**header_06(B, "Herons"), "content": "", "chunks": chunks},
        {**header_06(C, "Kingfishers"), "content": "Kingfishers dive."},
    ]}))

    kb = KnowledgeBaseManager(tmp_path, use_mock=True)

    assert _titles(kb) == ["Kingfishers"]
    assert [w["position"] for w in log.warnings("kb_import_skipped")] == [0, 1]


def test_a_header_whose_id_is_a_path_writes_nothing_outside(tmp_path):
    """From test_kb_document_id_trust.py: a shipped 0.6 folder's crafted ID."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "notes.json").write_text(json.dumps({"content": "outside notes", "chunks": {}}))
    folder = tmp_path / "project" / ".app" / "knowledge_base"
    write_06(folder, [(A, "Balloons", "a sensible plan")],
             extra_headers=[{**header_06("../../../../elsewhere/notes", "Plan"), "content": "a plan"}])

    kb = KnowledgeBaseManager(folder, use_mock=True)

    assert _titles(kb) == ["Balloons"]
    assert sorted(p.name for p in elsewhere.iterdir()) == ["notes.json"]


class TestKBManagerMigrationV1ToV2:
    """The 0.6 formats, v1 (text in metadata.json) and v2, are imported (from test_kb_manager.py)."""

    @pytest.fixture
    def kb_dir(self, tmp_path):
        kb = tmp_path / "kb"
        kb.mkdir()
        (kb / "documents").mkdir()
        (kb / "embeddings").mkdir()
        (kb / "files").mkdir()
        return kb

    def _make_v1_metadata(self, kb_dir):
        metadata = {
            "documents": [
                {
                    "id": "doc-v1-001",
                    "title": "V1 Document",
                    "content": "This is old-format content.",
                    "source_type": "user",
                    "source_url": None,
                    "file_path": None,
                    "created_at": "2024-06-01T12:00:00",
                    "updated_at": "2024-06-01T12:00:00",
                    "metadata": {},
                    "chunks": [
                        {
                            "id": "chunk-v1-001",
                            "document_id": "doc-v1-001",
                            "content": "This is old-format content.",
                            "chunk_index": 0,
                            "metadata": {"title": "V1 Document"},
                        }
                    ],
                }
            ],
            "updated_at": "2024-06-01T12:00:00",
        }
        (kb_dir / "metadata.json").write_text(json.dumps(metadata))
        return metadata

    def test_v1_loaded_correctly(self, kb_dir):
        """V1 format loads documents with content."""
        self._make_v1_metadata(kb_dir)
        kb = KnowledgeBaseManager(kb_dir, use_mock=True)

        doc = kb.get_document("doc-v1-001")
        assert doc is not None
        assert doc.title == "V1 Document"
        assert doc.content == "This is old-format content."
        assert len(doc.chunks) == 1
        assert doc.chunks[0].content == "This is old-format content."

    def test_v1_imported_into_the_database(self, kb_dir):
        """V1 format is imported on open; its metadata.json moves aside unchanged."""
        written = self._make_v1_metadata(kb_dir)
        KnowledgeBaseManager(kb_dir, use_mock=True)

        assert read_rows(kb_dir, "SELECT id, content FROM documents") == [("doc-v1-001", "This is old-format content.")]
        assert not (kb_dir / "metadata.json").exists()
        assert json.loads((kb_dir / "pre-0.7.0" / "metadata.json").read_text()) == written

    def test_v2_imported(self, kb_dir):
        """V2 format (text in documents/<id>.json, no summary) is imported."""
        metadata = {
            "version": 2,
            "documents": [
                {
                    "id": "doc-v2-001",
                    "title": "V2 Document",
                    "source_type": "user",
                    "source_url": None,
                    "file_path": None,
                    "created_at": "2024-06-01T12:00:00",
                    "updated_at": "2024-06-01T12:00:00",
                    "metadata": {},
                    "chunks": [
                        {
                            "id": "chunk-v2-001",
                            "document_id": "doc-v2-001",
                            "chunk_index": 0,
                            "metadata": {"title": "V2 Document"},
                        }
                    ],
                }
            ],
            "updated_at": "2024-06-01T12:00:00",
        }
        (kb_dir / "metadata.json").write_text(json.dumps(metadata))

        content_data = {
            "content": "V2 content here.",
            "chunks": {"chunk-v2-001": "V2 content here."},
        }
        (kb_dir / "documents" / "doc-v2-001.json").write_text(json.dumps(content_data))

        kb = KnowledgeBaseManager(kb_dir, use_mock=True)

        doc = kb.get_document("doc-v2-001")
        assert doc is not None
        assert doc.content == "V2 content here."
        assert doc.chunks[0].content == "V2 content here."
