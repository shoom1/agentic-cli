"""The knowledge base on SQLite: what a change, a crash, a model or a filter does.

Review findings (plans/2026-11-04-codebase-review/areas/memory-kb.md): K4
(F5) a crash between files tore the vector index from its chunks; K7 (F8)
keyword-only mode computed and saved hash vectors for every chunk; F11 a
failed ingest left the user's file in files/; F12 date_to dropped the day it
named and a date with an offset failed the search; F13 the search ran again
with more candidates until every chunk was ranked.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from agentic_cli.memory import EmbeddingService, MockEmbeddingService
from agentic_cli.memory.kb import KnowledgeBaseManager, SourceType, _db, _search
from agentic_cli.memory.kb import manager as manager_module
from agentic_cli.memory.kb._db import KnowledgeBaseBusy
from agentic_cli.memory.kb._vectors import VectorIndex, normalized
from tests.memory.fake_embeddings import FakeEmbeddingService
from tests.memory.kb_storage import LogRecorder, raw_write, read_rows

RIVER = "Otters nest in burrows along the river. " * 30


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    monkeypatch.setattr(EmbeddingService, "is_available", staticmethod(lambda: False))


class CountingMock(MockEmbeddingService):
    """A keyword-only service that counts the texts it is asked to embed."""

    def __init__(self) -> None:
        super().__init__()
        self.embedded = 0

    def embed_batch(self, texts):
        self.embedded += len(texts)
        return super().embed_batch(texts)

    def embed_text(self, text):
        self.embedded += 1
        return super().embed_text(text)


def _ingest(kb, title, text, **kwargs):
    return kb.ingest_document(content=text, title=title, source_type=SourceType.USER, **kwargs)


def test_keyword_only_mode_embeds_nothing_and_writes_no_vectors(tmp_path):
    service = CountingMock()
    kb = KnowledgeBaseManager(tmp_path, embedding_service=service)

    _ingest(kb, "Otters", RIVER)
    kb.search("burrows")

    assert service.embedded == 0
    assert not list(tmp_path.rglob("index.mock"))
    assert read_rows(tmp_path, "SELECT count(*) FROM chunks WHERE vector IS NOT NULL") == [(0,)]


def test_a_failed_ingest_leaves_no_file_behind(tmp_path):
    kb = KnowledgeBaseManager(tmp_path, embedding_service=FakeEmbeddingService(fail_after=0))

    with pytest.raises(RuntimeError):
        _ingest(kb, "Otters", "Otters nest in burrows.", file_bytes=b"%PDF-1.4 otters", file_extension=".pdf")

    assert list(kb.files_dir.iterdir()) == []
    assert kb.list_documents() == []


def test_date_to_keeps_the_whole_day_it_names(tmp_path):
    kb = KnowledgeBaseManager(tmp_path, use_mock=True)
    doc = _ingest(kb, "Otters", "Otters nest in burrows.")

    hits = kb.search("otters", filters={"date_to": doc.created_at.date().isoformat()})["results"]

    assert [hit["document_id"] for hit in hits] == [doc.id]


def test_a_date_with_an_offset_filters_instead_of_failing(tmp_path):
    kb = KnowledgeBaseManager(tmp_path, use_mock=True)
    doc = _ingest(kb, "Otters", "Otters nest in burrows.")
    an_hour_ago = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()

    assert [h["document_id"] for h in kb.search("otters", filters={"date_from": an_hour_ago})["results"]] == [doc.id]
    assert kb.search("otters", filters={"date_from": tomorrow})["results"] == []


REPO = Path(__file__).resolve().parents[2]


def _vectors_match_their_chunks(kb_dir: Path, fake: FakeEmbeddingService) -> bool:
    """Every stored vector is the one its own chunk's text makes (no vector belongs to another chunk)."""
    rows = read_rows(kb_dir, "SELECT content, vector FROM chunks")
    return all(
        vector is not None
        and np.allclose(np.frombuffer(vector, dtype="<f4"), normalized([fake.vector(text)])[0], atol=1e-6)
        for text, vector in rows
    )


_CHILD = textwrap.dedent('''
    import os, sys
    sys.path.insert(0, sys.argv[3])
    from agentic_cli.memory import EmbeddingService
    EmbeddingService.is_available = staticmethod(lambda: False)
    from agentic_cli.memory.kb import KnowledgeBaseManager, SourceType, _db, manager
    from tests.memory.fake_embeddings import FakeEmbeddingService
    folder, step = sys.argv[1], sys.argv[2]
    def die(*args, **kwargs):
        os._exit(17)
    if step == "inside-ingest":
        real = _db.Database.insert_document
        def insert_then_die(self, *args, **kwargs):
            real(self, *args, **kwargs)
            os._exit(17)
        _db.Database.insert_document = insert_then_die
        kb = KnowledgeBaseManager(folder, embedding_service=FakeEmbeddingService())
        kb.ingest_document(content="Kingfishers dive for fish. " * 40, title="Kingfishers",
                           source_type=SourceType.USER, file_bytes=b"kingfishers", file_extension=".txt")
    elif step == "after-delete-commit":
        kb = KnowledgeBaseManager(folder, embedding_service=FakeEmbeddingService())
        doc = kb.find_document("Herons")
        manager.KnowledgeBaseManager._delete_sidecar = die
        kb.delete_document(doc.id)
    elif step == "between-embedding-batches":
        fake = FakeEmbeddingService(model_name="fake-2")
        real = fake.embed_batch
        calls = []
        def second_dies(texts):
            calls.append(1)
            if len(calls) == 2:
                os._exit(17)
            return real(texts)
        fake.embed_batch = second_dies
        KnowledgeBaseManager(folder, embedding_service=fake)
    os._exit(0)
''')


def _crash(folder: Path, step: str) -> None:
    result = subprocess.run([sys.executable, "-c", _CHILD, str(folder), step, str(REPO)], capture_output=True, text=True)
    assert result.returncode == 17, result.stderr


def test_a_crash_inside_an_ingest_leaves_neither_the_document_nor_its_files(tmp_path):
    fake = FakeEmbeddingService()
    _ingest(KnowledgeBaseManager(tmp_path, embedding_service=fake), "Otters", RIVER)

    _crash(tmp_path, "inside-ingest")

    kb = KnowledgeBaseManager(tmp_path, embedding_service=fake)
    assert [d.title for d in kb.list_documents()] == ["Otters"]
    assert list(kb.files_dir.iterdir()) == []
    assert [p.name for p in kb.documents_dir.iterdir()] == [f"{kb.list_documents()[0].id}.md"]
    assert _vectors_match_their_chunks(tmp_path, fake)


def test_a_crash_after_a_delete_commits_leaves_no_files_of_it(tmp_path):
    fake = FakeEmbeddingService()
    kb = KnowledgeBaseManager(tmp_path, embedding_service=fake)
    _ingest(kb, "Otters", RIVER)
    herons = _ingest(kb, "Herons", "Herons wade at dawn. " * 30, file_bytes=b"herons", file_extension=".txt")

    _crash(tmp_path, "after-delete-commit")

    kb = KnowledgeBaseManager(tmp_path, embedding_service=fake)
    assert [d.title for d in kb.list_documents()] == ["Otters"]
    assert not (kb.documents_dir / f"{herons.id}.md").exists()
    assert list(kb.files_dir.iterdir()) == []


def test_an_interrupted_re_embedding_continues_at_the_next_open(tmp_path):
    kb = KnowledgeBaseManager(tmp_path, embedding_service=FakeEmbeddingService(model_name="fake-1"))
    for i in range(6):
        _ingest(kb, f"Note {i}", f"Herons wade at dawn number {i}. " * 1700)  # ~110 chunks each
    total = read_rows(tmp_path, "SELECT count(*) FROM chunks")[0][0]
    assert total > 512

    _crash(tmp_path, "between-embedding-batches")  # the first batch of 256 is saved

    fake = FakeEmbeddingService(model_name="fake-2")
    KnowledgeBaseManager(tmp_path, embedding_service=fake)
    assert fake.calls == total - 256
    assert _vectors_match_their_chunks(tmp_path, fake)


def test_after_a_delete_and_more_ingests_every_vector_is_its_own_chunk_s(tmp_path):
    fake = FakeEmbeddingService()
    kb = KnowledgeBaseManager(tmp_path, embedding_service=fake)
    docs = [_ingest(kb, name, f"{name} live by the river. " * 20) for name in ("Herons", "Kingfishers", "Otters")]
    kb.delete_document(docs[1].id)
    _ingest(kb, "Beavers", "Beavers build dams. " * 20)

    assert _vectors_match_their_chunks(tmp_path, fake)
    for doc in kb.list_documents():
        text = kb.get_document(doc.id).chunks[0].content
        assert kb.search(text)["results"][0]["document_id"] == doc.id


def test_the_next_open_with_a_model_embeds_exactly_the_chunks_written_without_one(tmp_path):
    doc = _ingest(KnowledgeBaseManager(tmp_path, use_mock=True), "Otters", RIVER)

    fake = FakeEmbeddingService()
    kb = KnowledgeBaseManager(tmp_path, embedding_service=fake)

    assert fake.calls == len(doc.chunks)
    assert _vectors_match_their_chunks(tmp_path, fake)
    assert kb.search("burrows")["results"][0]["document_id"] == doc.id


def test_a_model_change_elsewhere_leaves_this_session_searching_by_keyword(tmp_path, monkeypatch):
    log = LogRecorder().install(monkeypatch, manager_module)
    first = KnowledgeBaseManager(tmp_path, embedding_service=FakeEmbeddingService(model_name="fake-1"))
    _ingest(first, "Otters", RIVER)
    second_fake = FakeEmbeddingService(model_name="fake-2")
    KnowledgeBaseManager(tmp_path, embedding_service=second_fake)

    assert first.search("burrows")["results"][0]["document_title"] == "Otters"
    assert len(log.warnings("kb_model_changed_elsewhere")) == 1
    _ingest(first, "Herons", "Herons wade at dawn.")
    assert read_rows(tmp_path, "SELECT count(*) FROM chunks WHERE vector IS NULL") == [(1,)]
    assert read_rows(tmp_path, "SELECT value FROM info WHERE key = 'embedding_model'") == [("fake-2",)]


def test_another_session_s_document_is_found_by_meaning(tmp_path):
    fake = FakeEmbeddingService()
    first = KnowledgeBaseManager(tmp_path, embedding_service=fake)
    second = KnowledgeBaseManager(tmp_path, embedding_service=fake)
    _ingest(first, "Otters", RIVER)

    doc = _ingest(second, "Herons", "Herons wade at dawn. " * 20)
    text = second.get_document(doc.id).chunks[0].content

    assert first.search(text)["results"][0]["document_id"] == doc.id
    assert sorted(d.title for d in first.list_documents()) == ["Herons", "Otters"]


def test_embedding_happens_outside_the_write_lock(tmp_path):
    started, release = threading.Event(), threading.Event()

    class Slow(FakeEmbeddingService):
        def embed_batch(self, texts):
            started.set()
            assert release.wait(10)
            return super().embed_batch(texts)

    slow = KnowledgeBaseManager(tmp_path, embedding_service=Slow())
    other = KnowledgeBaseManager(tmp_path, use_mock=True)
    worker = threading.Thread(target=_ingest, args=(slow, "Otters", RIVER))
    worker.start()
    try:
        assert started.wait(10)
        _ingest(other, "Herons", "Herons wade.")  # would wait 60 s if the embedding held the lock
    finally:
        release.set()
        worker.join(10)
    assert sorted(d.title for d in other.list_documents()) == ["Herons", "Otters"]


def test_a_change_past_the_busy_timeout_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(_db, "BUSY_TIMEOUT_S", 0.2)
    kb = KnowledgeBaseManager(tmp_path, use_mock=True)
    raw = sqlite3.connect(tmp_path / "kb.sqlite", isolation_level=None)
    raw.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(KnowledgeBaseBusy):
            _ingest(kb, "Otters", "Otters nest.")
    finally:
        raw.execute("ROLLBACK")
        raw.close()


def test_a_failure_while_storing_rolls_back_and_removes_the_file(tmp_path, monkeypatch):
    kb = KnowledgeBaseManager(tmp_path, use_mock=True)

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(_db.Database, "insert_document", fail)
    with pytest.raises(OSError, match="disk full"):
        _ingest(kb, "Otters", "Otters nest.", file_bytes=b"otters", file_extension=".txt")
    monkeypatch.undo()

    assert list(kb.files_dir.iterdir()) == []
    assert kb.list_documents() == []


def test_a_search_ranks_once_whatever_the_filters_keep(tmp_path, monkeypatch):
    kb = KnowledgeBaseManager(tmp_path, use_mock=True)
    for i in range(30):
        _ingest(kb, f"Note {i}", f"Herons wade at dawn number {i}. " * 40)
    calls = []
    real = _search.keyword_ranked
    monkeypatch.setattr(_search, "keyword_ranked", lambda *a, **k: calls.append(1) or real(*a, **k))

    assert kb.search("herons", filters={"source_type": "arxiv"})["results"] == []
    assert len(calls) == 1


def test_files_a_crash_left_are_removed_and_others_kept(tmp_path):
    kb = KnowledgeBaseManager(tmp_path, use_mock=True)
    kept = _ingest(kb, "Otters", "Otters nest.", file_bytes=b"otters", file_extension=".txt")
    orphan = "99999999-9999-4999-8999-999999999999"
    for path in (kb.documents_dir / f"{orphan}.md", kb.files_dir / f"{orphan}.pdf",
                 kb.documents_dir / f".{orphan}.md.abc123.tmp", kb.kb_dir / ".index.md.def456.tmp",
                 kb.documents_dir / "README.md", kb.files_dir / "notes.pdf"):
        path.write_text("x")

    KnowledgeBaseManager(tmp_path, use_mock=True)

    assert sorted(p.name for p in kb.documents_dir.iterdir()) == sorted([f"{kept.id}.md", "README.md"])
    assert sorted(p.name for p in kb.files_dir.iterdir()) == sorted([f"{kept.id}.txt", "notes.pdf"])
    assert not (kb.kb_dir / ".index.md.def456.tmp").exists()


def test_a_missing_sidecar_is_rendered_from_the_stored_summary_without_the_model(tmp_path):
    class NoCalls:
        async def summarize(self, content, prompt):
            raise AssertionError("the summarizer was called")

    kb = KnowledgeBaseManager(tmp_path, use_mock=True, summarizer=NoCalls())
    doc = kb.ingest_document(
        content="Otters nest.", title="Otters", source_type=SourceType.USER, summary="About otters.",
        sidecar_payload={"summary": "About otters.", "claims": ["Otters nest in burrows."], "entities": {}},
    )
    kb.sidecar_path(doc.id).unlink()

    text = kb.read_sidecar(doc.id)

    assert "Otters nest in burrows." in text
    assert kb.sidecar_path(doc.id).exists()


def test_an_ingest_with_a_chunk_size_under_one_stores_nothing(tmp_path):
    kb = KnowledgeBaseManager(tmp_path, use_mock=True)

    with pytest.raises(ValueError):
        _ingest(kb, "Otters", "Otters nest.", file_bytes=b"otters", file_extension=".txt", chunk_size=0)

    assert list(kb.files_dir.iterdir()) == []
    assert kb.list_documents() == []


def test_this_session_s_vectors_are_used_only_under_its_lock(tmp_path, monkeypatch):
    # A search on another thread must not meet the matrix half changed.
    kb = KnowledgeBaseManager(tmp_path, embedding_service=FakeEmbeddingService())
    held: list[tuple[str, bool]] = []
    for name in ("apply", "load", "ranked"):
        real = getattr(VectorIndex, name)

        def wrapped(self, *args, _real=real, _name=name, **kwargs):
            held.append((_name, kb._lock.locked()))
            return _real(self, *args, **kwargs)

        monkeypatch.setattr(VectorIndex, name, wrapped)

    doc = _ingest(kb, "Otters", RIVER)
    kb.search("burrows")
    kb.delete_document(doc.id)
    kb.clear()

    assert {name for name, _ in held} == {"apply", "load", "ranked"}
    assert [call for call in held if not call[1]] == []


def test_a_delete_waits_only_briefly_for_another_session_s_read(tmp_path, monkeypatch):
    # The checkpoint that empties the log after a delete cannot finish while
    # another session reads; it is tried again at close.
    monkeypatch.setattr(_db, "BUSY_TIMEOUT_S", 10.0)
    kb = KnowledgeBaseManager(tmp_path, use_mock=True)
    doc = _ingest(kb, "Otters", "Otters nest in burrows: zqxmarkerword.")
    reader = sqlite3.connect(tmp_path / "kb.sqlite", isolation_level=None)
    try:
        reader.execute("BEGIN")
        reader.execute("SELECT count(*) FROM documents").fetchone()  # another session mid-search
        began = time.monotonic()
        kb.delete_document(doc.id)
        waited = time.monotonic() - began
        reader.execute("COMMIT")  # its read ends; the session stays open
        kb.close()

        assert waited < 5
        assert [p.name for p in tmp_path.rglob("*") if p.is_file() and b"zqxmarkerword" in p.read_bytes()] == []
    finally:
        reader.close()


def test_a_tokenizer_change_tokenizes_outside_the_write_lock(tmp_path, monkeypatch):
    kb = KnowledgeBaseManager(tmp_path, use_mock=True)
    for i in range(3):
        _ingest(kb, f"Note {i}", f"Herons wade at dawn number {i}.")
    kb.close()
    raw_write(tmp_path, "UPDATE info SET value = 'words-v2' WHERE key = 'tokenizer'")
    locked: list[bool] = []
    real = manager_module.tokenize

    def tokenize(text):
        other = sqlite3.connect(tmp_path / "kb.sqlite", timeout=0, isolation_level=None)
        try:
            other.execute("BEGIN IMMEDIATE")
            other.execute("ROLLBACK")
            locked.append(False)
        except sqlite3.OperationalError:
            locked.append(True)
        finally:
            other.close()
        return real(text)

    monkeypatch.setattr(manager_module, "tokenize", tokenize)
    reopened = KnowledgeBaseManager(tmp_path, use_mock=True)

    assert locked and not any(locked)
    assert read_rows(tmp_path, "SELECT value FROM info WHERE key = 'tokenizer'") == [("words-v3",)]
    assert len(reopened.search("herons")["results"]) == 3


def test_a_sidecar_rendered_again_keeps_the_document_s_chunk_count(tmp_path):
    # The row is read without its chunks; the sidecar still names how many there are.
    kb = KnowledgeBaseManager(tmp_path, use_mock=True)
    doc = _ingest(kb, "Otters", RIVER, sidecar_payload={"summary": "About otters.", "claims": [], "entities": {}})
    assert f"chunks: {len(doc.chunks)}\n" in kb.read_sidecar(doc.id)
    kb.sidecar_path(doc.id).unlink()

    assert f"chunks: {len(doc.chunks)}\n" in kb.read_sidecar(doc.id)
