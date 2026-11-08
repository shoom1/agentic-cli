"""``kb.sqlite``: created private, opened as untrusted, rows read one by one.

A project can ship its knowledge base, so the database is never opened
through a link (nor its -wal/-shm/-journal companions, through which SQLite
would write elsewhere) and only when its schema is exactly ours. A row that
does not parse is skipped, and no write touches it (review K1).
"""

from __future__ import annotations

import os
import sqlite3
import stat
from datetime import datetime
from pathlib import Path

import pytest

from agentic_cli.memory.kb import _db
from agentic_cli.memory.kb._db import Database, KnowledgeBaseBusy, UnsafeKnowledgeBase
from agentic_cli.memory.kb.models import Document, DocumentChunk, SourceType
from tests.memory.kb_storage import A, B, C, LogRecorder, raw_write, read_rows


def _doc(doc_id: str, title: str, text: str, chunks: int = 2) -> Document:
    doc = Document(id=doc_id, title=title, content=text, source_type=SourceType.USER, summary=title,
                   created_at=datetime(2026, 11, 1, 9), updated_at=datetime(2026, 11, 1, 9))
    doc.chunks = [DocumentChunk(id=f"{doc_id}-{i}", document_id=doc_id, content=f"{text} {i}", chunk_index=i)
                  for i in range(chunks)]
    return doc


def _store(path: Path, *docs: Document) -> Database:
    db = Database.open(path / "kb.sqlite")
    with db.write():
        for doc in docs:
            db.insert_document(doc, [c.content.lower() for c in doc.chunks], None, None)
    return db


def test_a_new_database_is_private_and_of_this_format(tmp_path):
    db = Database.open(tmp_path / "kb.sqlite")
    db.close()

    assert stat.S_IMODE(os.stat(tmp_path / "kb.sqlite").st_mode) == 0o600
    assert dict(read_rows(tmp_path, "SELECT key, value FROM info"))["format"] == "1"


def test_rows_read_back_as_written(tmp_path):
    db = _store(tmp_path, _doc(A, "Otters", "otters nest"))
    with db.read():
        whole, header = db.document(A), db.header(A)

    assert [c.content for c in whole.chunks] == ["otters nest 0", "otters nest 1"]
    assert whole.content == "otters nest" and whole.chunk_count == 2
    assert header.content == "" and header.chunks == [] and header.chunk_count == 2


def test_a_database_that_links_elsewhere_is_refused(tmp_path):
    elsewhere = tmp_path / "elsewhere.sqlite"
    Database.open(elsewhere).close()
    folder = tmp_path / "kb"
    folder.mkdir()
    (folder / "kb.sqlite").symlink_to(elsewhere)

    with pytest.raises(UnsafeKnowledgeBase):
        Database.open(folder / "kb.sqlite")


def test_a_wal_file_that_links_elsewhere_is_refused(tmp_path):
    Database.open(tmp_path / "kb.sqlite").close()
    target = tmp_path / "shell-rc"
    target.write_text("")
    (tmp_path / "kb.sqlite-wal").symlink_to(target)

    with pytest.raises(UnsafeKnowledgeBase):
        Database.open(tmp_path / "kb.sqlite")
    assert target.read_text() == ""


@pytest.mark.parametrize("statement", [
    "CREATE TRIGGER t AFTER INSERT ON chunks BEGIN DELETE FROM documents; END",
    "CREATE VIEW v AS SELECT id FROM documents",
    "CREATE TABLE extra (x)",
])
def test_a_shipped_database_with_anything_else_in_it_is_refused(tmp_path, statement):
    Database.open(tmp_path / "kb.sqlite").close()
    raw_write(tmp_path, statement)

    with pytest.raises(UnsafeKnowledgeBase):
        Database.open(tmp_path / "kb.sqlite")


def test_a_file_that_is_not_a_database_does_not_open(tmp_path):
    (tmp_path / "kb.sqlite").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:0\n")

    with pytest.raises(sqlite3.DatabaseError):
        Database.open(tmp_path / "kb.sqlite")


def test_a_database_of_another_format_does_not_open(tmp_path):
    Database.open(tmp_path / "kb.sqlite").close()
    raw_write(tmp_path, "UPDATE info SET value = '2' WHERE key = 'format'")

    with pytest.raises(ValueError, match="format"):
        Database.open(tmp_path / "kb.sqlite")


def test_a_row_that_does_not_parse_is_skipped_and_kept(tmp_path, monkeypatch):
    log = LogRecorder().install(monkeypatch, _db)
    _store(tmp_path, _doc(A, "Otters", "otters")).close()
    raw_write(tmp_path, ("INSERT INTO documents VALUES (?, 'Podcast', 'podcast', NULL, NULL, "
                         "'2026-11-01T09:00:00', '2026-11-01T09:00:00', '', '{}', NULL, 'x')", [B]))

    db = _store(tmp_path, _doc(C, "Herons", "herons"))
    with db.read():
        titles = [d.title for d in db.headers()]

    assert sorted(titles) == ["Herons", "Otters"]
    assert read_rows(tmp_path, "SELECT source_type FROM documents WHERE id = ?", [B]) == [("podcast",)]
    assert len(log.warnings("kb_row_unreadable")) == 1


def test_a_change_waits_for_another_session_s_then_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(_db, "BUSY_TIMEOUT_S", 0.2)
    db = Database.open(tmp_path / "kb.sqlite")
    other = sqlite3.connect(tmp_path / "kb.sqlite", isolation_level=None)
    other.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(KnowledgeBaseBusy):
            with db.write():
                pass
    finally:
        other.execute("ROLLBACK")
        other.close()
        db.close()


def test_without_fts5_the_database_opens_and_marks_the_index_stale(tmp_path, monkeypatch):
    _store(tmp_path, _doc(A, "Otters", "otters")).close()
    monkeypatch.setattr(_db, "fts5_available", lambda conn: False)

    db = _store(tmp_path, _doc(B, "Herons", "herons"))
    assert db.fts is False
    assert db.info()["keyword_index"] == "stale"
    db.close()

    monkeypatch.undo()
    db = Database.open(tmp_path / "kb.sqlite")
    assert db.fts is True and db.info()["keyword_index"] == "current"
    assert read_rows(tmp_path, "SELECT count(*) FROM chunk_terms WHERE chunk_terms MATCH '\"herons\"'") == [(2,)]
