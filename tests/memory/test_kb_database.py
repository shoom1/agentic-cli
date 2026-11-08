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
import struct
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

import pytest

from agentic_cli.memory.kb import _db
from agentic_cli.memory.kb._db import Database, KnowledgeBaseBusy, UnsafeKnowledgeBase
from agentic_cli.memory.kb._search import keyword_ranked
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


def _hot_journal(super_journal: Path) -> bytes:
    """A rollback journal SQLite takes for a hot one, naming ``super_journal``
    as its super-journal: playing it back, SQLite deletes that file."""
    name = str(super_journal).encode()
    magic = bytes([0xD9, 0xD5, 0x05, 0xF9, 0x20, 0xA1, 0x63, 0xD7])
    return (b"\x01" * 512 + struct.pack(">I", 0) + name + struct.pack(">I", len(name))
            + struct.pack(">I", sum(name) & 0xFFFFFFFF) + magic)


def test_a_shipped_rollback_journal_is_refused_before_sqlite_plays_it_back(tmp_path):
    victim = tmp_path / "outside" / "notes.txt"
    victim.parent.mkdir()
    victim.write_text("keep me\n")
    folder = tmp_path / "kb"
    folder.mkdir()
    Database.open(folder / "kb.sqlite").close()
    (folder / "kb.sqlite-journal").write_bytes(_hot_journal(victim))

    with pytest.raises(UnsafeKnowledgeBase, match="journal"):
        Database.open(folder / "kb.sqlite")
    assert victim.read_text() == "keep me\n"


def test_an_empty_rollback_journal_is_left_alone(tmp_path):
    Database.open(tmp_path / "kb.sqlite").close()
    (tmp_path / "kb.sqlite-journal").write_bytes(b"")

    Database.open(tmp_path / "kb.sqlite").close()


def test_a_database_swapped_for_a_link_after_the_check_is_refused(tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere.sqlite"
    Database.open(elsewhere).close()
    folder = tmp_path / "kb"
    folder.mkdir()
    (folder / "kb.sqlite").symlink_to(elsewhere)
    monkeypatch.setattr(_db, "check_files", lambda path: None)  # the link appears after the check

    with pytest.raises(UnsafeKnowledgeBase):
        Database.open(folder / "kb.sqlite")


def test_a_knowledge_base_reached_through_a_linked_folder_opens(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "linked").symlink_to(tmp_path / "real")

    Database.open(tmp_path / "linked" / "kb.sqlite").close()


def test_a_keyword_index_table_defined_otherwise_is_refused(tmp_path):
    Database.open(tmp_path / "kb.sqlite").close()
    raw_write(tmp_path, "DROP TABLE chunk_terms_data",
              "CREATE TABLE 'chunk_terms_data'(id INTEGER PRIMARY KEY, block BLOB CHECK (length(block) < 0))")

    with pytest.raises(UnsafeKnowledgeBase):
        Database.open(tmp_path / "kb.sqlite")


def test_keyword_index_tables_without_their_index_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(_db, "fts5_available", lambda conn: False)
    Database.open(tmp_path / "kb.sqlite").close()
    monkeypatch.undo()
    raw_write(tmp_path, "CREATE TABLE 'chunk_terms_data'(id INTEGER PRIMARY KEY, block BLOB)")

    with pytest.raises(UnsafeKnowledgeBase):
        Database.open(tmp_path / "kb.sqlite")


def test_a_refused_database_is_not_written_to(tmp_path):
    Database.open(tmp_path / "kb.sqlite").close()
    raw_write(tmp_path, "PRAGMA journal_mode = DELETE", "CREATE VIEW v AS SELECT id FROM documents")
    before = (tmp_path / "kb.sqlite").read_bytes()

    with pytest.raises(UnsafeKnowledgeBase):
        Database.open(tmp_path / "kb.sqlite")
    assert (tmp_path / "kb.sqlite").read_bytes() == before


def test_the_connection_is_opened_defensively(tmp_path):
    db = Database.open(tmp_path / "kb.sqlite")
    names = ("trusted_schema", "cell_size_check", "secure_delete", "mmap_size", "journal_mode")
    pragmas = {name: db.conn.execute(f"PRAGMA {name}").fetchone()[0] for name in names}
    db.close()

    assert pragmas == {"trusted_schema": 0, "cell_size_check": 1, "secure_delete": 1, "mmap_size": 0, "journal_mode": "wal"}


def test_a_keyword_index_this_sqlite_cannot_read_is_searched_by_scan_then_rebuilt(tmp_path, monkeypatch):
    # FTS5's secure-delete raises the index to a format SQLite before 3.44 cannot read.
    log = LogRecorder().install(monkeypatch, _db)
    _store(tmp_path, _doc(A, "Otters", "otters")).close()
    raw_write(tmp_path, "UPDATE chunk_terms_config SET v = 99 WHERE k = 'version'")

    db = _store(tmp_path, _doc(B, "Herons", "herons"))
    with db.read():
        found = [doc for _, doc in keyword_ranked(db.conn, ["herons"], fts=db.fts)]
    assert db.fts is False and found == [B, B]
    assert db.info()["keyword_index"] == "stale"
    assert len(log.warnings("kb_fts5_unavailable")) == 1
    db.close()

    raw_write(tmp_path, "UPDATE chunk_terms_config SET v = 4 WHERE k = 'version'")  # a SQLite that reads it
    db = Database.open(tmp_path / "kb.sqlite")
    assert db.fts is True and db.info()["keyword_index"] == "current"
    db.close()
    assert read_rows(tmp_path, "SELECT count(*) FROM chunk_terms WHERE chunk_terms MATCH '\"herons\"'") == [(2,)]


_OPENER = textwrap.dedent("""
    import sys, time
    from pathlib import Path
    from agentic_cli.memory.kb._db import Database
    start = float(sys.argv[2])
    while time.time() < start:
        time.sleep(0.0005)
    Database.open(Path(sys.argv[1])).close()
""")


@pytest.mark.parametrize("kind", ["new", "rollback-mode"])
def test_sessions_opening_one_database_at_once_all_open_it(tmp_path, kind):
    for attempt in range(2):
        path = tmp_path / f"{kind}-{attempt}" / "kb.sqlite"
        path.parent.mkdir()
        if kind == "rollback-mode":
            Database.open(path).close()
            raw_write(path.parent, "PRAGMA journal_mode = DELETE")
        start = time.time() + 2.0
        children = [
            subprocess.Popen([sys.executable, "-c", _OPENER, str(path), str(start)], stderr=subprocess.PIPE, text=True)
            for _ in range(4)
        ]
        results = [(child.wait(60), child.stderr.read()[-300:]) for child in children]

        assert [code for code, _ in results] == [0, 0, 0, 0], results
