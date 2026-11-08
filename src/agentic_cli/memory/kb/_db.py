"""The database of one knowledge base folder: ``kb.sqlite``.

It holds the documents, their chunks with each chunk's terms and vector, a
keyword index over the terms (FTS5) and what all of it was made with (the
``info`` table). It is the record of what the knowledge base holds: the
sidecars and ``index.md`` are rendered from it, and each session's search
vectors are loaded from it. Every change is one transaction, so a crash
leaves the database as it was before the change or after it.

A project can ship its knowledge base, database included, so the file is
opened as untrusted: never through a symlink, nor its ``-wal``, ``-shm`` and
``-journal`` companions (through a link there SQLite itself would write to a
file elsewhere), nor with a rollback journal SQLite would play back; in
SQLite's defensive mode, with triggers and views off; and only when its
schema is exactly this module's, checked before anything is written to it.

Deleted text does not stay on disk (``secure_delete``, the FTS5 table's own
``secure-delete``, and the manager's checkpoint after a delete).

A ``Database`` is not thread-safe: its manager holds a lock around each use.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import stat
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any

from agentic_cli.memory._core.io import create_private_file
from agentic_cli.memory._core.log import get_logger
from agentic_cli.memory.kb.models import Document, DocumentChunk, SourceType

logger = get_logger("agentic_cli.memory.kb")

DB_NAME = "kb.sqlite"
COMPANION_SUFFIXES = ("-wal", "-shm", "-journal")
FORMAT = "1"
# How long a change waits for another session's change before giving up.
BUSY_TIMEOUT_S = 60.0
# How long an open waits for a rollback journal to go: another session
# switching a new database to WAL keeps one for a moment.
JOURNAL_WAIT_S = 2.0

# Every document ID the framework assigns is a UUID; a row whose ID is not a
# plain name is never read, so never turned into a path.
PLAIN_DOC_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")

INFO_DDL = "CREATE TABLE info (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
DOCUMENTS_DDL = (
    "CREATE TABLE documents (id TEXT PRIMARY KEY, title TEXT NOT NULL, source_type TEXT NOT NULL, "
    "source_url TEXT, file_name TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, "
    "summary TEXT NOT NULL, metadata TEXT NOT NULL, enrichment TEXT, content TEXT NOT NULL)"
)
CHUNKS_DDL = (
    "CREATE TABLE chunks (rowid INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, document_id TEXT NOT NULL, "
    "position INTEGER NOT NULL, content TEXT NOT NULL, terms TEXT NOT NULL, vector BLOB)"
)
CHUNKS_INDEX_DDL = "CREATE INDEX chunks_by_document ON chunks(document_id)"
# The terms are split already (_tokenize.tokenize, space-separated): the ascii
# tokenizer splits at the spaces only, every non-ASCII character and "_"
# being part of a term. The external content table reads them from
# chunks.terms, so they are stored once.
FTS_DDL = (
    "CREATE VIRTUAL TABLE chunk_terms USING fts5(terms, content='chunks', content_rowid='rowid', "
    "tokenize=\"ascii tokenchars '_'\")"
)
_OWN = {
    "info": ("table", INFO_DDL),
    "documents": ("table", DOCUMENTS_DDL),
    "chunks": ("table", CHUNKS_DDL),
    "chunks_by_document": ("index", CHUNKS_INDEX_DDL),
}
_FTS_SHADOWS = frozenset({"chunk_terms_config", "chunk_terms_data", "chunk_terms_docsize", "chunk_terms_idx"})
_HEADER = "id, title, source_type, source_url, file_name, created_at, updated_at, summary, metadata"
_DEFENSIVE = (
    ("SQLITE_DBCONFIG_DEFENSIVE", True),
    ("SQLITE_DBCONFIG_TRUSTED_SCHEMA", False),
    ("SQLITE_DBCONFIG_ENABLE_TRIGGER", False),
    ("SQLITE_DBCONFIG_ENABLE_VIEW", False),
)


class UnsafeKnowledgeBase(OSError):
    """A knowledge base's folder holds a symlink, or something else, where one
    of its own folders (``documents``, ``files``, ``concepts``), its database
    (``kb.sqlite`` and companions) or its lock file belongs; or its database
    holds more than a knowledge base's schema (a trigger, a view, a table).

    A project can ship its knowledge base, and through a link the knowledge
    base would read, write and delete files elsewhere, so it does not work
    there: ``KnowledgeBase.open`` leaves that scope out, with a warning.
    """


class KnowledgeBaseBusy(TimeoutError):
    """Another session kept the knowledge base's write lock past ``BUSY_TIMEOUT_S``."""


class _RollbackJournal(UnsafeKnowledgeBase):
    """A non-empty ``kb.sqlite-journal``: SQLite would play it back."""


def refuse(path: Path, mode: int, kind: str) -> UnsafeKnowledgeBase:
    found = "a symbolic link" if stat.S_ISLNK(mode) else f"not {kind}"
    return UnsafeKnowledgeBase(f"{path} is {found}: a knowledge base works only in its own folders")


def check_files(db_path: Path) -> None:
    """Refuse a database file or companion that is not a regular file (a link,
    a folder), and a rollback journal that is not empty.

    SQLite plays a hot rollback journal back at its first read, and a crafted
    one names a "super-journal" that SQLite then deletes, wherever it is. A
    WAL database never leaves a journal behind, so a non-empty one was put
    there.

    Raises:
        UnsafeKnowledgeBase: one of them is.
    """
    for path in (db_path, *(db_path.with_name(db_path.name + s) for s in COMPANION_SUFFIXES)):
        try:
            found = os.lstat(path)
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(found.st_mode):
            raise refuse(path, found.st_mode, "a regular file")
        if path.name.endswith("-journal") and found.st_size:
            raise _RollbackJournal(
                f"{path} is a rollback journal, which a knowledge base never leaves: "
                "SQLite would play it back, so the knowledge base is not opened"
            )


def _check_files_at_open(db_path: Path) -> None:
    """``check_files``, giving a rollback journal ``JOURNAL_WAIT_S`` to go:
    another session switching a new database to WAL holds one for a moment."""
    deadline = time.monotonic() + JOURNAL_WAIT_S
    while True:
        try:
            check_files(db_path)
            return
        except _RollbackJournal:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def _is_busy(error: sqlite3.Error) -> bool:
    """Whether ``error`` is SQLite's "busy" or "locked" (extended codes included)."""
    code = getattr(error, "sqlite_errorcode", None)
    return code is not None and code & 0xFF in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)


def _busy_error(path: Path) -> KnowledgeBaseBusy:
    return KnowledgeBaseBusy(f"{path}: another session kept the knowledge base locked for {BUSY_TIMEOUT_S:g} s")


def _check_opened(conn: sqlite3.Connection, path: Path) -> None:
    """Refuse a connection whose database is not the regular file at ``path``.

    SQLite has no option that refuses a symlink for the database itself (it
    does for ``-wal`` and ``-shm``), so a link put there after
    ``check_files`` would be followed: the file SQLite opened (``PRAGMA
    database_list``) must be the very file at ``path``.
    """
    opened = next((row[2] for row in conn.execute("PRAGMA database_list") if row[1] == "main"), "")
    try:
        theirs, ours = os.stat(opened), os.lstat(path)
    except OSError:
        theirs = ours = None
    if theirs is None or not stat.S_ISREG(ours.st_mode) or (theirs.st_dev, theirs.st_ino) != (ours.st_dev, ours.st_ino):
        raise UnsafeKnowledgeBase(f"{path} is not the file SQLite opened ({opened}): a knowledge base works only in its own folders")


def _use_wal(conn: sqlite3.Connection, path: Path) -> None:
    """Switch to WAL, waiting for other sessions up to ``BUSY_TIMEOUT_S``.

    SQLite's busy timeout does not cover this switch: on a new or
    rollback-mode file another session switching it at once makes it fail at
    once, so it is tried again until the timeout.

    Raises:
        KnowledgeBaseBusy: other sessions kept it locked that long.
        sqlite3.OperationalError: the file stays in another mode (a
            filesystem without the shared memory WAL needs).
    """
    deadline = time.monotonic() + BUSY_TIMEOUT_S
    while True:
        try:
            mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            break
        except sqlite3.OperationalError as e:
            if not _is_busy(e):
                raise
            if time.monotonic() >= deadline:
                raise _busy_error(path) from e
            time.sleep(0.01)
    if str(mode).lower() != "wal":
        raise sqlite3.OperationalError(f"{path} stays in journal mode {mode!r}: a knowledge base needs WAL")


def fts5_available(conn: sqlite3.Connection) -> bool:
    """Whether this SQLite has FTS5 (nearly every build does)."""
    try:
        conn.execute("CREATE VIRTUAL TABLE temp.kb_fts5_probe USING fts5(x)")
    except sqlite3.OperationalError:
        return False
    conn.execute("DROP TABLE temp.kb_fts5_probe")
    return True


def naive(moment: datetime) -> datetime:
    """``moment`` as naive local time, the form every stored time has."""
    return moment if moment.tzinfo is None else moment.astimezone().replace(tzinfo=None)


def to_text(moment: datetime) -> str:
    """How a time is stored: naive local ISO 8601, so stored times compare as text."""
    return naive(moment).isoformat()


def _casefold(text: Any) -> Any:
    return text.casefold() if isinstance(text, str) else text


def _normalized(sql: str | None) -> str | None:
    return None if sql is None else " ".join(sql.split())


@cache
def _fts_shadow_tables() -> dict[str, str | None]:
    """The tables FTS5 makes for ``chunk_terms``, each with the definition this
    SQLite gives it (made in a scratch database to see). Without FTS5, only
    their names are known (such a session never reads them)."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(FTS_DDL)
        rows = conn.execute("SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name != 'chunk_terms'")
        return {str(name): _normalized(sql) for name, sql in rows}
    except sqlite3.OperationalError:
        return dict.fromkeys(_FTS_SHADOWS)
    finally:
        conn.close()


class Database:
    """One knowledge base folder's ``kb.sqlite``, through one connection."""

    def __init__(self, conn: sqlite3.Connection, path: Path, *, fts_module: bool) -> None:
        self.conn = conn
        self.path = path
        self._fts_module = fts_module
        self._fts_table = False
        # False when this SQLite cannot read the index (the format FTS5's
        # secure-delete writes is unreadable before SQLite 3.44).
        self._fts_readable = True
        # Before SQLite 3.44 FTS5 has no secure-delete: merge the index after a delete instead.
        self.optimize_after_delete = False
        self._reported: set[str] = set()

    @property
    def fts(self) -> bool:
        """Whether keyword search can use the FTS5 index: its module loads,
        the table exists and this SQLite reads its format."""
        return self._fts_module and self._fts_table and self._fts_readable

    # -- opening -----------------------------------------------------------

    @classmethod
    def open(cls, path: Path) -> "Database":
        """Open the knowledge base database at ``path``, creating it when missing.

        The file is checked (``check_files``, the schema, the format) before
        anything is written to it, the switch to WAL included.

        Raises:
            UnsafeKnowledgeBase: the file or a companion is a link or not a
                regular file, a rollback journal is there, or the schema is
                not this module's.
            ValueError: the database is of a format this version does not read.
            KnowledgeBaseBusy: other sessions kept it locked past ``BUSY_TIMEOUT_S``.
            sqlite3.DatabaseError: the file is not a database (a git-LFS
                pointer) or is corrupt.
        """
        _check_files_at_open(path)
        create_private_file(path)
        conn = sqlite3.connect(
            path.absolute().as_uri() + "?mode=rw",
            uri=True,
            timeout=BUSY_TIMEOUT_S,
            isolation_level=None,
            check_same_thread=False,
        )
        try:
            _check_opened(conn, path)
            for name, value in _DEFENSIVE:
                option = getattr(sqlite3, name, None)
                if option is not None:
                    conn.setconfig(option, value)
            conn.execute("PRAGMA trusted_schema = OFF")
            conn.execute("PRAGMA mmap_size = 0")
            conn.execute("PRAGMA secure_delete = ON")
            conn.execute("PRAGMA cell_size_check = ON")
            conn.create_function("kb_casefold", 1, _casefold, deterministic=True)
            db = cls(conn, path, fts_module=fts5_available(conn))
            db._check()
            _use_wal(conn, path)
            db._prepare()
            return db
        except BaseException:
            conn.close()
            raise

    def _check(self) -> set[str]:
        """The schema's names, after checking the schema and the format (none in a new file).

        Raises:
            UnsafeKnowledgeBase: see :meth:`_check_schema`.
            ValueError: a format this version does not read.
        """
        names = self._check_schema()
        if names:
            found = self.info().get("format")
            if found != FORMAT:
                raise ValueError(f"{self.path} is a knowledge base of format {found!r}; this version reads format {FORMAT}")
        return names

    def _prepare(self) -> None:
        """Create the schema in a new database (checked again: another session
        may have created it meanwhile); then the keyword index."""
        with self.write():
            names = self._check()
            if not names:
                for ddl in (INFO_DDL, DOCUMENTS_DDL, CHUNKS_DDL, CHUNKS_INDEX_DDL):
                    self.conn.execute(ddl)
                self.set_info(format=FORMAT, generation="0", keyword_index="current")
                names = set(_OWN)
            self._fts_table = "chunk_terms" in names
            if not self._fts_module:
                logger.warning("kb_fts5_unavailable", path=str(self.path))
                return
            if not self._fts_table:
                self.conn.execute(FTS_DDL)
                self._fts_table = True
                self.rebuild_keyword_index()
            elif not self._keyword_index_readable():
                self._fts_readable = False
                logger.warning("kb_fts5_unavailable", path=str(self.path),
                               reason="this SQLite cannot read the keyword index's format; a newer one rebuilds it")
                return
            elif self.info().get("keyword_index") == "stale":
                self.rebuild_keyword_index()
            if sqlite3.sqlite_version_info >= (3, 44, 0):
                self.conn.execute("INSERT INTO chunk_terms (chunk_terms, rank) VALUES ('secure-delete', 1)")
            else:
                self.optimize_after_delete = True

    def _keyword_index_readable(self) -> bool:
        """Whether this SQLite reads the keyword index: the format FTS5's
        secure-delete writes (from SQLite 3.44) is one an older SQLite refuses."""
        try:
            self.conn.execute("SELECT rowid FROM chunk_terms WHERE chunk_terms MATCH '\"kb\"' LIMIT 1").fetchall()
        except sqlite3.OperationalError:
            return False
        return True

    def _check_schema(self) -> set[str]:
        """The schema's names, after checking each one is this module's.

        Raises:
            UnsafeKnowledgeBase: anything else (a trigger, a view, another
                table or index, one of ours defined otherwise), or ours missing.
        """
        shadows = _fts_shadow_tables()
        names: set[str] = set()
        for kind, name, table, sql in self.conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_master"):
            names.add(name)
            own = _OWN.get(name)
            if own is not None and own[0] == kind and _normalized(sql) == _normalized(own[1]):
                continue
            if name == "chunk_terms" and kind == "table" and _normalized(sql) == _normalized(FTS_DDL):
                continue
            if name in shadows and kind == "table" and table == name and shadows[name] in (None, _normalized(sql)):
                continue
            if kind == "index" and sql is None and name.startswith("sqlite_autoindex_") and table in ("info", "documents", "chunks"):
                continue
            raise UnsafeKnowledgeBase(f"{self.path}: its {kind} {name!r} is not part of a knowledge base")
        if names and not set(_OWN) <= names:
            raise UnsafeKnowledgeBase(f"{self.path} is not a knowledge base: {sorted(set(_OWN) - names)} missing")
        present = names & set(shadows)
        if present != (set(shadows) if "chunk_terms" in names else set()):
            raise UnsafeKnowledgeBase(f"{self.path}: its keyword index tables do not match its keyword index")
        return names

    def close(self) -> None:
        """Close the connection (idempotent)."""
        self.conn.close()

    # -- transactions --------------------------------------------------------

    @contextmanager
    def write(self) -> Iterator["Database"]:
        """One change: committed when the block ends, rolled back if it raises.

        Raises:
            KnowledgeBaseBusy: another session held the write lock for
                ``BUSY_TIMEOUT_S``.
        """
        try:
            self.conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as e:
            if _is_busy(e):
                raise _busy_error(self.path) from e
            raise
        try:
            yield self
            self.conn.execute("COMMIT")
        except BaseException:
            if self.conn.in_transaction:
                self.conn.execute("ROLLBACK")
            raise

    @contextmanager
    def read(self) -> Iterator["Database"]:
        """A consistent view for several reads (WAL: it never waits for a writer)."""
        self.conn.execute("BEGIN")
        try:
            yield self
        finally:
            if self.conn.in_transaction:
                self.conn.execute("COMMIT")

    # -- info ----------------------------------------------------------------

    def info(self) -> dict[str, str]:
        return {str(key): str(value) for key, value in self.conn.execute("SELECT key, value FROM info")}

    def set_info(self, **values: str) -> None:
        self.conn.executemany("INSERT OR REPLACE INTO info (key, value) VALUES (?, ?)", list(values.items()))

    def generation(self) -> int:
        """A counter every change to the chunks or vectors increments."""
        row = self.conn.execute("SELECT value FROM info WHERE key = 'generation'").fetchone()
        try:
            return int(row[0]) if row else 0
        except (TypeError, ValueError):
            return 0

    def bump_generation(self) -> int:
        value = self.generation() + 1
        self.set_info(generation=str(value))
        return value

    # -- changes -------------------------------------------------------------

    def insert_document(
        self,
        doc: Document,
        terms: Sequence[str],
        vectors: Sequence[bytes] | None,
        enrichment: dict[str, Any] | None,
    ) -> list[int]:
        """Insert ``doc`` and its chunks, each with its terms and (when given)
        its vector; returns the chunks' rowids, in order."""
        self.conn.execute(
            "INSERT INTO documents (id, title, source_type, source_url, file_name, created_at, updated_at, "
            "summary, metadata, enrichment, content) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                doc.id, doc.title, doc.source_type.value, doc.source_url,
                doc.file_path.name if doc.file_path else None,
                to_text(doc.created_at), to_text(doc.updated_at), doc.summary,
                json.dumps(doc.metadata), None if enrichment is None else json.dumps(enrichment), doc.content,
            ),
        )
        rowids: list[int] = []
        for i, chunk in enumerate(doc.chunks):
            cursor = self.conn.execute(
                "INSERT INTO chunks (id, document_id, position, content, terms, vector) VALUES (?, ?, ?, ?, ?, ?)",
                (chunk.id, doc.id, chunk.chunk_index, chunk.content, terms[i], None if vectors is None else vectors[i]),
            )
            rowid = int(cursor.lastrowid)
            rowids.append(rowid)
            if self.fts:
                self.conn.execute("INSERT INTO chunk_terms (rowid, terms) VALUES (?, ?)", (rowid, terms[i]))
        if rowids and self._fts_table and not self.fts:
            self.set_info(keyword_index="stale")
        return rowids

    def delete_document(self, doc_id: str) -> list[int]:
        """Delete a document's rows (readable or not); returns its chunks' rowids."""
        rows = self.conn.execute("SELECT rowid, terms FROM chunks WHERE document_id = ?", (doc_id,)).fetchall()
        if self.fts:
            self.conn.executemany("INSERT INTO chunk_terms (chunk_terms, rowid, terms) VALUES ('delete', ?, ?)", rows)
            if rows and self.optimize_after_delete:
                self.conn.execute("INSERT INTO chunk_terms (chunk_terms) VALUES ('optimize')")
        elif rows and self._fts_table:
            self.set_info(keyword_index="stale")
        self.conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        self.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        return [int(rowid) for rowid, _ in rows]

    def delete_all(self) -> None:
        if self.fts:
            self.conn.execute("INSERT INTO chunk_terms (chunk_terms) VALUES ('delete-all')")
        elif self._fts_table:
            self.set_info(keyword_index="stale")
        self.conn.execute("DELETE FROM chunks")
        self.conn.execute("DELETE FROM documents")

    def set_enrichment(self, doc_id: str, payload: dict[str, Any]) -> None:
        self.conn.execute("UPDATE documents SET enrichment = ? WHERE id = ?", (json.dumps(payload), doc_id))

    def set_vectors(self, pairs: Iterable[tuple[int, bytes]]) -> None:
        """Store vectors for chunks that still have none (another session may have embedded them)."""
        self.conn.executemany(
            "UPDATE chunks SET vector = ? WHERE rowid = ? AND vector IS NULL",
            [(blob, rowid) for rowid, blob in pairs],
        )

    def clear_vectors(self) -> None:
        self.conn.execute("UPDATE chunks SET vector = NULL WHERE vector IS NOT NULL")

    def clear_vectors_of_wrong_size(self, size: int) -> int:
        """Clear vectors that are not ``size`` bytes (a hand-made or shipped row); they are embedded again."""
        return self.conn.execute(
            "UPDATE chunks SET vector = NULL WHERE vector IS NOT NULL AND (typeof(vector) != 'blob' OR length(vector) != ?)",
            (size,),
        ).rowcount

    def recompute_terms(self, terms_of: Callable[[str], str]) -> None:
        """Every chunk's terms again (a new tokenizer), and the keyword index from them."""
        rows = self.conn.execute("SELECT rowid, content FROM chunks").fetchall()
        self.conn.executemany(
            "UPDATE chunks SET terms = ? WHERE rowid = ?",
            [(terms_of(text if isinstance(text, str) else ""), rowid) for rowid, text in rows],
        )
        if self.fts:
            self.rebuild_keyword_index()
        elif self._fts_table:
            self.set_info(keyword_index="stale")

    def rebuild_keyword_index(self) -> None:
        self.conn.execute("INSERT INTO chunk_terms (chunk_terms) VALUES ('rebuild')")
        self.set_info(keyword_index="current")

    # -- reads ---------------------------------------------------------------

    def _report(self, doc_id: Any, error: Exception) -> None:
        key = repr(doc_id)
        if key not in self._reported:
            self._reported.add(key)
            logger.warning("kb_row_unreadable", path=str(self.path), doc_id=key, error=f"{type(error).__name__}: {error}")

    def _document(self, row: Sequence[Any], *, content: str = "", chunk_count: int = 0) -> Document | None:
        """The document in a ``_HEADER`` row, or None (reported once) when it does not parse."""
        try:
            doc_id, title, source_type, source_url, file_name, created, updated, summary, metadata = row
            if not isinstance(doc_id, str) or not PLAIN_DOC_ID.fullmatch(doc_id):
                raise ValueError("its ID is not a plain name")
            if not all(isinstance(v, str) for v in (title, created, updated, summary)):
                raise TypeError("a text column holds something else")
            if not all(v is None or isinstance(v, str) for v in (source_url, file_name)):
                raise TypeError("a text column holds something else")
            meta = json.loads(metadata)
            if not isinstance(meta, dict):
                raise TypeError("its metadata is not an object")
            return Document(
                id=doc_id, title=title, content=content, source_type=SourceType(source_type), summary=summary,
                source_url=source_url, file_path=Path(file_name) if file_name else None,
                created_at=naive(datetime.fromisoformat(created)), updated_at=naive(datetime.fromisoformat(updated)),
                metadata=meta, chunks=[], chunk_count=chunk_count,
            )
        except Exception as e:  # any malformed value: the row is skipped, never rewritten
            self._report(row[0] if row else None, e)
            return None

    def header(self, doc_id: str) -> Document | None:
        """A document without its text or chunks (with its chunk count)."""
        row = self.conn.execute(
            f"SELECT {_HEADER}, (SELECT count(*) FROM chunks WHERE document_id = documents.id) FROM documents WHERE id = ?",
            (doc_id,),
        ).fetchone()
        return None if row is None else self._document(row[:-1], chunk_count=int(row[-1]))

    def document(self, doc_id: str) -> Document | None:
        """The whole document: its text and its chunks."""
        row = self.conn.execute(f"SELECT {_HEADER}, content FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if row is None:
            return None
        doc = self._document(row[:-1], content=row[-1] if isinstance(row[-1], str) else "")
        if doc is None:
            return None
        doc.chunks = [
            DocumentChunk(id=str(cid), document_id=doc.id, content=text, chunk_index=int(position), metadata={"title": doc.title})
            for cid, position, text in self.conn.execute(
                "SELECT id, position, content FROM chunks WHERE document_id = ? ORDER BY position", (doc.id,)
            )
            if isinstance(text, str)
        ]
        doc.chunk_count = len(doc.chunks)
        return doc

    def headers(self, *, source_type: SourceType | None = None, limit: int | None = None) -> list[Document]:
        """Every readable document without its text or chunks, the latest change first."""
        sql = f"SELECT {_HEADER}, (SELECT count(*) FROM chunks WHERE document_id = documents.id) FROM documents"
        params: list[Any] = []
        if source_type is not None:
            sql += " WHERE source_type = ?"
            params.append(source_type.value)
        sql += " ORDER BY updated_at DESC, id"
        docs = [d for row in self.conn.execute(sql, params).fetchall() if (d := self._document(row[:-1], chunk_count=int(row[-1]))) is not None]
        return docs if limit is None else docs[:limit]

    def find(self, ref: str) -> Document | None:
        """By exact ID, then ID prefix, then title substring (case-insensitive); the oldest match."""
        found = self.document(ref)
        if found is not None:
            return found
        for sql, params in (
            ("SELECT id FROM documents WHERE substr(id, 1, ?) = ? ORDER BY created_at, id", (len(ref), ref)),
            ("SELECT id FROM documents WHERE instr(kb_casefold(title), ?) > 0 ORDER BY created_at, id", (ref.casefold(),)),
        ):
            for (doc_id,) in self.conn.execute(sql, params).fetchall():
                found = self.document(doc_id)
                if found is not None:
                    return found
        return None

    def enrichment(self, doc_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT enrichment FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if row is None or row[0] is None:
            return None
        try:
            value = json.loads(row[0])
        except (TypeError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def document_ids(self) -> set[str]:
        """Every row's ID, readable or not: the leftover sweep keeps their files."""
        return {str(row[0]) for row in self.conn.execute("SELECT id FROM documents")}

    def file_names(self) -> set[str]:
        return {str(row[0]) for row in self.conn.execute("SELECT file_name FROM documents WHERE file_name IS NOT NULL")}

    def chunks_without_vector(self, *, after: int, limit: int) -> list[tuple[int, str]]:
        return [
            (int(rowid), text)
            for rowid, text in self.conn.execute(
                "SELECT rowid, content FROM chunks WHERE vector IS NULL AND rowid > ? AND typeof(content) = 'text' "
                "ORDER BY rowid LIMIT ?",
                (after, limit),
            )
        ]

    def count_without_vector(self) -> int:
        return int(self.conn.execute("SELECT count(*) FROM chunks WHERE vector IS NULL").fetchone()[0])

    def vectors(self) -> list[tuple[int, str, bytes]]:
        return self.conn.execute(
            "SELECT rowid, document_id, vector FROM chunks WHERE vector IS NOT NULL ORDER BY rowid"
        ).fetchall()

    def hits(self, rowids: Sequence[int]) -> dict[int, tuple[Document, DocumentChunk]]:
        """Each chunk's document header and chunk; rows that do not parse are left out."""
        if not rowids:
            return {}
        marks = ",".join("?" * len(rowids))
        headers: dict[str, Document | None] = {}
        found: dict[int, tuple[Document, DocumentChunk]] = {}
        rows = self.conn.execute(
            f"SELECT rowid, id, document_id, position, content FROM chunks WHERE rowid IN ({marks})",
            [int(r) for r in rowids],
        ).fetchall()
        for rowid, chunk_id, doc_id, position, text in rows:
            if doc_id not in headers:
                headers[doc_id] = self.header(doc_id)
            doc = headers[doc_id]
            if doc is None or not isinstance(text, str):
                continue
            found[int(rowid)] = (
                doc,
                DocumentChunk(id=str(chunk_id), document_id=doc.id, content=text, chunk_index=int(position), metadata={"title": doc.title}),
            )
        return found

    def counts(self) -> dict[str, Any]:
        documents = int(self.conn.execute("SELECT count(*) FROM documents").fetchone()[0])
        chunks, vectors = self.conn.execute("SELECT count(*), count(vector) FROM chunks").fetchone()
        sources = {str(k): int(v) for k, v in self.conn.execute("SELECT source_type, count(*) FROM documents GROUP BY source_type")}
        return {"documents": documents, "chunks": int(chunks), "vectors": int(vectors), "sources": sources}
