"""Knowledge Base Manager: the knowledge base kept in one folder.

The folder:
    kb.sqlite          the record: documents, chunks, vectors, keyword index (``_db``)
    documents/<id>.md  each document's sidecar, rendered from the database
    files/<id>.<ext>   stored originals
    concepts/          concept pages (``ConceptStore``)
    index.md           the catalogue, rendered from the database
    ingest_log.md      one line per ingest and delete
"""

from __future__ import annotations

import asyncio
import os
import re
import sqlite3
import stat
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from agentic_cli.memory._core.chunking import chunk_text
from agentic_cli.memory._core.embeddings import EmbeddingConfig, EmbeddingService
from agentic_cli.memory._core.io import (
    append_text,
    atomic_write_bytes,
    atomic_write_text,
    is_regular_file,
    read_if_regular,
)
from agentic_cli.memory._core.log import get_logger
from agentic_cli.memory._core.mock_embeddings import MockEmbeddingService
from agentic_cli.memory.kb import _search, enrich
from agentic_cli.memory.kb._db import (  # noqa: F401 - KnowledgeBaseBusy, UnsafeKnowledgeBase: re-exported
    DB_NAME,
    Database,
    KnowledgeBaseBusy,
    UnsafeKnowledgeBase,
    check_files,
    refuse,
)
from agentic_cli.memory.kb._tokenize import TOKENIZER, tokenize
from agentic_cli.memory.kb._vectors import VectorIndex, normalized, to_blob
from agentic_cli.memory.kb.models import Document, DocumentChunk, SearchResult, SourceType
from agentic_cli.memory.kb.sidecar import render_index_md, render_sidecar_markdown

logger = get_logger("agentic_cli.memory.kb")

# How long the checkpoint after a delete waits for other sessions' reads; one
# it could not finish is tried again at the next delete and at close.
CHECKPOINT_WAIT_S = 2.0

_UUID = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_SIDECAR_NAME = re.compile(rf"({_UUID})\.md")
_ORIGINAL_NAME = re.compile(rf"({_UUID})\.[A-Za-z0-9]{{1,16}}")
_TEMP_NAME = re.compile(rf"\.(?:{_UUID}\.[A-Za-z0-9]{{1,16}}|index\.md)\..+\.tmp")


class Summarizer(Protocol):
    """What a knowledge base calls to write summaries and sidecars.

    ``prompt`` is the full instruction the knowledge base wants answered for
    ``content``. Whether answering it means an LLM call is up to the
    implementation.
    """

    async def summarize(self, content: str, prompt: str) -> str: ...


def _utc_iso_now() -> str:
    """Return current UTC time as ISO-8601 with a trailing 'Z'."""
    from datetime import timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _printable(text: str) -> str:
    """``text`` with control characters as spaces and whitespace runs collapsed."""
    return " ".join("".join(ch if ch.isprintable() else " " for ch in text).split())


def _folders_below(within: Path, base_dir: Path) -> tuple[Path, ...]:
    """Each folder from just below ``within`` down to ``base_dir``, in order.

    Raises:
        ValueError: ``base_dir`` is not inside ``within`` (a ``..`` counts as
            leaving it).
    """
    try:
        parts = base_dir.relative_to(within).parts
    except ValueError:
        raise ValueError(f"{base_dir} is not inside {within}") from None
    if ".." in parts:
        raise ValueError(f"{base_dir} is not inside {within}")
    folders: list[Path] = []
    for part in parts:
        folders.append((folders[-1] if folders else within) / part)
    return tuple(folders)


def _truncate(text: str, max_length: int = 200) -> str:
    """``text`` cut to ``max_length`` characters, with "..." when cut."""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


def _terms_of(text: str) -> str:
    """A chunk's terms as stored: the tokenizer's terms, space-separated."""
    return " ".join(tokenize(text))


def _loaded_or_keyword(service: Any) -> Any:
    """``service`` with its model loaded, or a mock service for keyword search.

    Reading ``embedding_dim`` loads an ``EmbeddingService``'s model. Whatever
    stops it (an offline first run with the model not cached, a broken
    install, a device that fails) is logged once, and the knowledge base
    searches by keyword rather than not opening.
    """
    if isinstance(service, MockEmbeddingService):
        return service
    try:
        service.embedding_dim
    except Exception as e:
        model = getattr(service, "model_name", "unknown")
        logger.warning("kb_keyword_search_only", reason="the embedding model did not load", model=model,
                       error=f"{type(e).__name__}: {e}")
        return MockEmbeddingService(model_name=model, batch_size=getattr(service, "batch_size", 32))
    return service


class _Threads:
    """One async operation's blocking calls, each on a worker thread in turn.

    Cancelling the coroutine that awaits a call does not stop its thread.
    What the operation holds (a sidecar lock, the backfill flag) is let go
    by :meth:`release_after`: at once, or when the call still running ends,
    so a second caller cannot start the same work while the first one's
    thread still writes.
    """

    def __init__(self) -> None:
        self._running: asyncio.Future[Any] | None = None

    async def run(self, func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        """``func(*args, **kwargs)`` on a worker thread."""
        self._running = asyncio.ensure_future(asyncio.to_thread(func, *args, **kwargs))
        return await asyncio.shield(self._running)

    def release_after(self, release: Callable[[], None]) -> None:
        """Call ``release`` now, or when the call still running ends."""
        running = self._running
        if running is None or running.done():
            release()
            return

        def ended(future: asyncio.Future[Any]) -> None:
            if not future.cancelled() and future.exception() is not None:
                logger.debug("kb_cancelled_call_failed", exc_info=future.exception())
            release()

        running.add_done_callback(ended)


class KnowledgeBaseManager:
    """One knowledge base folder: ingestion, hybrid search, documents, sidecars.

    Concurrency
    -----------
    * ``_lock`` (``threading.Lock``) keeps this manager's threads apart
      around every use of its database connection and of its search
      vectors. Changes from other processes are kept apart by SQLite's write
      lock (``Database.write``), which a change waits for up to
      ``BUSY_TIMEOUT_S``; a read never waits. Embedding happens before a
      change takes the write lock, so a change holds it briefly.
    * ``_quick_lock`` guards what the event loop itself touches: the
      ``_sidecar_locks`` inserts and the check-and-set of
      ``_backfill_running``. Never held across I/O.
    * ``_sidecar_locks`` (``dict[str, asyncio.Lock]``): per-document locks
      that keep two sidecar generations for one document apart; obtained
      through :meth:`get_or_create_sidecar_lock`. Bound to the event loop
      they were made on.
    """

    _EMBED_BATCH = 256
    _TERMS_BATCH = 1000

    def __init__(
        self,
        base_dir: Path | str,
        *,
        embedding: EmbeddingConfig | None = None,
        summarizer: "Summarizer | None" = None,
        use_mock: bool = False,
        embedding_service: Any = None,
        within: Path | str | None = None,
    ) -> None:
        """Open (or create) the knowledge base kept in ``base_dir``.

        Args:
            base_dir: Its folder (``kb.sqlite``, ``documents/``, ``files/``,
                ``concepts/``).
            embedding: The model to load when it builds its own embedding
                service. Defaults to ``EmbeddingConfig()``.
            summarizer: Called to write summaries and sidecars. Without one,
                the summary is the document's first ~500 characters.
            use_mock: Search by keyword only: nothing is embedded.
            embedding_service: A ready service, for example one shared with
                another knowledge base. A ``MockEmbeddingService`` means
                keyword-only too.
            within: A folder the caller trusts, which ``base_dir`` lies in
                (for a project's knowledge base, the working directory).
                Every folder from it down to ``base_dir`` must then be a real
                folder, never a symlink: a repository can ship
                ``.{app}/knowledge_base`` linked to a folder elsewhere.
                Without it, ``base_dir`` itself may be a link (a knowledge
                base kept on another disk).

        Raises:
            ValueError: ``base_dir`` is not inside ``within``, or its database
                is of a format this version does not read.
            UnsafeKnowledgeBase: see :meth:`_check_layout`.
            KnowledgeBaseBusy: other sessions kept its database locked.
            sqlite3.DatabaseError: its database is not one (a git-LFS
                pointer) or is corrupt.
        """
        base_dir = Path(base_dir)
        self._root_folders = () if within is None else _folders_below(Path(within), base_dir)
        self._lock = threading.Lock()
        self._sidecar_locks: dict[str, asyncio.Lock] = {}
        self._quick_lock = threading.Lock()
        self._backfill_running = False
        self._concepts_store: "ConceptStore | None" = None
        self._summarizer = summarizer
        # A checkpoint after a delete that could not empty the log (see _after_delete).
        self._checkpoint_pending = False
        self._closed = False

        self.kb_dir = base_dir
        self.documents_dir = base_dir / "documents"
        self.files_dir = base_dir / "files"
        self.concepts_dir = base_dir / "concepts"
        self.db_path = base_dir / DB_NAME
        self._lock_path = base_dir / ".lock"
        self._check_layout()
        for folder in (self.kb_dir, self.documents_dir, self.files_dir):
            folder.mkdir(parents=True, exist_ok=True)

        self._service, self._semantic = self._choose_service(embedding or EmbeddingConfig(), embedding_service, use_mock)
        self._model_name = str(getattr(self._service, "model_name", "unknown"))
        self._vectors: VectorIndex | None = None
        self._db = Database.open(self.db_path)
        try:
            with self._lock:
                self._reconcile_terms()
                self._sweep()
                self._ensure_index_md()
                if self._semantic:
                    self._vectors = self._reconcile_vectors()
        except BaseException:
            self._db.close()
            raise

    @staticmethod
    def _choose_service(embedding: EmbeddingConfig, given: Any, use_mock: bool) -> tuple[Any, bool]:
        """The embedding service, and whether this knowledge base searches by meaning."""
        if use_mock:
            if isinstance(given, MockEmbeddingService):
                return given, False
            return MockEmbeddingService(model_name=embedding.model_name, batch_size=embedding.batch_size), False
        service = given
        if service is None and EmbeddingService.is_available():
            service = EmbeddingService(model_name=embedding.model_name, batch_size=embedding.batch_size, device=embedding.device)
        if service is None:
            return MockEmbeddingService(model_name=embedding.model_name, batch_size=embedding.batch_size), False
        service = _loaded_or_keyword(service)
        return service, not isinstance(service, MockEmbeddingService)

    def close(self) -> None:
        """Close the database connection, emptying the write-ahead log first
        if a delete could not (see :meth:`_after_delete`). Idempotent; the
        manager is not used after it."""
        with self._lock:
            if self._closed:
                return
            if self._checkpoint_pending:
                self._checkpoint()
            self._closed = True
            self._db.close()

    def _check_layout(self) -> None:
        """Refuse a knowledge base whose own folders, database or lock file are not what they should be.

        A subfolder (``documents``, ``files``, ``concepts``, ``pre-0.7.0``)
        that is a symlink would put its reads, writes and deletes in a folder
        elsewhere, and so would the database or one of its companions
        (``kb.sqlite``, ``-wal``, ``-shm``, ``-journal``) or the lock file
        (or create the file a dangling link names); a non-empty rollback
        journal is refused too (see ``_db.check_files``). The knowledge
        base's folder itself may be a symlink, unless it was opened
        ``within`` a folder: then each folder from there down to it must be a
        real one, so a repository cannot point ``.{app}`` or
        ``.{app}/knowledge_base`` elsewhere. Checked when the knowledge base
        opens and before each read or change, since a ``git pull`` can
        replace a folder while a session runs.

        Each check is an ``lstat`` and the open comes after it. That holds
        while nothing swaps these folders in between: a repository's files
        are static, and the model's code cannot write under ``.{app}`` (the
        permission rules and the sandboxes deny it). A sandbox that mounted
        the project read-write would need opens anchored on directory
        descriptors instead (``O_DIRECTORY|O_NOFOLLOW`` and ``dir_fd=``).

        Raises:
            UnsafeKnowledgeBase: one of them is a symlink, or the wrong kind of file.
        """
        for folder in self._root_folders:
            try:
                mode = os.lstat(folder).st_mode
            except FileNotFoundError:
                break  # created when the knowledge base opens, and all below it
            except NotADirectoryError:
                raise UnsafeKnowledgeBase(
                    f"{folder} is not in a folder: a knowledge base works only in its own folders"
                ) from None
            if not stat.S_ISDIR(mode):
                raise refuse(folder, mode, "a folder")
        # pre-0.7.0: where the 0.6 import moves the old files (_import_06.LEGACY_DIR)
        folders = (self.documents_dir, self.files_dir, self.concepts_dir, self.kb_dir / "pre-0.7.0")
        expected = [(folder, stat.S_ISDIR, "a folder") for folder in folders]
        expected.append((self._lock_path, stat.S_ISREG, "a regular file"))
        for path, is_expected, kind in expected:
            try:
                mode = os.lstat(path).st_mode
            except FileNotFoundError:
                continue
            if not is_expected(mode):
                raise refuse(path, mode, kind)
        check_files(self.db_path)

    @contextmanager
    def _write(self) -> Iterator[Database]:
        """One change, as every process sees it (see the class docstring)."""
        with self._lock:
            self._check_layout()
            with self._db.write():
                yield self._db

    @contextmanager
    def _read(self) -> Iterator[Database]:
        with self._lock:
            self._check_layout()
            with self._db.read():
                yield self._db

    @property
    def concepts(self) -> "ConceptStore":
        """Lazy-loaded ConceptStore pointed at ``<kb_dir>/concepts/``.

        Raises:
            UnsafeKnowledgeBase: see :meth:`_check_layout`, checked on each use.
        """
        self._check_layout()
        if self._concepts_store is None:
            from agentic_cli.memory.kb.concepts import ConceptStore
            self._concepts_store = ConceptStore(self.concepts_dir)
        return self._concepts_store

    # -- opening ---------------------------------------------------------------

    def _reconcile_terms(self) -> None:
        """Make every chunk's terms this version's tokenizer's (once after it changes).

        The terms are computed outside the write lock, a batch at a time, and
        each batch is written in its own short transaction (the keyword index
        kept in step), as long as the database still records the tokenizer
        this run started from: another session may have finished the change
        meanwhile, or made another. The last transaction takes the chunks
        added meanwhile and records this tokenizer.
        """
        while True:
            started = self._db.info().get("tokenizer")
            if started == TOKENIZER or self._recompute_terms(started):
                return

    def _recompute_terms(self, started: str | None) -> bool:
        """One pass of :meth:`_reconcile_terms`; False when another session
        changed the recorded tokenizer meanwhile."""
        last = 0
        while True:
            with self._db.read():
                batch = self._db.chunk_texts(after=last, limit=self._TERMS_BATCH)
            if not batch:
                break
            pairs = [(rowid, _terms_of(text)) for rowid, text in batch]
            with self._db.write():
                if self._db.info().get("tokenizer") != started:
                    return False
                self._db.set_terms(pairs)
            last = batch[-1][0]
        with self._db.write():
            if self._db.info().get("tokenizer") != started:
                return False
            self._db.set_terms([(rowid, _terms_of(text)) for rowid, text in self._db.chunk_texts(after=last)])
            self._db.set_info(tokenizer=TOKENIZER)
        return True

    def _sweep(self) -> None:
        """Remove what a crash left: a sidecar or stored file whose document has
        no row, and the temporary file of an unfinished write.

        Runs as the knowledge base opens (and after ``clear``), holding the
        write lock: every other session writes its files inside its own
        write transaction, so none is half way through one. Only names the
        knowledge base gives its files are touched, and a link is removed,
        never followed.
        """
        removed = 0
        with self._db.write():
            ids = self._db.document_ids()
            for folder, pattern in ((self.documents_dir, _SIDECAR_NAME), (self.files_dir, _ORIGINAL_NAME), (self.kb_dir, None)):
                with os.scandir(folder) as entries:
                    for entry in entries:
                        if entry.is_dir(follow_symlinks=False):
                            continue
                        match = pattern.fullmatch(entry.name) if pattern is not None else None
                        if _TEMP_NAME.fullmatch(entry.name) or (match is not None and match.group(1) not in ids):
                            Path(entry.path).unlink(missing_ok=True)
                            removed += 1
        if removed:
            logger.info("kb_leftovers_removed", path=str(self.kb_dir), files=removed)

    def _ensure_index_md(self, *, rewrite: bool = False) -> None:
        """Render ``index.md`` when asked (after an import) or when it is missing."""
        if rewrite or not self._index_md_path().exists():
            with self._db.write():
                if rewrite or self._db.document_ids():
                    self._write_index_md(self._db)

    def _reconcile_vectors(self) -> VectorIndex | None:
        """Make every chunk's vector one this session's model made, then load them.

        A database that names another model, or none, has its vectors cleared
        and this model recorded, in one transaction. Then every chunk without
        a vector is embedded, a batch at a time outside the write lock, each
        batch saved in its own transaction, so an interrupted run is
        continued by the next open. If embedding fails, or another session
        records another model meanwhile, this session searches by keyword.
        """
        model, dim = self._model_name, int(self._service.embedding_dim)
        with self._db.write():
            info = self._db.info()
            if (info.get("embedding_model"), info.get("embedding_dim")) != (model, str(dim)):
                self._db.clear_vectors()
                self._db.set_info(embedding_model=model, embedding_dim=str(dim))
                self._db.bump_generation()
            elif self._db.clear_vectors_of_wrong_size(dim * 4):
                self._db.bump_generation()
            missing = self._db.count_without_vector()
        if missing:
            logger.info("kb_reembedding", path=str(self.kb_dir), model=model, chunks=missing)
        started, done, last = time.monotonic(), 0, 0
        while True:
            with self._db.read():
                batch = self._db.chunks_without_vector(after=last, limit=self._EMBED_BATCH)
            if not batch:
                break
            try:
                vectors = normalized(self._service.embed_batch([text for _, text in batch]))
            except Exception as e:
                logger.warning("kb_reembed_failed", path=str(self.kb_dir), error=f"{type(e).__name__}: {e}", exc_info=True)
                return self._keyword_only()
            with self._db.write():
                if self._db.info().get("embedding_model") != model:
                    logger.warning("kb_model_changed_elsewhere", path=str(self.kb_dir), model=model)
                    return self._keyword_only()
                self._db.set_vectors([(rowid, to_blob(vector)) for (rowid, _), vector in zip(batch, vectors)])
                self._db.bump_generation()
            done += len(batch)
            last = batch[-1][0]
        if done:
            logger.info("kb_reembedded", path=str(self.kb_dir), model=model, chunks=done,
                        seconds=round(time.monotonic() - started, 2))
        index = VectorIndex(dim)
        with self._db.read():
            index.load(self._db.vectors(), self._db.generation())
        return index

    def _keyword_only(self) -> None:
        """Search by keyword for the rest of this session (the stored vectors stay)."""
        self._semantic = False
        self._vectors = None
        return None

    def _current_vectors(self, db: Database) -> VectorIndex | None:
        """This session's vectors, reloaded when another session changed them."""
        if self._vectors is None:
            return None
        info = db.info()
        if info.get("embedding_model") != self._model_name:
            logger.warning("kb_model_changed_elsewhere", path=str(self.kb_dir), model=self._model_name,
                           recorded=info.get("embedding_model"))
            return self._keyword_only()
        generation = db.generation()
        if self._vectors.generation != generation:
            self._vectors.load(db.vectors(), generation)
        return self._vectors

    # -- rendered files --------------------------------------------------------

    def sidecar_path(self, doc_id: str) -> Path:
        """Path to a document's markdown sidecar, whether or not it exists.

        To read it, use :meth:`read_sidecar`, which never follows a symlink.
        """
        return self.documents_dir / f"{doc_id}.md"

    def _has_sidecar(self, doc_id: str) -> bool:
        return is_regular_file(self.sidecar_path(doc_id))

    def get_or_create_sidecar_lock(self, doc_id: str) -> asyncio.Lock:
        """Return the per-doc async lock used to serialize sidecar writes.

        Same lock is returned for the same ``doc_id`` across repeat calls,
        so ``backfill_sidecars`` and the lazy ``kb_read`` fallback cannot
        double-generate a sidecar for the same document.

        The dict insert is guarded by ``_quick_lock`` so two threads racing
        to create the lock for a new doc will not produce two distinct
        ``asyncio.Lock`` instances (not by ``_lock``, which a thread holds
        while it waits for another session's write lock: the event loop
        calls this).
        """
        existing = self._sidecar_locks.get(doc_id)
        if existing is not None:
            return existing
        with self._quick_lock:
            return self._sidecar_locks.setdefault(doc_id, asyncio.Lock())

    def _index_md_path(self) -> Path:
        return self.kb_dir / "index.md"

    def _write_rendered(self, path: Path, text: str, event: str) -> None:
        """Write a rendered file; failing to must not fail the change (the database has it)."""
        try:
            atomic_write_text(path, text)
        except OSError as e:
            logger.warning(event, path=str(path), error=str(e))

    def _write_index_md(self, db: Database) -> None:
        self._write_rendered(
            self._index_md_path(), render_index_md(db.headers(), updated_at_iso=_utc_iso_now()), "kb_index_md_not_written"
        )

    def _delete_sidecar(self, doc_id: str) -> None:
        self.sidecar_path(doc_id).unlink(missing_ok=True)

    def _ingest_log_path(self) -> Path:
        return self.kb_dir / "ingest_log.md"

    def _append_ingest_log(self, action: str, doc: Document) -> None:
        """Append one audit line to ingest_log.md.

        The title (a web page's ``<title>``, or the model's argument) and the
        arXiv ID are written on one line, without control characters. A log
        that is a symlink, or not a regular file, is not written to (a
        warning instead): a project can ship one linked to a shell rc. Nor
        does any other failure to write it fail the change it records.
        """
        ts = _utc_iso_now()
        ident = doc.metadata.get("arxiv_id") if doc.metadata else None
        title_quoted = f'"{_printable(doc.title)}"'
        parts = [
            "-",
            ts,
            "·",
            action,
            "·",
            doc.source_type.value,
        ]
        if ident:
            parts += ["·", _printable(str(ident))]
        parts += ["·", title_quoted]
        if action == "ingest":
            parts += ["·", f"{len(doc.chunks)} chunks"]
        line = " ".join(parts) + "\n"
        path = self._ingest_log_path()
        try:
            append_text(path, line)
        except OSError as e:
            logger.warning("kb_ingest_log_not_written", path=str(path), error=str(e))

    # ------------------------------------------------------------------
    # File storage
    # ------------------------------------------------------------------

    def store_file(self, doc_id: str, file_bytes: bytes, extension: str) -> Path:
        """Store a binary file (e.g. PDF) for a document.

        Args:
            doc_id: Document ID.
            file_bytes: Raw file bytes.
            extension: File extension including dot (e.g. ".pdf").

        Returns:
            Relative path within files_dir.

        Raises:
            UnsafeKnowledgeBase: see :meth:`_check_layout`.
        """
        self._check_layout()
        self.files_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{doc_id}{extension}"
        atomic_write_bytes(self.files_dir / filename, file_bytes)
        return Path(filename)

    def _stored_file_name(self, doc: Document) -> str | None:
        """The stored file's name, if ``file_path`` names a file directly
        inside ``files_dir``.

        ``file_path`` comes from the database, which a project can ship with
        its knowledge base, so any other path is refused.
        """
        if not doc.file_path:
            return None
        name = doc.file_path.name
        if name in ("", ".", "..") or Path(name) != doc.file_path:
            logger.warning(
                "kb_stored_file_outside_files_dir",
                doc_id=doc.id,
                file_path=str(doc.file_path),
            )
            return None
        return name

    def get_file_path(self, doc_id: str) -> Path | None:
        """Get the absolute file path for a document's stored file.

        A project can ship ``files/`` or a stored file as a symlink, so links
        are followed and the result must still be a file inside the knowledge
        base.

        Args:
            doc_id: Document ID.

        Returns:
            The resolved Path if the stored file is inside the knowledge base,
            None otherwise.
        """
        with self._read() as db:
            doc = db.header(doc_id)
        name = self._stored_file_name(doc) if doc else None
        if name is None:
            return None
        try:
            resolved = (self.files_dir / name).resolve()
            inside = resolved.is_relative_to(self.kb_dir.resolve())
            if inside and resolved.is_file():
                return resolved
        except (OSError, RuntimeError):  # a symlink loop, or unreadable
            return None
        if not inside:
            logger.warning("kb_stored_file_outside_kb_dir", doc_id=doc_id, resolved=str(resolved))
        return None

    def _delete_stored_file(self, doc: Document) -> None:
        """Delete a document's stored file (see :meth:`_stored_file_name`)."""
        name = self._stored_file_name(doc)
        if name is None:
            return
        try:
            (self.files_dir / name).unlink(missing_ok=True)
        except OSError as e:
            logger.warning("kb_stored_file_not_deleted", doc_id=doc.id, error=str(e))

    @staticmethod
    def _truncate_summary(content: str) -> str:
        """Return the deterministic fallback summary (first ~500 chars)."""
        return enrich.preview(content)

    async def generate_summary(self, content: str, title: str = "") -> str:
        """Generate a document summary via the summarizer given at construction.

        Falls back to the first ~500 chars of ``content`` if there is no
        summarizer, if the summarizer raises, or if it returns empty.

        Args:
            content: Full document text.
            title: Document title, embedded in the prompt for context.

        Returns:
            Summary string (~500 chars).
        """
        return await enrich.summarize(self._summarizer, content, title)

    async def generate_sidecar_payload(
        self, content: str, title: str = ""
    ) -> dict[str, Any]:
        """Generate the structured payload for a document sidecar.

        Returns a dict with keys ``summary`` (str), ``claims`` (list[str]),
        and ``entities`` (dict[str, list[str]]). Falls back to
        ``{summary: enrich.preview(content), claims: [], entities: {}}`` if there
        is no summarizer or the LLM call fails.
        """
        return await enrich.sidecar_payload(self._summarizer, content, title)

    # -- changes ---------------------------------------------------------------

    def ingest_document(
        self,
        content: str,
        title: str,
        source_type: SourceType,
        source_url: str | None = None,
        metadata: dict[str, Any] | None = None,
        file_bytes: bytes | None = None,
        file_extension: str = ".pdf",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        summary: str | None = None,
        sidecar_payload: dict[str, Any] | None = None,
    ) -> Document:
        """Ingest a new document into the knowledge base.

        Chunking and embedding happen first, outside the write lock; then one
        transaction stores the original, the document, its chunks and keyword
        terms (and vectors), and renders its sidecar and ``index.md``. If any
        of that fails, nothing is stored and the original is removed.

        Args:
            content: Document content.
            title: Document title.
            source_type: Type of source (ARXIV, SSRN, WEB, etc.).
            source_url: Optional URL of the source.
            metadata: Optional additional metadata.
            file_bytes: Optional raw file bytes to store.
            file_extension: Extension for stored file (default ".pdf").
            chunk_size: Size of chunks for embedding.
            chunk_overlap: Overlap between chunks.
            summary: Optional pre-computed summary. When ``None``, falls
                back to the deterministic truncation of ``content``.
                Async callers that want an LLM-generated summary and sidecar
                should ingest through ``KnowledgeBase.ingest_text`` /
                ``ingest_bytes``, which make the summarizer call and pass
                the result in here.

        Returns:
            The created Document object.

        Raises:
            ValueError: ``chunk_size`` under 1 or ``chunk_overlap`` under 0
                (nothing is stored).
            KnowledgeBaseBusy: other sessions kept the knowledge base locked.
        """
        if summary is None:
            summary = self._truncate_summary(content)
        doc = Document.create(title=title, content=content, source_type=source_type, summary=summary,
                              source_url=source_url, metadata=metadata)
        texts = chunk_text(content, chunk_size, chunk_overlap)
        doc.chunks = [DocumentChunk.create(document_id=doc.id, content=text, chunk_index=i, metadata={"title": title})
                      for i, text in enumerate(texts)]
        doc.chunk_count = len(doc.chunks)
        terms = [_terms_of(text) for text in texts]
        vectors = normalized(self._service.embed_batch(texts)) if self._semantic and texts else None
        payload = sidecar_payload or {"summary": summary, "claims": [], "entities": {}}
        stored: Path | None = None
        blobs: list[bytes] | None = None
        try:
            with self._write() as db:
                if file_bytes is not None:
                    doc.file_path = self.store_file(doc.id, file_bytes, file_extension)
                    stored = self.files_dir / doc.file_path
                if vectors is not None:
                    if db.info().get("embedding_model") == self._model_name:
                        blobs = [to_blob(v) for v in vectors]
                    else:
                        logger.warning("kb_model_changed_elsewhere", path=str(self.kb_dir), model=self._model_name)
                        self._keyword_only()
                rowids = db.insert_document(doc, terms, blobs, payload)
                generation = db.bump_generation()
                self._write_rendered(self.sidecar_path(doc.id), render_sidecar_markdown(doc, payload), "kb_sidecar_not_written")
                self._write_index_md(db)
        except BaseException:
            if stored is not None:
                stored.unlink(missing_ok=True)
            raise
        self._append_ingest_log("ingest", doc)
        if blobs is not None:
            with self._lock:
                if self._vectors is not None:
                    self._vectors.apply(generation, add=(rowids, [doc.id] * len(rowids), vectors))
        return doc

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document: its rows in one transaction, then its sidecar and stored file.

        Returns:
            True if deleted, False if not found (or its row does not parse).
        """
        with self._write() as db:
            doc = db.header(doc_id)
            if doc is None:
                return False
            rowids = db.delete_document(doc_id)
            generation = db.bump_generation()
            self._write_index_md(db)
        with self._lock:
            if self._vectors is not None:
                self._vectors.apply(generation, remove=rowids)
        self._sidecar_locks.pop(doc_id, None)
        self._after_delete()
        self._delete_sidecar(doc_id)
        self._delete_stored_file(doc)
        self._append_ingest_log("delete", doc)
        return True

    def clear(self) -> None:
        """Remove every document, then every sidecar and stored file."""
        with self._write() as db:
            docs = db.headers()
            db.delete_all()
            db.bump_generation()
            self._write_index_md(db)
        self._sidecar_locks.clear()
        self._after_delete()
        for doc in docs:
            self._delete_sidecar(doc.id)
            self._delete_stored_file(doc)
        with self._lock:
            self._sweep()
            if self._vectors is not None:
                with self._db.read():
                    self._vectors.load(self._db.vectors(), self._db.generation())

    def _after_delete(self) -> None:
        """Empty the write-ahead log, so a deleted document's text leaves the disk.

        While another session reads an older snapshot the log cannot be
        emptied: the checkpoint waits ``CHECKPOINT_WAIT_S`` at most, and one
        it could not finish is tried again at the next delete and at close.
        """
        with self._lock:
            self._checkpoint()

    def _checkpoint(self) -> None:
        """The checkpoint of :meth:`_after_delete` (holding ``_lock``); never raises."""
        try:
            self._checkpoint_pending = not self._db.checkpoint(CHECKPOINT_WAIT_S)
        except sqlite3.Error as e:  # never fails the delete it follows
            self._checkpoint_pending = True
            logger.debug("kb_checkpoint_failed", error=str(e))

    # -- reads -----------------------------------------------------------------

    def search(self, query: str, filters: dict[str, Any] | None = None, top_k: int = 10) -> dict[str, Any]:
        """Hybrid search: every chunk ranked by meaning (with a model) and by
        keyword (BM25), the rankings fused by reciprocal rank (k = 60), the
        best chunk of each document kept. Filters apply before ranking.

        In keyword-only mode (no ``kb`` extra, ``use_mock``, a model that did
        not load) the search is by keyword alone.
        """
        start = time.time()
        embed_ms = 0.0
        query_vector = None
        if self._semantic:
            began = time.time()
            query_vector = normalized([self._service.embed_text(query)])[0]
            embed_ms = (time.time() - began) * 1000
        terms = tokenize(query)
        began = time.time()
        with self._read() as db:
            allowed = _search.allowed_documents(db.conn, filters) if filters else None
            keyword = _search.keyword_ranked(db.conn, terms, fts=db.fts, allowed=allowed)
            vectors = self._current_vectors(db)
            rankings: list[np.ndarray] = []
            owners: list[np.ndarray] = []
            if vectors is not None and query_vector is not None:
                rowids, documents = vectors.ranked(query_vector, allowed)
                rankings.append(rowids)
                owners.append(documents)
            rankings.append(np.fromiter((rowid for rowid, _ in keyword), dtype=np.int64, count=len(keyword)))
            owners.append(np.array([doc_id for _, doc_id in keyword], dtype=str))
            picked = _search.fuse(rankings, owners=owners, top_k=top_k)
            found = db.hits([rowid for rowid, _ in picked])
        search_ms = (time.time() - began) * 1000
        results = [
            SearchResult(document=found[rowid][0], chunk=found[rowid][1], score=score, highlight=_truncate(found[rowid][1].content))
            for rowid, score in picked
            if rowid in found
        ]
        return {
            "results": [r.to_dict() for r in results],
            "total_matches": len(results),
            "query_embedding_time_ms": round(embed_ms, 2),
            "search_time_ms": round(search_ms, 2),
            "total_time_ms": round((time.time() - start) * 1000, 2),
        }

    def get_document(self, doc_id: str) -> Document | None:
        """The whole document (text and chunks), or None."""
        with self._read() as db:
            return db.document(doc_id)

    def find_document(self, id_or_title: str) -> Document | None:
        """Find a document by exact ID, ID prefix, or title substring.

        Lookup order:
        1. Exact ID match
        2. ID prefix match
        3. Case-insensitive title substring match

        Returns:
            The whole document (text and chunks) if found, None otherwise.
        """
        with self._read() as db:
            return db.find(id_or_title)

    def list_documents(self, source_type: SourceType | None = None, limit: int | None = 100) -> list[Document]:
        """Documents without their text or chunks (``chunk_count`` set), the latest change first.

        Args:
            source_type: Optional source type filter.
            limit: Maximum number of documents to return; None for all.
        """
        with self._read() as db:
            return db.headers(source_type=source_type, limit=limit)

    def get_stats(self) -> dict[str, Any]:
        """Get knowledge base statistics.

        Returns:
            Dict with document count, chunk count, and other stats.
        """
        with self._read() as db:
            counts = db.counts()
        return {
            "document_count": counts["documents"],
            "chunk_count": counts["chunks"],
            "vector_count": counts["vectors"],
            "source_counts": counts["sources"],
            "embedding_model": self._model_name,
            "embedding_dim": self._service.embedding_dim,
        }

    # -- sidecars --------------------------------------------------------------

    def read_sidecar(self, doc_id: str) -> str | None:
        """A document's sidecar, or None when it has none.

        Public so a host can read a sidecar without generating one, the
        lazy-sidecar pattern ``kb_read`` uses. A sidecar file that is
        missing (or a symlink, or not a regular file) is rendered from the
        summarizer's output stored with the document, when there is one,
        without calling the model; the file written replaces a link.

        Raises:
            UnsafeKnowledgeBase: see :meth:`_check_layout`.
        """
        self._check_layout()
        data = read_if_regular(self.sidecar_path(doc_id))
        if data is not None:
            return data.decode("utf-8")
        with self._write() as db:
            doc = db.header(doc_id)
            payload = db.enrichment(doc_id) if doc is not None else None
            if doc is None or payload is None:
                return None
            text = render_sidecar_markdown(doc, payload)
            atomic_write_text(self.sidecar_path(doc_id), text)
            return text

    def write_sidecar_if_present(self, doc: Document, payload: dict[str, Any]) -> bool:
        """Store a sidecar payload a host built outside the lock and render
        it, unless the document is gone.

        Public for hosts that generate the payload themselves (an LLM call
        made without holding ``_lock``, derived from the document's text) and
        then need to write it back safely — the pattern ``kb_read`` uses: a
        ``delete_document`` (or ``clear``) during that call must not be
        followed by the write. Both remove a document in a write transaction,
        and this check and write are one too, so they cannot interleave, in
        this process or another. The document is looked up by its id.

        Returns:
            True if the sidecar was written.
        """
        with self._write() as db:
            current = db.header(doc.id)
            if current is None:
                return False
            db.set_enrichment(doc.id, payload)
            atomic_write_text(self.sidecar_path(doc.id), render_sidecar_markdown(current, payload))
            return True

    async def backfill_sidecars(
        self,
        progress_cb: "Any | None" = None,
    ) -> int:
        """Generate missing sidecars for all documents in the KB.

        Iterates the documents, generates a sidecar payload via the
        knowledge base's summarizer for any doc that doesn't have a sidecar
        file, and writes it. A document whose summarizer output is stored
        gets its sidecar rendered from it, with no model call. Returns the
        count of sidecars written. Existing sidecars are not touched.

        Per-doc locks (``_sidecar_locks``) coordinate with the lazy
        sidecar-generation path in ``KnowledgeBase.sidecar``: concurrent
        backfill + read on the same doc never double-LLM.

        Raises ``BackfillAlreadyRunning`` if another ``backfill_sidecars``
        call is already in progress on this manager. The in-progress flag
        is set synchronously (before any ``await``), so two concurrent
        callers cannot both acquire it. Reads and writes of the knowledge
        base run on a worker thread: a change waits for SQLite's write lock. A
        cancelled backfill keeps the flag, and the document's sidecar lock,
        until the call its thread is in ends.

        Skips ingest_log/index updates — they're already current; only
        the sidecar files are out of date.

        Args:
            progress_cb: Optional callable ``(done, total, doc)`` invoked
                BEFORE each per-doc LLM call. ``done`` is zero-indexed
                (0 on the first doc). Use to surface progress in UIs.
                Called inside the per-doc lock, so raising from it will
                abort the backfill.

        Returns:
            Number of sidecars written this call.
        """
        # Check-and-set the running flag under one lock acquisition so a
        # second thread cannot slip past the guard between the read and the
        # write.
        with self._quick_lock:
            if self._backfill_running:
                raise BackfillAlreadyRunning(
                    "backfill_sidecars is already running on this KB"
                )
            self._backfill_running = True
        threads = _Threads()
        try:
            todo = await threads.run(self._documents_without_sidecar)
            written = 0
            total = len(todo)
            for i, doc in enumerate(todo):
                # Guard against delete-during-backfill — doc may have
                # been removed since the snapshot was taken.
                if not await threads.run(self._needs_sidecar, doc.id):
                    continue
                lock = self.get_or_create_sidecar_lock(doc.id)
                await lock.acquire()
                try:
                    # Double-check inside the lock — another task may
                    # have written it (e.g. lazy kb_read fallback).
                    if not await threads.run(self._needs_sidecar, doc.id):
                        continue
                    if await threads.run(self.read_sidecar, doc.id) is not None:
                        written += 1  # rendered from the stored summarizer output
                        continue
                    if progress_cb is not None:
                        progress_cb(i, total, doc)
                    full = await threads.run(self.get_document, doc.id)  # the listing has no text
                    if full is None:
                        continue
                    payload = await self.generate_sidecar_payload(full.content, title=full.title)
                    if await threads.run(self.write_sidecar_if_present, full, payload):
                        written += 1
                finally:
                    threads.release_after(lock.release)
            return written
        finally:
            threads.release_after(self._end_backfill)

    def _end_backfill(self) -> None:
        with self._quick_lock:
            self._backfill_running = False

    def _documents_without_sidecar(self) -> list[Document]:
        """The documents that have no sidecar, as the knowledge base is now."""
        with self._read() as db:
            headers = db.headers()
        return [doc for doc in headers if not self._has_sidecar(doc.id)]

    def _needs_sidecar(self, doc_id: str) -> bool:
        """Whether the document is still there, still without a sidecar.

        A delete after this check is caught by ``write_sidecar_if_present``.
        """
        with self._read() as db:
            present = db.header(doc_id) is not None
        return present and not self._has_sidecar(doc_id)


class BackfillAlreadyRunning(RuntimeError):
    """Raised when ``backfill_sidecars`` is called while another call is
    already in progress on the same ``KnowledgeBaseManager``."""
