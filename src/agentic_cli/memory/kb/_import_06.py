"""A 0.6 knowledge base folder, imported once into ``kb.sqlite``.

0.6 kept a folder's documents in ``metadata.json`` (the headers; in its v1
format the text too), ``documents/<id>.json`` (each document's text and its
chunks' texts) and ``embeddings/`` (FAISS, a keyword index, a model record).
The import reads each header on its own, so one that does not parse costs
only itself, and chunks each document again with this version's chunker. The
vectors are left for the first open with a model: FAISS is no longer
installed, and an old index may be torn (review K4). The old files move into
``pre-0.7.0/``, where nothing reads them; ``embeddings/`` holds nothing that
is needed and is deleted.

Delete this module when 0.6 folders are no longer read.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentic_cli.memory._core.chunking import chunk_text
from agentic_cli.memory._core.filelock import folder_lock
from agentic_cli.memory._core.io import is_regular_file, read_if_regular
from agentic_cli.memory._core.log import get_logger
from agentic_cli.memory.kb import _db
from agentic_cli.memory.kb._db import (
    COMPANION_SUFFIXES,
    DB_NAME,
    PLAIN_DOC_ID,
    Database,
    KnowledgeBaseBusy,
    naive,
    refuse,
)
from agentic_cli.memory.kb._tokenize import TOKENIZER, tokenize
from agentic_cli.memory.kb.models import Document, DocumentChunk

logger = get_logger("agentic_cli.memory.kb")

LEGACY_DIR = "pre-0.7.0"
_PENDING = "import_cleanup"


def needs_import(kb_dir: Path) -> bool:
    """A 0.6 folder: ``metadata.json`` and no database yet."""
    return is_regular_file(kb_dir / "metadata.json") and not os.path.lexists(kb_dir / DB_NAME)


def run(kb_dir: Path) -> None:
    """Import the folder once, under the folder lock 0.6 sessions also take.

    Raises:
        ValueError: ``metadata.json`` is not JSON at all (a git-LFS pointer
            not pulled yet, say): nothing is imported, created or moved,
            and the next open tries again.
        KnowledgeBaseBusy: an older version's session held the folder lock
            for ``BUSY_TIMEOUT_S``.
    """
    _headers(kb_dir)  # before anything is created, the lock file included
    with _legacy_lock(kb_dir):
        if needs_import(kb_dir):  # another process may have imported it meanwhile
            _build(kb_dir)


@contextmanager
def _legacy_lock(kb_dir: Path) -> Iterator[None]:
    """The ``.lock`` folder lock 0.6 sessions take around each change, waited
    for up to ``BUSY_TIMEOUT_S`` (an older version's session may be changing
    the folder), then ``KnowledgeBaseBusy``."""

    def waiting() -> None:
        logger.warning("kb_legacy_lock_held", path=str(kb_dir),
                       reason="an older version's session is changing this knowledge base; waiting")

    with ExitStack() as stack:
        try:
            stack.enter_context(folder_lock(kb_dir / ".lock", timeout=_db.BUSY_TIMEOUT_S, on_wait=waiting))
        except TimeoutError as e:
            raise KnowledgeBaseBusy(
                f"{kb_dir}: an older version's session kept the knowledge base locked for {_db.BUSY_TIMEOUT_S:g} s"
            ) from e
        yield


def tidy(kb_dir: Path, db: Database) -> bool:
    """Finish an import (move the old files aside) or move aside 0.6 files an
    old version wrote after it. Returns whether files moved."""
    if db.info().get(_PENDING) != "pending" and not _legacy_files(kb_dir):
        return False
    # The folder lock keeps 0.6 sessions out; the write lock keeps out this
    # version's, whose files are written inside their own transactions.
    with _legacy_lock(kb_dir), db.write():
        if db.info().get(_PENDING) == "pending":
            _move_aside(kb_dir, kb_dir / LEGACY_DIR, imported=db.document_ids(), referenced=db.file_names())
            db.set_info(**{_PENDING: "done"})
            logger.info("kb_import_files_moved", path=str(kb_dir / LEGACY_DIR),
                        note="nothing reads it any more: delete it once the knowledge base looks right")
        elif _legacy_files(kb_dir):
            # What the old session wrote: its headers and texts, and the
            # sidecars and stored files no row names (the leftover sweep
            # would delete them).
            target = kb_dir / LEGACY_DIR / f"after-import-{_stamp()}"
            _move_aside(kb_dir, target, imported=db.document_ids(), referenced=db.file_names())
            logger.warning("kb_legacy_files_ignored", path=str(target),
                           reason="an older version wrote to this knowledge base after it was imported; "
                                  "what it wrote is not in the knowledge base")
        else:
            return False
    # .lock stays: deleting a lock file another process waits on would let two hold it.
    return True


def _build(kb_dir: Path) -> None:
    building = kb_dir / f"{DB_NAME}.importing"
    for leftover in (building, *(building.with_name(building.name + s) for s in COMPANION_SUFFIXES)):
        leftover.unlink(missing_ok=True)
    version, headers = _headers(kb_dir)
    imported, skipped, seen = 0, [], set()
    db = Database.open(building)
    try:
        with db.write():
            for position, raw in enumerate(headers):
                doc = _document(kb_dir, raw, version, position, skipped)
                if doc is None or doc.id in seen:
                    if doc is not None:
                        skipped.append(doc.id)
                        logger.warning("kb_import_skipped", path=str(kb_dir), position=position, doc_id=doc.id, reason="a second header with this ID")
                    continue
                seen.add(doc.id)
                db.insert_document(doc, [" ".join(tokenize(c.content)) for c in doc.chunks], None, None)
                imported += 1
            db.set_info(tokenizer=TOKENIZER, **{_PENDING: "pending"})
    finally:
        db.close()  # the last connection: SQLite empties the log into the file
    log = building.with_name(building.name + "-wal")
    if os.path.lexists(log) and os.lstat(log).st_size:
        raise OSError(f"{log} was not emptied: the import runs again at the next open")
    os.replace(building, kb_dir / DB_NAME)
    for leftover in (building.with_name(building.name + s) for s in COMPANION_SUFFIXES):
        leftover.unlink(missing_ok=True)
    logger.info("kb_imported", path=str(kb_dir), documents=imported, skipped=len(skipped), skipped_ids=skipped)


def _headers(kb_dir: Path) -> tuple[int, list[Any]]:
    """``metadata.json``'s format version and headers.

    Raises:
        ValueError: it is not JSON at all.
    """
    raw = read_if_regular(kb_dir / "metadata.json")
    try:
        data = json.loads(raw) if raw is not None else {}
    except (ValueError, UnicodeDecodeError) as e:
        raise ValueError(
            f"{kb_dir / 'metadata.json'} could not be read as JSON (a git-LFS pointer not pulled yet?): "
            f"the knowledge base is left out until it can be imported ({e})"
        ) from None
    if isinstance(data, dict) and isinstance(data.get("documents"), list):
        version = data.get("version", 1)
        return (version if isinstance(version, int) else 1), data["documents"]
    logger.warning("kb_import_skipped", path=str(kb_dir), reason="metadata.json holds no document list")
    return 1, []


def _document(kb_dir: Path, raw: Any, version: int, position: int, skipped: list[str]) -> Document | None:
    """The document of one 0.6 header, chunked again; None (logged) when it does not parse.

    Everything read from the header is read inside one ``try``: a field of the
    wrong type (a v1 text that is not a string, chunk positions of mixed
    types) costs that document only.
    """
    doc_id = raw.get("id") if isinstance(raw, dict) else None
    try:
        if not isinstance(doc_id, str) or not PLAIN_DOC_ID.fullmatch(doc_id):
            raise ValueError("its ID is not a plain name")
        doc = Document.from_dict(raw)
        if not all(isinstance(value, str) for value in (doc.title, doc.summary, doc.content)):
            raise TypeError("its title, summary or text is not a string")
        if not isinstance(doc.metadata, dict) or not (doc.source_url is None or isinstance(doc.source_url, str)):
            raise TypeError("its metadata or source URL is of the wrong type")
        chunk_texts = {c.id: c.content for c in doc.chunks}
        if version >= 2:
            content, chunk_texts = _stored_text(kb_dir, doc_id)
            doc.content = content
        if doc.file_path is not None and (doc.file_path.name in ("", ".", "..") or Path(doc.file_path.name) != doc.file_path):
            logger.warning("kb_stored_file_outside_files_dir", doc_id=doc_id, file_path=str(doc.file_path))
            doc.file_path = None
        texts = chunk_text(doc.content) if doc.content.strip() else [
            text
            for c in sorted(doc.chunks, key=lambda c: c.chunk_index)
            if isinstance(text := chunk_texts.get(c.id, ""), str) and text.strip()
        ]
        doc.chunks = [DocumentChunk.create(document_id=doc_id, content=t, chunk_index=i, metadata={"title": doc.title})
                      for i, t in enumerate(texts)]
        doc.chunk_count = len(doc.chunks)
        doc.created_at, doc.updated_at = naive(doc.created_at), naive(doc.updated_at)
    except Exception as e:  # any malformed header costs only itself
        skipped.append(repr(doc_id))
        logger.warning("kb_import_skipped", path=str(kb_dir), position=position, doc_id=repr(doc_id),
                       reason=f"{type(e).__name__}: {e}")
        return None
    return doc


def _stored_text(kb_dir: Path, doc_id: str) -> tuple[str, dict[str, str]]:
    """A v2/v3 document's text and chunk texts from ``documents/<id>.json``; empty when unreadable."""
    try:
        raw = read_if_regular(kb_dir / "documents" / f"{doc_id}.json")
        data = json.loads(raw) if raw is not None else None
        content = data.get("content") if isinstance(data, dict) else None
        chunks = data.get("chunks") if isinstance(data, dict) else None
    except (ValueError, UnicodeDecodeError, OSError):
        content = chunks = None
    if not isinstance(content, str):
        logger.warning("kb_import_text_unreadable", path=str(kb_dir), doc_id=doc_id)
        content = ""
    texts = {str(k): v for k, v in chunks.items() if isinstance(v, str)} if isinstance(chunks, dict) else {}
    return content, texts


def _legacy_files(kb_dir: Path) -> bool:
    return (
        os.path.lexists(kb_dir / "metadata.json")
        or os.path.lexists(kb_dir / "embeddings")
        or any(p.suffix == ".json" for p in _entries(kb_dir / "documents"))
    )


def _entries(folder: Path) -> list[Path]:
    """The files (and links) directly in ``folder``, a real folder; none otherwise."""
    try:
        if not stat.S_ISDIR(os.lstat(folder).st_mode):
            return []
    except FileNotFoundError:
        return []
    with os.scandir(folder) as entries:
        return [Path(e.path) for e in entries if not e.is_dir(follow_symlinks=False)]


def _real_folder(path: Path) -> Path:
    """``path`` as a real folder, created if missing.

    Raises:
        UnsafeKnowledgeBase: it is a link or not a folder (a shipped
            ``pre-0.7.0`` linked elsewhere would receive the old files).
    """
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        path.mkdir()
        return path
    if not stat.S_ISDIR(mode):
        raise refuse(path, mode, "a folder")
    return path


def _move_aside(kb_dir: Path, target: Path, *, imported: set[str] | None, referenced: set[str] | None) -> None:
    """Move 0.6's files into ``target``; with ``imported``, also the sidecars
    and stored files of documents that were not imported (the leftover sweep
    would delete them). Then delete ``embeddings/``."""
    _real_folder(kb_dir / LEGACY_DIR)
    if target != kb_dir / LEGACY_DIR:
        _real_folder(target)

    def move(path: Path, *parts: str) -> None:
        folder = target
        for part in parts[:-1]:
            folder = _real_folder(folder / part)
        destination = folder / parts[-1]
        if os.path.lexists(destination):
            destination = destination.with_name(f"{destination.name}.{_stamp()}")
        os.replace(path, destination)

    if os.path.lexists(kb_dir / "metadata.json"):
        move(kb_dir / "metadata.json", "metadata.json")
    for path in _entries(kb_dir / "documents"):
        stem, suffix = path.stem, path.suffix
        if suffix == ".json" or (imported is not None and suffix == ".md" and stem not in imported):
            move(path, "documents", path.name)
    if imported is not None and referenced is not None:
        for path in _entries(kb_dir / "files"):
            if path.name not in referenced:
                move(path, "files", path.name)
    embeddings = kb_dir / "embeddings"
    if os.path.islink(embeddings):
        embeddings.unlink()
    elif embeddings.is_dir():
        shutil.rmtree(embeddings)


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
