"""One knowledge base over several directories (scopes).

``KnowledgeBaseManager`` is the store for one directory. ``KnowledgeBase``
holds one manager per scope (in agentic-cli: the project's and the user's)
and does what spans them: search merged by reciprocal rank fusion, lookups
in scope order, one catalogue, sidecars, concept pages, backfill. Method
names follow the manager's.

**Provisional:** its name and methods may change in 0.7.0, when the
knowledge base becomes its own library.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from agentic_cli.memory._core.embeddings import EmbeddingConfig, EmbeddingService
from agentic_cli.memory._core.log import get_logger
from agentic_cli.memory.kb import convert as _convert
from agentic_cli.memory.kb.manager import KnowledgeBaseManager
from agentic_cli.memory.kb.models import Document, SourceType

logger = get_logger("agentic_cli.memory.kb")

# Reciprocal rank fusion constant: the standard 60.
RRF_K = 60

ProgressCallback = Callable[[str, int, int, Document], None]


class DocumentNotFound(LookupError):
    """The document went away while an operation on it was under way."""


class EmptyDocument(ValueError):
    """An ingest with no text and no bytes."""


class ConversionFailed(Exception):
    """The converter failed unexpectedly; the original exception is ``__cause__``."""


# A title made from a document's text: its first characters.
_TITLE_CHARS = 80


def _cut(text: str, length: int) -> str:
    return text[:length] + "..." if len(text) > length else text


def _url_title(url: str) -> str:
    """A title from a URL: its last path segment (percent-decoded), else its host."""
    parsed = urlparse(url)
    segment = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    return unquote(segment) if segment else (parsed.netloc or url)


@dataclass(frozen=True)
class ScopedDocument:
    """A document, and the scope it was found in."""

    document: Document
    scope: str


@dataclass(frozen=True)
class BackfillResult:
    """What ``backfill_sidecars`` did in one scope.

    ``written`` counts the sidecars of a scope that finished; it is 0 when
    ``error`` is set.
    """

    scope: str
    written: int
    error: BaseException | None = None


def _same_dir(a: Path | str, b: Path | str) -> bool:
    """Whether two paths are one directory, however they are spelled.

    ``samefile`` sees through symlinks and, on a case-insensitive filesystem,
    letter case, which ``resolve()`` does not. A path that does not exist
    yet falls back to comparing resolved paths.
    """
    try:
        a, b = os.fspath(a), os.fspath(b)
    except (TypeError, ValueError):
        return False
    if not isinstance(a, (str, bytes)) or not isinstance(b, (str, bytes)):
        return False
    try:
        return os.path.samefile(a, b)
    except OSError:
        pass
    try:
        return Path(os.fsdecode(a)).resolve() == Path(os.fsdecode(b)).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _same_store(a: Any, b: Any) -> bool:
    """The same manager, or two managers on one directory."""
    if a is b:
        return True
    dir_a, dir_b = getattr(a, "kb_dir", None), getattr(b, "kb_dir", None)
    if dir_a is None or dir_b is None:
        return False
    return _same_dir(dir_a, dir_b)


def _check_scope(name: Any, manager: Any) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError(f"a scope name must be a non-empty string, got {name!r}")
    if manager is None:
        raise ValueError(f"scope {name!r} has no knowledge base")


def _matches(doc: Document, query: str) -> bool:
    """Whether ``query`` (lower-case) is in the title or in an author."""
    if query in doc.title.lower():
        return True
    authors = doc.metadata.get("authors")
    if not isinstance(authors, (list, tuple)):
        return False
    return any(query in a.lower() for a in authors if isinstance(a, str))


def _fuse(ranked: list[list[dict[str, Any]]], top_k: int) -> list[dict[str, Any]]:
    """Merge ranked hit lists by reciprocal rank fusion.

    Scores from separate indexes are on different scales, so the merge uses
    rank positions: each list adds ``1 / (RRF_K + rank + 1)`` to a document's
    score. A document in several lists keeps the first list's hit, and every
    rank it has adds to its score.
    """
    hits: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    for results in ranked:
        for rank, hit in enumerate(results):
            doc_id = hit.get("document_id", "")
            if not doc_id:
                continue
            hits.setdefault(doc_id, hit)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
    merged = []
    for doc_id in sorted(scores, key=lambda d: scores[d], reverse=True)[:top_k]:
        hit = hits[doc_id]
        hit["score"] = round(scores[doc_id], 4)
        merged.append(hit)
    return merged


class KnowledgeBase:
    """The scopes of a knowledge base (project, user, ...), used as one.

    **Provisional:** the name and the methods may change in 0.7.0.

    Scope order is lookup precedence, and the first scope takes writes.
    Two names for one directory are one scope, named by the first; the
    other name still works with :meth:`manager` and ``scope=``.

    In ``search`` and ``list_documents``, an error in the first scope
    raises; an error in a later scope is logged and that scope skipped.
    """

    def __init__(self, managers: Mapping[str, KnowledgeBaseManager]) -> None:
        self._managers: dict[str, KnowledgeBaseManager] = {}
        self._aliases: dict[str, str] = {}
        for name, manager in managers.items():
            _check_scope(name, manager)
            same = next((n for n, m in self._managers.items() if _same_store(m, manager)), None)
            if same is None:
                self._managers[name] = manager
            else:
                self._aliases[name] = same
        if not self._managers:
            raise ValueError("a knowledge base needs at least one scope")

    @classmethod
    def open(
        cls,
        dirs: Mapping[str, Path | str],
        *,
        embedding_service: Any = None,
        embedding: EmbeddingConfig | None = None,
        summarizer: Any = None,
        use_mock: bool = False,
    ) -> "KnowledgeBase":
        """Open (or create) a knowledge base in each directory of ``dirs``.

        Args:
            dirs: Scope name → directory; the order is lookup precedence and
                the first scope takes writes. Directories that resolve to the
                same path become one scope, named by the first.
            embedding_service: A service to share across the scopes (for
                example with a memory store). Without one, the knowledge base
                builds one from ``embedding`` and shares it, unless
                ``use_mock`` is set or sentence-transformers is missing; the
                scopes then search by keyword.
            embedding: The model to build a service for; default
                ``EmbeddingConfig()``.
            summarizer: Passed to every scope (see ``KnowledgeBaseManager``).
            use_mock: Mock embeddings and vector store in every scope.

        With the ``kb`` extra and no ``use_mock``, this builds one embedding
        service; the model loads once, when the first real index is opened,
        to size or reconcile it.
        """
        if not dirs:
            raise ValueError("a knowledge base needs at least one scope")
        for name, directory in dirs.items():
            _check_scope(name, directory)
        service = embedding_service
        if service is None and not use_mock and EmbeddingService.is_available():
            config = embedding or EmbeddingConfig()
            service = EmbeddingService(
                model_name=config.model_name, batch_size=config.batch_size, device=config.device
            )
        opened: list[KnowledgeBaseManager] = []
        managers: dict[str, KnowledgeBaseManager] = {}
        for name, directory in dirs.items():
            path = Path(directory)
            manager = next((m for m in opened if _same_dir(m.kb_dir, path)), None)
            if manager is None:
                manager = KnowledgeBaseManager(
                    path,
                    embedding=embedding,
                    summarizer=summarizer,
                    use_mock=use_mock,
                    embedding_service=service,
                )
                opened.append(manager)
            managers[name] = manager
        return cls(managers)

    # -- scopes --------------------------------------------------------------

    @property
    def scopes(self) -> tuple[str, ...]:
        """The scope names, in lookup order (an alias is not a scope)."""
        return tuple(self._managers)

    def manager(self, name: str) -> KnowledgeBaseManager:
        """The manager of scope ``name`` (or of the scope it is an alias of).

        For per-directory work such as ``delete_document`` or ``clear``.

        Raises:
            KeyError: no scope has that name.
        """
        return self._managers[self._scope_name(name)]

    def _scope_name(self, name: str) -> str:
        name = self._aliases.get(name, name)
        if name not in self._managers:
            raise KeyError(f"no scope {name!r}; scopes: {', '.join(self._managers)}")
        return name

    def _first(self) -> tuple[str, KnowledgeBaseManager]:
        name = next(iter(self._managers))
        return name, self._managers[name]

    # -- reading -------------------------------------------------------------

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 10
    ) -> dict[str, Any]:
        """Search every scope's chunks (semantic and keyword) and merge the hits.

        Returns the dict ``KnowledgeBaseManager.search`` returns, every hit
        tagged with ``scope``. When a later scope answered too, the hits are
        merged by reciprocal rank fusion (k = 60): ``score`` is the fused
        score and ``total_matches`` the merged count.
        """
        items = iter(self._managers.items())
        first_name, first = next(items)
        result = first.search(query, filters=filters, top_k=top_k)
        for hit in result.get("results", []):
            hit["scope"] = first_name
        ranked = [result.get("results", [])]
        for name, manager in items:
            try:
                other = manager.search(query, filters=filters, top_k=top_k)
            except Exception:
                logger.debug("kb_scope_search_failed", scope=name, exc_info=True)
                continue
            for hit in other.get("results", []):
                hit["scope"] = name
            ranked.append(other.get("results", []))
        if len(ranked) > 1:
            result["results"] = _fuse(ranked, top_k)
            result["total_matches"] = len(result["results"])
        return result

    def find_document(self, ref: str) -> ScopedDocument | None:
        """One document by exact ID, then ID prefix, then title substring.

        Each scope is searched with that whole chain before the next scope.
        """
        for name, manager in self._managers.items():
            doc = manager.find_document(ref)
            if doc is not None:
                return ScopedDocument(doc, name)
        return None

    def get_document(self, doc_id: str) -> ScopedDocument | None:
        """One document by exact ID, scope by scope."""
        for name, manager in self._managers.items():
            doc = manager.get_document(doc_id)
            if doc is not None:
                return ScopedDocument(doc, name)
        return None

    def list_documents(
        self,
        *,
        title_or_author: str = "",
        source_type: SourceType | None = None,
        scope: str | None = None,
        limit: int | None = 20,
    ) -> list[ScopedDocument]:
        """The catalogue, newest first.

        Each scope's documents are filtered first (``title_or_author``: a
        case-insensitive substring of the title or of an author;
        ``source_type``, a ``SourceType``), then merged newest first by ``updated_at``, without
        listing an ID twice, and cut at ``limit`` (None: all). ``scope``
        limits it to one scope; an unknown one raises ``KeyError``.
        """
        names = [self._scope_name(scope)] if scope is not None else list(self._managers)
        query = title_or_author.lower()
        listed: list[ScopedDocument] = []
        seen: set[str] = set()
        for position, name in enumerate(names):
            try:
                docs = self._managers[name].list_documents(source_type=source_type, limit=None)
                matches = [d for d in docs if not query or _matches(d, query)]
            except Exception:
                if position == 0:
                    raise
                logger.debug("kb_scope_list_failed", scope=name, exc_info=True)
                continue
            for doc in matches:
                if doc.id in seen:
                    continue
                seen.add(doc.id)
                listed.append(ScopedDocument(doc, name))
        listed.sort(key=lambda s: s.document.updated_at, reverse=True)
        return listed if limit is None else listed[:limit]

    # -- ingestion -----------------------------------------------------------

    async def ingest_text(
        self,
        content: str,
        *,
        title: str = "",
        source_type: SourceType = SourceType.USER,
        source_url: str | None = None,
        metadata: dict[str, Any] | None = None,
        scope: str | None = None,
    ) -> ScopedDocument:
        """Store ``content`` as a document of ``scope`` (default: the first scope).

        The title defaults to the text's first 80 characters. One summarizer
        call writes the summary and the sidecar.

        Raises:
            EmptyDocument: ``content`` is empty.
        """
        return await self._store(
            scope, text=content, data=None, extension=".txt", title=title,
            source_type=source_type, source_url=source_url, metadata=dict(metadata or {}),
        )

    async def ingest_bytes(
        self,
        data: bytes,
        *,
        extension: str,
        content_type: str | None = None,
        charset: str | None = None,
        fetched_url: str | None = None,
        fetched_at: str | None = None,
        title: str = "",
        default_title: str = "",
        source_type: SourceType | None = None,
        source_url: str | None = None,
        metadata: dict[str, Any] | None = None,
        fallback_text: str | None = None,
        scope: str | None = None,
    ) -> ScopedDocument:
        """Convert ``data`` and store it, with the bytes, in ``scope`` (default: the first).

        Convert (on a worker thread), fill in defaults, enrich (one
        summarizer call), store. A value the caller passed always wins:

        - Title: the page's title, then the last segment of ``fetched_url``
          (or its host), then ``default_title``, then the text's first 80
          characters.
        - Source type: ``web`` with a ``fetched_url``, otherwise ``local``.
        - Source URL: ``fetched_url``.
        - Metadata: the page's authors, its description (as ``abstract``),
          ``published``, then ``fetched_at`` and ``file_size_bytes``; the
          caller's keys win.
        - Text: the converted text, or ``fallback_text`` when that is empty.

        Args:
            data: The bytes.
            extension: Their extension, for example ``.pdf``.
            content_type: The fetched page's Content-Type.
            charset: The fetched page's charset. None means a local file,
                which must be UTF-8 unless it is a PDF.
            fetched_url: Where the bytes were downloaded; it helps
                trafilatura and marks the document as a web page.
            fetched_at: When (ISO 8601).
            source_url: What the document cites (arXiv: the abstract page,
                while the bytes come from the PDF URL).

        Raises:
            UnsupportedDocument: the converter cannot read the bytes.
            ConversionFailed: the converter failed unexpectedly.
            EmptyDocument: no text and no bytes.
        """
        try:
            converted = await asyncio.to_thread(
                _convert.convert_document,
                data, content_type=content_type, extension=extension, charset=charset, url=fetched_url,
            )
        except _convert.UnsupportedDocument:
            raise
        except Exception as e:
            raise ConversionFailed(f"could not convert it ({type(e).__name__})") from e

        found: dict[str, Any] = {}
        if converted.authors:
            found["authors"] = list(converted.authors)
        if converted.description:
            found["abstract"] = converted.description
        if converted.published:
            found["published"] = converted.published
        if fetched_at:
            found["fetched_at"] = fetched_at
        found["file_size_bytes"] = len(data)
        found.update(metadata or {})

        if fetched_url:
            page_title = converted.title or _url_title(fetched_url)
            default_type = SourceType.WEB
        else:
            page_title = converted.title or default_title
            default_type = SourceType.LOCAL
        return await self._store(
            scope,
            text=converted.text or (fallback_text or ""),
            data=data,
            extension=extension,
            title=title or page_title,
            source_type=source_type or default_type,
            source_url=source_url or fetched_url,
            metadata=found,
        )

    async def _store(
        self,
        scope: str | None,
        *,
        text: str,
        data: bytes | None,
        extension: str,
        title: str,
        source_type: SourceType,
        source_url: str | None,
        metadata: dict[str, Any],
    ) -> ScopedDocument:
        """Enrich (one summarizer call) and store through the scope's manager."""
        if not text and not data:
            raise EmptyDocument("no text and no bytes to ingest")
        name = self._scope_name(scope) if scope is not None else self.scopes[0]
        manager = self._managers[name]
        if not title:
            title = _cut(text, _TITLE_CHARS)
        payload = await manager.generate_sidecar_payload(text, title=title)
        document = manager.ingest_document(
            content=text,
            title=title,
            source_type=source_type,
            source_url=source_url,
            metadata=metadata or None,
            file_bytes=data,
            file_extension=extension,
            summary=payload["summary"] or None,
            sidecar_payload=payload,
        )
        return ScopedDocument(document, name)

    def read_text(self, doc: ScopedDocument, *, max_chars: int | None = None) -> tuple[str, bool]:
        """The document's stored text, and whether it was cut at ``max_chars``.

        A PDF stored without text (from before text was kept) is read from
        its stored file.
        """
        document = doc.document
        text = document.content
        if not text and document.file_path:
            file_path = self.manager(doc.scope).get_file_path(document.id)
            if file_path is not None and str(file_path).endswith(".pdf"):
                text = _convert.extract_pdf_text(file_path)
        if max_chars is not None and len(text) > max_chars:
            return text[:max_chars], True
        return text, False

    async def sidecar(self, doc: ScopedDocument) -> str:
        """The document's sidecar, written first when it is missing.

        The summarizer is called under the document's sidecar lock, so two
        first reads make one call.

        Raises:
            DocumentNotFound: the document was deleted during the
                summarizer call (nothing is written).
        """
        manager = self.manager(doc.scope)
        doc_id = doc.document.id
        path = manager.sidecar_path(doc_id)
        if not path.exists():
            async with manager.get_or_create_sidecar_lock(doc_id):
                if not path.exists():
                    live = manager.get_document(doc_id)
                    if live is None:
                        raise DocumentNotFound(doc_id)
                    text, _ = self.read_text(ScopedDocument(live, doc.scope))
                    payload = await manager.generate_sidecar_payload(text, title=live.title)
                    if not manager.write_sidecar_if_present(live, payload):
                        raise DocumentNotFound(doc_id)
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise DocumentNotFound(doc_id) from None

    # -- concepts --------------------------------------------------------------

    def write_concept(
        self, title: str, body: str, sources: list[str], *, slug: str = ""
    ) -> dict[str, Any]:
        """Write a concept page to the first scope; each source must exist in some scope.

        Returns what ``ConceptStore.write`` returns.
        """
        _, first = self._first()

        def exists(doc_id: str) -> bool:
            return any(m.get_document(doc_id) is not None for m in self._managers.values())

        return first.concepts.write(
            title=title, body=body, sources=sources, slug=slug, valid_ids_check=exists
        )

    def search_concepts(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Search the first scope's concept pages."""
        _, first = self._first()
        return first.concepts.search(query, limit=limit)

    # -- maintenance -------------------------------------------------------------

    async def backfill_sidecars(
        self, *, progress: ProgressCallback | None = None
    ) -> list[BackfillResult]:
        """Write the missing sidecars of every scope, in order.

        ``progress(scope, done, total, document)`` is called before each
        summarizer call. A scope's error (``BackfillAlreadyRunning``, or
        whatever stopped it) is its result's ``error``; the next scope still
        runs.
        """
        results: list[BackfillResult] = []
        for name, manager in self._managers.items():
            try:
                written = await manager.backfill_sidecars(progress_cb=self._scoped(progress, name))
            except Exception as e:
                results.append(BackfillResult(name, 0, e))
                continue
            results.append(BackfillResult(name, written))
        return results

    @staticmethod
    def _scoped(progress: ProgressCallback | None, scope: str) -> Callable[[int, int, Document], None] | None:
        """``progress`` as the manager's callback, with ``scope`` bound."""
        if progress is None:
            return None

        def callback(done: int, total: int, document: Document) -> None:
            progress(scope, done, total, document)

        return callback

    def stats(self) -> dict[str, dict[str, Any]]:
        """Each scope's ``get_stats()``, by scope name."""
        return {name: manager.get_stats() for name, manager in self._managers.items()}
