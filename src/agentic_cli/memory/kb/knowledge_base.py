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

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_cli.memory._core.embeddings import EmbeddingConfig, EmbeddingService
from agentic_cli.memory._core.log import get_logger
from agentic_cli.memory.kb.manager import KnowledgeBaseManager
from agentic_cli.memory.kb.models import Document, SourceType

logger = get_logger("agentic_cli.memory.kb")

# Reciprocal rank fusion constant: the standard 60.
RRF_K = 60

ProgressCallback = Callable[[str, int, int, Document], None]


class DocumentNotFound(LookupError):
    """The document went away while an operation on it was under way."""


@dataclass(frozen=True)
class ScopedDocument:
    """A document, and the scope it was found in."""

    document: Document
    scope: str


@dataclass(frozen=True)
class BackfillResult:
    """What ``backfill_sidecars`` did in one scope."""

    scope: str
    written: int
    error: BaseException | None = None


def _same_store(a: Any, b: Any) -> bool:
    """The same manager, or two managers on one directory."""
    if a is b:
        return True
    dir_a, dir_b = getattr(a, "kb_dir", None), getattr(b, "kb_dir", None)
    if dir_a is None or dir_b is None:
        return False
    try:
        return Path(dir_a).resolve() == Path(dir_b).resolve()
    except (OSError, RuntimeError):
        return False


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
            if not isinstance(name, str) or not name:
                raise ValueError(f"a scope name must be a non-empty string, got {name!r}")
            if manager is None:
                raise ValueError(f"scope {name!r} has no knowledge base")
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
        """
        if not dirs:
            raise ValueError("a knowledge base needs at least one scope")
        service = embedding_service
        if service is None and not use_mock and EmbeddingService.is_available():
            config = embedding or EmbeddingConfig()
            service = EmbeddingService(
                model_name=config.model_name, batch_size=config.batch_size, device=config.device
            )
        opened: list[tuple[Path, KnowledgeBaseManager]] = []
        managers: dict[str, KnowledgeBaseManager] = {}
        for name, directory in dirs.items():
            path = Path(directory)
            resolved = path.resolve()
            manager = next((m for r, m in opened if r == resolved), None)
            if manager is None:
                manager = KnowledgeBaseManager(
                    path,
                    embedding=embedding,
                    summarizer=summarizer,
                    use_mock=use_mock,
                    embedding_service=service,
                )
                opened.append((resolved, manager))
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
        ``source_type``), then merged newest first by ``updated_at``, without
        listing an ID twice, and cut at ``limit`` (None: all). ``scope``
        limits it to one scope.
        """
        names = [self._scope_name(scope)] if scope is not None else list(self._managers)
        query = title_or_author.lower()
        listed: list[ScopedDocument] = []
        seen: set[str] = set()
        for position, name in enumerate(names):
            try:
                docs = self._managers[name].list_documents(source_type=source_type, limit=None)
            except Exception:
                if position == 0:
                    raise
                logger.debug("kb_scope_list_failed", scope=name, exc_info=True)
                continue
            for doc in docs:
                if doc.id in seen:
                    continue
                if query and not (
                    query in doc.title.lower()
                    or any(query in a.lower() for a in doc.metadata.get("authors", []))
                ):
                    continue
                seen.add(doc.id)
                listed.append(ScopedDocument(doc, name))
        listed.sort(key=lambda s: s.document.updated_at, reverse=True)
        return listed if limit is None else listed[:limit]
