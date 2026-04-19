"""Knowledge Base Manager.

Main interface for knowledge base operations including document ingestion,
search, and management.

Persistence format (v2):
    metadata.json    - Document headers and chunk metadata (no content)
    documents/{id}.json - Per-document content and chunk texts
    embeddings/      - FAISS index and mappings

Legacy format (v1): All content stored in metadata.json.
Auto-migrated to v2 on first load.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentic_cli.knowledge_base.embeddings import EmbeddingService
from agentic_cli.knowledge_base.models import (
    Document,
    DocumentChunk,
    SearchResult,
    SourceType,
)
from agentic_cli.knowledge_base.vector_store import VectorStore
from agentic_cli.constants import truncate
from agentic_cli.logging import Loggers
from agentic_cli.file_utils import atomic_write_json, atomic_write_text

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings

logger = Loggers.knowledge_base()

# Current persistence format version
_FORMAT_VERSION = 3


def _utc_iso_now() -> str:
    """Return current UTC time as ISO-8601 with a trailing 'Z'."""
    from datetime import timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def matches_document_filters(doc: Document, filters: dict[str, Any]) -> bool:
    """Check if a document matches the given filters.

    Standalone function for reuse outside the manager.

    Args:
        doc: Document to check.
        filters: Filter dict with optional keys:
            - source_type: SourceType or string
            - date_from: ISO date string (inclusive lower bound)
            - date_to: ISO date string (inclusive upper bound)
            - keywords: list of strings (all must be present, case-insensitive)

    Returns:
        True if document matches all filters.
    """
    if "source_type" in filters:
        expected = filters["source_type"]
        if isinstance(expected, str):
            expected = SourceType(expected)
        if doc.source_type != expected:
            return False

    if "date_from" in filters:
        date_from = datetime.fromisoformat(filters["date_from"])
        if doc.created_at < date_from:
            return False

    if "date_to" in filters:
        date_to = datetime.fromisoformat(filters["date_to"])
        if doc.created_at > date_to:
            return False

    if "keywords" in filters:
        content_lower = doc.content.lower()
        for keyword in filters["keywords"]:
            if keyword.lower() not in content_lower:
                return False

    return True


class KnowledgeBaseManager:
    """Main interface for knowledge base operations.

    Provides document ingestion, semantic search, and document management.

    Concurrency model
    -----------------
    Two lock surfaces with different semantics:

    * ``_lock`` (``threading.Lock``) — guards every mutation to the
      in-memory document/chunk/vector/BM25 state and the check-and-set
      of ``_backfill_running``. The sync mutation API
      (``ingest_document``, ``search``, ``delete_document``, ``clear``)
      is safe to call from any thread.
    * ``_sidecar_locks`` (``dict[str, asyncio.Lock]``) — per-doc locks
      that serialize concurrent LLM-backed sidecar generation for the
      same document. Always obtain these through
      :meth:`get_or_create_sidecar_lock`; never index the dict directly.

    The async sidecar path (``backfill_sidecars`` and the lazy
    ``kb_read`` fallback in ``tools/knowledge_tools``) is safe within a
    single asyncio event loop. Sharing a manager across multiple event
    loops is NOT supported — the per-doc ``asyncio.Lock`` is bound to
    the loop it was created on.
    """

    def __init__(
        self,
        settings: "BaseSettings | None" = None,
        use_mock: bool = False,
        base_dir: Path | None = None,
        embedding_service: Any = None,
        vector_store: Any = None,
    ) -> None:
        """Initialize the knowledge base manager.

        Args:
            settings: Application settings. Required for paths configuration.
            use_mock: If True, use mock services (for testing without ML models).
            base_dir: Optional override for all KB paths. When provided, all
                paths (kb_dir, documents_dir, embeddings_dir, files_dir) are
                derived from this directory instead of from settings. Embedding
                model and batch size still come from settings for consistency.
            embedding_service: Optional pre-configured embedding service.
            vector_store: Optional pre-configured vector store.
        """
        self._lock = threading.Lock()
        self._sidecar_locks: dict[str, asyncio.Lock] = {}
        self._backfill_running: bool = False
        self._concepts_store: "ConceptStore | None" = None
        self._settings = settings
        self._use_mock = use_mock

        if base_dir is not None:
            # Override: derive all paths from base_dir
            self.kb_dir = base_dir
            self.documents_dir = base_dir / "documents"
            self.embeddings_dir = base_dir / "embeddings"
            # Embedding config still from settings
            if settings:
                embedding_model = settings.embedding_model
                batch_size = settings.embedding_batch_size
                embedding_device = settings.embedding_device
            else:
                embedding_model = "all-MiniLM-L6-v2"
                batch_size = 32
                embedding_device = "auto"
        elif settings:
            # Get paths from settings
            self.kb_dir = settings.knowledge_base_dir
            self.documents_dir = settings.knowledge_base_documents_dir
            self.embeddings_dir = settings.knowledge_base_embeddings_dir
            embedding_model = settings.embedding_model
            batch_size = settings.embedding_batch_size
            embedding_device = settings.embedding_device
        else:
            # Fallback defaults for standalone use
            self.kb_dir = Path.home() / ".agentic" / "knowledge_base"
            self.documents_dir = self.kb_dir / "documents"
            self.embeddings_dir = self.kb_dir / "embeddings"
            embedding_model = "all-MiniLM-L6-v2"
            batch_size = 32
            embedding_device = "auto"

        self.metadata_path = self.kb_dir / "metadata.json"
        self.files_dir = self.kb_dir / "files"

        # Ensure directories exist
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.files_dir.mkdir(parents=True, exist_ok=True)

        # Use injected services or create them
        if embedding_service and vector_store:
            self._embedding_service = embedding_service
            self._vector_store = vector_store
        else:
            self._embedding_service, self._vector_store = self._create_services(
                embedding_model, batch_size, embedding_device
            )

        # Load document metadata
        self._documents: dict[str, Document] = {}
        self._chunks: dict[str, DocumentChunk] = {}
        self._load_metadata()

        # BM25 index for hybrid search
        self._bm25_index = None
        try:
            from agentic_cli.knowledge_base.bm25_index import create_bm25_index
            self._bm25_index = create_bm25_index(use_mock=use_mock)
            if self.embeddings_dir.exists():
                self._bm25_index.load(self.embeddings_dir)
        except Exception:
            logger.debug("bm25_init_skipped")

    @property
    def concepts(self) -> "ConceptStore":
        """Lazy-loaded ConceptStore pointed at ``<kb_dir>/concepts/``."""
        if self._concepts_store is None:
            from agentic_cli.knowledge_base.concepts import ConceptStore
            self._concepts_store = ConceptStore(self.kb_dir / "concepts")
        return self._concepts_store

    def _create_services(
        self, embedding_model: str, batch_size: int, embedding_device: str = "auto"
    ) -> tuple[Any, Any]:
        """Create embedding service and vector store.

        Tries real implementations first; falls back to mocks if
        dependencies (sentence_transformers, faiss) are unavailable.
        """
        if not self._use_mock and EmbeddingService.is_available():
            try:
                emb = EmbeddingService(
                    model_name=embedding_model,
                    batch_size=batch_size,
                    device=embedding_device,
                )
                vs = VectorStore(
                    index_path=self.embeddings_dir / "index.faiss",
                    embedding_dim=emb.embedding_dim,
                )
                return emb, vs
            except ImportError:
                pass
        self._use_mock = True

        from agentic_cli.knowledge_base._mocks import (
            MockEmbeddingService,
            MockVectorStore,
        )

        emb = MockEmbeddingService(
            model_name=embedding_model,
            batch_size=batch_size,
        )
        vs = MockVectorStore(
            index_path=self.embeddings_dir / "index.mock",
            embedding_dim=emb.embedding_dim,
        )
        return emb, vs

    # ------------------------------------------------------------------
    # Persistence — v2/v3 format (content in per-document files)
    # ------------------------------------------------------------------

    def _document_content_path(self, doc_id: str) -> Path:
        """Get the path for a document's content file."""
        return self.documents_dir / f"{doc_id}.json"

    def _save_document_content(self, doc: Document) -> None:
        """Save document content to its individual file."""
        chunk_contents = {c.id: c.content for c in doc.chunks}
        data = {
            "content": doc.content,
            "chunks": chunk_contents,
        }
        atomic_write_json(self._document_content_path(doc.id), data)

    def _load_document_content(self, doc_id: str) -> dict[str, Any] | None:
        """Load document content from its individual file.

        Returns:
            Dict with 'content' and 'chunks' keys, or None if not found.
        """
        path = self._document_content_path(doc_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "failed_to_load_document_content",
                doc_id=doc_id,
                error=str(e),
            )
            return None

    def _delete_document_content(self, doc_id: str) -> None:
        """Delete a document's content file."""
        path = self._document_content_path(doc_id)
        if path.exists():
            path.unlink()

    def _sidecar_path(self, doc_id: str) -> Path:
        return self.documents_dir / f"{doc_id}.md"

    def _write_sidecar(
        self, doc: Document, payload: dict[str, Any] | None = None
    ) -> Path:
        """Write the per-document markdown sidecar.

        If ``payload`` is None, falls back to a summary-only payload built
        from ``doc.summary``. Idempotent — overwrites existing sidecar.
        """
        from agentic_cli.knowledge_base.sidecar import render_sidecar_markdown

        if payload is None:
            payload = {"summary": doc.summary or "", "claims": [], "entities": {}}
        path = self._sidecar_path(doc.id)
        atomic_write_text(path, render_sidecar_markdown(doc, payload))
        return path

    def _delete_sidecar(self, doc_id: str) -> None:
        path = self._sidecar_path(doc_id)
        if path.exists():
            path.unlink()

    def get_or_create_sidecar_lock(self, doc_id: str) -> asyncio.Lock:
        """Return the per-doc async lock used to serialize sidecar writes.

        Same lock is returned for the same ``doc_id`` across repeat calls,
        so ``backfill_sidecars`` and the lazy ``kb_read`` fallback cannot
        double-generate a sidecar for the same document.

        The dict insert is guarded by ``_lock`` so two threads racing to
        create the lock for a new doc will not produce two distinct
        ``asyncio.Lock`` instances.
        """
        existing = self._sidecar_locks.get(doc_id)
        if existing is not None:
            return existing
        with self._lock:
            return self._sidecar_locks.setdefault(doc_id, asyncio.Lock())

    def _index_md_path(self) -> Path:
        return self.kb_dir / "index.md"

    def _rebuild_index_md(self) -> None:
        """Rebuild index.md from current in-memory documents."""
        from agentic_cli.knowledge_base.sidecar import render_index_md

        text = render_index_md(
            list(self._documents.values()),
            updated_at_iso=_utc_iso_now(),
        )
        atomic_write_text(self._index_md_path(), text)

    def _ingest_log_path(self) -> Path:
        return self.kb_dir / "ingest_log.md"

    def _append_ingest_log(self, action: str, doc: Document) -> None:
        """Append one audit line to ingest_log.md."""
        ts = _utc_iso_now()
        ident = doc.metadata.get("arxiv_id") if doc.metadata else None
        title_quoted = f'"{doc.title}"'
        parts = [
            "-",
            ts,
            "·",
            action,
            "·",
            doc.source_type.value,
        ]
        if ident:
            parts += ["·", ident]
        parts += ["·", title_quoted]
        if action == "ingest":
            parts += ["·", f"{len(doc.chunks)} chunks"]
        line = " ".join(parts) + "\n"
        path = self._ingest_log_path()
        with path.open("a", encoding="utf-8") as f:
            f.write(line)

    def _load_metadata(self) -> None:
        """Load document metadata from disk.

        Handles v1 (content in metadata.json), v2 (content in per-document
        files, no summary), and v3 (content in per-document files, with
        summary) formats. Auto-migrates older formats to v3.
        """
        if not self.metadata_path.exists():
            return

        try:
            data = json.loads(self.metadata_path.read_text())
            version = data.get("version", 1)

            for doc_data in data.get("documents", []):
                if version >= 2:
                    # v2/v3: Load header from index, content from per-doc file
                    doc = Document.from_dict(doc_data)
                    content_data = self._load_document_content(doc.id)
                    if content_data:
                        doc.content = content_data["content"]
                        chunk_contents = content_data.get("chunks", {})
                        for chunk in doc.chunks:
                            if chunk.id in chunk_contents:
                                chunk.content = chunk_contents[chunk.id]
                else:
                    # v1: Everything is in metadata.json
                    doc = Document.from_dict(doc_data)

                self._documents[doc.id] = doc
                for chunk in doc.chunks:
                    self._chunks[chunk.id] = chunk

            # Migrate older formats → v3
            if version < _FORMAT_VERSION and self._documents:
                logger.info(
                    "migrating_metadata_format",
                    from_version=version,
                    to_version=_FORMAT_VERSION,
                    document_count=len(self._documents),
                )
                if version < 2:
                    for doc in self._documents.values():
                        self._save_document_content(doc)
                self._save_metadata()

            # Ensure index.md exists for legacy KBs (no extra cost; ~ms)
            if self._documents and not self._index_md_path().exists():
                self._rebuild_index_md()

        except (json.JSONDecodeError, KeyError) as e:
            # Log error but continue with empty state
            logger.warning(
                "failed_to_load_metadata",
                error=str(e),
                path=str(self.metadata_path),
            )

    def _save_metadata(self) -> None:
        """Save document metadata index to disk (without content)."""
        doc_headers = []
        for doc in self._documents.values():
            header = doc.to_dict()
            # Strip content from index — stored in per-document files
            header.pop("content", None)
            for chunk_header in header.get("chunks", []):
                chunk_header.pop("content", None)
            doc_headers.append(header)

        data = {
            "version": _FORMAT_VERSION,
            "documents": doc_headers,
            "updated_at": datetime.now().isoformat(),
        }
        atomic_write_json(self.metadata_path, data)

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
        """
        self.files_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{doc_id}{extension}"
        dest = self.files_dir / filename
        dest.write_bytes(file_bytes)
        return Path(filename)

    def get_file_path(self, doc_id: str) -> Path | None:
        """Get the absolute file path for a document's stored file.

        Args:
            doc_id: Document ID.

        Returns:
            Absolute Path if file exists, None otherwise.
        """
        doc = self._documents.get(doc_id)
        if doc and doc.file_path:
            abs_path = self.files_dir / doc.file_path
            if abs_path.exists():
                return abs_path
        return None

    @staticmethod
    def extract_text_from_pdf(file_path: Path) -> str:
        """Extract text from a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Extracted text, or empty string on failure.
        """
        from agentic_cli.tools.pdf_utils import extract_pdf_text

        return extract_pdf_text(file_path)

    @staticmethod
    def _truncate_summary(content: str) -> str:
        """Return the deterministic fallback summary (first ~500 chars)."""
        return truncate(content, 500) if content else ""

    # Cap the amount of content we hand to the LLM summarizer. Long PDFs
    # can easily exceed sensible prompt budgets, and the summary only
    # needs the lead-in to be useful.
    _SUMMARY_INPUT_CHAR_LIMIT = 12_000

    _SUMMARY_PROMPT_TEMPLATE = (
        "Summarize the following document in 2-3 sentences (max ~500 "
        "characters). Focus on the main topic, key contributions, and "
        "why the document is relevant. Return only the summary text "
        "with no preamble or markdown.\n\n"
        "Title: {title}\n\n"
        "Content:\n---\n{content}\n---"
    )

    async def generate_summary(self, content: str, title: str = "") -> str:
        """Generate a document summary via the registered LLM summarizer.

        Falls back to the first ~500 chars of ``content`` if no summarizer
        is registered, if the summarizer raises, or if it returns empty.

        Args:
            content: Full document text.
            title: Document title, embedded in the prompt for context.

        Returns:
            Summary string (~500 chars).
        """
        if not content:
            return ""

        try:
            from agentic_cli.workflow.service_registry import (
                get_service,
                LLM_SUMMARIZER,
            )

            summarizer = get_service(LLM_SUMMARIZER)
            if summarizer is not None:
                capped = content[: self._SUMMARY_INPUT_CHAR_LIMIT]
                prompt = self._SUMMARY_PROMPT_TEMPLATE.format(
                    title=title or "(untitled)",
                    content=capped,
                )
                summary = await summarizer.summarize(capped, prompt)
                if summary:
                    return summary.strip()
        except Exception:
            logger.debug("kb_summary_generation_failed", exc_info=True)

        return self._truncate_summary(content)

    _SIDECAR_PROMPT_TEMPLATE = (
        "Extract a structured payload from the document below. "
        "Return PLAIN TEXT in exactly this layout, with section headers "
        "in ALL CAPS followed by a colon:\n\n"
        "SUMMARY:\n"
        "<3-5 paragraph synthesis of the document, roughly 1500-3000 "
        "chars total. Use blank lines between paragraphs. Cover, in "
        "order: (1) the problem the work addresses and why it matters, "
        "(2) the technical approach / methodology with enough detail "
        "to convey how it works, (3) the main results with concrete "
        "numbers or outcomes, (4) limitations or open questions, and "
        "(5) how this fits into the broader landscape. Write for a "
        "reader who will use this instead of re-reading the paper.>\n\n"
        "CLAIMS:\n"
        "- <one specific claim with numbers or concrete evidence>\n"
        "- <one specific claim with numbers or concrete evidence>\n"
        "- <aim for 5-10 claims, each standalone and specific>\n\n"
        "ENTITIES:\n"
        "<KindLabel>: <comma-separated names>\n"
        "<KindLabel>: <comma-separated names>\n"
        "Use kind labels like Models, Datasets, Methods, Authors, Tools, "
        "Organizations as relevant; be exhaustive within each kind. "
        "Omit any section that has no real content.\n\n"
        "STRICT OUTPUT RULES:\n"
        "- No preamble, no meta-commentary, no thinking trace.\n"
        "- No markdown bold/italic on the SUMMARY/CLAIMS/ENTITIES headers.\n"
        "- Start your response directly with 'SUMMARY:'.\n"
        "- Always emit all three sections when content supports them.\n"
        "- The summary is prose (paragraphs), not a bullet list.\n\n"
        "Title: {title}\n\n"
        "Content:\n---\n{content}\n---"
    )

    async def generate_sidecar_payload(
        self, content: str, title: str = ""
    ) -> dict[str, Any]:
        """Generate the structured payload for a document sidecar.

        Returns a dict with keys ``summary`` (str), ``claims`` (list[str]),
        and ``entities`` (dict[str, list[str]]). Falls back to
        ``{summary: truncate(content), claims: [], entities: {}}`` if no
        summarizer is registered or the LLM call fails.
        """
        fallback = {
            "summary": self._truncate_summary(content),
            "claims": [],
            "entities": {},
        }
        if not content:
            return fallback

        try:
            from agentic_cli.workflow.service_registry import (
                get_service,
                LLM_SUMMARIZER,
            )

            summarizer = get_service(LLM_SUMMARIZER)
            if summarizer is None:
                return fallback

            capped = content[: self._SUMMARY_INPUT_CHAR_LIMIT]
            prompt = self._SIDECAR_PROMPT_TEMPLATE.format(
                title=title or "(untitled)",
                content=capped,
            )
            raw = await summarizer.summarize(capped, prompt)
        except Exception:
            logger.debug("kb_sidecar_payload_failed", exc_info=True)
            return fallback

        return self._parse_sidecar_response(raw) or fallback

    # Matches section headers like "SUMMARY:", "**SUMMARY:**", "  Claims :",
    # tolerating leading whitespace and optional markdown bold wrappers
    # (including a trailing ``**`` immediately after the colon).
    _SECTION_HEADER_RE = re.compile(
        r"^\s*\**\s*(SUMMARY|CLAIMS|ENTITIES)\s*\**\s*:\**",
        re.MULTILINE | re.IGNORECASE,
    )

    @classmethod
    def _parse_sidecar_response(cls, raw: str) -> dict[str, Any] | None:
        """Parse the SUMMARY/CLAIMS/ENTITIES blocks. Returns None on garbage.

        Sections are split by regex so multi-paragraph summaries preserve
        their blank-line breaks verbatim. CLAIMS is parsed as a bullet
        list; ENTITIES as ``Kind: comma, separated, names`` lines.
        """
        if not raw:
            return None

        matches = list(cls._SECTION_HEADER_RE.finditer(raw))
        if not matches:
            return None

        sections: dict[str, str] = {}
        for i, m in enumerate(matches):
            key = m.group(1).lower()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
            sections[key] = raw[start:end].strip()

        summary = sections.get("summary", "")

        claims: list[str] = []
        for line in sections.get("claims", "").splitlines():
            stripped = line.strip().lstrip("-*").strip()
            if stripped:
                claims.append(stripped)

        entities: dict[str, list[str]] = {}
        for line in sections.get("entities", "").splitlines():
            stripped = line.strip().lstrip("-*").strip()
            if ":" in stripped:
                kind, _, names = stripped.partition(":")
                items = [n.strip() for n in names.split(",") if n.strip()]
                if items:
                    entities[kind.strip()] = items

        if not summary and not claims and not entities:
            return None
        return {"summary": summary, "claims": claims, "entities": entities}

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------

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
                Async callers that want an LLM-generated summary should
                ``await self.generate_summary(...)`` first and pass the
                result in here.

        Returns:
            The created Document object.
        """
        if summary is None:
            summary = self._truncate_summary(content)

        # Create document
        doc = Document.create(
            title=title,
            content=content,
            source_type=source_type,
            summary=summary,
            source_url=source_url,
            metadata=metadata,
        )

        # Store file if provided
        if file_bytes is not None:
            rel_path = self.store_file(doc.id, file_bytes, file_extension)
            doc.file_path = rel_path

        # Chunk and embed outside lock (CPU-bound, no shared state yet)
        chunk_texts = self._embedding_service.chunk_document(
            content, chunk_size, chunk_overlap
        )

        for i, chunk_text in enumerate(chunk_texts):
            chunk = DocumentChunk.create(
                document_id=doc.id,
                content=chunk_text,
                chunk_index=i,
                metadata={"title": title},
            )
            doc.chunks.append(chunk)

        embeddings: list[list[float]] = []
        if doc.chunks:
            chunk_texts = [c.content for c in doc.chunks]
            embeddings = self._embedding_service.embed_batch(chunk_texts)
            for chunk, embedding in zip(doc.chunks, embeddings):
                chunk.embedding = embedding

        # Mutate shared state under lock
        with self._lock:
            # Add embeddings first — if this fails, no state is modified
            if doc.chunks and embeddings:
                chunk_ids = [c.id for c in doc.chunks]
                self._vector_store.add_embeddings(chunk_ids, embeddings)

            # Now safe to update in-memory state
            for chunk in doc.chunks:
                self._chunks[chunk.id] = chunk

            self._documents[doc.id] = doc

            # Persist (content in per-doc file, headers in metadata index)
            self._save_document_content(doc)
            self._write_sidecar(doc, sidecar_payload)
            self._rebuild_index_md()
            self._append_ingest_log("ingest", doc)
            self._save_metadata()
            self._vector_store.save()

            # Add to BM25 index
            bm25_index = getattr(self, "_bm25_index", None)
            if bm25_index is not None and doc.chunks:
                bm25_ids = [c.id for c in doc.chunks]
                bm25_texts = [c.content for c in doc.chunks]
                bm25_index.add_documents(bm25_ids, bm25_texts)
                bm25_index.save(self.embeddings_dir)

        return doc

    def search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Search the knowledge base using hybrid retrieval.

        Uses both semantic (vector) and keyword (BM25) search, merged via
        Reciprocal Rank Fusion. Falls back to semantic-only if BM25 is unavailable.
        """
        start_time = time.time()

        # Semantic search
        embed_start = time.time()
        query_embedding = self._embedding_service.embed_text(query)
        embed_time = (time.time() - embed_start) * 1000

        with self._lock:
            search_start = time.time()
            semantic_results = self._vector_store.search(query_embedding, top_k * 2)
            search_time = (time.time() - search_start) * 1000

        # BM25 search
        bm25_results = []
        bm25_index = getattr(self, "_bm25_index", None)
        if bm25_index is not None:
            bm25_results = bm25_index.search(query, top_k=top_k * 2)

        # Fuse results
        if bm25_results:
            fused = self._fuse_results(semantic_results, bm25_results)
        else:
            fused = semantic_results

        # Build SearchResult objects, apply filters, deduplicate
        results: list[SearchResult] = []
        seen_docs: set[str] = set()
        with self._lock:
            for chunk_id, score in fused:
                if len(results) >= top_k:
                    break
                chunk = self._chunks.get(chunk_id)
                if not chunk:
                    continue
                doc = self._documents.get(chunk.document_id)
                if not doc:
                    continue
                if doc.id in seen_docs:
                    continue
                if filters and not self._matches_filters(doc, filters):
                    continue
                seen_docs.add(doc.id)
                highlight = truncate(chunk.content)
                results.append(
                    SearchResult(
                        document=doc,
                        chunk=chunk,
                        score=score,
                        highlight=highlight,
                    )
                )

        total_time = (time.time() - start_time) * 1000

        return {
            "results": [r.to_dict() for r in results],
            "total_matches": len(results),
            "query_embedding_time_ms": round(embed_time, 2),
            "search_time_ms": round(search_time, 2),
            "total_time_ms": round(total_time, 2),
        }

    @staticmethod
    def _fuse_results(
        semantic_results: list[tuple[str, float]],
        bm25_results: list[tuple[str, float]],
        k: int = 60,
    ) -> list[tuple[str, float]]:
        """Merge ranked lists using Reciprocal Rank Fusion."""
        from collections import defaultdict
        scores: dict[str, float] = defaultdict(float)
        for rank, (chunk_id, _) in enumerate(semantic_results):
            scores[chunk_id] += 1.0 / (k + rank + 1)
        for rank, (chunk_id, _) in enumerate(bm25_results):
            scores[chunk_id] += 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _matches_filters(self, doc: Document, filters: dict[str, Any]) -> bool:
        """Check if document matches the given filters."""
        return matches_document_filters(doc, filters)

    def get_document(self, doc_id: str) -> Document | None:
        """Retrieve a specific document by ID.

        Args:
            doc_id: Document ID.

        Returns:
            Document if found, None otherwise.
        """
        return self._documents.get(doc_id)

    def find_document(self, id_or_title: str) -> Document | None:
        """Find a document by exact ID, ID prefix, or title substring.

        Lookup order:
        1. Exact ID match
        2. ID prefix match
        3. Case-insensitive title substring match

        Args:
            id_or_title: Document ID (or prefix) or title substring.

        Returns:
            Document if found, None otherwise.
        """
        # Exact ID
        doc = self._documents.get(id_or_title)
        if doc:
            return doc

        # ID prefix
        for doc_id, d in self._documents.items():
            if doc_id.startswith(id_or_title):
                return d

        # Title substring (case-insensitive)
        query_lower = id_or_title.lower()
        for d in self._documents.values():
            if query_lower in d.title.lower():
                return d

        return None

    def list_documents(
        self,
        source_type: SourceType | None = None,
        limit: int = 100,
    ) -> list[Document]:
        """List documents with optional filtering.

        Args:
            source_type: Optional source type filter.
            limit: Maximum number of documents to return.

        Returns:
            List of documents.
        """
        docs = list(self._documents.values())

        if source_type:
            docs = [d for d in docs if d.source_type == source_type]

        # Sort by updated_at descending
        docs.sort(key=lambda d: d.updated_at, reverse=True)

        return docs[:limit]

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document from the knowledge base.

        Args:
            doc_id: Document ID.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            doc = self._documents.get(doc_id)
            if not doc:
                return False

            # Remove chunks from vector store
            chunk_ids = [c.id for c in doc.chunks]
            self._vector_store.remove_embeddings(chunk_ids)

            bm25_index = getattr(self, "_bm25_index", None)
            if bm25_index is not None:
                bm25_index.remove_documents(chunk_ids)
                bm25_index.save(self.embeddings_dir)

            # Remove from memory
            for chunk_id in chunk_ids:
                self._chunks.pop(chunk_id, None)
            del self._documents[doc_id]
            self._sidecar_locks.pop(doc_id, None)

            # Persist
            self._delete_document_content(doc_id)
            self._delete_sidecar(doc_id)
            self._rebuild_index_md()
            self._append_ingest_log("delete", doc)
            self._save_metadata()
            self._vector_store.save()

        return True

    def get_stats(self) -> dict[str, Any]:
        """Get knowledge base statistics.

        Returns:
            Dict with document count, chunk count, and other stats.
        """
        source_counts: dict[str, int] = {}
        for doc in self._documents.values():
            source = doc.source_type.value
            source_counts[source] = source_counts.get(source, 0) + 1

        embedding_model = "unknown"
        if self._settings:
            embedding_model = self._settings.embedding_model

        return {
            "document_count": len(self._documents),
            "chunk_count": len(self._chunks),
            "vector_count": self._vector_store.size,
            "source_counts": source_counts,
            "embedding_model": embedding_model,
            "embedding_dim": self._embedding_service.embedding_dim,
        }

    def clear(self) -> None:
        """Clear all documents from the knowledge base."""
        with self._lock:
            # Delete per-document content files and sidecars
            for doc_id in list(self._documents):
                self._delete_document_content(doc_id)
                self._delete_sidecar(doc_id)

            self._documents = {}
            self._chunks = {}
            self._sidecar_locks.clear()
            self._vector_store.clear()
            self._save_metadata()
            self._vector_store.save()
            self._rebuild_index_md()

    async def backfill_sidecars(
        self,
        progress_cb: "Any | None" = None,
    ) -> int:
        """Generate missing sidecars for all documents in the KB.

        Iterates the in-memory document set, generates a sidecar payload
        via the registered LLM summarizer for any doc that doesn't have a
        sidecar file, and writes it. Returns the count of sidecars written.
        Existing sidecars are not touched.

        Per-doc locks (``_sidecar_locks``) coordinate with the lazy
        sidecar-generation path in ``kb_read`` (Task 8): concurrent
        backfill + read on the same doc never double-LLM.

        Raises ``BackfillAlreadyRunning`` if another ``backfill_sidecars``
        call is already in progress on this manager. The in-progress flag
        is set and cleared synchronously (before/after any ``await``), so
        two concurrent callers cannot both acquire it.

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
        # Check-and-set the running flag and snapshot the work list under
        # one lock acquisition so a second thread cannot slip past the
        # guard between the read and the write.
        with self._lock:
            if self._backfill_running:
                raise BackfillAlreadyRunning(
                    "backfill_sidecars is already running on this KB"
                )
            self._backfill_running = True
            todo = [
                doc
                for doc in self._documents.values()
                if not self._sidecar_path(doc.id).exists()
            ]
        try:
            written = 0
            total = len(todo)
            for i, doc in enumerate(todo):
                # Guard against delete-during-backfill — doc may have
                # been removed since the snapshot was taken. Checked
                # under the lock so a concurrent delete_document cannot
                # interleave between the check and use.
                with self._lock:
                    if doc.id not in self._documents:
                        continue
                lock = self.get_or_create_sidecar_lock(doc.id)
                async with lock:
                    # Double-check inside the lock — another task may
                    # have written it (e.g. lazy kb_read fallback).
                    if self._sidecar_path(doc.id).exists():
                        continue
                    with self._lock:
                        if doc.id not in self._documents:
                            continue
                    if progress_cb is not None:
                        progress_cb(i, total, doc)
                    payload = await self.generate_sidecar_payload(
                        doc.content, title=doc.title
                    )
                    self._write_sidecar(doc, payload)
                    written += 1
            return written
        finally:
            with self._lock:
                self._backfill_running = False


class BackfillAlreadyRunning(RuntimeError):
    """Raised when ``backfill_sidecars`` is called while another call is
    already in progress on the same ``KnowledgeBaseManager``."""
