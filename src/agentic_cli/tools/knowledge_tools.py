"""Knowledge base tools for agentic workflows.

Provides tools for managing documents in the unified knowledge base:
- kb_ingest: Ingest text, files, or URLs into KB
- kb_search: Semantic search across all documents
- kb_read: Read a stored document (sidecar by default, full text with full=True)
- kb_list: List documents with summaries

Each tool comes in two flavors that share a single implementation:

- ``@register_tool``-decorated module functions look up the KB managers
  from the service registry and call the shared helpers below.
- The closure-bound versions in ``tools.factories.make_kb_tools`` capture
  the KB managers in a closure and call the same helpers.

The helpers (``_search_kbs``, ``_ingest_document_with_kb``,
``_read_document_from_kbs``, ``_list_documents_in_kbs``) take the KB
managers as explicit args so both call paths stay in sync.
"""

from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

from agentic_cli.constants import truncate
from agentic_cli.tools.registry import (
    register_tool,
    ToolCategory,
)
from agentic_cli.workflow.permissions import Capability
from agentic_cli.workflow.service_registry import (
    get_service,
    require_service,
    KB_MANAGER,
    USER_KB_MANAGER,
)


# Max chars of extracted text to return via kb_read (full=True).
READ_DOCUMENT_MAX_CHARS = 30_000


# ---------------------------------------------------------------------------
# Pure helpers (no service registry dependency)
# ---------------------------------------------------------------------------


def _build_document_item(d, scope: str) -> dict[str, Any]:
    """Build a document summary dict from a Document object.

    Args:
        d: Document instance.
        scope: "project" or "user".

    Returns:
        Dict with document metadata suitable for kb_list output.
    """
    item: dict[str, Any] = {
        "id": d.id,
        "title": d.title,
        "summary": d.summary,
        "source_type": d.source_type.value,
        "created_at": d.created_at.isoformat(),
        "chunks": len(d.chunks),
        "scope": scope,
    }
    if d.metadata.get("authors"):
        item["authors"] = d.metadata["authors"]
    if d.metadata.get("arxiv_id"):
        item["arxiv_id"] = d.metadata["arxiv_id"]
    if d.metadata.get("tags"):
        item["tags"] = d.metadata["tags"]
    if d.file_path:
        item["has_file"] = True
    return item


def _extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    from agentic_cli.tools.pdf_utils import extract_pdf_text

    return extract_pdf_text(pdf_bytes)


def _detect_extension(url: str, content_type: str = "") -> str:
    """Detect file extension from URL and/or Content-Type header.

    The Content-Type header is the authoritative source; URL suffix is
    a fallback. Many services (arxiv, S3, CDNs) serve PDFs from URLs
    that don't end in ``.pdf``.
    """
    if "application/pdf" in content_type:
        return ".pdf"
    path = url.split("?")[0].split("#")[0]
    if path.endswith(".pdf"):
        return ".pdf"
    return ".bin"


# ---------------------------------------------------------------------------
# Shared implementations — take KB managers as explicit args
# ---------------------------------------------------------------------------


def _find_doc_in_kbs(kb_manager, user_kb_manager, doc_id_or_title: str) -> tuple:
    """Find a document across project + user KBs.

    Lookup order: project KB first, then user KB on miss.

    Returns:
        (document, source_kb) tuple. ``document`` is None if not found
        in either KB. ``source_kb`` is the project KB on miss (so callers
        that want to print / open against the project KB still have a
        handle), or ``(None, None)`` if no project KB was supplied.
    """
    if kb_manager is None:
        return None, None
    doc = kb_manager.find_document(doc_id_or_title)
    if doc is not None:
        return doc, kb_manager
    if user_kb_manager is not None and user_kb_manager is not kb_manager:
        doc = user_kb_manager.find_document(doc_id_or_title)
        if doc is not None:
            return doc, user_kb_manager
    return None, kb_manager


def _merge_kb_results_rrf(
    project_results: list[dict],
    user_results: list[dict],
    top_k: int,
    k: int = 60,
) -> list[dict]:
    """Merge two KB result lists via Reciprocal Rank Fusion.

    Scores from separate FAISS / BM25 indexes live on different
    scales and are not comparable in absolute terms, so a "concat +
    sort by score" merge produces a degenerate ordering whenever the
    two KBs happen to score on different magnitudes. RRF works on
    rank position instead, so the merge is well-defined regardless
    of how the underlying KBs assign scores.

    On document_id collisions, the project entry wins (its dict is
    kept), but both KBs' ranks contribute to the fused score — so a
    document that ranks high in both KBs ends up higher in the merged
    list than one that's only ranked high in one.

    Args:
        project_results: Ordered list of search-result dicts from project KB.
        user_results: Ordered list of search-result dicts from user KB.
        top_k: Maximum results to return after merge.
        k: RRF constant (default 60, the standard value).

    Returns:
        Merged list (length <= top_k) with the fused RRF score in
        each entry's ``score`` field. The original raw KB score is
        discarded; absolute KB scores from independent indexes do
        not compose meaningfully across a merge.
    """
    fused: dict[str, dict] = {}
    fused_scores: dict[str, float] = {}

    for rank, r in enumerate(project_results):
        doc_id = r.get("document_id", "")
        if not doc_id:
            continue
        fused[doc_id] = r
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    for rank, r in enumerate(user_results):
        doc_id = r.get("document_id", "")
        if not doc_id:
            continue
        if doc_id not in fused:
            fused[doc_id] = r
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    merged = []
    for doc_id in sorted_ids[:top_k]:
        entry = fused[doc_id]
        entry["score"] = round(fused_scores[doc_id], 4)
        merged.append(entry)
    return merged


def _search_kbs(
    kb_manager,
    user_kb_manager,
    query: str,
    filters: str = "",
    top_k: int = 10,
) -> dict[str, Any]:
    """Shared implementation for kb_search.

    Caller must pass a non-None ``kb_manager`` — registry-based wrappers
    handle the missing-kb error case before reaching this helper.
    """
    import json as _json

    parsed_filters = None
    if filters:
        try:
            parsed_filters = _json.loads(filters)
        except _json.JSONDecodeError:
            return {"success": False, "error": f"Invalid JSON in filters: {filters}"}

    try:
        result = kb_manager.search(query, filters=parsed_filters, top_k=top_k)

        # Tag project results with scope
        for r in result.get("results", []):
            r["scope"] = "project"

        # Merge user KB results (non-fatal if unavailable)
        if user_kb_manager is not None and user_kb_manager is not kb_manager:
            try:
                user_result = user_kb_manager.search(query, filters=parsed_filters, top_k=top_k)
                for r in user_result.get("results", []):
                    r["scope"] = "user"
                result["results"] = _merge_kb_results_rrf(
                    result.get("results", []),
                    user_result.get("results", []),
                    top_k,
                )
                result["total_matches"] = len(result["results"])
            except Exception:
                logger.debug("user_kb_search_failed", query=query, exc_info=True)

        return {"success": True, **result}
    except Exception as e:
        return {"success": False, "error": f"Search failed: {e}"}


async def _ingest_document_with_kb(
    kb_manager,
    content: str = "",
    url_or_path: str = "",
    title: str = "",
    source_type: str = "user",
    source_url: str | None = None,
    authors: list[str] | None = None,
    abstract: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Shared implementation for kb_ingest.

    Handles all three input modes (text content, local file, remote URL),
    PDF text extraction, source type auto-detection, and ingestion into
    the supplied ``kb_manager``.
    """
    from agentic_cli.knowledge_base.models import SourceType

    # Build metadata dict from optional fields
    meta: dict[str, Any] = {}
    if authors:
        meta["authors"] = authors
    if abstract:
        meta["abstract"] = abstract
    if tags:
        meta["tags"] = tags

    file_bytes: bytes | None = None
    file_extension = ".pdf"

    # --- URL / file path mode ---
    if url_or_path:
        # Auto-detect source type
        if url_or_path.startswith(("http://", "https://")) and source_type == "user":
            source_type = "web"
        elif not url_or_path.startswith(("http://", "https://")) and source_type == "user":
            source_type = "local"

        source_url = source_url or url_or_path

        if url_or_path.startswith(("http://", "https://")):
            # Generic URL download
            try:
                import httpx
            except ImportError:
                return {"success": False, "error": "httpx not installed, cannot download URLs"}

            try:
                async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
                    response = await client.get(url_or_path)
                    response.raise_for_status()
                    file_bytes = response.content
            except httpx.HTTPStatusError as e:
                return {"success": False, "error": f"HTTP {e.response.status_code} downloading file"}
            except httpx.RequestError as e:
                return {"success": False, "error": f"Failed to download: {e}"}

            # Detect extension from Content-Type header, fall back to URL
            content_type = response.headers.get("content-type", "")
            file_extension = _detect_extension(url_or_path, content_type)

            # Extract text if PDF
            if file_extension == ".pdf" and file_bytes:
                content = _extract_text_from_bytes(file_bytes)

            if not title:
                title = url_or_path.split("/")[-1] or url_or_path

        else:
            # Local file
            source_path = Path(url_or_path).expanduser().resolve()
            if not source_path.exists():
                return {"success": False, "error": f"File not found: {url_or_path}"}

            file_bytes = source_path.read_bytes()
            file_extension = source_path.suffix.lower() or ".bin"

            # Extract text if PDF
            if file_extension == ".pdf":
                from agentic_cli.knowledge_base.manager import KnowledgeBaseManager
                content = KnowledgeBaseManager.extract_text_from_pdf(source_path)

            if not title:
                title = source_path.stem

            meta["file_size_bytes"] = len(file_bytes)

    # --- Validate ---
    if not content and not file_bytes:
        return {
            "success": False,
            "error": (
                "No content or file provided. "
                "You must supply either 'content' (text string) or 'url_or_path' (URL or file path). "
                "For ArXiv papers, use ingest_arxiv_paper instead."
            ),
        }

    if not title:
        title = truncate(content, 80)

    # Validate source_type
    try:
        source = SourceType(source_type)
    except ValueError:
        valid = ", ".join(t.value for t in SourceType)
        return {"success": False, "error": f"Invalid source_type: {source_type!r}. Valid: {valid}"}

    # --- Generate sidecar payload via async LLM call (outside lock) ---
    payload = await kb_manager.generate_sidecar_payload(content, title=title)

    # --- Ingest ---
    try:
        doc = kb_manager.ingest_document(
            content=content,
            title=title,
            source_type=source,
            source_url=source_url,
            metadata=meta or None,
            file_bytes=file_bytes,
            file_extension=file_extension,
            summary=payload["summary"] or None,
            sidecar_payload=payload,
        )
    except Exception as e:
        return {"success": False, "error": f"Ingestion failed: {e}"}

    return {
        "success": True,
        "document_id": doc.id,
        "title": doc.title,
        "chunks_created": len(doc.chunks),
        "summary": doc.summary,
    }


async def _read_document_from_kbs(
    kb_manager,
    user_kb_manager,
    doc_id_or_title: str,
    full: bool = False,
    max_chars: int = READ_DOCUMENT_MAX_CHARS,
) -> dict[str, Any]:
    """Shared implementation for kb_read.

    Default returns the sidecar payload (summary + claims + entities).
    With ``full=True``, returns the raw extracted text up to ``max_chars``.
    Lazily generates a missing sidecar via the manager's per-doc lock.
    """
    doc, source_kb = _find_doc_in_kbs(kb_manager, user_kb_manager, doc_id_or_title)
    if doc is None:
        return {"success": False, "error": f"Document not found: {doc_id_or_title}"}

    if full:
        content = doc.content
        if not content and doc.file_path:
            file_path = source_kb.get_file_path(doc.id)
            if file_path and str(file_path).endswith(".pdf"):
                from agentic_cli.knowledge_base.manager import KnowledgeBaseManager
                content = KnowledgeBaseManager.extract_text_from_pdf(file_path)
        truncated = len(content) > max_chars
        if truncated:
            content = content[:max_chars]
        return {
            "success": True,
            "full": True,
            "document_id": doc.id,
            "title": doc.title,
            "scope": "user" if user_kb_manager is not None and source_kb is user_kb_manager else "project",
            "content": content,
            "truncated": truncated,
            "source_type": doc.source_type.value,
        }

    # Sidecar mode (default). Lazily generate if missing.
    sidecar_path = source_kb._sidecar_path(doc.id)
    if not sidecar_path.exists():
        lock = source_kb.get_or_create_sidecar_lock(doc.id)
        async with lock:
            if not sidecar_path.exists():
                # Resolve content the same way full=True does — extract from
                # PDF when in-memory content is empty. Avoids caching a
                # useless empty sidecar for legacy PDFs.
                content_for_payload = doc.content
                if not content_for_payload and doc.file_path:
                    file_path = source_kb.get_file_path(doc.id)
                    if file_path and str(file_path).endswith(".pdf"):
                        from agentic_cli.knowledge_base.manager import (
                            KnowledgeBaseManager,
                        )
                        content_for_payload = (
                            KnowledgeBaseManager.extract_text_from_pdf(file_path)
                        )
                payload = await source_kb.generate_sidecar_payload(
                    content_for_payload, title=doc.title
                )
                source_kb._write_sidecar(doc, payload)

    sidecar_text = sidecar_path.read_text()
    return {
        "success": True,
        "full": False,
        "document_id": doc.id,
        "title": doc.title,
        "source_type": doc.source_type.value,
        "scope": "user" if user_kb_manager is not None and source_kb is user_kb_manager else "project",
        "summary": doc.summary,
        "sidecar": sidecar_text,
    }


def _list_documents_in_kbs(
    kb_manager,
    user_kb_manager,
    query: str = "",
    source_type: str = "",
    limit: int = 20,
) -> dict[str, Any]:
    """Shared implementation for kb_list."""
    from agentic_cli.knowledge_base.models import SourceType as ST

    # Parse source_type filter
    st_filter = None
    if source_type:
        try:
            st_filter = ST(source_type)
        except ValueError:
            pass

    docs = kb_manager.list_documents(source_type=st_filter, limit=limit)

    # Apply query filter
    if query:
        query_lower = query.lower()
        docs = [
            d for d in docs
            if query_lower in d.title.lower()
            or any(query_lower in a.lower() for a in d.metadata.get("authors", []))
        ]

    items = []
    seen_ids: set[str] = set()
    for d in docs:
        items.append(_build_document_item(d, "project"))
        seen_ids.add(d.id)

    # Merge user KB documents
    if user_kb_manager is not None and user_kb_manager is not kb_manager:
        try:
            user_docs = user_kb_manager.list_documents(source_type=st_filter, limit=limit)
            if query:
                query_lower = query.lower()
                user_docs = [
                    d for d in user_docs
                    if query_lower in d.title.lower()
                    or any(query_lower in a.lower() for a in d.metadata.get("authors", []))
                ]
            for d in user_docs:
                if d.id not in seen_ids:
                    items.append(_build_document_item(d, "user"))
                    seen_ids.add(d.id)
        except Exception:
            logger.debug("user_kb_list_documents_failed", exc_info=True)

    return {
        "success": True,
        "documents": items,
        "count": len(items),
    }


async def _write_concept_with_kb(
    kb_manager,
    user_kb_manager,
    title: str,
    body: str,
    sources: list[str],
    slug: str = "",
) -> dict[str, Any]:
    """Shared implementation for kb_write_concept.

    Writes to the project KB's concepts directory. Source IDs are
    validated against BOTH KBs (project first, then user if distinct) so
    cross-KB citations work.
    """
    if kb_manager is None:
        return {"success": False, "error": "kb manager not available"}

    def _is_valid_id(doc_id: str) -> bool:
        if kb_manager.get_document(doc_id) is not None:
            return True
        if user_kb_manager is not None and user_kb_manager is not kb_manager:
            if user_kb_manager.get_document(doc_id) is not None:
                return True
        return False

    try:
        return kb_manager.concepts.write(
            title=title,
            body=body,
            sources=sources,
            slug=slug,
            valid_ids_check=_is_valid_id,
        )
    except Exception as e:
        return {"success": False, "error": f"write_concept failed: {e}"}


async def _search_concepts_with_kb(
    kb_manager,
    user_kb_manager,
    query: str,
    limit: int = 10,
) -> dict[str, Any]:
    """Shared implementation for kb_search_concepts.

    Searches the project KB's concepts directory. Async signature
    matches the other tool helpers even though the underlying
    implementation is synchronous grep.
    """
    if kb_manager is None:
        return {"success": False, "error": "kb manager not available"}

    try:
        hits = kb_manager.concepts.search(query, limit=limit)
    except Exception as e:
        return {"success": False, "error": f"search_concepts failed: {e}"}

    return {
        "success": True,
        "concepts": hits,
        "count": len(hits),
    }


# ---------------------------------------------------------------------------
# Registry-bound helper (back-compat for tests/callers that don't have
# explicit KB handles)
# ---------------------------------------------------------------------------


def _find_document_in_kbs(doc_id_or_title: str) -> tuple:
    """Find a document across main and user KBs via the service registry.

    Thin registry-based wrapper around ``_find_doc_in_kbs``. Used by the
    module-level ``@register_tool`` functions and exercised directly by
    ``tests/test_kb_helpers.py``.
    """
    kb = get_service(KB_MANAGER)
    if kb is None:
        return None, None
    user_kb = get_service(USER_KB_MANAGER)
    return _find_doc_in_kbs(kb, user_kb, doc_id_or_title)


# ---------------------------------------------------------------------------
# Module-level @register_tool wrappers
# ---------------------------------------------------------------------------


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    capabilities=[Capability("kb.read")],
    description="Search the local knowledge base for relevant documents using semantic similarity. Use this when you need to find previously ingested papers, notes, or documents.",
)
def kb_search(
    query: str,
    filters: str = "",
    top_k: int = 10,
) -> dict[str, Any]:
    """Search the knowledge base for relevant information.

    Args:
        query: Natural language search query
        filters: Optional JSON string with filters (e.g. '{"source_type": "arxiv", "date_from": "2024-01-01"}')
        top_k: Maximum number of results

    Returns:
        Dictionary with search results and timing information
    """
    kb = require_service(KB_MANAGER)
    if isinstance(kb, dict):
        return kb
    user_kb = get_service(USER_KB_MANAGER)
    return _search_kbs(kb, user_kb, query, filters, top_k)


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    capabilities=[Capability("kb.write")],
    description=(
        "Ingest a document into the knowledge base. "
        "REQUIRED: provide either 'content' (text) or 'url_or_path' (file path or URL). "
        "ArXiv URLs auto-fetch metadata and PDF. "
        "Valid source_type values: arxiv, ssrn, web, internal, user, local."
    ),
)
async def kb_ingest(
    content: str = "",
    url_or_path: str = "",
    title: str = "",
    source_type: str = "user",
    source_url: str | None = None,
    authors: list[str] | None = None,
    abstract: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Ingest a document into the knowledge base.

    You MUST provide at least one of `content` or `url_or_path`:
    1. Text content: provide `content` directly
    2. Local file: provide `url_or_path` pointing to a local file
    3. URL: provide `url_or_path` with an HTTP(S) URL

    For arXiv papers, use `ingest_arxiv_paper` instead — it handles
    metadata, PDF download, and rate limiting automatically.

    Args:
        content: Document text content (REQUIRED if url_or_path not given)
        url_or_path: URL or local file path (REQUIRED if content not given)
        title: Document title
        source_type: Source type (ssrn, web, internal, user, local)
        source_url: Optional URL of the source
        authors: Optional list of author names
        abstract: Optional paper abstract
        tags: Optional tags for categorization

    Returns:
        Dictionary with ingestion result
    """
    kb = get_service(KB_MANAGER)
    if kb is None:
        return {"success": False, "error": "kb manager not available"}
    return await _ingest_document_with_kb(
        kb,
        content=content,
        url_or_path=url_or_path,
        title=title,
        source_type=source_type,
        source_url=source_url,
        authors=authors,
        abstract=abstract,
        tags=tags,
    )


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    capabilities=[Capability("kb.read")],
    description=(
        "Read a stored document by ID or title. Returns the per-document "
        "sidecar (summary, key claims, entities) by default. Pass full=True "
        "to get the raw extracted text up to max_chars."
    ),
)
async def kb_read(
    doc_id_or_title: str,
    full: bool = False,
    max_chars: int = READ_DOCUMENT_MAX_CHARS,
) -> dict[str, Any]:
    """Read a stored document.

    Args:
        doc_id_or_title: Document ID or title substring.
        full: If True, return raw text. Default returns the sidecar payload.
        max_chars: Maximum characters of raw text to return when full=True.
    """
    kb = get_service(KB_MANAGER)
    if kb is None:
        return {"success": False, "error": f"Document not found: {doc_id_or_title}"}
    user_kb = get_service(USER_KB_MANAGER)
    return await _read_document_from_kbs(kb, user_kb, doc_id_or_title, full, max_chars)


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    capabilities=[Capability("kb.read")],
    description="List documents in the knowledge base with summaries. Filter by query or source type. Returns summaries, not full content.",
)
def kb_list(
    query: str = "",
    source_type: str = "",
    limit: int = 20,
) -> dict[str, Any]:
    """List documents with summaries.

    Args:
        query: Optional filter by title substring (case-insensitive).
        source_type: Optional filter by source type.
        limit: Maximum number of documents to return.

    Returns:
        Dictionary with document list.
    """
    kb = get_service(KB_MANAGER)
    if kb is None:
        return {"success": False, "error": "kb manager not available"}
    user_kb = get_service(USER_KB_MANAGER)
    return _list_documents_in_kbs(kb, user_kb, query, source_type, limit)


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    capabilities=[Capability("kb.write")],
    description=(
        "Save an agent-curated concept page summarizing what the KB "
        "knows about a topic. Pages live at concepts/{slug}.md and are "
        "agent-writable, grep-searchable, and human-readable. `sources` "
        "must cite at least one valid document ID from the KB."
    ),
)
async def kb_write_concept(
    title: str,
    body: str,
    sources: list[str],
    slug: str = "",
) -> dict[str, Any]:
    """Create or overwrite a concept page.

    Args:
        title: Human-readable concept title. Used to derive the slug
            when ``slug`` is empty.
        body: Markdown body. Free-form; no required sections.
        sources: Document IDs this concept cites. Must include at least
            one valid ID (verified against the KB).
        slug: Optional explicit slug. If given and already exists, the
            existing concept is overwritten (body replaced, sources
            merged as union). If empty, auto-generated from title and
            collision-suffixed on conflict.
    """
    kb = get_service(KB_MANAGER)
    if kb is None:
        return {"success": False, "error": "kb manager not available"}
    user_kb = get_service(USER_KB_MANAGER)
    return await _write_concept_with_kb(
        kb, user_kb, title=title, body=body, sources=sources, slug=slug,
    )


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    capabilities=[Capability("kb.read")],
    description=(
        "Search concept pages (agent-curated synthesis notes). "
        "Case-insensitive substring match; title hits rank above body "
        "hits. Use when asking 'what does the KB know about X?'."
    ),
)
async def kb_search_concepts(
    query: str,
    limit: int = 10,
) -> dict[str, Any]:
    """Search concept pages.

    Args:
        query: Case-insensitive substring.
        limit: Max hits to return.
    """
    kb = get_service(KB_MANAGER)
    if kb is None:
        return {"success": False, "error": "kb manager not available"}
    user_kb = get_service(USER_KB_MANAGER)
    return await _search_concepts_with_kb(kb, user_kb, query=query, limit=limit)


