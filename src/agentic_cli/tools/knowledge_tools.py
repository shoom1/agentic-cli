"""Knowledge base tools for agentic workflows.

Provides tools for managing documents in the unified knowledge base:
- ingest_document: Ingest text, files, or URLs into KB
- search_knowledge_base: Semantic search across all documents
- read_document: Extract and return text from a stored document
- list_documents: List documents with summaries
- open_document: Open a document's file in the system viewer
"""

import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

from agentic_cli.constants import truncate
from agentic_cli.tools import requires, require_context
from agentic_cli.tools.registry import (
    register_tool,
    ToolCategory,
    PermissionLevel,
)
from agentic_cli.workflow.context import get_context_kb_manager, get_context_user_kb_manager


# Max chars of extracted text to return via read_document.
READ_DOCUMENT_MAX_CHARS = 30_000

# Extensions considered safe to open in the system viewer.
# Excluded: .html (JS execution), .doc/.xls/.ppt (VBA macros),
# .odt/.ods/.odp (LibreOffice macros), .docm/.xlsm/.pptm (Office macros).
SAFE_OPEN_EXTENSIONS: frozenset[str] = frozenset({
    # Documents
    ".pdf", ".txt", ".md", ".csv", ".json", ".xml", ".rtf", ".epub",
    # Modern Office (macro-free by design)
    ".docx", ".xlsx", ".pptx",
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".bmp", ".tiff", ".webp",
})


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    permission_level=PermissionLevel.SAFE,
    description="Search the local knowledge base for relevant documents using semantic similarity. Use this when you need to find previously ingested papers, notes, or documents.",
)
@requires("kb_manager")
@require_context("KB manager", get_context_kb_manager)
def search_knowledge_base(
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
    import json as _json

    parsed_filters = None
    if filters:
        try:
            parsed_filters = _json.loads(filters)
        except _json.JSONDecodeError:
            return {"success": False, "error": f"Invalid JSON in filters: {filters}"}

    try:
        kb = get_context_kb_manager()
        result = kb.search(query, filters=parsed_filters, top_k=top_k)

        # Tag project results with scope
        for r in result.get("results", []):
            r["scope"] = "project"

        # Merge user KB results (non-fatal if unavailable)
        user_kb = get_context_user_kb_manager()
        if user_kb is not None and user_kb is not kb:
            try:
                user_result = user_kb.search(query, filters=parsed_filters, top_k=top_k)
                # Deduplicate by document_id (project wins)
                seen_doc_ids = {r["document_id"] for r in result.get("results", [])}
                for r in user_result.get("results", []):
                    if r["document_id"] not in seen_doc_ids:
                        r["scope"] = "user"
                        result["results"].append(r)
                        seen_doc_ids.add(r["document_id"])
                # Re-sort by score descending and trim to top_k
                result["results"].sort(key=lambda r: r.get("score", 0), reverse=True)
                result["results"] = result["results"][:top_k]
                result["total_matches"] = len(result["results"])
            except Exception:
                logger.debug("user_kb_search_failed", query=query, exc_info=True)

        return {"success": True, **result}
    except Exception as e:
        return {"success": False, "error": f"Search failed: {e}"}


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    permission_level=PermissionLevel.CAUTION,
    description=(
        "Ingest a document into the knowledge base. Accepts text content, a local file path, "
        "or a URL (including ArXiv). ArXiv URLs auto-fetch metadata and PDF. "
        "Valid source_type values: arxiv, ssrn, web, internal, user, local."
    ),
)
@requires("kb_manager")
@require_context("KB manager", get_context_kb_manager)
async def ingest_document(
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

    Supports three modes:
    1. Text content: provide `content` directly
    2. Local file: provide `url_or_path` pointing to a local file
    3. URL: provide `url_or_path` with an HTTP(S) URL
       - ArXiv URLs auto-fetch metadata and download PDF

    Args:
        content: Document text content (for text-based ingestion)
        url_or_path: URL or local file path (for file-based ingestion)
        title: Document title (auto-fetched for ArXiv if empty)
        source_type: Source type (arxiv, ssrn, web, internal, user, local)
        source_url: Optional URL of the source
        authors: Optional list of author names
        abstract: Optional paper abstract
        tags: Optional tags for categorization

    Returns:
        Dictionary with ingestion result
    """
    from agentic_cli.knowledge_base.models import SourceType

    kb = get_context_kb_manager()

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
        if "arxiv.org" in url_or_path:
            source_type = "arxiv"
        elif url_or_path.startswith(("http://", "https://")) and source_type == "user":
            source_type = "web"
        elif not url_or_path.startswith(("http://", "https://")) and source_type == "user":
            source_type = "local"

        source_url = source_url or url_or_path

        if source_type == "arxiv":
            # ArXiv: fetch metadata + download PDF
            result = await _ingest_arxiv(
                url_or_path, title, authors, abstract, meta, kb
            )
            return result

        elif url_or_path.startswith(("http://", "https://")):
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

            # Detect extension from URL
            file_extension = _detect_extension(url_or_path)

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
        return {"success": False, "error": "No content or file provided. Supply 'content' or 'url_or_path'."}

    if not title:
        title = truncate(content, 80)

    # Validate source_type
    try:
        source = SourceType(source_type)
    except ValueError:
        valid = ", ".join(t.value for t in SourceType)
        return {"success": False, "error": f"Invalid source_type: {source_type!r}. Valid: {valid}"}

    # --- Ingest ---
    try:
        doc = kb.ingest_document(
            content=content,
            title=title,
            source_type=source,
            source_url=source_url,
            metadata=meta or None,
            file_bytes=file_bytes,
            file_extension=file_extension,
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


async def _ingest_arxiv(
    url_or_path: str,
    title: str,
    authors: list[str] | None,
    abstract: str,
    meta: dict[str, Any],
    kb,
) -> dict[str, Any]:
    """Handle ArXiv-specific ingestion: fetch metadata + download PDF.

    Args:
        url_or_path: ArXiv URL or ID.
        title: User-provided title (may be empty).
        authors: User-provided authors (may be None).
        abstract: User-provided abstract (may be empty).
        meta: Metadata dict to populate.
        kb: KnowledgeBaseManager instance.

    Returns:
        Tool result dict.
    """
    from agentic_cli.knowledge_base.models import SourceType

    # Extract arxiv ID
    arxiv_id = ""
    match = re.search(r"(\d{4}\.\d{4,5})", url_or_path)
    if match:
        arxiv_id = match.group(1)

    if not arxiv_id:
        return {"success": False, "error": f"Could not extract ArXiv ID from: {url_or_path}"}

    # Fetch metadata via fetch_arxiv_paper
    try:
        from agentic_cli.tools.arxiv_tools import fetch_arxiv_paper
        metadata_result = await fetch_arxiv_paper(arxiv_id)
        if metadata_result.get("success") and "paper" in metadata_result:
            paper_info = metadata_result["paper"]
            title = title or paper_info.get("title", "")
            authors = authors or paper_info.get("authors", [])
            abstract = abstract or paper_info.get("abstract", "")
            meta["arxiv_id"] = arxiv_id
            meta["pdf_url"] = paper_info.get("pdf_url", "")
            meta["categories"] = paper_info.get("categories", [])
    except Exception:
        logger.warning("arxiv_metadata_fetch_failed", arxiv_id=arxiv_id, exc_info=True)

    if authors:
        meta["authors"] = authors
    if abstract:
        meta["abstract"] = abstract

    source_url = f"https://arxiv.org/abs/{arxiv_id}"

    # Download PDF
    file_bytes: bytes | None = None
    content = ""
    try:
        import httpx

        from agentic_cli.tools.arxiv_tools import _get_arxiv_source
        source = _get_arxiv_source()
        source.wait_for_rate_limit()

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            response = await client.get(pdf_url)
            response.raise_for_status()
            file_bytes = response.content

        # Extract text
        content = _extract_text_from_bytes(file_bytes)
        meta["file_size_bytes"] = len(file_bytes)
    except Exception:
        logger.warning("arxiv_pdf_download_failed", arxiv_id=arxiv_id, exc_info=True)
        # Use abstract as fallback content if PDF download fails
        content = abstract or title

    if not title:
        title = f"ArXiv paper {arxiv_id}"

    try:
        doc = kb.ingest_document(
            content=content,
            title=title,
            source_type=SourceType.ARXIV,
            source_url=source_url,
            metadata=meta or None,
            file_bytes=file_bytes,
            file_extension=".pdf",
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


def _extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    try:
        import pypdf
        import io
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except (ImportError, Exception):
        logger.debug("pdf_text_extraction_failed", exc_info=True)
        return ""


def _detect_extension(url: str) -> str:
    """Detect file extension from URL."""
    # Strip query params
    path = url.split("?")[0].split("#")[0]
    if path.endswith(".pdf"):
        return ".pdf"
    return ".bin"


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    permission_level=PermissionLevel.SAFE,
    description="Read and return the text content of a stored document by ID or title. Returns full text (up to max_chars limit).",
)
@requires("kb_manager")
@require_context("KB manager", get_context_kb_manager)
def read_document(
    doc_id_or_title: str,
    max_chars: int = READ_DOCUMENT_MAX_CHARS,
) -> dict[str, Any]:
    """Extract and return text from a stored document.

    Args:
        doc_id_or_title: Document ID or title substring.
        max_chars: Maximum characters to return (default 30K).

    Returns:
        Dictionary with document text and metadata.
    """
    kb = get_context_kb_manager()
    doc = kb.find_document(doc_id_or_title)
    source_kb = kb

    # Fall back to user KB
    if doc is None:
        user_kb = get_context_user_kb_manager()
        if user_kb is not None and user_kb is not kb:
            doc = user_kb.find_document(doc_id_or_title)
            source_kb = user_kb

    if doc is None:
        return {"success": False, "error": f"Document not found: {doc_id_or_title}"}

    content = doc.content

    # If no content, try extracting from stored file
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
        "document_id": doc.id,
        "title": doc.title,
        "content": content,
        "truncated": truncated,
        "source_type": doc.source_type.value,
    }


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    permission_level=PermissionLevel.SAFE,
    description="List documents in the knowledge base with summaries. Filter by query or source type. Returns summaries, not full content.",
)
@requires("kb_manager")
@require_context("KB manager", get_context_kb_manager)
def list_documents(
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
    from agentic_cli.knowledge_base.models import SourceType as ST

    kb = get_context_kb_manager()

    # Parse source_type filter
    st_filter = None
    if source_type:
        try:
            st_filter = ST(source_type)
        except ValueError:
            pass

    docs = kb.list_documents(source_type=st_filter, limit=limit)

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
        item: dict[str, Any] = {
            "id": d.id,
            "title": d.title,
            "summary": d.summary,
            "source_type": d.source_type.value,
            "created_at": d.created_at.isoformat(),
            "chunks": len(d.chunks),
            "scope": "project",
        }
        if d.metadata.get("authors"):
            item["authors"] = d.metadata["authors"]
        if d.metadata.get("arxiv_id"):
            item["arxiv_id"] = d.metadata["arxiv_id"]
        if d.metadata.get("tags"):
            item["tags"] = d.metadata["tags"]
        if d.file_path:
            item["has_file"] = True
        items.append(item)
        seen_ids.add(d.id)

    # Merge user KB documents
    user_kb = get_context_user_kb_manager()
    if user_kb is not None and user_kb is not kb:
        try:
            user_docs = user_kb.list_documents(source_type=st_filter, limit=limit)
            if query:
                query_lower = query.lower()
                user_docs = [
                    d for d in user_docs
                    if query_lower in d.title.lower()
                    or any(query_lower in a.lower() for a in d.metadata.get("authors", []))
                ]
            for d in user_docs:
                if d.id in seen_ids:
                    continue
                item = {
                    "id": d.id,
                    "title": d.title,
                    "summary": d.summary,
                    "source_type": d.source_type.value,
                    "created_at": d.created_at.isoformat(),
                    "chunks": len(d.chunks),
                    "scope": "user",
                }
                if d.metadata.get("authors"):
                    item["authors"] = d.metadata["authors"]
                if d.metadata.get("arxiv_id"):
                    item["arxiv_id"] = d.metadata["arxiv_id"]
                if d.metadata.get("tags"):
                    item["tags"] = d.metadata["tags"]
                if d.file_path:
                    item["has_file"] = True
                items.append(item)
                seen_ids.add(d.id)
        except Exception:
            logger.debug("user_kb_list_documents_failed", exc_info=True)

    return {
        "success": True,
        "documents": items,
        "count": len(items),
    }


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    permission_level=PermissionLevel.DANGEROUS,
    description="Open a document's stored file (e.g. PDF) in the system default viewer. Provide a document ID or title.",
)
@requires("kb_manager")
@require_context("KB manager", get_context_kb_manager)
def open_document(
    doc_id_or_title: str,
) -> dict[str, Any]:
    """Open a document's file in the system viewer.

    Args:
        doc_id_or_title: Document ID or title substring.

    Returns:
        Dictionary with result.
    """
    kb = get_context_kb_manager()
    doc = kb.find_document(doc_id_or_title)
    source_kb = kb

    # Fall back to user KB
    if doc is None:
        user_kb = get_context_user_kb_manager()
        if user_kb is not None and user_kb is not kb:
            doc = user_kb.find_document(doc_id_or_title)
            source_kb = user_kb

    if doc is None:
        return {"success": False, "error": f"Document not found: {doc_id_or_title}"}

    file_path = source_kb.get_file_path(doc.id)
    if file_path is None:
        return {"success": False, "error": f"No file stored for document: {doc.title}"}

    ext = file_path.suffix.lower()
    if ext not in SAFE_OPEN_EXTENSIONS:
        logger.warning(
            "open_document_blocked_extension",
            file_path=str(file_path),
            title=doc.title,
            extension=ext,
        )
        return {
            "success": False,
            "error": f"File type '{ext}' is not allowed. Supported: documents, images, and office files.",
        }

    system = platform.system()
    try:
        if system == "Darwin":
            cmd = ["open", str(file_path)]
        elif system == "Linux":
            cmd = ["xdg-open", str(file_path)]
        elif system == "Windows":
            cmd = ["start", "", str(file_path)]
        else:
            return {"success": False, "error": f"Unsupported platform: {system}"}

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            shell=(system == "Windows"),
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            logger.warning(
                "open_document_failed",
                file_path=str(file_path),
                returncode=result.returncode,
                stderr=stderr,
            )
            return {"success": False, "error": f"Failed to open file: {stderr or 'unknown error'}"}
    except subprocess.TimeoutExpired:
        # open/xdg-open may block if viewer takes time — treat as success
        pass
    except OSError as e:
        return {"success": False, "error": f"Failed to open file: {e}"}

    logger.info(
        "open_document_success",
        file_path=str(file_path),
        title=doc.title,
        extension=ext,
    )
    return {
        "success": True,
        "title": doc.title,
        "file_path": str(file_path),
        "message": f"Opened: {doc.title}",
    }


