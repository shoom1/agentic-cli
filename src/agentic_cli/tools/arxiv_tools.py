"""ArXiv tools for agentic workflows.

Provides tools for searching and fetching arXiv paper metadata.

The ``search_arxiv`` and ``fetch_arxiv_paper`` functions below are the
ones discovered by ``@register_tool`` and reached from ad-hoc call sites
and tests. At runtime inside an agent, ``_get_service_tool_map`` replaces
them with closure-bound versions produced by ``factories.make_arxiv_tools``.

Both paths share the same implementation via ``_search_arxiv_with_source``
and ``_fetch_arxiv_paper_with_source`` — the module versions fetch the
``ArxivSearchSource`` from the service registry, the factory versions
capture it in a closure.
"""

from typing import Any

from agentic_cli.tools.arxiv_source import _clean_arxiv_id  # re-exported for tests/back-compat
from agentic_cli.tools.pdf_utils import extract_pdf_text
from agentic_cli.tools.registry import (
    register_tool,
    ToolCategory,
    PermissionLevel,
)
from agentic_cli.workflow.permissions import Capability
from agentic_cli.workflow.service_registry import (
    ARXIV_SOURCE,
    KB_MANAGER,
    require_service,
)


_VALID_SORT_BY = ("relevance", "lastUpdatedDate", "submittedDate")
_VALID_SORT_ORDER = ("ascending", "descending")


def _validate_sort_options(sort_by: str, sort_order: str) -> dict[str, Any] | None:
    """Return an error dict if sort options are invalid, else None."""
    if sort_by not in _VALID_SORT_BY:
        return {"success": False, "error": f"sort_by must be one of {_VALID_SORT_BY}, got '{sort_by}'"}
    if sort_order not in _VALID_SORT_ORDER:
        return {"success": False, "error": f"sort_order must be one of {_VALID_SORT_ORDER}, got '{sort_order}'"}
    return None


def _search_arxiv_with_source(
    source,
    query: str,
    max_results: int = 10,
    categories: list[str] | None = None,
    sort_by: str = "relevance",
    sort_order: str = "descending",
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """Run an arXiv search against an explicit source instance.

    Shared implementation used by both the module-level ``search_arxiv``
    and the factory-bound version. Callers must validate sort options
    before reaching here.
    """
    results = source.search(
        query=query,
        max_results=max_results,
        categories=categories,
        sort_by=sort_by,
        sort_order=sort_order,
        date_from=date_from,
        date_to=date_to,
    )

    if source.last_error is not None:
        return {
            "success": False,
            "error": source.last_error,
            "query": query,
        }

    papers = []
    for result in results:
        paper = {
            "title": result.title,
            "authors": result.metadata.get("authors", []),
            "abstract": result.snippet,
            "url": result.url,
            "abs_url": result.metadata.get("abs_url", result.url),
            "pdf_url": result.metadata.get("pdf_url", ""),
            "src_url": result.metadata.get("src_url", ""),
            "published_date": result.metadata.get("published", ""),
            "categories": result.metadata.get("categories", []),
            "arxiv_id": result.metadata.get("arxiv_id", ""),
        }
        papers.append(paper)

    return {
        "success": True,
        "papers": papers,
        "total_found": len(papers),
        "query": query,
    }


async def _fetch_arxiv_paper_with_source(source, arxiv_id: str) -> dict[str, Any]:
    """Fetch a single arXiv paper's metadata against an explicit source instance.

    Delegates to ``source.fetch_by_id`` and wraps exceptions into the
    ``{"success": bool, ...}`` contract expected by tool callers. The
    arxiv URL and feedparser call live on the source — this wrapper
    only adapts the shape and adds ``url`` as a backward-compat alias
    for ``abs_url``.
    """
    arxiv_id = _clean_arxiv_id(arxiv_id)

    try:
        paper = await source.fetch_by_id(arxiv_id)
    except LookupError as exc:
        return {"success": False, "error": str(exc)}
    except RuntimeError as exc:
        return {"success": False, "error": str(exc)}

    return {"success": True, "paper": {**paper, "url": paper["abs_url"]}}


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    permission_level=PermissionLevel.SAFE,
    capabilities=[Capability("http.read")],
    description="Search arXiv for academic papers by query, category, or date range. Use this to find research papers on a topic.",
)
def search_arxiv(
    query: str,
    max_results: int = 10,
    categories: list[str] | None = None,
    sort_by: str = "relevance",
    sort_order: str = "descending",
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """Search arXiv for academic papers.

    Args:
        query: Search query for arXiv papers
        max_results: Maximum number of results to return
        categories: Optional list of arXiv categories to filter (e.g., ['cs.LG', 'cs.CV'])
        sort_by: Sort by 'relevance', 'lastUpdatedDate', or 'submittedDate'
        sort_order: Sort order 'ascending' or 'descending'
        date_from: Filter papers submitted after this date (YYYY-MM-DD)
        date_to: Filter papers submitted before this date (YYYY-MM-DD)

    Returns:
        Dictionary with search results and metadata
    """
    err = _validate_sort_options(sort_by, sort_order)
    if err is not None:
        return err

    source = require_service(ARXIV_SOURCE)
    if isinstance(source, dict):
        return source

    return _search_arxiv_with_source(
        source,
        query=query,
        max_results=max_results,
        categories=categories,
        sort_by=sort_by,
        sort_order=sort_order,
        date_from=date_from,
        date_to=date_to,
    )


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    permission_level=PermissionLevel.SAFE,
    capabilities=[Capability("http.read")],
    description="Fetch metadata for a specific arXiv paper by ID or URL. Returns title, authors, abstract, categories, and PDF URL.",
)
async def fetch_arxiv_paper(
    arxiv_id: str,
) -> dict[str, Any]:
    """Fetch detailed information about a specific arXiv paper.

    Args:
        arxiv_id: The arXiv paper ID, e.g. '1706.03762' or '1706.03762v2'.
                  Also accepts full arXiv URLs.

    Returns:
        Dictionary with paper metadata or error information.
    """
    source = require_service(ARXIV_SOURCE)
    if isinstance(source, dict):
        return source

    return await _fetch_arxiv_paper_with_source(source, arxiv_id)


async def _ingest_arxiv_paper_with_services(
    source,
    kb_manager,
    arxiv_id: str,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Download an arXiv paper's PDF, extract text, and ingest into the KB.

    Shared implementation used by both the module-level ``ingest_arxiv_paper``
    tool and the closure-bound factory version. Composes the arxiv layer
    (metadata + PDF download, with shared rate limiter) and the KB layer
    (chunking + indexing + storage of the raw bytes).

    Behavior:
    1. Normalize arxiv_id (handles URLs and version suffixes).
    2. Fetch metadata via ``source.fetch_by_id`` — a cache hit if the
       paper was returned by a prior search, else a single API call.
    3. Wait on the shared rate limiter before the PDF download (arxiv's
       crawl policy doesn't distinguish the API host from arxiv.org).
    4. Download the PDF from the feed-provided ``pdf_url``.
    5. Extract text and ingest into KB with full metadata.
    6. On PDF download failure, fall back to ingesting the abstract so
       the agent at least has searchable content.
    """
    from agentic_cli.knowledge_base.models import SourceType

    arxiv_id = _clean_arxiv_id(arxiv_id)

    # 1. Metadata (cache hit if available)
    try:
        paper = await source.fetch_by_id(arxiv_id)
    except LookupError as exc:
        return {"success": False, "error": str(exc)}
    except RuntimeError as exc:
        return {"success": False, "error": str(exc)}

    title = paper.get("title", "") or f"ArXiv paper {arxiv_id}"
    abstract = paper.get("abstract", "")

    meta: dict[str, Any] = {
        "arxiv_id": arxiv_id,
        "abs_url": paper.get("abs_url", ""),
        "pdf_url": paper.get("pdf_url", ""),
        "src_url": paper.get("src_url", ""),
        "authors": paper.get("authors", []),
        "abstract": abstract,
        "categories": paper.get("categories", []),
        "primary_category": paper.get("primary_category", ""),
        "published_date": paper.get("published_date", ""),
        "updated_date": paper.get("updated_date", ""),
    }
    if tags:
        meta["tags"] = tags

    # 2. PDF download via the source's encapsulated downloader (shared
    # rate limiter, shared httpx import). On any failure we fall back to
    # the abstract so the caller still ends up with searchable content.
    file_bytes: bytes | None = None
    content = ""
    pdf_url = paper.get("pdf_url", "")

    if pdf_url:
        try:
            file_bytes = await source.download_pdf(pdf_url)
            content = extract_pdf_text(file_bytes)
            meta["file_size_bytes"] = len(file_bytes)
        except Exception as exc:
            file_bytes = None
            content = abstract or title
            meta["pdf_download_error"] = str(exc)

    if not content:
        content = abstract or title

    # 3. Generate LLM summary (async, safe before taking the KB lock)
    summary = await kb_manager.generate_summary(content, title=title)

    # 4. Ingest into KB
    try:
        doc = kb_manager.ingest_document(
            content=content,
            title=title,
            source_type=SourceType.ARXIV,
            source_url=paper.get("abs_url", "") or f"https://arxiv.org/abs/{arxiv_id}",
            metadata=meta,
            file_bytes=file_bytes,
            file_extension=".pdf",
            summary=summary,
        )
    except Exception as exc:
        return {"success": False, "error": f"Ingestion failed: {exc}"}

    return {
        "success": True,
        "document_id": doc.id,
        "title": doc.title,
        "chunks_created": len(doc.chunks),
        "summary": doc.summary,
        "pdf_downloaded": file_bytes is not None,
    }


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    permission_level=PermissionLevel.SAFE,
    capabilities=[Capability("http.read"), Capability("kb.write")],
    description="Download an arXiv paper's PDF, extract text, and ingest it into the knowledge base. Use this to add a specific arXiv paper to long-term storage so it can be searched later.",
)
async def ingest_arxiv_paper(
    arxiv_id: str,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Download and ingest an arXiv paper into the knowledge base.

    Args:
        arxiv_id: The arXiv paper ID, e.g. '1706.03762' or full URL.
        tags: Optional list of tags to attach to the stored document.

    Returns:
        Dictionary with document_id, title, chunks_created, and a
        pdf_downloaded flag (False if PDF fetch failed and the abstract
        was used as fallback content).
    """
    source = require_service(ARXIV_SOURCE)
    if isinstance(source, dict):
        return source

    kb_manager = require_service(KB_MANAGER)
    if isinstance(kb_manager, dict):
        return kb_manager

    return await _ingest_arxiv_paper_with_services(source, kb_manager, arxiv_id, tags=tags)
