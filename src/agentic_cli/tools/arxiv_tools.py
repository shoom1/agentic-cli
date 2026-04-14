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

from agentic_cli.tools.registry import (
    register_tool,
    ToolCategory,
    PermissionLevel,
)
from agentic_cli.workflow.service_registry import ARXIV_SOURCE, require_service


def _clean_arxiv_id(arxiv_id: str) -> str:
    """Clean and normalize an arXiv paper ID.

    Handles various formats:
    - New plain ID: '1706.03762'
    - With version: '1706.03762v2'
    - Old format: 'math/0607733', 'hep-th/9901001v1'
    - Full URL: 'https://arxiv.org/abs/1706.03762'
    - Old URL: 'https://arxiv.org/abs/math/0607733'
    - PDF URL: 'https://arxiv.org/pdf/1706.03762.pdf'

    Args:
        arxiv_id: The arXiv ID in any supported format

    Returns:
        Cleaned arXiv ID (e.g., '1706.03762' or 'math/0607733')
    """
    import re

    # Extract ID from URLs
    if "arxiv.org" in arxiv_id:
        # New format: YYMM.NNNNN
        match = re.search(r"(\d{4}\.\d{4,5})", arxiv_id)
        if match:
            arxiv_id = match.group(1)
        else:
            # Old format: subject/NNNNNNN (e.g., math/0607733)
            match = re.search(r"([a-zA-Z-]+/\d{7})", arxiv_id)
            if match:
                arxiv_id = match.group(1)

    # Remove version suffix (e.g., v1, v2)
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

    return arxiv_id


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
