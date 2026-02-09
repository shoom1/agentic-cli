"""ArXiv tools for agentic workflows.

Provides tools for searching, fetching, and analyzing arXiv papers.
"""

from typing import Any

from agentic_cli.tools.registry import (
    register_tool,
    ToolCategory,
    PermissionLevel,
)

# Max chars of extracted PDF text to return in tool results.
# ~7.5K tokens — enough for key content without blowing up the context window.
PDF_TEXT_MAX_CHARS = 30_000


# Module-level ArxivSearchSource instance for rate limiting and caching
_arxiv_source = None


def _get_arxiv_source():
    """Get or create the ArxivSearchSource instance."""
    global _arxiv_source
    if _arxiv_source is None:
        from agentic_cli.knowledge_base.sources import ArxivSearchSource
        _arxiv_source = ArxivSearchSource()
    return _arxiv_source


def _clean_arxiv_id(arxiv_id: str) -> str:
    """Clean and normalize an arXiv paper ID.

    Handles various formats:
    - Plain ID: '1706.03762'
    - With version: '1706.03762v2'
    - Full URL: 'https://arxiv.org/abs/1706.03762'
    - PDF URL: 'https://arxiv.org/pdf/1706.03762.pdf'

    Args:
        arxiv_id: The arXiv ID in any supported format

    Returns:
        Cleaned arXiv ID (e.g., '1706.03762')
    """
    import re

    # Extract ID from URLs (handles both abs and pdf URLs)
    if "arxiv.org" in arxiv_id:
        match = re.search(r"(\d{4}\.\d{4,5})", arxiv_id)
        if match:
            arxiv_id = match.group(1)

    # Remove version suffix (e.g., v1, v2)
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

    return arxiv_id


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

    Raises:
        ValueError: If sort_by or sort_order has an invalid value
    """
    # Validate sort options
    valid_sort_by = ("relevance", "lastUpdatedDate", "submittedDate")
    valid_sort_order = ("ascending", "descending")

    if sort_by not in valid_sort_by:
        raise ValueError(f"sort_by must be one of {valid_sort_by}, got '{sort_by}'")
    if sort_order not in valid_sort_order:
        raise ValueError(f"sort_order must be one of {valid_sort_order}, got '{sort_order}'")

    source = _get_arxiv_source()
    results = source.search(
        query=query,
        max_results=max_results,
        categories=categories,
        sort_by=sort_by,
        sort_order=sort_order,
        date_from=date_from,
        date_to=date_to,
    )

    # Check for errors (rate limiting, parse failures, etc.)
    if source.last_error is not None:
        return {
            "success": False,
            "error": source.last_error,
            "query": query,
        }

    # Convert SearchSourceResult to paper dict format
    papers = []
    for result in results:
        paper = {
            "title": result.title,
            "authors": result.metadata.get("authors", []),
            "abstract": result.snippet,
            "url": result.url,
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


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    permission_level=PermissionLevel.SAFE,
    description="Fetch metadata for a specific arXiv paper by ID or URL. Optionally download the PDF and extract text with download='pdf'.",
)
async def fetch_arxiv_paper(
    arxiv_id: str,
    download: str = "",
) -> dict[str, Any]:
    """Fetch detailed information about a specific arXiv paper.

    Args:
        arxiv_id: The arXiv paper ID, e.g. '1706.03762' or '1706.03762v2'.
                  Also accepts full arXiv URLs.
        download: Set to 'pdf' to download the PDF and extract text.
                  Empty string (default) returns metadata only.

    Returns:
        Dictionary with paper details or error information.
        When download='pdf', also includes 'pdf_text' (truncated to PDF_TEXT_MAX_CHARS).
    """
    try:
        import feedparser
    except ImportError:
        return {"success": False, "error": "feedparser not installed"}

    # Clean the arxiv_id
    arxiv_id = _clean_arxiv_id(arxiv_id)

    # Enforce rate limiting using shared ArxivSearchSource
    source = _get_arxiv_source()
    source.wait_for_rate_limit()

    # Fetch using ArXiv API id_list parameter
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"

    try:
        feed = feedparser.parse(url)
    except Exception as e:
        return {"success": False, "error": f"Failed to fetch paper: {e}"}

    if not feed.entries:
        return {"success": False, "error": f"Paper with ID '{arxiv_id}' not found"}

    entry = feed.entries[0]

    # Extract paper details
    paper = {
        "arxiv_id": arxiv_id,
        "title": entry.get("title", "").replace("\n", " ").strip(),
        "authors": [author.get("name", "") for author in entry.get("authors", [])],
        "abstract": entry.get("summary", "").replace("\n", " ").strip(),
        "url": entry.get("link", ""),
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        "published_date": entry.get("published", ""),
        "updated_date": entry.get("updated", ""),
        "categories": [tag.get("term", "") for tag in entry.get("tags", [])],
        "primary_category": entry.get("arxiv_primary_category", {}).get("term", ""),
    }

    result = {"success": True, "paper": paper}

    # Optionally download PDF and extract text
    if download == "pdf":
        pdf_result = await _download_arxiv_pdf(arxiv_id, source)
        if pdf_result.get("error"):
            result["download_error"] = pdf_result["error"]
        else:
            pdf_text = pdf_result["pdf_text"]
            truncated = len(pdf_text) > PDF_TEXT_MAX_CHARS
            if truncated:
                pdf_text = pdf_text[:PDF_TEXT_MAX_CHARS]
            result["pdf_text"] = pdf_text
            result["pdf_text_truncated"] = truncated
            result["pdf_size_bytes"] = pdf_result["pdf_size_bytes"]

    return result


async def _download_arxiv_pdf(
    arxiv_id: str,
    source,
) -> dict[str, Any]:
    """Download a PDF from ArXiv and extract text.

    Args:
        arxiv_id: Cleaned arXiv paper ID.
        source: ArxivSearchSource instance for rate limiting.

    Returns:
        Dict with pdf_text, pdf_bytes (base64), pdf_size_bytes, or error.
    """
    try:
        import httpx
    except ImportError:
        return {"error": "httpx not installed, cannot download PDFs"}

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    # Rate limit the PDF download
    source.wait_for_rate_limit()

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            response = await client.get(pdf_url)
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code} downloading PDF"}
    except httpx.RequestError as e:
        return {"error": f"Failed to download PDF: {e}"}

    pdf_bytes = response.content
    pdf_size = len(pdf_bytes)

    # Extract text from PDF
    pdf_text = ""
    try:
        import pypdf
        import io

        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        pdf_text = "\n\n".join(pages)
    except ImportError:
        return {"error": "pypdf not installed, cannot extract PDF text"}
    except Exception as e:
        return {"error": f"Failed to extract PDF text: {e}"}

    # Base64-encode the PDF bytes for transport
    import base64
    pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")

    return {
        "pdf_text": pdf_text,
        "pdf_bytes": pdf_b64,
        "pdf_size_bytes": pdf_size,
    }


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    permission_level=PermissionLevel.SAFE,
    description="Analyze an arXiv paper's abstract page using LLM-powered content extraction. Use this when you need deeper analysis beyond metadata (e.g., key contributions, methodology).",
)
async def analyze_arxiv_paper(arxiv_id: str, prompt: str) -> dict[str, Any]:
    """Analyze an arXiv paper using LLM-powered content extraction.

    Fetches the arXiv abstract page and uses an LLM to analyze it
    based on the provided prompt.

    Args:
        arxiv_id: The arXiv paper ID (e.g., '1706.03762')
        prompt: The analysis prompt (e.g., 'What is the main contribution?')

    Returns:
        Dictionary with analysis results or error information
    """
    from agentic_cli.tools.webfetch_tool import web_fetch

    # Clean the arxiv_id
    arxiv_id = _clean_arxiv_id(arxiv_id)

    # Build the abstract URL
    url = f"https://arxiv.org/abs/{arxiv_id}"

    # Enforce rate limiting using shared ArxivSearchSource
    source = _get_arxiv_source()
    source.wait_for_rate_limit()

    # Use web_fetch to get LLM analysis
    result = await web_fetch(url, prompt)

    if not result.get("success"):
        return {
            "success": False,
            "arxiv_id": arxiv_id,
            "error": result.get("error", "Failed to analyze paper"),
        }

    return {
        "success": True,
        "arxiv_id": arxiv_id,
        "url": url,
        "prompt": prompt,
        "analysis": result.get("summary", ""),
    }
