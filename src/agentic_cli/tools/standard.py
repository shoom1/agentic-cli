"""Standard tools for agentic workflows.

These tools can be used directly by agents without additional configuration.
They use the context settings to get API keys and configuration.

Design Note:
    Tools accept an optional `settings` parameter for explicit dependency injection.
    When not provided, they fall back to get_settings() for backward compatibility
    and to work seamlessly within workflow manager's settings context.
"""

from typing import TYPE_CHECKING, Any

from agentic_cli.config import get_settings
from agentic_cli.tools.registry import (
    register_tool,
    ToolCategory,
    PermissionLevel,
)

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings
    from agentic_cli.knowledge_base import KnowledgeBaseManager


def _get_knowledge_base_manager(
    settings: "BaseSettings | None" = None,
) -> "KnowledgeBaseManager":
    """Factory function to create a KnowledgeBaseManager instance.

    Eliminates duplication of KB manager instantiation across tools.

    Args:
        settings: Optional settings instance. Uses get_settings() if not provided.

    Returns:
        Configured KnowledgeBaseManager instance
    """
    from agentic_cli.knowledge_base import KnowledgeBaseManager

    resolved_settings = settings or get_settings()
    return KnowledgeBaseManager(
        settings=resolved_settings,
        use_mock=resolved_settings.knowledge_base_use_mock,
    )


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    permission_level=PermissionLevel.SAFE,
    description="Search the local knowledge base for relevant documents using semantic similarity. Use this when you need to find previously ingested papers, notes, or documents.",
)
def search_knowledge_base(
    query: str,
    filters: dict | None = None,
    top_k: int = 10,
    *,
    settings: "BaseSettings | None" = None,
) -> dict[str, Any]:
    """Search the knowledge base for relevant information.

    Args:
        query: Natural language search query
        filters: Optional filters (source_type, date_from, date_to)
        top_k: Maximum number of results
        settings: Optional settings instance for explicit dependency injection.
                  If not provided, uses the current context settings.

    Returns:
        Dictionary with search results and timing information
    """
    kb = _get_knowledge_base_manager(settings)
    return kb.search(query, filters=filters, top_k=top_k)


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    permission_level=PermissionLevel.CAUTION,
    description="Ingest a document into the knowledge base for later semantic search. Use this to store papers, articles, or notes for future reference.",
)
def ingest_to_knowledge_base(
    content: str,
    title: str,
    source_type: str = "user",
    source_url: str | None = None,
    *,
    settings: "BaseSettings | None" = None,
) -> dict[str, Any]:
    """Ingest a document into the knowledge base.

    Args:
        content: Document content to ingest
        title: Document title
        source_type: Type of source (arxiv, web, user, internal)
        source_url: Optional URL of the source
        settings: Optional settings instance for explicit dependency injection.
                  If not provided, uses the current context settings.

    Returns:
        Dictionary with ingestion result
    """
    from agentic_cli.knowledge_base import SourceType

    kb = _get_knowledge_base_manager(settings)

    source = SourceType(source_type)
    doc = kb.ingest_document(
        content=content,
        title=title,
        source_type=source,
        source_url=source_url,
    )

    return {
        "success": True,
        "document_id": doc.id,
        "title": doc.title,
        "chunks_created": len(doc.chunks),
    }


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
        "papers": papers,
        "total_found": len(papers),
        "query": query,
    }


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    permission_level=PermissionLevel.SAFE,
    description="Fetch metadata for a specific arXiv paper by ID or URL (title, authors, abstract, categories). Use this when you have a paper ID and need details.",
)
def fetch_arxiv_paper(arxiv_id: str) -> dict[str, Any]:
    """Fetch detailed information about a specific arXiv paper.

    Args:
        arxiv_id: The arXiv paper ID (e.g., '1706.03762', '1706.03762v2',
                  or full URL 'https://arxiv.org/abs/1706.03762')

    Returns:
        Dictionary with paper details or error information
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

    return {"success": True, "paper": paper}


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


@register_tool(
    category=ToolCategory.EXECUTION,
    permission_level=PermissionLevel.CAUTION,
    description="Execute Python code in a sandboxed environment with restricted imports. Use this for calculations, data processing, or prototyping. Only whitelisted modules (math, numpy, pandas, json, etc.) are available.",
)
def execute_python(
    code: str,
    context: dict | None = None,
    timeout_seconds: int | None = None,
    *,
    settings: "BaseSettings | None" = None,
) -> dict[str, Any]:
    """Execute Python code safely.

    Args:
        code: Python code to execute
        context: Optional variables to inject
        timeout_seconds: Maximum execution time
        settings: Optional settings instance for explicit dependency injection.
                  If not provided, uses the current context settings.

    Returns:
        Dictionary with execution results
    """
    from agentic_cli.tools.executor import SafePythonExecutor

    resolved_settings = settings or get_settings()
    executor = SafePythonExecutor(default_timeout=resolved_settings.python_executor_timeout)
    return executor.execute(code, context=context, timeout_seconds=timeout_seconds)


@register_tool(
    category=ToolCategory.INTERACTION,
    permission_level=PermissionLevel.SAFE,
    description="Ask the user a clarifying question and wait for their response. Use this when requirements are ambiguous or you need user input to proceed.",
)
async def ask_clarification(
    question: str,
    options: list[str] | None = None,
) -> dict[str, Any]:
    """Ask the user for clarification.

    This tool pauses execution and requests input from the user via
    the CLI. It uses the workflow context to emit a USER_INPUT_REQUIRED
    event that the CLI will handle.

    Args:
        question: The question to ask
        options: Optional list of suggested answers (shown as choices)

    Returns:
        Dictionary with the user's response
    """
    import uuid
    from agentic_cli.config import get_context_workflow
    from agentic_cli.workflow.events import UserInputRequest

    workflow = get_context_workflow()

    if workflow is None:
        # Fallback for when not running within a workflow context
        return {
            "question": question,
            "options": options or [],
            "error": "No workflow context available for user interaction",
            "response": None,
        }

    from agentic_cli.workflow.events import InputType

    # Create user input request
    request = UserInputRequest(
        request_id=str(uuid.uuid4()),
        tool_name="ask_clarification",
        prompt=question,
        input_type=InputType.CHOICE if options else InputType.TEXT,
        choices=options,
    )

    # Request user input (this will block until CLI provides response)
    response = await workflow.request_user_input(request)

    return {
        "question": question,
        "options": options or [],
        "response": response,
        "summary": f"User responded: {response[:50]}{'...' if len(response) > 50 else ''}",
    }
