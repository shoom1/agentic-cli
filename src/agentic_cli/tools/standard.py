"""Standard tools for agentic workflows.

These tools can be used directly by agents without additional configuration.
They use the global settings to get API keys and configuration.
"""

from typing import Any

from agentic_cli.config import get_settings


def search_knowledge_base(
    query: str,
    filters: dict | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Search the knowledge base for relevant information.

    Args:
        query: Natural language search query
        filters: Optional filters (source_type, date_from, date_to)
        top_k: Maximum number of results

    Returns:
        Dictionary with search results and timing information
    """
    from agentic_cli.knowledge_base import KnowledgeBaseManager

    settings = get_settings()
    kb = KnowledgeBaseManager(
        settings=settings,
        use_mock=settings.knowledge_base_use_mock,
    )
    return kb.search(query, filters=filters, top_k=top_k)


def ingest_to_knowledge_base(
    content: str,
    title: str,
    source_type: str = "user",
    source_url: str | None = None,
) -> dict[str, Any]:
    """Ingest a document into the knowledge base.

    Args:
        content: Document content to ingest
        title: Document title
        source_type: Type of source (arxiv, web, user, internal)
        source_url: Optional URL of the source

    Returns:
        Dictionary with ingestion result
    """
    from agentic_cli.knowledge_base import KnowledgeBaseManager, SourceType

    settings = get_settings()
    kb = KnowledgeBaseManager(
        settings=settings,
        use_mock=settings.knowledge_base_use_mock,
    )

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


def search_arxiv(
    query: str,
    max_results: int = 10,
    categories: list[str] | None = None,
) -> dict[str, Any]:
    """Search arXiv for academic papers.

    Args:
        query: Search query for arXiv papers
        max_results: Maximum number of results to return
        categories: Optional list of arXiv categories to filter (e.g., ['cs.LG', 'cs.CV'])

    Returns:
        Dictionary with search results and metadata
    """
    try:
        import feedparser
    except ImportError:
        return {"papers": [], "total_found": 0, "error": "feedparser not installed"}

    # Build arXiv API URL
    base_url = "http://export.arxiv.org/api/query?"
    search_query = f"search_query=all:{query}"

    if categories:
        cat_query = " OR ".join(f"cat:{cat}" for cat in categories)
        search_query = f"search_query=(all:{query}) AND ({cat_query})"

    url = f"{base_url}{search_query}&start=0&max_results={max_results}"

    feed = feedparser.parse(url)

    papers = []
    for entry in feed.entries:
        entry_categories = [tag.get("term", "") for tag in entry.get("tags", [])]

        paper = {
            "title": entry.get("title", "").replace("\n", " "),
            "authors": [author.get("name", "") for author in entry.get("authors", [])],
            "abstract": entry.get("summary", "").replace("\n", " "),
            "url": entry.get("link", ""),
            "published_date": entry.get("published", ""),
            "categories": entry_categories,
            "arxiv_id": entry.get("id", "").split("/abs/")[-1],
        }
        papers.append(paper)

    return {
        "papers": papers,
        "total_found": len(papers),
        "query": query,
    }


def web_search(
    query: str,
    max_results: int = 10,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
) -> dict[str, Any]:
    """Search the web for relevant information.

    Args:
        query: Search query
        max_results: Maximum number of results
        allowed_domains: Only include results from these domains
        blocked_domains: Exclude results from these domains

    Returns:
        Dictionary with search results
    """
    from agentic_cli.tools.search import WebSearchClient

    settings = get_settings()
    client = WebSearchClient(api_key=settings.serper_api_key)
    return client.search(
        query,
        max_results=max_results,
        allowed_domains=allowed_domains,
        blocked_domains=blocked_domains,
    )


def execute_python(
    code: str,
    context: dict | None = None,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Execute Python code safely.

    Args:
        code: Python code to execute
        context: Optional variables to inject
        timeout_seconds: Maximum execution time

    Returns:
        Dictionary with execution results
    """
    from agentic_cli.tools.executor import SafePythonExecutor

    settings = get_settings()
    executor = SafePythonExecutor(default_timeout=settings.python_executor_timeout)
    return executor.execute(code, context=context, timeout_seconds=timeout_seconds)


def ask_clarification(
    question: str,
    options: list[str] | None = None,
) -> dict[str, Any]:
    """Ask the user for clarification.

    Note: This is a placeholder that returns a message indicating user input is needed.
    The actual user interaction is handled by the CLI layer.

    Args:
        question: The question to ask
        options: Optional list of suggested answers

    Returns:
        Dictionary with the question (actual response comes from CLI)
    """
    return {
        "question": question,
        "options": options or [],
        "response": f"[User clarification needed: {question}]",
    }
