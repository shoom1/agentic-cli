"""Standard tools for agentic workflows.

These tools can be used directly by agents without additional configuration.
They use the context settings to get API keys and configuration.

Design Note:
    Tools accept an optional `settings` parameter for explicit dependency injection.
    When not provided, they fall back to get_settings() for backward compatibility
    and to work seamlessly within WorkflowManager's settings context.
"""

from typing import TYPE_CHECKING, Any

from agentic_cli.config import get_settings

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
    *,
    settings: "BaseSettings | None" = None,
) -> dict[str, Any]:
    """Search the web for relevant information.

    Args:
        query: Search query
        max_results: Maximum number of results
        allowed_domains: Only include results from these domains
        blocked_domains: Exclude results from these domains
        settings: Optional settings instance for explicit dependency injection.
                  If not provided, uses the current context settings.

    Returns:
        Dictionary with search results
    """
    from agentic_cli.tools.search import WebSearchClient

    resolved_settings = settings or get_settings()
    client = WebSearchClient(api_key=resolved_settings.serper_api_key)
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

    # Create user input request
    request = UserInputRequest(
        request_id=str(uuid.uuid4()),
        tool_name="ask_clarification",
        prompt=question,
        input_type="choice" if options else "text",
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
