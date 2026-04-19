"""Service tool factories — create closure-bound tool functions.

Each factory produces tool functions with the service dependency captured
in a closure, eliminating the need for ContextVar lookups at call time.

The inner functions contain the tool logic directly (no delegation to
legacy modules), so the service registry is never in the dependency chain.

Usage:
    from agentic_cli.tools.factories import make_memory_tools

    memory_store = MemoryStore(settings)
    save_memory, search_memory = make_memory_tools(memory_store)
"""

from __future__ import annotations

from typing import Any, Callable


# ---------------------------------------------------------------------------
# Memory tools
# ---------------------------------------------------------------------------

def make_memory_tools(memory_store, embedding_service=None) -> list[Callable]:
    """Create memory tools bound to a MemoryStore.

    Thin closure-bound wrappers around the shared helpers in
    ``tools.memory_tools``. The helpers take the store as an explicit arg
    so the closure-bound and module-level (registry-bound) tool versions
    stay in lockstep.

    Args:
        memory_store: A MemoryStore instance.
        embedding_service: Optional EmbeddingService (unused, kept for interface compat).

    Returns:
        [save_memory, search_memory, update_memory, delete_memory]
    """
    from agentic_cli.tools.memory_tools import (
        _SENTINEL,
        _delete_memory_with_store,
        _save_memory_with_store,
        _search_memory_with_store,
        _update_memory_with_store,
        save_memory as _orig_save,
        search_memory as _orig_search,
        update_memory as _orig_update,
        delete_memory as _orig_delete,
    )

    def save_memory(
        content: str,
        tags: list[str] | None = None,
        importance: int = 5,
    ) -> dict[str, Any]:
        return _save_memory_with_store(memory_store, content, tags, importance)

    def search_memory(
        query: str,
        limit: int = 10,
        include_archived: bool = False,
    ) -> dict[str, Any]:
        return _search_memory_with_store(
            memory_store, query, limit, include_archived
        )

    def update_memory(
        item_id: str,
        content: str | None = None,
        tags: list[str] | None = _SENTINEL,
    ) -> dict[str, Any]:
        return _update_memory_with_store(memory_store, item_id, content, tags)

    def delete_memory(
        item_id: str,
        purge: bool = False,
    ) -> dict[str, Any]:
        return _delete_memory_with_store(memory_store, item_id, purge)

    save_memory.__name__ = "save_memory"
    save_memory.__doc__ = _orig_save.__doc__
    search_memory.__name__ = "search_memory"
    search_memory.__doc__ = _orig_search.__doc__
    update_memory.__name__ = "update_memory"
    update_memory.__doc__ = _orig_update.__doc__
    delete_memory.__name__ = "delete_memory"
    delete_memory.__doc__ = _orig_delete.__doc__

    return [save_memory, search_memory, update_memory, delete_memory]


# ---------------------------------------------------------------------------
# Knowledge base tools
# ---------------------------------------------------------------------------

def make_kb_tools(kb_manager, user_kb_manager=None) -> list[Callable]:
    """Create KB tools bound to KBManager instances.

    Thin closure-bound wrappers around the shared helpers in
    ``tools.knowledge_tools``. The helpers take the KB managers as
    explicit args so the closure-bound and module-level (registry-bound)
    tool versions stay in lockstep.

    Args:
        kb_manager: Project-scoped KnowledgeBaseManager.
        user_kb_manager: Optional user-scoped KnowledgeBaseManager.

    Returns:
        [kb_search, kb_ingest, kb_read, kb_list, kb_write_concept, kb_search_concepts]
        (six tools — the concept pages pair was added in Phase 2.)
    """
    from agentic_cli.tools.knowledge_tools import (
        READ_DOCUMENT_MAX_CHARS,
        _ingest_document_with_kb,
        _list_documents_in_kbs,
        _read_document_from_kbs,
        _search_concepts_with_kb,
        _search_kbs,
        _write_concept_with_kb,
        kb_ingest as _orig_ingest,
        kb_list as _orig_list,
        kb_read as _orig_read,
        kb_search as _orig_search,
        kb_search_concepts as _orig_search_concepts,
        kb_write_concept as _orig_write_concept,
    )

    def kb_search(
        query: str,
        filters: str = "",
        top_k: int = 10,
    ) -> dict[str, Any]:
        return _search_kbs(kb_manager, user_kb_manager, query, filters, top_k)

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
        return await _ingest_document_with_kb(
            kb_manager,
            content=content,
            url_or_path=url_or_path,
            title=title,
            source_type=source_type,
            source_url=source_url,
            authors=authors,
            abstract=abstract,
            tags=tags,
        )

    async def kb_read(
        doc_id_or_title: str,
        full: bool = False,
        max_chars: int = READ_DOCUMENT_MAX_CHARS,
    ) -> dict[str, Any]:
        return await _read_document_from_kbs(
            kb_manager, user_kb_manager, doc_id_or_title, full, max_chars
        )

    def kb_list(
        query: str = "",
        source_type: str = "",
        limit: int = 20,
    ) -> dict[str, Any]:
        return _list_documents_in_kbs(
            kb_manager, user_kb_manager, query, source_type, limit
        )

    async def kb_write_concept(
        title: str,
        body: str,
        sources: list[str],
        slug: str = "",
    ) -> dict[str, Any]:
        return await _write_concept_with_kb(
            kb_manager, user_kb_manager,
            title=title, body=body, sources=sources, slug=slug,
        )

    async def kb_search_concepts(
        query: str,
        limit: int = 10,
    ) -> dict[str, Any]:
        return await _search_concepts_with_kb(
            kb_manager, user_kb_manager, query=query, limit=limit,
        )

    kb_search.__name__ = "kb_search"
    kb_search.__doc__ = _orig_search.__doc__
    kb_ingest.__name__ = "kb_ingest"
    kb_ingest.__doc__ = _orig_ingest.__doc__
    kb_read.__name__ = "kb_read"
    kb_read.__doc__ = _orig_read.__doc__
    kb_list.__name__ = "kb_list"
    kb_list.__doc__ = _orig_list.__doc__
    kb_write_concept.__name__ = "kb_write_concept"
    kb_write_concept.__doc__ = _orig_write_concept.__doc__
    kb_search_concepts.__name__ = "kb_search_concepts"
    kb_search_concepts.__doc__ = _orig_search_concepts.__doc__

    return [
        kb_search,
        kb_ingest,
        kb_read,
        kb_list,
        kb_write_concept,
        kb_search_concepts,
    ]


# ---------------------------------------------------------------------------
# Web fetch tool
# ---------------------------------------------------------------------------

def make_webfetch_tool(summarizer) -> Callable:
    """Create web_fetch bound to an LLM summarizer.

    Args:
        summarizer: Object with an async ``summarize(content, prompt)`` method.

    Returns:
        Async web_fetch function.
    """
    from agentic_cli.tools.webfetch_tool import get_or_create_fetcher
    from agentic_cli.tools.webfetch import HTMLToMarkdown, build_summarize_prompt

    async def web_fetch(url: str, prompt: str, timeout: int = 30) -> dict[str, Any]:
        """Fetch web content and summarize it using an LLM.

        Args:
            url: The URL to fetch content from.
            prompt: The prompt describing what information to extract.
            timeout: Request timeout in seconds (default: 30).

        Returns:
            Dictionary with success, summary, url, truncated, cached keys.
        """
        fetcher = get_or_create_fetcher()

        # Fetch the content
        fetch_result = await fetcher.fetch(url, timeout=timeout)

        # Handle fetch failure
        if not fetch_result.success:
            if fetch_result.redirect is not None:
                return {
                    "success": False,
                    "redirect": True,
                    "redirect_url": fetch_result.redirect.to_url,
                    "redirect_host": fetch_result.redirect.to_host,
                    "message": f"Redirect to different host: {fetch_result.redirect.to_host}",
                    "url": url,
                }
            return {
                "success": False,
                "error": fetch_result.error,
                "url": url,
            }

        # Convert HTML to markdown
        converter = HTMLToMarkdown()
        markdown_content = converter.convert(
            fetch_result.content,
            fetch_result.content_type or "text/html",
        )

        # Build the summarization prompt
        full_prompt = build_summarize_prompt(markdown_content, prompt)

        # Summarize using the LLM
        try:
            summary = await summarizer.summarize(markdown_content, full_prompt)
        except Exception as e:
            return {
                "success": False,
                "error": f"LLM summarization failed: {e}",
                "url": url,
            }

        return {
            "success": True,
            "summary": summary,
            "url": url,
            "truncated": fetch_result.truncated,
            "cached": fetch_result.from_cache,
        }

    web_fetch.__name__ = "web_fetch"
    return web_fetch


# ---------------------------------------------------------------------------
# Sandbox tool
# ---------------------------------------------------------------------------

def make_sandbox_tool(sandbox_manager) -> Callable:
    """Create sandbox_execute bound to a SandboxManager.

    Args:
        sandbox_manager: SandboxManager instance.

    Returns:
        sandbox_execute function.
    """

    def sandbox_execute(
        code: str,
        session_id: str = "default",
        timeout_seconds: int = 120,
    ) -> dict[str, Any]:
        """Execute Python code in a stateful sandbox.

        Args:
            code: Python code to execute.
            session_id: Session identifier for state persistence.
            timeout_seconds: Maximum execution time in seconds.

        Returns:
            Dictionary with execution results.
        """
        result = sandbox_manager.execute(
            code=code,
            session_id=session_id,
            timeout_seconds=timeout_seconds,
        )
        return {
            "success": result.success,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "result": result.result,
            "artifacts": result.artifacts,
            "execution_time": round(result.execution_time, 3),
            "error": result.error,
        }

    sandbox_execute.__name__ = "sandbox_execute"
    return sandbox_execute


# ---------------------------------------------------------------------------
# ArXiv tools
# ---------------------------------------------------------------------------

def make_arxiv_tools(arxiv_source) -> list[Callable]:
    """Create arXiv tools bound to an ArxivSearchSource.

    Args:
        arxiv_source: An ArxivSearchSource instance (shared rate limiter + cache).

    Returns:
        [search_arxiv, fetch_arxiv_paper]
    """
    from agentic_cli.tools.arxiv_tools import (
        _validate_sort_options,
        _search_arxiv_with_source,
        _fetch_arxiv_paper_with_source,
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
        return _search_arxiv_with_source(
            arxiv_source,
            query=query,
            max_results=max_results,
            categories=categories,
            sort_by=sort_by,
            sort_order=sort_order,
            date_from=date_from,
            date_to=date_to,
        )

    async def fetch_arxiv_paper(arxiv_id: str) -> dict[str, Any]:
        """Fetch detailed information about a specific arXiv paper.

        Args:
            arxiv_id: The arXiv paper ID, e.g. '1706.03762' or '1706.03762v2'.
                      Also accepts full arXiv URLs.

        Returns:
            Dictionary with paper metadata or error information.
        """
        return await _fetch_arxiv_paper_with_source(arxiv_source, arxiv_id)

    search_arxiv.__name__ = "search_arxiv"
    fetch_arxiv_paper.__name__ = "fetch_arxiv_paper"
    return [search_arxiv, fetch_arxiv_paper]


def make_ingest_arxiv_tool(arxiv_source, kb_manager) -> Callable:
    """Create ingest_arxiv_paper bound to an arxiv source and KB manager.

    The composed tool downloads a paper's PDF via the arxiv layer (with
    shared rate limiter and id cache) and stores it in the KB layer.

    Args:
        arxiv_source: An ArxivSearchSource instance.
        kb_manager: A KnowledgeBaseManager instance.

    Returns:
        ingest_arxiv_paper async function with both services captured
        in the closure.
    """
    from agentic_cli.tools.arxiv_tools import _ingest_arxiv_paper_with_services

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
            pdf_downloaded flag (False if PDF fetch failed and the
            abstract was used as fallback content).
        """
        return await _ingest_arxiv_paper_with_services(
            arxiv_source, kb_manager, arxiv_id, tags=tags
        )

    ingest_arxiv_paper.__name__ = "ingest_arxiv_paper"
    return ingest_arxiv_paper


# ---------------------------------------------------------------------------
# Interaction tools
# ---------------------------------------------------------------------------

def make_interaction_tools(workflow_manager) -> list[Callable]:
    """Create interaction tools bound to a workflow manager.

    Args:
        workflow_manager: BaseWorkflowManager instance.

    Returns:
        [ask_clarification]
    """

    async def ask_clarification(
        question: str,
        options: list[str] | None = None,
    ) -> dict[str, Any]:
        """Ask the user for clarification.

        Args:
            question: The question to ask.
            options: Optional list of suggested answers.

        Returns:
            Dictionary with the user's response.
        """
        import uuid
        from agentic_cli.workflow.events import UserInputRequest, InputType

        if workflow_manager is None:
            return {
                "success": False,
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
            input_type=InputType.CHOICE if options else InputType.TEXT,
            choices=options,
        )

        # Request user input (this will block until CLI provides response)
        response = await workflow_manager.request_user_input(request)

        return {
            "success": True,
            "question": question,
            "options": options or [],
            "response": response,
            "summary": f"User responded: {response[:50]}{'...' if len(response) > 50 else ''}",
        }

    ask_clarification.__name__ = "ask_clarification"
    return [ask_clarification]
