"""Service tool factories — create closure-bound tool functions.

Each factory produces tool functions with the service dependency captured
in a closure, eliminating the need for ContextVar lookups at call time.

The inner functions preserve ``__name__`` and ``__doc__`` to match the
originals so that LLM tool schemas remain identical.

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

def make_memory_tools(memory_store) -> list[Callable]:
    """Create memory tools bound to a MemoryStore.

    Args:
        memory_store: A MemoryStore instance.

    Returns:
        [save_memory, search_memory]
    """

    def save_memory(
        content: str,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Save information to persistent memory.

        Use this to remember important facts, preferences, or learnings
        that should persist across sessions.

        Args:
            content: The content to store.
            tags: Optional tags for categorization.

        Returns:
            A dict with the stored item ID.
        """
        item_id = memory_store.store(content, tags=tags)
        return {
            "success": True,
            "item_id": item_id,
            "message": "Saved to persistent memory",
        }

    def search_memory(
        query: str,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Search persistent memory for stored information.

        Args:
            query: The search query (substring match, case-insensitive).
            limit: Maximum number of results to return.

        Returns:
            A dict with matching memory items.
        """
        results = memory_store.search(query, limit=limit)
        items = [
            {
                "id": item.id,
                "content": item.content,
                "tags": item.tags,
            }
            for item in results
        ]
        return {
            "success": True,
            "query": query,
            "items": items,
            "count": len(items),
        }

    save_memory.__name__ = "save_memory"
    search_memory.__name__ = "search_memory"

    return [save_memory, search_memory]


# ---------------------------------------------------------------------------
# Knowledge base tools
# ---------------------------------------------------------------------------

def make_kb_tools(kb_manager, user_kb_manager=None) -> list[Callable]:
    """Create KB tools bound to KBManager instances.

    The factories delegate to the original module functions after ensuring
    the service registry has the correct managers set. This avoids
    duplicating the complex KB logic (dual-KB merging, ArXiv detection,
    etc.).

    Args:
        kb_manager: Project-scoped KnowledgeBaseManager.
        user_kb_manager: Optional user-scoped KnowledgeBaseManager.

    Returns:
        [search_knowledge_base, ingest_document, read_document,
         list_documents, open_document]
    """
    from agentic_cli.tools import knowledge_tools as kt
    from agentic_cli.workflow.service_registry import (
        get_service_registry,
        KB_MANAGER,
        USER_KB_MANAGER,
    )

    def _ensure_services():
        registry = get_service_registry()
        registry[KB_MANAGER] = kb_manager
        if user_kb_manager is not None:
            registry[USER_KB_MANAGER] = user_kb_manager

    def search_knowledge_base(
        query: str,
        filters: str = "",
        top_k: int = 10,
    ) -> dict[str, Any]:
        _ensure_services()
        return kt.search_knowledge_base(query, filters=filters, top_k=top_k)

    search_knowledge_base.__name__ = "search_knowledge_base"
    search_knowledge_base.__doc__ = kt.search_knowledge_base.__doc__

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
        _ensure_services()
        return await kt.ingest_document(
            content=content,
            url_or_path=url_or_path,
            title=title,
            source_type=source_type,
            source_url=source_url,
            authors=authors,
            abstract=abstract,
            tags=tags,
        )

    ingest_document.__name__ = "ingest_document"
    ingest_document.__doc__ = kt.ingest_document.__doc__

    def read_document(
        doc_id_or_title: str,
        max_chars: int = kt.READ_DOCUMENT_MAX_CHARS,
    ) -> dict[str, Any]:
        _ensure_services()
        return kt.read_document(doc_id_or_title, max_chars=max_chars)

    read_document.__name__ = "read_document"
    read_document.__doc__ = kt.read_document.__doc__

    def list_documents(
        query: str = "",
        source_type: str = "",
        limit: int = 20,
    ) -> dict[str, Any]:
        _ensure_services()
        return kt.list_documents(query=query, source_type=source_type, limit=limit)

    list_documents.__name__ = "list_documents"
    list_documents.__doc__ = kt.list_documents.__doc__

    def open_document(doc_id_or_title: str) -> dict[str, Any]:
        _ensure_services()
        return kt.open_document(doc_id_or_title)

    open_document.__name__ = "open_document"
    open_document.__doc__ = kt.open_document.__doc__

    return [search_knowledge_base, ingest_document, read_document, list_documents, open_document]


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
    from agentic_cli.workflow.service_registry import (
        get_service_registry,
        LLM_SUMMARIZER,
    )

    async def web_fetch(url: str, prompt: str, timeout: int = 30) -> dict[str, Any]:
        """Fetch web content and summarize it using an LLM.

        Args:
            url: The URL to fetch content from.
            prompt: The prompt describing what information to extract.
            timeout: Request timeout in seconds (default: 30).

        Returns:
            Dictionary with success, summary, url, truncated, cached keys.
        """
        # Ensure summarizer is in registry for require_service() inside the tool
        registry = get_service_registry()
        registry[LLM_SUMMARIZER] = summarizer

        from agentic_cli.tools.webfetch_tool import web_fetch as _original_web_fetch
        return await _original_web_fetch(url=url, prompt=prompt, timeout=timeout)

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
    from agentic_cli.workflow.service_registry import (
        get_service_registry,
        SANDBOX_MANAGER,
    )

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
        registry = get_service_registry()
        registry[SANDBOX_MANAGER] = sandbox_manager

        from agentic_cli.tools.sandbox import sandbox_execute as _original
        return _original(code=code, session_id=session_id, timeout_seconds=timeout_seconds)

    sandbox_execute.__name__ = "sandbox_execute"
    return sandbox_execute


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
    from agentic_cli.workflow.service_registry import (
        get_service_registry,
        WORKFLOW,
    )

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
        registry = get_service_registry()
        registry[WORKFLOW] = workflow_manager

        from agentic_cli.tools.interaction_tools import (
            ask_clarification as _original,
        )
        return await _original(question=question, options=options)

    ask_clarification.__name__ = "ask_clarification"
    return [ask_clarification]
