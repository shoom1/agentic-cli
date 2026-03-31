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

    Args:
        memory_store: A MemoryStore instance.
        embedding_service: Optional EmbeddingService (unused, kept for interface compat).

    Returns:
        [save_memory, search_memory, update_memory, delete_memory]
    """

    def save_memory(
        content: str,
        tags: list[str] | None = None,
        importance: int = 5,
    ) -> dict[str, Any]:
        """Save information to persistent memory.

        Use this to remember important facts, preferences, or learnings
        that should persist across sessions.

        Args:
            content: The content to store.
            tags: Optional tags for categorization.
            importance: Importance rating 1-10 (default 5). Higher = more important.

        Returns:
            A dict with the stored item ID and any similar existing memories.
        """
        result = memory_store.store_with_similarity_check(content, tags=tags, importance=importance)
        return {
            "success": True,
            "item_id": result["item_id"],
            "message": "Saved to persistent memory",
            "similar_existing": result["similar_existing"],
        }

    def search_memory(
        query: str,
        limit: int = 10,
        include_archived: bool = False,
    ) -> dict[str, Any]:
        """Search persistent memory for stored information.

        Args:
            query: The search query.
            limit: Maximum number of results to return.
            include_archived: If True, include archived (soft-deleted) memories.

        Returns:
            A dict with matching memory items.
        """
        results = memory_store.search(query, limit=limit, include_archived=include_archived)
        items = [
            {
                "id": item.id,
                "content": item.content,
                "tags": item.tags,
                "importance": item.importance,
            }
            for item in results
        ]
        return {
            "success": True,
            "query": query,
            "items": items,
            "count": len(items),
        }

    def update_memory(
        item_id: str,
        content: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Update an existing memory item.

        Args:
            item_id: ID of the memory to update.
            content: New content (optional).
            tags: New tags (optional). Pass explicitly to update; omit (None) to leave unchanged.

        Returns:
            A dict indicating success.
        """
        # Don't forward tags to store.update() unless explicitly provided,
        # since the store's sentinel default means "leave unchanged" but None means "clear tags".
        if tags is None:
            updated = memory_store.update(item_id, content=content)
        else:
            updated = memory_store.update(item_id, content=content, tags=tags)
        return {"success": True, "updated": updated}

    def delete_memory(
        item_id: str,
        purge: bool = False,
    ) -> dict[str, Any]:
        """Delete a memory item (soft-delete by default).

        Args:
            item_id: ID of the memory to delete.
            purge: If True, permanently remove. If False, archive.

        Returns:
            A dict indicating success.
        """
        deleted = memory_store.delete(item_id, purge=purge)
        return {"success": True, "deleted": deleted}

    save_memory.__name__ = "save_memory"
    search_memory.__name__ = "search_memory"
    update_memory.__name__ = "update_memory"
    delete_memory.__name__ = "delete_memory"

    return [save_memory, search_memory, update_memory, delete_memory]


# ---------------------------------------------------------------------------
# Knowledge base tools
# ---------------------------------------------------------------------------

def make_kb_tools(kb_manager, user_kb_manager=None) -> list[Callable]:
    """Create KB tools bound to KBManager instances.

    The inner functions contain the KB tool logic directly, using
    ``kb_manager`` and ``user_kb_manager`` from the closure instead
    of looking them up in the service registry.

    Args:
        kb_manager: Project-scoped KnowledgeBaseManager.
        user_kb_manager: Optional user-scoped KnowledgeBaseManager.

    Returns:
        [search_knowledge_base, ingest_document, read_document,
         list_documents, open_document]
    """
    import structlog

    from agentic_cli.tools.knowledge_tools import (
        _build_document_item,
        _extract_arxiv_id,
        _extract_text_from_bytes,
        _detect_extension,
        _ingest_arxiv,
        SAFE_OPEN_EXTENSIONS,
        READ_DOCUMENT_MAX_CHARS,
    )
    from agentic_cli.tools.knowledge_tools import (
        search_knowledge_base as _orig_search,
        ingest_document as _orig_ingest,
        read_document as _orig_read,
        list_documents as _orig_list,
        open_document as _orig_open,
    )

    logger = structlog.get_logger("agentic_cli.tools.knowledge_tools")

    # --- Closure-local helper ---
    def _find_doc(doc_id_or_title: str) -> tuple:
        """Find a document across main and user knowledge bases."""
        doc = kb_manager.find_document(doc_id_or_title)
        source_kb = kb_manager

        if doc is None and user_kb_manager is not None and user_kb_manager is not kb_manager:
            doc = user_kb_manager.find_document(doc_id_or_title)
            if doc is not None:
                source_kb = user_kb_manager

        return doc, source_kb

    # --- search_knowledge_base ---
    def search_knowledge_base(
        query: str,
        filters: str = "",
        top_k: int = 10,
    ) -> dict[str, Any]:
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
                    seen_doc_ids = {r["document_id"] for r in result.get("results", [])}
                    for r in user_result.get("results", []):
                        if r["document_id"] not in seen_doc_ids:
                            r["scope"] = "user"
                            result["results"].append(r)
                            seen_doc_ids.add(r["document_id"])
                    result["results"].sort(key=lambda r: r.get("score", 0), reverse=True)
                    result["results"] = result["results"][:top_k]
                    result["total_matches"] = len(result["results"])
                except Exception:
                    logger.debug("user_kb_search_failed", query=query, exc_info=True)

            return {"success": True, **result}
        except Exception as e:
            return {"success": False, "error": f"Search failed: {e}"}

    search_knowledge_base.__name__ = "search_knowledge_base"
    search_knowledge_base.__doc__ = _orig_search.__doc__

    # --- ingest_document ---
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
        from pathlib import Path
        from agentic_cli.constants import truncate
        from agentic_cli.knowledge_base.models import SourceType

        kb = kb_manager

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
            if "arxiv.org" in url_or_path:
                source_type = "arxiv"
            elif url_or_path.startswith(("http://", "https://")) and source_type == "user":
                source_type = "web"
            elif not url_or_path.startswith(("http://", "https://")) and source_type == "user":
                source_type = "local"

            source_url = source_url or url_or_path

            if source_type == "arxiv":
                result = await _ingest_arxiv(
                    url_or_path, title, authors, abstract, meta, kb
                )
                return result

            elif url_or_path.startswith(("http://", "https://")):
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

                file_extension = _detect_extension(url_or_path)

                if file_extension == ".pdf" and file_bytes:
                    content = _extract_text_from_bytes(file_bytes)

                if not title:
                    title = url_or_path.split("/")[-1] or url_or_path

            else:
                source_path = Path(url_or_path).expanduser().resolve()
                if not source_path.exists():
                    return {"success": False, "error": f"File not found: {url_or_path}"}

                file_bytes = source_path.read_bytes()
                file_extension = source_path.suffix.lower() or ".bin"

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
                    "For ArXiv papers, pass the ArXiv URL as url_or_path."
                ),
            }

        if not title:
            title = truncate(content, 80)

        try:
            source = SourceType(source_type)
        except ValueError:
            valid = ", ".join(t.value for t in SourceType)
            return {"success": False, "error": f"Invalid source_type: {source_type!r}. Valid: {valid}"}

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

    ingest_document.__name__ = "ingest_document"
    ingest_document.__doc__ = _orig_ingest.__doc__

    # --- read_document ---
    def read_document(
        doc_id_or_title: str,
        max_chars: int = READ_DOCUMENT_MAX_CHARS,
    ) -> dict[str, Any]:
        doc, source_kb = _find_doc(doc_id_or_title)

        if doc is None:
            return {"success": False, "error": f"Document not found: {doc_id_or_title}"}

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
            "document_id": doc.id,
            "title": doc.title,
            "content": content,
            "truncated": truncated,
            "source_type": doc.source_type.value,
        }

    read_document.__name__ = "read_document"
    read_document.__doc__ = _orig_read.__doc__

    # --- list_documents ---
    def list_documents(
        query: str = "",
        source_type: str = "",
        limit: int = 20,
    ) -> dict[str, Any]:
        from agentic_cli.knowledge_base.models import SourceType as ST

        kb = kb_manager

        st_filter = None
        if source_type:
            try:
                st_filter = ST(source_type)
            except ValueError:
                pass

        docs = kb.list_documents(source_type=st_filter, limit=limit)

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

        if user_kb_manager is not None and user_kb_manager is not kb:
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

    list_documents.__name__ = "list_documents"
    list_documents.__doc__ = _orig_list.__doc__

    # --- open_document ---
    def open_document(doc_id_or_title: str) -> dict[str, Any]:
        import platform
        import subprocess

        doc, source_kb = _find_doc(doc_id_or_title)

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

    open_document.__name__ = "open_document"
    open_document.__doc__ = _orig_open.__doc__

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
