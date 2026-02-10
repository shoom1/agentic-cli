"""Knowledge base tools for agentic workflows.

Provides tools for searching and ingesting documents into the local
knowledge base using semantic similarity.
"""

from typing import Any

from agentic_cli.tools import requires, require_context
from agentic_cli.tools.registry import (
    register_tool,
    ToolCategory,
    PermissionLevel,
)
from agentic_cli.workflow.context import get_context_kb_manager


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
        return {"success": True, **result}
    except Exception as e:
        return {"success": False, "error": f"Search failed: {e}"}


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    permission_level=PermissionLevel.CAUTION,
    description="Ingest a document into the knowledge base for later semantic search. Use this to store papers, articles, or notes for future reference.",
)
@requires("kb_manager")
@require_context("KB manager", get_context_kb_manager)
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
    from agentic_cli.knowledge_base import SourceType

    try:
        source = SourceType(source_type)
    except ValueError:
        valid = ", ".join(t.value for t in SourceType)
        return {"success": False, "error": f"Invalid source_type: {source_type!r}. Valid: {valid}"}

    try:
        kb = get_context_kb_manager()
        doc = kb.ingest_document(
            content=content,
            title=title,
            source_type=source,
            source_url=source_url,
        )
    except Exception as e:
        return {"success": False, "error": f"Ingestion failed: {e}"}

    return {
        "success": True,
        "document_id": doc.id,
        "title": doc.title,
        "chunks_created": len(doc.chunks),
    }
