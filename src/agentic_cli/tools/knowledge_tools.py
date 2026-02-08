"""Knowledge base tools for agentic workflows.

Provides tools for searching and ingesting documents into the local
knowledge base using semantic similarity.
"""

from typing import TYPE_CHECKING, Any

from agentic_cli.config import get_settings
from agentic_cli.tools.registry import (
    register_tool,
    ToolCategory,
    PermissionLevel,
)

if TYPE_CHECKING:
    from agentic_cli.knowledge_base import KnowledgeBaseManager


def _get_knowledge_base_manager() -> "KnowledgeBaseManager":
    """Factory function to create a KnowledgeBaseManager instance.

    Eliminates duplication of KB manager instantiation across tools.
    Uses context settings via get_settings().

    Returns:
        Configured KnowledgeBaseManager instance
    """
    from agentic_cli.knowledge_base import KnowledgeBaseManager

    settings = get_settings()
    return KnowledgeBaseManager(
        settings=settings,
        use_mock=settings.knowledge_base_use_mock,
    )


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    permission_level=PermissionLevel.SAFE,
    description="Search the local knowledge base for relevant documents using semantic similarity. Use this when you need to find previously ingested papers, notes, or documents.",
)
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

    kb = _get_knowledge_base_manager()
    return kb.search(query, filters=parsed_filters, top_k=top_k)


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

    kb = _get_knowledge_base_manager()

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
