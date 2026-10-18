"""Knowledge base tools for agentic workflows.

- kb_ingest_text / kb_ingest_file: Ingest content into KB.
  Two separate tools so each declares the right capability for the
  permissions engine — text-only (kb.write), and a local file or a page
  web_fetch saved (kb.write + filesystem.read).
- kb_search: Semantic search across all documents
- kb_read: Read a stored document (sidecar by default, full text with full=True)
- kb_list: List documents with summaries
- kb_write_concept / kb_search_concepts: Concept pages

Each tool is an adapter over ``agentic_cli.memory.kb.KnowledgeBase``:
arguments in, one call, a ``{"success": ...}`` dict out. It comes in two
flavors that share one implementation: the ``@register_tool`` module
functions find the knowledge base in the service registry
(``tools._knowledge_base.current_knowledge_base``), and the closures in
``tools.factories.make_kb_tools`` capture it.
"""

from typing import Any

import structlog

logger = structlog.get_logger(__name__)

from agentic_cli.config import get_settings
from agentic_cli.paths import resolve_path
from agentic_cli.tools.registry import (
    register_tool,
    ToolCategory,
)
from agentic_cli.tools.webfetch.saved import (
    is_saved_page_metadata,
    read_saved_page_meta,
    saved_pages_dir,
)
from agentic_cli.workflow.permissions import Capability
from agentic_cli.tools._knowledge_base import NO_KNOWLEDGE_BASE, current_knowledge_base


# Max chars of extracted text to return via kb_read (full=True).
READ_DOCUMENT_MAX_CHARS = 30_000


# ---------------------------------------------------------------------------
# Pure helpers (no service registry dependency)
# ---------------------------------------------------------------------------


def _build_document_item(d, scope: str) -> dict[str, Any]:
    """Build a document summary dict from a Document object.

    Args:
        d: Document instance.
        scope: "project" or "user".

    Returns:
        Dict with document metadata suitable for kb_list output.
    """
    item: dict[str, Any] = {
        "id": d.id,
        "title": d.title,
        "summary": d.summary,
        "source_type": d.source_type.value,
        "created_at": d.created_at.isoformat(),
        "chunks": len(d.chunks),
        "scope": scope,
    }
    if d.metadata.get("authors"):
        item["authors"] = d.metadata["authors"]
    if d.metadata.get("arxiv_id"):
        item["arxiv_id"] = d.metadata["arxiv_id"]
    if d.metadata.get("tags"):
        item["tags"] = d.metadata["tags"]
    if d.file_path:
        item["has_file"] = True
    return item


# ---------------------------------------------------------------------------
# Shared implementations — take KB managers as explicit args
# ---------------------------------------------------------------------------


def _search(kb, query: str, filters: str = "", top_k: int = 10) -> dict[str, Any]:
    """Shared implementation for kb_search."""
    import json as _json

    parsed_filters = None
    if filters:
        try:
            parsed_filters = _json.loads(filters)
        except _json.JSONDecodeError:
            return {"success": False, "error": f"Invalid JSON in filters: {filters}"}

    try:
        result = kb.search(query, filters=parsed_filters, top_k=top_k)
    except Exception as e:
        return {"success": False, "error": f"Search failed: {e}"}
    return {"success": True, **result}


_NO_CONTENT = (
    "No content or file provided. "
    "Pass non-empty content (kb_ingest_text) or a readable file "
    "(kb_ingest_file); for a web page, read it with web_fetch and "
    "pass its saved_path to kb_ingest_file. "
    "For ArXiv papers, use ingest_arxiv_paper instead."
)


def _source_type(value: str):
    """``value`` as a SourceType, and an error dict when it is not one."""
    from agentic_cli.memory.kb import SourceType

    try:
        return SourceType(value), None
    except ValueError:
        valid = ", ".join(t.value for t in SourceType)
        return None, {"success": False, "error": f"Invalid source_type: {value!r}. Valid: {valid}"}


def _caller_metadata(authors: list[str] | None, abstract: str, tags: list[str] | None) -> dict[str, Any]:
    """The tool's ``authors``, ``abstract`` and ``tags``, the ones given."""
    meta: dict[str, Any] = {}
    if authors:
        meta["authors"] = authors
    if abstract:
        meta["abstract"] = abstract
    if tags:
        meta["tags"] = tags
    return meta


async def _ingested(ingest, display_name: str) -> dict[str, Any]:
    """Await one pipeline call and turn its result, or its error, into the tool's dict."""
    from agentic_cli.memory.kb import ConversionFailed, EmptyDocument, UnsupportedDocument

    try:
        found = await ingest
    except EmptyDocument:
        return {"success": False, "error": _NO_CONTENT}
    except UnsupportedDocument as e:
        return {"success": False, "error": f"Cannot ingest {display_name}: {e}"}
    except ConversionFailed as e:
        cause = type(e.__cause__).__name__
        logger.warning("kb_convert_failed", file=display_name, error=cause, exc_info=True)
        return {"success": False, "error": f"Cannot ingest {display_name}: could not convert it ({cause})"}
    except Exception as e:
        return {"success": False, "error": f"Ingestion failed: {e}"}
    doc = found.document
    return {
        "success": True,
        "document_id": doc.id,
        "title": doc.title,
        "chunks_created": len(doc.chunks),
        "summary": doc.summary,
    }


async def _ingest_text(
    kb,
    content: str,
    title: str = "",
    source_type: str = "user",
    source_url: str | None = None,
    authors: list[str] | None = None,
    abstract: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Ingest a plain text document — no filesystem or network access."""
    if not content:
        return {"success": False, "error": _NO_CONTENT}
    st, error = _source_type(source_type)
    if error:
        return error
    return await _ingested(
        kb.ingest_text(
            content, title=title, source_type=st, source_url=source_url,
            metadata=_caller_metadata(authors, abstract, tags),
        ),
        display_name="",
    )


async def _ingest_file(
    kb,
    path: str,
    title: str = "",
    source_type: str = "",
    source_url: str | None = None,
    authors: list[str] | None = None,
    abstract: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Ingest a local file, or a page ``web_fetch`` saved. Caller's
    permissions engine has already gated ``filesystem.read`` for ``path``."""
    if not path:
        return {"success": False, "error": "path is required"}

    try:
        source_path = resolve_path(path)
        if not source_path.is_file():
            return {"success": False, "error": f"File not found: {path}"}
        file_bytes = source_path.read_bytes()
    except ValueError:
        return {"success": False, "error": f"Not a valid path: {path!r}"}
    except OSError as e:
        return {"success": False, "error": f"Cannot read file {path}: {e.strerror or e}"}

    try:
        folder = saved_pages_dir(get_settings().app_name)
    except OSError:
        # saved_pages_dir reads Path.cwd(); a deleted working directory
        # must not stop an absolute-path ingest. Treat the file as local.
        folder = None

    saved = None
    if folder is not None:
        if is_saved_page_metadata(source_path, folder):
            return {
                "success": False,
                "error": f"{path} is a saved page's metadata; pass the page itself",
            }
        saved = read_saved_page_meta(source_path, folder)

    st = None
    if source_type:
        st, error = _source_type(source_type)
        if error:
            return error

    result = await _ingested(
        kb.ingest_bytes(
            file_bytes,
            extension=source_path.suffix.lower() or ".bin",
            content_type=saved.content_type if saved else None,
            charset=(saved.charset or "utf-8") if saved else None,
            fetched_url=saved.url if saved else None,
            fetched_at=saved.fetched_at.isoformat() if saved else None,
            title=title,
            default_title=source_path.stem,
            source_type=st,
            source_url=source_url or (None if saved else str(source_path)),
            metadata=_caller_metadata(authors, abstract, tags),
        ),
        display_name=path,
    )
    if saved is not None and saved.truncated and result.get("success"):
        result["truncated"] = True
    return result


def _read_scope(kb, found) -> str:
    """The scope ``kb_read`` reports.

    0.6.3 said "user" whenever the document came from the user scope's
    manager, also when both names point at one directory (where kb_search
    and kb_list say "project"). Kept until the knowledge-base redesign.
    """
    try:
        user = kb.manager("user")
    except KeyError:
        return found.scope
    return "user" if kb.manager(found.scope) is user else found.scope


async def _read(
    kb, doc_id_or_title: str, full: bool = False, max_chars: int = READ_DOCUMENT_MAX_CHARS
) -> dict[str, Any]:
    """Shared implementation for kb_read: the sidecar by default, the text with ``full``."""
    from agentic_cli.memory.kb import DocumentNotFound

    found = kb.find_document(doc_id_or_title)
    if found is None:
        return {"success": False, "error": f"Document not found: {doc_id_or_title}"}
    doc = found.document
    if full:
        content, truncated = kb.read_text(found, max_chars=max_chars)
        return {
            "success": True,
            "full": True,
            "document_id": doc.id,
            "title": doc.title,
            "scope": _read_scope(kb, found),
            "content": content,
            "truncated": truncated,
            "source_type": doc.source_type.value,
        }
    try:
        sidecar = await kb.sidecar(found)
    except DocumentNotFound:
        return {"success": False, "error": f"Document not found: {doc_id_or_title}"}
    return {
        "success": True,
        "full": False,
        "document_id": doc.id,
        "title": doc.title,
        "source_type": doc.source_type.value,
        "scope": _read_scope(kb, found),
        "summary": doc.summary,
        "sidecar": sidecar,
    }


def _list(kb, query: str = "", source_type: str = "", limit: int = 20) -> dict[str, Any]:
    """Shared implementation for kb_list."""
    from agentic_cli.memory.kb import SourceType as ST

    st_filter = None
    if source_type:
        try:
            st_filter = ST(source_type)
        except ValueError:
            valid = ", ".join(t.value for t in ST)
            return {
                "success": False,
                "error": f"Invalid source_type: {source_type!r}. Valid: {valid}",
            }

    listed = kb.list_documents(title_or_author=query, source_type=st_filter, limit=limit)
    items = [_build_document_item(s.document, s.scope) for s in listed]
    return {"success": True, "documents": items, "count": len(items)}


async def _write_concept(kb, title: str, body: str, sources: list[str], slug: str = "") -> dict[str, Any]:
    """Shared implementation for kb_write_concept (the first scope's concept pages)."""
    try:
        return kb.write_concept(title, body, sources, slug=slug)
    except Exception as e:
        return {"success": False, "error": f"write_concept failed: {e}"}


async def _search_concepts(kb, query: str, limit: int = 10) -> dict[str, Any]:
    """Shared implementation for kb_search_concepts."""
    try:
        hits = kb.search_concepts(query, limit=limit)
    except Exception as e:
        return {"success": False, "error": f"search_concepts failed: {e}"}
    return {"success": True, "concepts": hits, "count": len(hits)}


# ---------------------------------------------------------------------------
# Module-level @register_tool wrappers
# ---------------------------------------------------------------------------


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    capabilities=[Capability("kb.read")],
    description="Search the local knowledge base for relevant documents using semantic similarity. Use this when you need to find previously ingested papers, notes, or documents.",
    requires="knowledge_base",
)
def kb_search(
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
    kb = current_knowledge_base()
    if kb is None:
        return {"success": False, "error": NO_KNOWLEDGE_BASE}
    return _search(kb, query, filters, top_k)


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    capabilities=[Capability("kb.write")],
    description=(
        "Ingest text content into the knowledge base. Use this for content "
        "you already have in memory; use kb_ingest_file for local files and "
        "for pages web_fetch saved."
    ),
    requires="knowledge_base",
)
async def kb_ingest_text(
    content: str,
    title: str = "",
    source_type: str = "user",
    source_url: str | None = None,
    authors: list[str] | None = None,
    abstract: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Ingest text content into the knowledge base.

    Args:
        content: Document text content (required, non-empty).
        title: Document title.
        source_type: Source type (ssrn, web, internal, user, local).
        source_url: Optional URL of the source.
        authors: Optional list of author names.
        abstract: Optional paper abstract.
        tags: Optional tags for categorization.
    """
    kb = current_knowledge_base()
    if kb is None:
        return {"success": False, "error": NO_KNOWLEDGE_BASE}
    return await _ingest_text(
        kb, content=content, title=title, source_type=source_type, source_url=source_url,
        authors=authors, abstract=abstract, tags=tags,
    )


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    capabilities=[
        Capability("kb.write"),
        Capability("filesystem.read", target_arg="path"),
    ],
    description=(
        "Ingest a local file, or a page web_fetch saved (its saved_path), into "
        "the knowledge base: a PDF's text, an HTML page's main text, or a UTF-8 "
        "text file as it is. Other binary files are refused. Triggers a "
        "filesystem.read permission check for the supplied path."
    ),
    requires="knowledge_base",
)
async def kb_ingest_file(
    path: str,
    title: str = "",
    source_type: str = "",
    source_url: str | None = None,
    authors: list[str] | None = None,
    abstract: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Ingest a local file, or a page web_fetch saved, into the knowledge base.

    A PDF's text is extracted; an HTML page goes in as its main text; any
    other UTF-8 text file (Markdown, plain text, JSON, source code) goes in
    as it is. Other binary files are refused. For a page web_fetch saved
    (pass its ``saved_path``), the page's URL, title, authors and description
    fill in what you leave empty.

    Args:
        path: Path to the file, or a ``saved_path`` from web_fetch.
        title: Document title (defaults to the page title, else the file name).
        source_type: Source type (defaults to ``web`` for a saved page,
            ``local`` otherwise).
        source_url: URL of the source (defaults to the page URL for a saved
            page, the file path otherwise).
        authors: Author names (default: the page's authors, if any).
        abstract: Abstract (default: the page's description, if any).
        tags: Optional tags for categorization.
    """
    kb = current_knowledge_base()
    if kb is None:
        return {"success": False, "error": NO_KNOWLEDGE_BASE}
    return await _ingest_file(
        kb, path=path, title=title, source_type=source_type, source_url=source_url,
        authors=authors, abstract=abstract, tags=tags,
    )


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    capabilities=[Capability("kb.read")],
    description=(
        "Read a stored document by ID or title. Returns the per-document "
        "sidecar (summary, key claims, entities) by default. Pass full=True "
        "to get the raw extracted text up to max_chars."
    ),
    requires="knowledge_base",
)
async def kb_read(
    doc_id_or_title: str,
    full: bool = False,
    max_chars: int = READ_DOCUMENT_MAX_CHARS,
) -> dict[str, Any]:
    """Read a stored document.

    Args:
        doc_id_or_title: Document ID or title substring.
        full: If True, return raw text. Default returns the sidecar payload.
        max_chars: Maximum characters of raw text to return when full=True.
    """
    kb = current_knowledge_base()
    if kb is None:
        return {"success": False, "error": f"Document not found: {doc_id_or_title}"}
    return await _read(kb, doc_id_or_title, full, max_chars)


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    capabilities=[Capability("kb.read")],
    description="List documents in the knowledge base with summaries. Filter by query or source type. Returns summaries, not full content.",
    requires="knowledge_base",
)
def kb_list(
    query: str = "",
    source_type: str = "",
    limit: int = 20,
) -> dict[str, Any]:
    """List documents with summaries.

    Args:
        query: Optional filter by title substring (case-insensitive).
        source_type: Optional filter by source type.
        limit: Maximum number of documents to return.

    Returns:
        Dictionary with document list.
    """
    kb = current_knowledge_base()
    if kb is None:
        return {"success": False, "error": NO_KNOWLEDGE_BASE}
    return _list(kb, query, source_type, limit)


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    capabilities=[Capability("kb.write")],
    description=(
        "Save an agent-curated concept page summarizing what the KB "
        "knows about a topic. Pages live at concepts/{slug}.md and are "
        "agent-writable, grep-searchable, and human-readable. `sources` "
        "must cite at least one valid document ID from the KB."
    ),
    requires="knowledge_base",
)
async def kb_write_concept(
    title: str,
    body: str,
    sources: list[str],
    slug: str = "",
) -> dict[str, Any]:
    """Create or overwrite a concept page.

    Args:
        title: Human-readable concept title. Used to derive the slug
            when ``slug`` is empty.
        body: Markdown body. Free-form; no required sections.
        sources: Document IDs this concept cites. Must include at least
            one valid ID (verified against the KB).
        slug: Optional explicit slug. If given and already exists, the
            existing concept is overwritten (body replaced, sources
            merged as union). If empty, auto-generated from title and
            collision-suffixed on conflict.
    """
    kb = current_knowledge_base()
    if kb is None:
        return {"success": False, "error": NO_KNOWLEDGE_BASE}
    return await _write_concept(kb, title=title, body=body, sources=sources, slug=slug)


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    capabilities=[Capability("kb.read")],
    description=(
        "Search concept pages (agent-curated synthesis notes). "
        "Case-insensitive substring match; title hits rank above body "
        "hits. Use when asking 'what does the KB know about X?'."
    ),
    requires="knowledge_base",
)
async def kb_search_concepts(
    query: str,
    limit: int = 10,
) -> dict[str, Any]:
    """Search concept pages.

    Args:
        query: Case-insensitive substring.
        limit: Max hits to return.
    """
    kb = current_knowledge_base()
    if kb is None:
        return {"success": False, "error": NO_KNOWLEDGE_BASE}
    return await _search_concepts(kb, query=query, limit=limit)


