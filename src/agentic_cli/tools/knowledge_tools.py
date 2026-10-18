"""Knowledge base tools for agentic workflows.

- kb_ingest_text / kb_ingest_file / kb_ingest_url: Ingest content into KB.
  Three separate tools so each declares the right capability for the
  permissions engine — text-only (kb.write), local file or a page web_fetch
  saved (kb.write + filesystem.read), and URL (kb.write + http.read; deprecated
  until 0.7.0, use web_fetch then kb_ingest_file).
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

import asyncio
import warnings
from datetime import datetime, timezone
from typing import Any
from urllib.parse import unquote, urlparse

import structlog

logger = structlog.get_logger(__name__)

from agentic_cli.config import get_settings
from agentic_cli.constants import truncate
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


def _writes_to(kb):
    """The manager of the scope that takes writes (ingestion moves to the
    knowledge base itself in a later change)."""
    return kb.manager(kb.scopes[0])


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


def _build_meta(
    authors: list[str] | None,
    abstract: str,
    tags: list[str] | None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if authors:
        meta["authors"] = authors
    if abstract:
        meta["abstract"] = abstract
    if tags:
        meta["tags"] = tags
    return meta


async def _finalize_ingest(
    kb_manager,
    *,
    content: str,
    title: str,
    source_type: str,
    source_url: str | None,
    meta: dict[str, Any],
    file_bytes: bytes | None,
    file_extension: str,
) -> dict[str, Any]:
    """Validate, generate sidecar, and persist into the supplied ``kb_manager``.

    Shared tail used by all three ingest entry points (text/file/url).
    """
    from agentic_cli.memory.kb import SourceType

    if not content and not file_bytes:
        return {
            "success": False,
            "error": (
                "No content or file provided. "
                "Pass non-empty content (kb_ingest_text) or a readable file "
                "(kb_ingest_file); for a web page, read it with web_fetch and "
                "pass its saved_path to kb_ingest_file. "
                "For ArXiv papers, use ingest_arxiv_paper instead."
            ),
        }

    if not title:
        title = truncate(content, 80)

    try:
        source = SourceType(source_type)
    except ValueError:
        valid = ", ".join(t.value for t in SourceType)
        return {"success": False, "error": f"Invalid source_type: {source_type!r}. Valid: {valid}"}

    payload = await kb_manager.generate_sidecar_payload(content, title=title)

    try:
        doc = kb_manager.ingest_document(
            content=content,
            title=title,
            source_type=source,
            source_url=source_url,
            metadata=meta or None,
            file_bytes=file_bytes,
            file_extension=file_extension,
            summary=payload["summary"] or None,
            sidecar_payload=payload,
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


def _url_title(url: str) -> str:
    """A title from a URL: its last path segment (percent-decoded), else its host."""
    parsed = urlparse(url)
    segment = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    return unquote(segment) if segment else (parsed.netloc or url)


async def _ingest_bytes_with_kb(
    kb_manager,
    data: bytes,
    *,
    extension: str,
    content_type: str | None,
    charset: str | None,
    page_url: str | None,
    fetched_at: str | None,
    truncated: bool,
    fallback_title: str,
    fallback_source_url: str,
    display_name: str,
    title: str,
    source_type: str,
    source_url: str | None,
    authors: list[str] | None,
    abstract: str,
    tags: list[str] | None,
) -> dict[str, Any]:
    """Convert ``data`` and ingest it.

    Shared by ``kb_ingest_file`` and ``kb_ingest_url``. ``page_url`` marks a
    web page (a saved page or a fetched URL): it defaults to source type
    ``web``, the page URL as source, and the page's title, else the URL's
    last segment. ``charset`` None means a local file, which must be UTF-8
    unless it is a PDF. Arguments the model passed always win.
    """
    from agentic_cli.memory.kb import UnsupportedDocument, convert_document

    try:
        converted = await asyncio.to_thread(
            convert_document,
            data, content_type=content_type, extension=extension, charset=charset, url=page_url,
        )
    except UnsupportedDocument as e:
        return {"success": False, "error": f"Cannot ingest {display_name}: {e}"}
    except Exception as e:  # a future converter may raise something else; never escape the tool
        logger.warning(
            "kb_convert_failed", file=display_name, error=type(e).__name__, exc_info=True
        )
        return {
            "success": False,
            "error": f"Cannot ingest {display_name}: could not convert it ({type(e).__name__})",
        }

    meta = _build_meta(authors or list(converted.authors), abstract or converted.description or "", tags)
    meta["file_size_bytes"] = len(data)
    if converted.published:
        meta["published"] = converted.published
    if fetched_at:
        meta["fetched_at"] = fetched_at

    if page_url:
        default_title = converted.title or _url_title(page_url)
        default_type, default_source = "web", page_url
    else:
        default_title = converted.title or fallback_title
        default_type, default_source = "local", fallback_source_url

    result = await _finalize_ingest(
        kb_manager,
        content=converted.text,
        title=title or default_title,
        source_type=source_type or default_type,
        source_url=source_url or default_source,
        meta=meta,
        file_bytes=data,
        file_extension=extension,
    )
    if truncated and result.get("success"):
        result["truncated"] = True
    return result


async def _ingest_text_with_kb(
    kb_manager,
    content: str,
    title: str = "",
    source_type: str = "user",
    source_url: str | None = None,
    authors: list[str] | None = None,
    abstract: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Ingest a plain text document — no filesystem or network access."""
    return await _finalize_ingest(
        kb_manager,
        content=content,
        title=title,
        source_type=source_type,
        source_url=source_url,
        meta=_build_meta(authors, abstract, tags),
        file_bytes=None,
        file_extension=".txt",
    )


async def _ingest_file_with_kb(
    kb_manager,
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

    return await _ingest_bytes_with_kb(
        kb_manager,
        file_bytes,
        extension=source_path.suffix.lower() or ".bin",
        content_type=saved.content_type if saved else None,
        charset=(saved.charset or "utf-8") if saved else None,
        page_url=saved.url if saved else None,
        fetched_at=saved.fetched_at.isoformat() if saved else None,
        truncated=saved.truncated if saved else False,
        fallback_title=source_path.stem,
        fallback_source_url=str(source_path),
        display_name=path,
        title=title,
        source_type=source_type,
        source_url=source_url,
        authors=authors,
        abstract=abstract,
        tags=tags,
    )


async def _ingest_url_with_kb(
    kb_manager,
    url: str,
    title: str = "",
    source_type: str = "",
    source_url: str | None = None,
    authors: list[str] | None = None,
    abstract: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Fetch a URL, save the page, and ingest it (deprecated until 0.7.0).

    Routes the download through ``ContentFetcher`` (URL validation, redirect
    revalidation, DNS pinning); the caller's permissions engine has already
    gated ``http.read`` for ``url``. The page is saved as ``web_fetch`` saves
    it and converted like ``kb_ingest_file`` converts a saved page.
    """
    warnings.warn(
        "kb_ingest_url is deprecated and will be removed in 0.7.0; "
        "use web_fetch, then kb_ingest_file(saved_path)",
        DeprecationWarning,
        stacklevel=3,
    )
    logger.warning("kb_ingest_url_deprecated", host=urlparse(url).hostname)
    if not url or not url.startswith(("http://", "https://")):
        return {
            "success": False,
            "error": "url must be an http:// or https:// URL",
        }

    from agentic_cli.tools.webfetch.saved import saved_page_extension
    from agentic_cli.tools.webfetch_tool import get_or_create_fetcher, save_fetched_page

    fetcher = get_or_create_fetcher()
    fetch_result = await fetcher.fetch(url, timeout=60)
    if not fetch_result.success:
        if fetch_result.redirect is not None:
            return {
                "success": False,
                "error": (
                    f"Cross-host redirect blocked: "
                    f"{fetch_result.redirect.from_url} -> "
                    f"{fetch_result.redirect.to_url}"
                ),
                "redirect": True,
                "redirect_url": fetch_result.redirect.to_url,
                "redirect_host": fetch_result.redirect.to_host,
            }
        return {"success": False, "error": fetch_result.error or "fetch failed"}

    content_type = fetch_result.content_type or "text/html"
    extension = saved_page_extension(content_type)
    if extension is None:
        return {
            "success": False,
            "error": f"Cannot ingest {url}: {content_type} is not a type the knowledge base can ingest",
        }

    if isinstance(fetch_result.raw, bytes):
        data = fetch_result.raw
    elif isinstance(fetch_result.content, bytes):
        data = fetch_result.content
    else:
        data = (fetch_result.content or "").encode("utf-8")

    # Off the event loop, same as web_fetch's own save.
    saved = await asyncio.to_thread(save_fetched_page, url, fetch_result)
    result = await _ingest_bytes_with_kb(
        kb_manager,
        data,
        extension=extension,
        content_type=content_type,
        charset=fetch_result.charset or "utf-8",
        page_url=url,
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        truncated=fetch_result.raw_truncated,
        fallback_title=url,
        fallback_source_url=url,
        display_name=url,
        title=title,
        source_type=source_type,
        source_url=source_url,
        authors=authors,
        abstract=abstract,
        tags=tags,
    )
    return {**result, **saved}


def _read_scope(kb, found) -> str:
    """The scope ``kb_read`` reports.

    0.6.3 said "user" whenever the document came from the user scope's
    manager, also when both names point at one directory (where kb_search
    and kb_list say "project"). Kept until 0.7.0.
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
    requires="kb_manager",
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
    requires="kb_manager",
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
    return await _ingest_text_with_kb(
        _writes_to(kb),
        content=content,
        title=title,
        source_type=source_type,
        source_url=source_url,
        authors=authors,
        abstract=abstract,
        tags=tags,
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
    requires="kb_manager",
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
    return await _ingest_file_with_kb(
        _writes_to(kb),
        path=path,
        title=title,
        source_type=source_type,
        source_url=source_url,
        authors=authors,
        abstract=abstract,
        tags=tags,
    )


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    capabilities=[
        Capability("kb.write"),
        Capability("http.read", target_arg="url"),
    ],
    description=(
        "Deprecated, removed in 0.7.0: use web_fetch, then "
        "kb_ingest_file(saved_path). Fetches an http(s) URL through the "
        "hardened web fetcher and ingests it. Triggers an http.read "
        "permission check for the supplied URL."
    ),
    requires="kb_manager",
)
async def kb_ingest_url(
    url: str,
    title: str = "",
    source_type: str = "",
    source_url: str | None = None,
    authors: list[str] | None = None,
    abstract: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Deprecated, removed in 0.7.0: use web_fetch, then kb_ingest_file(saved_path).

    Fetches a URL and ingests it into the knowledge base.

    Args:
        url: An ``http://`` or ``https://`` URL.
        title: Document title (defaults to the page title, else the URL's last segment).
        source_type: Source type (defaults to ``web``).
        source_url: Optional URL of the source (defaults to ``url``).
        authors: Optional list of author names.
        abstract: Optional paper abstract.
        tags: Optional tags for categorization.
    """
    kb = current_knowledge_base()
    if kb is None:
        return {"success": False, "error": NO_KNOWLEDGE_BASE}
    return await _ingest_url_with_kb(
        _writes_to(kb),
        url=url,
        title=title,
        source_type=source_type,
        source_url=source_url,
        authors=authors,
        abstract=abstract,
        tags=tags,
    )


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    capabilities=[Capability("kb.read")],
    description=(
        "Read a stored document by ID or title. Returns the per-document "
        "sidecar (summary, key claims, entities) by default. Pass full=True "
        "to get the raw extracted text up to max_chars."
    ),
    requires="kb_manager",
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
    requires="kb_manager",
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
    requires="kb_manager",
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
    requires="kb_manager",
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


