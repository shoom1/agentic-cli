"""Web fetch tool: fetch a page, save it, and summarize it with an LLM.

``fetch_and_summarize`` is the body of both ``web_fetch`` variants: the
registered tool here and the one ``tools.factories.make_webfetch_tool`` builds.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import structlog

from agentic_cli.config import get_settings
from agentic_cli.tools.registry import register_tool, ToolCategory
from agentic_cli.workflow.permissions import Capability
from agentic_cli.tools.webfetch import (
    ContentFetcher,
    FetchResult,
    URLValidator,
    RobotsTxtChecker,
    HTMLToMarkdown,
    build_summarize_prompt,
)
from agentic_cli.tools.webfetch.saved import (
    SaveError,
    cleanup_saved_pages,
    page_name,
    save_page,
    saved_page_extension,
    saved_pages_dir,
)
from agentic_cli.tools.webfetch.transport import PinnedTransport
from agentic_cli.workflow.service_registry import require_service, LLM_SUMMARIZER

logger = structlog.get_logger(__name__)


# Module-level fetcher state (lazy-created, invalidated on settings change)
_fetcher: ContentFetcher | None = None
_fetcher_settings_snapshot: tuple | None = None


def _settings_key(settings) -> tuple:
    """Extract settings values relevant to the fetcher for change detection."""
    return (
        tuple(settings.webfetch_blocked_domains),
        settings.webfetch_cache_ttl_seconds,
        settings.webfetch_max_content_bytes,
        settings.webfetch_max_pdf_bytes,
        settings.webfetch_max_download_bytes,
    )


def get_or_create_fetcher(settings=None) -> ContentFetcher:
    """Get or create the module-level ContentFetcher.

    Re-creates the fetcher if relevant settings have changed since
    the last call, preventing stale configuration.

    Args:
        settings: Optional settings instance. If not provided,
            uses get_settings() to get current settings.

    Returns:
        ContentFetcher instance configured with current settings.
    """
    global _fetcher, _fetcher_settings_snapshot

    if settings is None:
        settings = get_settings()

    current_key = _settings_key(settings)

    if _fetcher is not None and _fetcher_settings_snapshot == current_key:
        return _fetcher

    validator = URLValidator(blocked_domains=settings.webfetch_blocked_domains)
    transport = PinnedTransport(validator)
    robots_checker = RobotsTxtChecker(transport=transport)

    _fetcher = ContentFetcher(
        validator=validator,
        robots_checker=robots_checker,
        transport=transport,
        cache_ttl_seconds=settings.webfetch_cache_ttl_seconds,
        max_content_bytes=settings.webfetch_max_content_bytes,
        max_pdf_bytes=settings.webfetch_max_pdf_bytes,
        max_download_bytes=settings.webfetch_max_download_bytes,
    )
    _fetcher_settings_snapshot = current_key

    return _fetcher


def save_fetched_page(url: str, result: FetchResult) -> dict[str, Any]:
    """Save a fetched page in the project's saved-pages folder.

    Returns ``{"saved_path": ...}`` (relative to the current directory),
    ``{"save_error": ...}``, or ``{}`` when the result carries no body or
    pages of its type are not saved.

    A planted file in the saved-pages folder (a cloned repository controls
    it) must never make this raise, so saving, computing the saved path, and
    the cleanup sweep are each guarded. A cleanup failure after a successful
    save is logged but never turns the result into a ``save_error`` — the
    page was already saved.
    """
    if not isinstance(result.raw, bytes) or saved_page_extension(result.content_type) is None:
        return {}
    settings = get_settings()
    folder = saved_pages_dir(settings.app_name)
    try:
        page = save_page(
            folder,
            url=url,
            final_url=result.final_url or url,
            content_type=result.content_type or "",
            charset=result.charset,
            data=result.raw,
            truncated=result.raw_truncated,
        )
        saved_path = os.path.relpath(page, Path.cwd())
    except Exception as e:
        logger.warning("webfetch_page_not_saved", host=urlparse(url).hostname or url, error=str(e))
        return {"save_error": str(e)}
    try:
        cleanup_saved_pages(
            folder,
            max_age_days=settings.webfetch_saved_max_age_days,
            max_bytes=settings.webfetch_saved_max_mb * 1024 * 1024,
            keep=page_name(url),
        )
    except Exception as e:
        # cleanup_saved_pages is already non-raising; this guards the call
        # itself (and any future change to that contract) without turning a
        # cleanup failure into a save_error for a page that did get saved.
        logger.warning("saved_pages_cleanup_failed", folder=str(folder), error=str(e))
    return {"saved_path": saved_path}


async def fetch_and_summarize(url: str, prompt: str, timeout: int, summarizer) -> dict[str, Any]:
    """Fetch ``url``, save the page, and summarize it with ``summarizer``."""
    fetcher = get_or_create_fetcher()
    fetch_result = await fetcher.fetch(url, timeout=timeout)

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

    saved = await asyncio.to_thread(save_fetched_page, url, fetch_result)

    markdown_content = HTMLToMarkdown().convert(
        fetch_result.content,
        fetch_result.content_type or "text/html",
    )
    full_prompt = build_summarize_prompt(markdown_content, prompt)
    try:
        summary = await summarizer.summarize(markdown_content, full_prompt)
    except Exception as e:
        return {
            "success": False,
            "error": f"LLM summarization failed: {e}",
            "url": url,
            **saved,
        }

    return {
        "success": True,
        "summary": summary,
        "url": url,
        "truncated": fetch_result.truncated,
        "cached": fetch_result.from_cache,
        **saved,
    }


@register_tool(
    category=ToolCategory.NETWORK,
    capabilities=[Capability("http.read", target_arg="url")],
    description=(
        "Fetch a web page, convert it to markdown, and summarize it using an LLM "
        "based on your prompt. The full page is saved (saved_path), so "
        "kb_ingest_file can add it to the knowledge base."
    ),
    requires="llm_summarizer",
)
async def web_fetch(url: str, prompt: str, timeout: int = 30) -> dict[str, Any]:
    """Fetch web content and summarize it using an LLM.

    Fetches content from the specified URL, converts it to markdown, and uses
    an LLM to summarize it based on the provided prompt. The full page is
    saved: pass ``saved_path`` to ``kb_ingest_file`` to add it to the
    knowledge base, or read a saved HTML or text page with ``read_file``.

    Args:
        url: The URL to fetch content from.
        prompt: The prompt describing what information to extract.
        timeout: Request timeout in seconds (default: 30).

    Returns:
        Dictionary with:
        - success: Whether the operation succeeded
        - summary: The LLM-generated summary (if successful)
        - url: The fetched URL
        - truncated: Whether the summarized content was truncated
        - cached: Whether content came from cache
        - saved_path: Where the full page was saved (HTML, text, JSON, XML and PDF pages)
        - save_error: Why the page could not be saved (the summary is still returned)
        - error: Error message (if failed)
        - redirect: Redirect info (if cross-host redirect occurred)
    """
    summarizer = require_service(LLM_SUMMARIZER)
    if isinstance(summarizer, dict):
        return summarizer
    return await fetch_and_summarize(url, prompt, timeout, summarizer)
