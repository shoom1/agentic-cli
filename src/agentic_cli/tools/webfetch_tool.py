"""Web fetch tool for fetching and summarizing web content.

Provides the main web_fetch tool that orchestrates content fetching,
markdown conversion, and LLM summarization.
"""

from __future__ import annotations

from typing import Any

from agentic_cli.config import get_settings
from agentic_cli.tools import requires
from agentic_cli.tools.registry import register_tool, ToolCategory, PermissionLevel
from agentic_cli.tools.webfetch import (
    ContentFetcher,
    URLValidator,
    RobotsTxtChecker,
    HTMLToMarkdown,
    build_summarize_prompt,
)
from agentic_cli.workflow.context import get_context_llm_summarizer


# Module-level fetcher instance (lazy-created)
_fetcher: ContentFetcher | None = None


def get_or_create_fetcher(settings=None) -> ContentFetcher:
    """Get or create the module-level ContentFetcher.

    Args:
        settings: Optional settings instance. If not provided,
            uses get_settings() to get current settings.

    Returns:
        ContentFetcher instance configured with current settings.
    """
    global _fetcher

    if _fetcher is not None:
        return _fetcher

    if settings is None:
        settings = get_settings()

    validator = URLValidator(blocked_domains=settings.webfetch_blocked_domains)
    robots_checker = RobotsTxtChecker()

    _fetcher = ContentFetcher(
        validator=validator,
        robots_checker=robots_checker,
        cache_ttl_seconds=settings.webfetch_cache_ttl_seconds,
        max_content_bytes=settings.webfetch_max_content_bytes,
    )

    return _fetcher


@register_tool(
    category=ToolCategory.NETWORK,
    permission_level=PermissionLevel.SAFE,
    description="Fetch web content and summarize it using an LLM",
)
@requires("llm_summarizer")
async def web_fetch(url: str, prompt: str, timeout: int = 30) -> dict[str, Any]:
    """Fetch web content and summarize it using an LLM.

    Fetches content from the specified URL, converts it to markdown,
    and uses an LLM to summarize based on the provided prompt.

    Args:
        url: The URL to fetch content from.
        prompt: The prompt describing what information to extract.
        timeout: Request timeout in seconds (default: 30).

    Returns:
        Dictionary with:
        - success: Whether the operation succeeded
        - summary: The LLM-generated summary (if successful)
        - url: The fetched URL
        - truncated: Whether content was truncated
        - cached: Whether content came from cache
        - error: Error message (if failed)
        - redirect: Redirect info (if cross-host redirect occurred)
    """
    # Get the LLM summarizer from context
    summarizer = get_context_llm_summarizer()
    if summarizer is None:
        return {
            "success": False,
            "error": "No LLM summarizer available in context. "
                     "Ensure the workflow manager has been configured with an LLM summarizer.",
        }

    # Get the fetcher
    fetcher = get_or_create_fetcher()

    # Fetch the content
    fetch_result = await fetcher.fetch(url, timeout=timeout)

    # Handle fetch failure
    if not fetch_result.success:
        # Check for redirect
        if fetch_result.redirect is not None:
            return {
                "success": False,
                "redirect": True,
                "redirect_url": fetch_result.redirect.to_url,
                "redirect_host": fetch_result.redirect.to_host,
                "message": f"Redirect to different host: {fetch_result.redirect.to_host}",
                "url": url,
            }
        # Other fetch error
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
