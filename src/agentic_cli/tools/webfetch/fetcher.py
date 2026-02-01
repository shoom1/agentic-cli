"""Content fetching with caching and redirect handling."""

from __future__ import annotations

import time
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from agentic_cli.tools.webfetch.validator import URLValidator
from agentic_cli.tools.webfetch.robots import RobotsTxtChecker


@dataclass
class RedirectInfo:
    """Information about a cross-host redirect."""
    from_url: str
    to_url: str
    to_host: str


@dataclass
class FetchResult:
    """Result of a content fetch operation."""
    success: bool
    content: str | None = None
    content_type: str | None = None
    redirect: RedirectInfo | None = None
    error: str | None = None
    truncated: bool = False
    from_cache: bool = False


@dataclass
class CachedResponse:
    """Cached HTTP response."""
    content: str
    content_type: str
    timestamp: float
    truncated: bool = False


class ContentFetcher:
    """Fetches web content with validation, caching, and redirect handling."""

    def __init__(
        self,
        validator: URLValidator,
        robots_checker: RobotsTxtChecker,
        cache_ttl_seconds: int = 900,
        max_content_bytes: int = 102400,
    ) -> None:
        self._validator = validator
        self._robots = robots_checker
        self._cache_ttl = cache_ttl_seconds
        self._max_content_bytes = max_content_bytes
        self._cache: dict[str, CachedResponse] = {}

    async def fetch(self, url: str, timeout: int = 30) -> FetchResult:
        """Fetch content from a URL.

        Args:
            url: The URL to fetch.
            timeout: Request timeout in seconds.

        Returns:
            FetchResult with content or error information.
        """
        # Check cache first
        cached = self._get_cached(url)
        if cached is not None:
            return FetchResult(
                success=True,
                content=cached.content,
                content_type=cached.content_type,
                truncated=cached.truncated,
                from_cache=True,
            )

        # Validate URL
        validation = self._validator.validate(url)
        if not validation.valid:
            return FetchResult(success=False, error=validation.error)

        # Check robots.txt
        if not await self._robots.can_fetch(url):
            return FetchResult(success=False, error=f"Blocked by robots.txt for {url}")

        # Fetch content
        try:
            async with httpx.AsyncClient(follow_redirects=True, max_redirects=5) as client:
                response = await client.get(url, timeout=timeout)

                # Check for cross-host redirect
                redirect_info = self._check_cross_host_redirect(url, response)
                if redirect_info is not None:
                    return FetchResult(
                        success=False,
                        redirect=redirect_info,
                        error=f"Cross-host redirect to {redirect_info.to_host}",
                    )

                content = response.text
                content_type = response.headers.get("content-type", "text/html")

                # Truncate if needed
                truncated = False
                if len(content) > self._max_content_bytes:
                    content = content[:self._max_content_bytes]
                    content += f"\n\n[Content truncated at {self._max_content_bytes} bytes]"
                    truncated = True

                # Cache the response
                self._cache[url] = CachedResponse(
                    content=content,
                    content_type=content_type,
                    timestamp=time.time(),
                    truncated=truncated,
                )

                return FetchResult(
                    success=True,
                    content=content,
                    content_type=content_type,
                    truncated=truncated,
                    from_cache=False,
                )

        except httpx.TimeoutException:
            return FetchResult(success=False, error=f"Request timeout after {timeout}s")
        except httpx.RequestError as e:
            return FetchResult(success=False, error=f"Request failed: {e}")

    def _get_cached(self, url: str) -> CachedResponse | None:
        """Get cached response if available and not expired.

        Args:
            url: The URL to look up.

        Returns:
            CachedResponse if found and valid, None otherwise.
        """
        if url not in self._cache:
            return None
        cached = self._cache[url]
        if time.time() - cached.timestamp > self._cache_ttl:
            del self._cache[url]
            return None
        return cached

    def _check_cross_host_redirect(
        self, original_url: str, response: httpx.Response
    ) -> RedirectInfo | None:
        """Check if response involved a cross-host redirect.

        Args:
            original_url: The originally requested URL.
            response: The final HTTP response.

        Returns:
            RedirectInfo if cross-host redirect occurred, None otherwise.
        """
        if not response.history:
            return None
        original_host = urlparse(original_url).netloc
        final_host = urlparse(str(response.url)).netloc
        if original_host.lower() != final_host.lower():
            return RedirectInfo(
                from_url=original_url,
                to_url=str(response.url),
                to_host=final_host,
            )
        return None

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()
