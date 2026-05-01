"""Content fetching with caching and redirect handling."""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

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
    content: str | bytes | None = None
    content_type: str | None = None
    redirect: RedirectInfo | None = None
    error: str | None = None
    truncated: bool = False
    from_cache: bool = False


@dataclass
class CachedResponse:
    """Cached HTTP response."""
    content: str | bytes
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
        max_pdf_bytes: int = 5242880,
    ) -> None:
        self._validator = validator
        self._robots = robots_checker
        self._cache_ttl = cache_ttl_seconds
        self._max_content_bytes = max_content_bytes
        self._max_pdf_bytes = max_pdf_bytes
        self._cache: dict[str, CachedResponse] = {}

    MAX_REDIRECTS = 5

    async def fetch(self, url: str, timeout: int = 30) -> FetchResult:
        """Fetch content from a URL.

        Redirects are followed manually so each Location is revalidated by
        the SSRF guard before the next request is issued — using
        ``follow_redirects=True`` would let httpx contact intermediate hosts
        (e.g., 169.254.169.254) before we get a chance to inspect them.

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

        # Validate the originally requested URL
        validation = self._validator.validate(url)
        if not validation.valid:
            return FetchResult(success=False, error=validation.error)

        # Check robots.txt for the originally requested URL
        if not await self._robots.can_fetch(url):
            return FetchResult(success=False, error=f"Blocked by robots.txt for {url}")

        original_url = url
        current_url = url
        history: list[str] = []
        response: httpx.Response | None = None

        try:
            async with httpx.AsyncClient(follow_redirects=False) as client:
                for _ in range(self.MAX_REDIRECTS + 1):
                    response = await client.get(current_url, timeout=timeout)

                    # Post-fetch IP revalidation (mitigates DNS rebinding)
                    final_host = urlparse(str(response.url)).hostname
                    if final_host:
                        try:
                            post_ip = socket.gethostbyname(final_host)
                            ip_check = self._validator.validate_ip(post_ip)
                            if not ip_check.valid:
                                return FetchResult(
                                    success=False,
                                    error=f"DNS rebinding detected: {ip_check.error}",
                                )
                        except socket.gaierror:
                            pass

                    if response.status_code not in (301, 302, 303, 307, 308):
                        break

                    location = response.headers.get("location")
                    if not location:
                        break

                    next_url = urljoin(str(response.url), location)
                    next_validation = self._validator.validate(next_url)
                    if not next_validation.valid:
                        return FetchResult(
                            success=False,
                            error=(
                                f"Redirect to disallowed URL blocked: "
                                f"{next_validation.error}"
                            ),
                        )

                    # Cross-host redirects must be reported BEFORE issuing
                    # the next GET — the user's permission grant covers the
                    # original host only, so contacting another origin is a
                    # side effect they did not approve.
                    next_host = urlparse(next_url).netloc
                    original_host = urlparse(original_url).netloc
                    if next_host.lower() != original_host.lower():
                        return FetchResult(
                            success=False,
                            redirect=RedirectInfo(
                                from_url=original_url,
                                to_url=next_url,
                                to_host=next_host,
                            ),
                            error=f"Cross-host redirect to {next_host}",
                        )

                    # Same-host redirect: the new path may itself be
                    # disallowed by robots.txt even though the original
                    # path was not.
                    if not await self._robots.can_fetch(next_url):
                        return FetchResult(
                            success=False,
                            error=f"Blocked by robots.txt for {next_url}",
                        )

                    history.append(current_url)
                    current_url = next_url
                else:
                    return FetchResult(
                        success=False,
                        error=f"Too many redirects (max {self.MAX_REDIRECTS})",
                    )

        except httpx.TimeoutException:
            return FetchResult(success=False, error=f"Request timeout after {timeout}s")
        except httpx.RequestError as e:
            return FetchResult(success=False, error=f"Request failed: {e}")

        assert response is not None  # loop runs at least once

        content_type = response.headers.get("content-type", "text/html")

        # Use bytes for PDF, text for everything else
        is_pdf = "application/pdf" in content_type.lower()
        if is_pdf:
            content: str | bytes = response.content
            truncated = False
            if len(content) > self._max_pdf_bytes:
                content = content[:self._max_pdf_bytes]
                truncated = True
        else:
            content = response.text
            truncated = False
            if len(content) > self._max_content_bytes:
                content = content[:self._max_content_bytes]
                content += f"\n\n[Content truncated at {self._max_content_bytes} bytes]"
                truncated = True

        # Cache the response under the originally requested URL
        self._cache[original_url] = CachedResponse(
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

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()
