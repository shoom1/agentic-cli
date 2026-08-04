"""Content fetching with caching and redirect handling."""

from __future__ import annotations

import time
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx

from agentic_cli.tools.webfetch.validator import URLValidator, BlockedAddressError
from agentic_cli.tools.webfetch.robots import RobotsTxtChecker
from agentic_cli.tools.webfetch.transport import PinnedTransport


@dataclass
class RedirectInfo:
    from_url: str
    to_url: str
    to_host: str


@dataclass
class FetchResult:
    success: bool
    content: str | bytes | None = None
    content_type: str | None = None
    redirect: RedirectInfo | None = None
    error: str | None = None
    truncated: bool = False
    from_cache: bool = False


@dataclass
class CachedResponse:
    content: str | bytes
    content_type: str
    timestamp: float
    truncated: bool = False


class ContentFetcher:
    """Fetches web content with SSRF-safe pinning, caching, and redirect handling."""

    MAX_REDIRECTS = 5

    def __init__(
        self,
        validator: URLValidator,
        robots_checker: RobotsTxtChecker,
        transport: PinnedTransport,
        cache_ttl_seconds: int = 900,
        max_content_bytes: int = 102400,
        max_pdf_bytes: int = 5242880,
    ) -> None:
        self._validator = validator
        self._robots = robots_checker
        self._transport = transport
        self._cache_ttl = cache_ttl_seconds
        self._max_content_bytes = max_content_bytes
        self._max_pdf_bytes = max_pdf_bytes
        self._cache: dict[str, CachedResponse] = {}

    async def fetch(self, url: str, timeout: int = 30) -> FetchResult:
        """Fetch content from a URL.

        Redirects are followed manually so each Location is revalidated before
        the next request; the PinnedTransport resolves+validates+pins every hop
        (and the robots fetch), connecting only to globally-routable IPs.
        """
        cached = self._get_cached(url)
        if cached is not None:
            return FetchResult(
                success=True, content=cached.content, content_type=cached.content_type,
                truncated=cached.truncated, from_cache=True,
            )

        validation = self._validator.validate(url)
        if not validation.valid:
            return FetchResult(success=False, error=validation.error)

        if not await self._robots.can_fetch(url):
            return FetchResult(success=False, error=f"Blocked by robots.txt for {url}")

        original_url = url
        current_url = url

        try:
            async with httpx.AsyncClient(transport=self._transport, follow_redirects=False) as client:
                for _ in range(self.MAX_REDIRECTS + 1):
                    async with client.stream("GET", current_url, timeout=timeout) as response:
                        status = response.status_code
                        headers = response.headers
                        content_type = headers.get("content-type", "text/html")
                        is_pdf = "application/pdf" in content_type.lower()
                        cap = self._max_pdf_bytes if is_pdf else self._max_content_bytes
                        encoding = response.charset_encoding or "utf-8"
                        buf = bytearray()
                        truncated = False
                        async for chunk in response.aiter_bytes():
                            buf.extend(chunk)
                            if len(buf) > cap:
                                del buf[cap:]
                                truncated = True
                                break

                    if status in (301, 302, 303, 307, 308):
                        location = headers.get("location")
                        if location:
                            next_url = urljoin(current_url, location)
                            next_validation = self._validator.validate(next_url)
                            if not next_validation.valid:
                                return FetchResult(
                                    success=False,
                                    error=f"Redirect to disallowed URL blocked: {next_validation.error}",
                                )
                            next_host = urlparse(next_url).netloc
                            original_host = urlparse(original_url).netloc
                            if next_host.lower() != original_host.lower():
                                return FetchResult(
                                    success=False,
                                    redirect=RedirectInfo(original_url, next_url, next_host),
                                    error=f"Cross-host redirect to {next_host}",
                                )
                            if not await self._robots.can_fetch(next_url):
                                return FetchResult(success=False, error=f"Blocked by robots.txt for {next_url}")
                            current_url = next_url
                            continue

                    # Final response (non-redirect, or redirect without Location).
                    if is_pdf:
                        content: str | bytes = bytes(buf)
                    else:
                        content = buf.decode(encoding, errors="replace")
                        if truncated:
                            content += f"\n\n[Content truncated at {cap} bytes]"
                    self._cache[original_url] = CachedResponse(
                        content=content, content_type=content_type,
                        timestamp=time.time(), truncated=truncated,
                    )
                    return FetchResult(
                        success=True, content=content, content_type=content_type,
                        truncated=truncated, from_cache=False,
                    )
                else:
                    return FetchResult(success=False, error=f"Too many redirects (max {self.MAX_REDIRECTS})")

        except BlockedAddressError as e:
            return FetchResult(success=False, error=f"Blocked (SSRF): {e}")
        except httpx.TimeoutException:
            return FetchResult(success=False, error=f"Request timeout after {timeout}s")
        except httpx.RequestError as e:
            return FetchResult(success=False, error=f"Request failed: {e}")

    def _get_cached(self, url: str) -> CachedResponse | None:
        if url not in self._cache:
            return None
        cached = self._cache[url]
        if time.time() - cached.timestamp > self._cache_ttl:
            del self._cache[url]
            return None
        return cached

    def clear_cache(self) -> None:
        self._cache.clear()
