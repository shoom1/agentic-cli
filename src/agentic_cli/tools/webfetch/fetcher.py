"""Content fetching with caching and redirect handling."""

from __future__ import annotations

import time
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx

from agentic_cli.tools.webfetch._text import decode_body
from agentic_cli.tools.webfetch.saved import saved_page_extension
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
    """The outcome of a fetch.

    ``content`` is what the summarizer sees: for text, the first
    ``max_content_bytes`` decoded, with a marker when cut (``truncated``); for
    a PDF, the bytes. ``raw`` is the body as received, up to the download
    limit (``raw_truncated`` when it went over), with its ``charset`` and the
    ``final_url`` after same-host redirects.
    """

    success: bool
    content: str | bytes | None = None
    content_type: str | None = None
    redirect: RedirectInfo | None = None
    error: str | None = None
    truncated: bool = False
    from_cache: bool = False
    status_code: int | None = None
    raw: bytes | None = None
    charset: str | None = None
    final_url: str | None = None
    raw_truncated: bool = False


@dataclass
class CachedResponse:
    raw: bytes
    content_type: str
    charset: str | None
    final_url: str
    timestamp: float
    raw_truncated: bool = False


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
        max_download_bytes: int = 5242880,
    ) -> None:
        self._validator = validator
        self._robots = robots_checker
        self._transport = transport
        self._cache_ttl = cache_ttl_seconds
        self._max_content_bytes = max_content_bytes
        self._max_pdf_bytes = max_pdf_bytes
        # A text body is read whole up to this limit, never less than what
        # the summarizer gets.
        self._max_download_bytes = max(max_download_bytes, max_content_bytes)
        self._cache: dict[str, CachedResponse] = {}

    async def fetch(self, url: str, timeout: int = 30) -> FetchResult:
        """Fetch content from a URL.

        Redirects are followed manually so each Location is revalidated before
        the next request; the PinnedTransport resolves+validates+pins every hop
        (and the robots fetch), connecting only to globally-routable IPs.
        """
        cached = self._get_cached(url)
        if cached is not None:
            return self._result(cached, from_cache=True)

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
                        reason = response.reason_phrase
                        headers = response.headers
                        content_type = headers.get("content-type", "text/html")
                        is_pdf = "application/pdf" in content_type.lower()
                        # Only types that are actually saved (saved.py's own
                        # list) get the generous download cap; everything
                        # else (images, archives, octet-stream, redirect and
                        # error bodies, ...) is capped at what the summarizer
                        # would see anyway, since it is never saved.
                        if is_pdf:
                            cap = self._max_pdf_bytes
                        elif saved_page_extension(content_type) is not None:
                            cap = self._max_download_bytes
                        else:
                            cap = self._max_content_bytes
                        charset = response.charset_encoding
                        buf = bytearray()
                        raw_truncated = False
                        async for chunk in response.aiter_bytes():
                            buf.extend(chunk)
                            if len(buf) > cap:
                                del buf[cap:]
                                raw_truncated = True
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
                    # An error page is not the content asked for: never
                    # returned as content, summarized, ingested or cached.
                    if status >= 400:
                        label = f"HTTP {status} {reason}".rstrip()
                        return FetchResult(
                            success=False, status_code=status,
                            error=f"{label} for {current_url}",
                        )
                    cached = CachedResponse(
                        raw=bytes(buf), content_type=content_type, charset=charset,
                        final_url=current_url, timestamp=time.time(),
                        raw_truncated=raw_truncated,
                    )
                    self._sweep_expired_cache()
                    self._cache[original_url] = cached
                    return self._result(cached, from_cache=False, status_code=status)
                else:
                    return FetchResult(success=False, error=f"Too many redirects (max {self.MAX_REDIRECTS})")

        except BlockedAddressError as e:
            return FetchResult(success=False, error=f"Blocked (SSRF): {e}")
        except httpx.TimeoutException:
            return FetchResult(success=False, error=f"Request timeout after {timeout}s")
        except httpx.RequestError as e:
            return FetchResult(success=False, error=f"Request failed: {e}")

    def _result(
        self, cached: CachedResponse, *, from_cache: bool, status_code: int | None = None,
    ) -> FetchResult:
        """Build a result; ``content`` is what the summarizer sees."""
        if "application/pdf" in cached.content_type.lower():
            content: str | bytes = cached.raw
            truncated = cached.raw_truncated
        else:
            limit = self._max_content_bytes
            truncated = cached.raw_truncated or len(cached.raw) > limit
            content = decode_body(cached.raw[:limit], cached.charset)
            if truncated:
                content += f"\n\n[Content truncated at {limit} bytes]"
        return FetchResult(
            success=True, content=content, content_type=cached.content_type,
            truncated=truncated, from_cache=from_cache, status_code=status_code,
            raw=cached.raw, charset=cached.charset, final_url=cached.final_url,
            raw_truncated=cached.raw_truncated,
        )

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

    def _sweep_expired_cache(self) -> None:
        """Drop expired entries so memory does not grow for the life of the
        process when distinct URLs are fetched (expiry was previously lazy:
        an entry was only dropped when that same URL was requested again)."""
        now = time.time()
        expired = [u for u, c in self._cache.items() if now - c.timestamp > self._cache_ttl]
        for u in expired:
            del self._cache[u]
