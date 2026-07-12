"""Robots.txt compliance checker."""

from __future__ import annotations

from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx


class RobotsTxtChecker:
    """Checks robots.txt compliance, fetching robots.txt through the shared
    SSRF-safe PinnedTransport (so the robots request is resolved-validated-pinned
    like every other webfetch request)."""

    USER_AGENT = "AgenticCLI/1.0"
    _MAX_ROBOTS_BYTES = 512 * 1024

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport
        self._cache: dict[str, RobotFileParser | None] = {}

    async def can_fetch(self, url: str) -> bool:
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        if domain not in self._cache:
            self._cache[domain] = await self._fetch_robots(domain)
        parser = self._cache[domain]
        if parser is None:
            return True  # no robots.txt / fetch error → permissive
        return parser.can_fetch(self.USER_AGENT, url)

    async def _fetch_robots(self, domain: str) -> RobotFileParser | None:
        robots_url = f"{domain}/robots.txt"
        try:
            async with httpx.AsyncClient(transport=self._transport) as client:
                async with client.stream("GET", robots_url, timeout=10.0) as response:
                    if response.status_code != 200:
                        return None
                    buf = bytearray()
                    async for chunk in response.aiter_bytes():
                        buf.extend(chunk)
                        if len(buf) > self._MAX_ROBOTS_BYTES:
                            break
                    text = buf.decode(response.charset_encoding or "utf-8", errors="replace")
            parser = RobotFileParser()
            parser.parse(text.splitlines())
            return parser
        except Exception:
            return None  # network / SSRF-block / timeout → permissive
