"""Robots.txt compliance checker."""

from __future__ import annotations

from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx


class RobotsTxtChecker:
    """Checks robots.txt compliance for URLs.

    Fetches and caches robots.txt per domain, then checks if our
    user agent is allowed to access specific paths.
    """

    USER_AGENT = "AgenticCLI/1.0"

    def __init__(self) -> None:
        """Initialize the checker with empty cache."""
        self._cache: dict[str, RobotFileParser | None] = {}

    async def can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt.

        Args:
            url: The URL to check.

        Returns:
            True if allowed (or on error), False if explicitly disallowed.
        """
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"

        # Get or fetch robots.txt for this domain
        if domain not in self._cache:
            self._cache[domain] = await self._fetch_robots(domain)

        parser = self._cache[domain]
        if parser is None:
            # No robots.txt or error fetching - be permissive
            return True

        return parser.can_fetch(self.USER_AGENT, url)

    async def _fetch_robots(self, domain: str) -> RobotFileParser | None:
        """Fetch and parse robots.txt for a domain.

        Args:
            domain: The domain (scheme + netloc) to fetch robots.txt for.

        Returns:
            Parsed RobotFileParser, or None if not found/error.
        """
        robots_url = f"{domain}/robots.txt"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(robots_url, timeout=10.0)

                if response.status_code != 200:
                    return None

                parser = RobotFileParser()
                parser.parse(response.text.splitlines())
                return parser

        except Exception:
            # Network errors, timeouts, etc. - be permissive
            return None

    def clear_cache(self) -> None:
        """Clear the robots.txt cache."""
        self._cache.clear()
