"""Web search client for agentic CLI applications.

Provides web search using Serper.dev API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class WebResult:
    """Web search result."""

    title: str
    url: str
    snippet: str
    domain: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "domain": self.domain,
        }


class WebSearchClient:
    """Client for web search.

    Uses Serper.dev API for web search, with fallback to mock results.
    """

    SERPER_URL = "https://google.serper.dev/search"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the web search client.

        Args:
            api_key: Serper.dev API key (optional).
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def search(
        self,
        query: str,
        max_results: int = 10,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> dict[str, Any]:
        """Search the web for relevant information.

        Args:
            query: Search query.
            max_results: Maximum number of results.
            allowed_domains: Only include results from these domains.
            blocked_domains: Exclude results from these domains.

        Returns:
            Dict with results and metadata.
        """
        # Build query with domain filters
        full_query = self._build_query(query, allowed_domains)

        if self.api_key:
            return self._search_serper(full_query, max_results, blocked_domains)
        return self._mock_search(query, max_results)

    def _build_query(self, query: str, allowed_domains: list[str] | None) -> str:
        """Build search query with domain filters."""
        if not allowed_domains:
            return query
        domain_filter = " OR ".join(f"site:{d}" for d in allowed_domains)
        return f"{query} ({domain_filter})"

    def _search_serper(
        self,
        query: str,
        max_results: int,
        blocked_domains: list[str] | None,
    ) -> dict[str, Any]:
        """Search using Serper.dev API."""
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

        data = {
            "q": query,
            "num": max_results,
        }

        try:
            response = self._client.post(
                self.SERPER_URL,
                headers=headers,
                json=data,
            )
            response.raise_for_status()
            result = response.json()

            results = []
            for item in result.get("organic", []):
                # Filter blocked domains
                domain = self._extract_domain(item.get("link", ""))
                if blocked_domains and domain in blocked_domains:
                    continue

                web_result = WebResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    domain=domain,
                )
                results.append(web_result.to_dict())

            return {
                "results": results[:max_results],
                "total_found": len(results),
            }

        except httpx.HTTPError as e:
            return {
                "results": [],
                "total_found": 0,
                "error": f"Web search error: {str(e)}",
            }

    def _mock_search(
        self,
        query: str,
        max_results: int,
    ) -> dict[str, Any]:
        """Return mock results when no API key is available."""
        return {
            "results": [],
            "total_found": 0,
            "message": (
                f"Web search for '{query}' - "
                "No API key configured. Set SERPER_API_KEY environment variable."
            ),
        }

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        match = re.search(r"https?://([^/]+)", url)
        return match.group(1) if match else ""


@dataclass
class MockWebSearchClient:
    """Mock web search client for testing."""

    api_key: str | None = None
    timeout: float = 30.0
    mock_results: list[dict] = field(default_factory=list)

    def search(
        self,
        query: str,
        max_results: int = 10,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return mock search results."""
        if self.mock_results:
            return {
                "results": self.mock_results[:max_results],
                "total_found": len(self.mock_results),
            }

        # Generate mock result
        mock_result = {
            "title": f"Mock Result: {query}",
            "url": "https://example.com/mock",
            "snippet": f"This is a mock search result for {query}.",
            "domain": "example.com",
        }

        return {
            "results": [mock_result],
            "total_found": 1,
        }
