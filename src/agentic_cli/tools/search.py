"""Web search tool with pluggable backends.

Provides a unified web search interface that can use different search
providers (Tavily, Brave) based on configuration.

Usage:
    # As a tool for agents
    from agentic_cli.tools.search import web_search

    agent_config = AgentConfig(
        name="research_agent",
        tools=[web_search, ...],
    )

    # Direct usage
    results = web_search("Python programming", max_results=5)
"""

from __future__ import annotations

import httpx
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class SearchResult:
    """A single search result."""

    title: str
    url: str
    snippet: str
    score: float | None = None


class SearchBackend(ABC):
    """Abstract base class for search backends."""

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Execute a search query.

        Args:
            query: The search query string
            max_results: Maximum number of results to return

        Returns:
            List of SearchResult objects
        """
        pass


class TavilyBackend(SearchBackend):
    """Tavily search backend.

    Tavily is an AI-optimized search engine that returns
    pre-processed results suitable for LLM consumption.

    API docs: https://docs.tavily.com/
    """

    BASE_URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Search using Tavily API."""
        response = httpx.post(
            self.BASE_URL,
            json={
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": False,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("results", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                score=item.get("score"),
            ))
        return results


class BraveBackend(SearchBackend):
    """Brave Search backend.

    Brave Search is a privacy-focused search engine with
    its own independent index.

    API docs: https://brave.com/search/api/
    """

    BASE_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Search using Brave Search API."""
        response = httpx.get(
            self.BASE_URL,
            params={
                "q": query,
                "count": max_results,
            },
            headers={
                "X-Subscription-Token": self.api_key,
                "Accept": "application/json",
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("web", {}).get("results", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
                score=None,  # Brave doesn't provide relevance scores
            ))
        return results


# Registry of available backends
SEARCH_BACKENDS: dict[str, type[SearchBackend]] = {
    "tavily": TavilyBackend,
    "brave": BraveBackend,
}


def _get_backend(settings: Any) -> SearchBackend:
    """Get the configured search backend.

    Args:
        settings: Application settings

    Returns:
        Configured SearchBackend instance

    Raises:
        ValueError: If backend is not configured or API key is missing
    """
    backend_name = getattr(settings, "search_backend", None)

    if not backend_name:
        raise ValueError(
            "No search backend configured. Set 'search_backend' in settings "
            "to 'tavily' or 'brave'."
        )

    if backend_name not in SEARCH_BACKENDS:
        raise ValueError(
            f"Unknown search backend '{backend_name}'. "
            f"Available: {', '.join(SEARCH_BACKENDS.keys())}"
        )

    # Get the appropriate API key
    if backend_name == "tavily":
        api_key = getattr(settings, "tavily_api_key", None)
        if not api_key:
            raise ValueError(
                "Tavily API key not configured. Set 'tavily_api_key' in settings "
                "or TAVILY_API_KEY environment variable."
            )
    elif backend_name == "brave":
        api_key = getattr(settings, "brave_api_key", None)
        if not api_key:
            raise ValueError(
                "Brave API key not configured. Set 'brave_api_key' in settings "
                "or BRAVE_API_KEY environment variable."
            )
    else:
        raise ValueError(f"No API key handler for backend '{backend_name}'")

    backend_class = SEARCH_BACKENDS[backend_name]
    return backend_class(api_key)


def web_search(
    query: str,
    max_results: int = 5,
) -> dict:
    """Search the web for information.

    Uses the configured search backend (Tavily or Brave) to search
    the web and return relevant results.

    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 5)

    Returns:
        Dictionary with search results:
        {
            "success": bool,
            "query": str,
            "results": [
                {"title": str, "url": str, "snippet": str, "score": float | None},
                ...
            ],
            "error": str | None
        }
    """
    from agentic_cli.config import get_settings

    resolved_settings = get_settings()

    try:
        backend = _get_backend(resolved_settings)
        results = backend.search(query, max_results)

        return {
            "success": True,
            "query": query,
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "score": r.score,
                }
                for r in results
            ],
            "error": None,
        }
    except ValueError as e:
        # Configuration errors
        return {
            "success": False,
            "query": query,
            "results": [],
            "error": str(e),
        }
    except httpx.HTTPStatusError as e:
        # API errors
        return {
            "success": False,
            "query": query,
            "results": [],
            "error": f"Search API error: {e.response.status_code} - {e.response.text}",
        }
    except httpx.RequestError as e:
        # Network errors
        return {
            "success": False,
            "query": query,
            "results": [],
            "error": f"Search request failed: {str(e)}",
        }
