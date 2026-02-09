"""Pluggable search source interface.

Provides an abstraction for external search sources (ArXiv, SSRN, web search, etc.)
that can be registered and used by the knowledge base and tools.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from agentic_cli.logging import Loggers

logger = Loggers.knowledge_base()


@dataclass
class SearchSourceResult:
    """Standard result from a search source.

    Provides a common structure for results from various sources.
    """

    title: str
    url: str
    snippet: str
    source_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source_name,
            "metadata": self.metadata,
        }


class SearchSource(ABC):
    """Abstract base class for search sources.

    Implement this interface to create custom search sources
    that can be registered with the SearchSourceRegistry.

    Example:
        class MyDatabaseSource(SearchSource):
            @property
            def name(self) -> str:
                return "my_database"

            @property
            def description(self) -> str:
                return "Search internal database"

            def search(self, query: str, **kwargs) -> list[SearchSourceResult]:
                results = self._db.query(query)
                return [
                    SearchSourceResult(
                        title=r.title,
                        url=r.url,
                        snippet=r.summary,
                        source_name=self.name,
                    )
                    for r in results
                ]
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this source."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this source provides."""
        pass

    @property
    def requires_api_key(self) -> str | None:
        """Name of required API key setting, if any."""
        return None

    @property
    def rate_limit(self) -> float:
        """Minimum seconds between requests (0 = no limit)."""
        return 0

    @abstractmethod
    def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs,
    ) -> list[SearchSourceResult]:
        """Execute a search query.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            **kwargs: Source-specific parameters

        Returns:
            List of SearchSourceResult objects
        """
        pass

    def is_available(self) -> bool:
        """Check if this source is available (has required keys, etc.)."""
        if self.requires_api_key:
            from agentic_cli.config import get_settings

            settings = get_settings()
            key = getattr(settings, self.requires_api_key, None)
            return bool(key)
        return True


class SearchSourceRegistry:
    """Registry for managing search sources.

    Allows registration of custom search sources and provides
    unified search across multiple sources.
    """

    def __init__(self):
        self._sources: dict[str, SearchSource] = {}

    def register(self, source: SearchSource) -> None:
        """Register a search source.

        Args:
            source: SearchSource implementation to register
        """
        self._sources[source.name] = source

    def unregister(self, name: str) -> None:
        """Unregister a search source.

        Args:
            name: Name of source to remove
        """
        self._sources.pop(name, None)

    def get(self, name: str) -> SearchSource | None:
        """Get a search source by name.

        Args:
            name: Source name

        Returns:
            SearchSource if found, None otherwise
        """
        return self._sources.get(name)

    def list_sources(self) -> list[SearchSource]:
        """List all registered sources."""
        return list(self._sources.values())

    def list_available(self) -> list[SearchSource]:
        """List sources that are currently available."""
        return [s for s in self._sources.values() if s.is_available()]

    def search(
        self,
        query: str,
        sources: list[str] | None = None,
        max_results: int = 10,
        **kwargs,
    ) -> dict[str, list[SearchSourceResult]]:
        """Search across multiple sources.

        Args:
            query: Search query
            sources: List of source names to search (None = all available)
            max_results: Maximum results per source
            **kwargs: Passed to individual sources

        Returns:
            Dict mapping source name to list of results
        """
        results = {}

        if sources is None:
            target_sources = self.list_available()
        else:
            target_sources = [
                self._sources[name]
                for name in sources
                if name in self._sources and self._sources[name].is_available()
            ]

        for source in target_sources:
            try:
                source_results = source.search(query, max_results=max_results, **kwargs)
                results[source.name] = source_results
            except Exception as e:
                # Log error but continue with other sources
                results[source.name] = []

        return results


# Default registry instance
_default_registry = SearchSourceRegistry()


def get_search_registry() -> SearchSourceRegistry:
    """Get the default search source registry."""
    return _default_registry


def register_search_source(source: SearchSource) -> None:
    """Register a search source with the default registry."""
    _default_registry.register(source)


# Built-in search source implementations


@dataclass
class CachedSearchResult:
    """Cached search result with timestamp."""

    results: list[SearchSourceResult]
    timestamp: float


class ArxivSearchSource(SearchSource):
    """ArXiv paper search source with rate limiting and caching."""

    def __init__(
        self,
        cache_ttl_seconds: int = 900,
        max_cache_size: int = 100,
    ) -> None:
        """Initialize with rate limiting and caching.

        Args:
            cache_ttl_seconds: Cache time-to-live in seconds (default: 900 = 15 minutes)
            max_cache_size: Maximum number of cached queries (default: 100)
        """
        self._last_request_time: float = 0.0
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_cache_size = max_cache_size
        self._cache: dict[str, CachedSearchResult] = {}
        self._last_error: str | None = None

    @property
    def last_error(self) -> str | None:
        """Last error message from search(), or None if last search succeeded."""
        return self._last_error

    @property
    def name(self) -> str:
        return "arxiv"

    @property
    def description(self) -> str:
        return "Search arXiv for academic papers"

    @property
    def rate_limit(self) -> float:
        return 3.0  # ArXiv requires 3 second delays

    def _make_cache_key(
        self,
        query: str,
        max_results: int,
        categories: list[str] | None,
        sort_by: str,
        sort_order: str,
        date_from: str | None,
        date_to: str | None,
    ) -> str:
        """Create a cache key from search parameters."""
        cat_str = ",".join(sorted(categories)) if categories else ""
        date_str = f"{date_from or ''}-{date_to or ''}"
        return f"{query}|{max_results}|{cat_str}|{sort_by}|{sort_order}|{date_str}"

    def _get_cached(self, cache_key: str) -> list[SearchSourceResult] | None:
        """Get cached results if available and not expired."""
        if cache_key not in self._cache:
            return None
        cached = self._cache[cache_key]
        if time.time() - cached.timestamp > self.cache_ttl_seconds:
            del self._cache[cache_key]
            return None
        return cached.results

    def _evict_oldest_if_full(self) -> None:
        """Evict oldest cache entry if cache is at max size."""
        if len(self._cache) >= self.max_cache_size:
            # Find oldest entry by timestamp
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
            del self._cache[oldest_key]

    def clear_cache(self) -> None:
        """Clear all cached search results."""
        self._cache.clear()

    def wait_for_rate_limit(self) -> None:
        """Wait if necessary to respect rate limiting.

        Call this before making any ArXiv API request.
        """
        current_time = time.time()
        elapsed = current_time - self._last_request_time
        if self._last_request_time > 0 and elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()

    def search(
        self,
        query: str,
        max_results: int = 10,
        categories: list[str] | None = None,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        date_from: str | None = None,
        date_to: str | None = None,
        **kwargs,
    ) -> list[SearchSourceResult]:
        """Search ArXiv.

        Args:
            query: Search query
            max_results: Maximum results
            categories: Optional ArXiv category filters (e.g., ['cs.AI', 'cs.LG'])
            sort_by: Sort results by 'relevance', 'lastUpdatedDate', or 'submittedDate'
            sort_order: Sort order 'ascending' or 'descending'
            date_from: Filter papers submitted after this date (YYYY-MM-DD)
            date_to: Filter papers submitted before this date (YYYY-MM-DD)
        """
        # Check cache first
        cache_key = self._make_cache_key(
            query, max_results, categories, sort_by, sort_order, date_from, date_to
        )
        cached_results = self._get_cached(cache_key)
        if cached_results is not None:
            return cached_results

        try:
            import feedparser
        except ImportError:
            self._last_error = "feedparser not installed"
            return []

        # Enforce rate limiting
        self.wait_for_rate_limit()

        base_url = "http://export.arxiv.org/api/query?"

        # Build search query
        query_parts = [f"all:{query}"]

        if categories:
            cat_query = " OR ".join(f"cat:{cat}" for cat in categories)
            query_parts.append(f"({cat_query})")

        # Add date range filter if specified
        if date_from or date_to:
            from_date = date_from.replace("-", "") if date_from else "*"
            to_date = date_to.replace("-", "") if date_to else "*"
            query_parts.append(f"submittedDate:[{from_date} TO {to_date}]")

        from urllib.parse import quote

        raw_query = " AND ".join(f"({p})" if " " in p else p for p in query_parts)
        search_query = "search_query=" + quote(raw_query, safe="")

        # Build URL with sort parameters
        url = f"{base_url}{search_query}&start=0&max_results={max_results}"
        url += f"&sortBy={sort_by}&sortOrder={sort_order}"

        try:
            feed = feedparser.parse(url)
        except Exception as e:
            self._last_error = f"Failed to parse ArXiv feed: {e}"
            return []

        # Detect ArXiv rate limiting / HTTP errors
        status = getattr(feed, "status", 200)
        if status in (403, 429):
            self._last_error = f"ArXiv rate limited (HTTP {status})"
            logger.warning("arxiv_rate_limited", status=status)
            return []
        if feed.bozo and not feed.entries:
            self._last_error = f"ArXiv feed error: {feed.bozo_exception}"
            logger.warning("arxiv_feed_error", error=str(feed.bozo_exception))
            return []

        results = []
        for entry in feed.entries:
            results.append(
                SearchSourceResult(
                    title=entry.get("title", "").replace("\n", " "),
                    url=entry.get("link", ""),
                    snippet=entry.get("summary", "").replace("\n", " ")[:500],
                    source_name=self.name,
                    metadata={
                        "authors": [
                            a.get("name", "") for a in entry.get("authors", [])
                        ],
                        "published": entry.get("published", ""),
                        "categories": [
                            t.get("term", "") for t in entry.get("tags", [])
                        ],
                        "arxiv_id": entry.get("id", "").split("/abs/")[-1],
                    },
                )
            )

        # Clear error on successful parse
        self._last_error = None

        # Only cache non-error results (avoid caching empty results from rate limiting)
        if results or (status == 200 and not feed.bozo):
            self._evict_oldest_if_full()
            self._cache[cache_key] = CachedSearchResult(
                results=results,
                timestamp=time.time(),
            )

        return results


# Register built-in sources
# Note: For web search, use google_search_tool from google.adk.tools with LlmAgent
register_search_source(ArxivSearchSource())
