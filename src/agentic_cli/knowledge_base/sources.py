"""Pluggable search source interface.

Provides an abstraction for external search sources (ArXiv, SSRN, web search, etc.)
that can be registered and used by the knowledge base and tools.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


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


class ArxivSearchSource(SearchSource):
    """ArXiv paper search source."""

    @property
    def name(self) -> str:
        return "arxiv"

    @property
    def description(self) -> str:
        return "Search arXiv for academic papers"

    @property
    def rate_limit(self) -> float:
        return 3.0  # ArXiv requires 3 second delays

    def search(
        self,
        query: str,
        max_results: int = 10,
        categories: list[str] | None = None,
        **kwargs,
    ) -> list[SearchSourceResult]:
        """Search ArXiv.

        Args:
            query: Search query
            max_results: Maximum results
            categories: Optional ArXiv category filters (e.g., ['cs.AI', 'cs.LG'])
        """
        try:
            import feedparser
        except ImportError:
            return []

        base_url = "http://export.arxiv.org/api/query?"
        search_query = f"search_query=all:{query}"

        if categories:
            cat_query = " OR ".join(f"cat:{cat}" for cat in categories)
            search_query = f"search_query=(all:{query}) AND ({cat_query})"

        url = f"{base_url}{search_query}&start=0&max_results={max_results}"

        try:
            feed = feedparser.parse(url)
        except Exception:
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

        return results


class WebSearchSource(SearchSource):
    """Web search source using Serper.dev API."""

    @property
    def name(self) -> str:
        return "web"

    @property
    def description(self) -> str:
        return "Search the web using Serper.dev"

    @property
    def requires_api_key(self) -> str | None:
        return "serper_api_key"

    def search(
        self,
        query: str,
        max_results: int = 10,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        **kwargs,
    ) -> list[SearchSourceResult]:
        """Search the web.

        Args:
            query: Search query
            max_results: Maximum results
            allowed_domains: Only include results from these domains
            blocked_domains: Exclude results from these domains
        """
        from agentic_cli.tools.search import WebSearchClient
        from agentic_cli.config import get_settings

        settings = get_settings()
        client = WebSearchClient(api_key=settings.serper_api_key)

        search_result = client.search(
            query,
            max_results=max_results,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
        )

        results = []
        for item in search_result.get("results", []):
            results.append(
                SearchSourceResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("snippet", ""),
                    source_name=self.name,
                    metadata={"domain": item.get("domain", "")},
                )
            )

        return results


# Register built-in sources
register_search_source(ArxivSearchSource())
register_search_source(WebSearchSource())
