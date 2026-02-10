"""Pluggable search source interface.

Provides an abstraction for external search sources (ArXiv, SSRN, web search, etc.)
that can be registered and used by the knowledge base and tools.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


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
