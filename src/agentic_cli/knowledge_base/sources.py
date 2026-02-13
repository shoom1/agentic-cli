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
