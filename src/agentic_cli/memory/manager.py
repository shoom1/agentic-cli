"""Memory manager for unified access to all memory tiers.

Provides a single interface for working with both session-scoped WorkingMemory
and persistent LongTermMemory, with support for cross-tier search.
"""

from dataclasses import dataclass, field
from typing import Any

from agentic_cli.config import BaseSettings
from agentic_cli.memory.longterm import LongTermMemory, MemoryEntry as LongTermMemoryEntry
from agentic_cli.memory.working import WorkingMemory


@dataclass
class MemorySearchResult:
    """Results from a cross-tier memory search.

    Attributes:
        query: The original search query.
        working_results: List of (key, value) tuples from working memory.
        longterm_results: List of MemoryEntry objects from long-term memory.
        kb_results: List of dicts for future knowledge base integration.
    """

    query: str
    working_results: list[tuple[str, Any]] = field(default_factory=list)
    longterm_results: list[LongTermMemoryEntry] = field(default_factory=list)
    kb_results: list[dict] = field(default_factory=list)


class MemoryManager:
    """Unified manager for all memory tiers.

    Provides a single interface for accessing WorkingMemory (session-scoped)
    and LongTermMemory (persistent). Supports cross-tier search and
    session persistence operations.

    Example:
        >>> from agentic_cli.memory import MemoryManager, MemoryType
        >>> manager = MemoryManager(settings)
        >>> manager.working.set("current_task", "analysis")
        >>> manager.longterm.store(type=MemoryType.FACT, content="Important", source="s1")
        >>> results = manager.search("task")
    """

    def __init__(self, settings: BaseSettings) -> None:
        """Initialize the memory manager.

        Args:
            settings: BaseSettings instance providing workspace_dir for LongTermMemory.
        """
        self._settings = settings
        self._working = WorkingMemory()
        self._longterm = LongTermMemory(settings)

    @property
    def working(self) -> WorkingMemory:
        """Access the session-scoped working memory.

        Returns:
            WorkingMemory instance for key-value storage.
        """
        return self._working

    @property
    def longterm(self) -> LongTermMemory:
        """Access the persistent long-term memory.

        Returns:
            LongTermMemory instance for persistent storage.
        """
        return self._longterm

    def search(
        self,
        query: str,
        include_working: bool = True,
        include_longterm: bool = True,
        include_kb: bool = False,
    ) -> MemorySearchResult:
        """Search across memory tiers.

        Searches for the query in enabled memory tiers:
        - Working memory: searches keys and string values for substring match
        - Long-term memory: calls recall(query) for substring match

        Args:
            query: The search query string.
            include_working: Include results from working memory.
            include_longterm: Include results from long-term memory.
            include_kb: Include results from knowledge base (future).

        Returns:
            MemorySearchResult with results from each tier.
        """
        result = MemorySearchResult(query=query)
        query_lower = query.lower()

        if include_working:
            # Search working memory by key and string value
            for key in self._working.list():
                value = self._working.get(key)
                # Check if query matches key
                if query_lower in key.lower():
                    result.working_results.append((key, value))
                # Check if query matches string value
                elif isinstance(value, str) and query_lower in value.lower():
                    result.working_results.append((key, value))

        if include_longterm:
            # Use long-term memory's built-in recall search
            result.longterm_results = self._longterm.recall(query)

        if include_kb:
            # Future: integrate with knowledge base
            pass

        return result

    def clear_working(self) -> None:
        """Clear all entries from working memory."""
        self._working.clear()

    def get_working_snapshot(self) -> dict[str, Any]:
        """Export working memory state for session persistence.

        Returns:
            A dictionary representation that can be serialized and restored.
        """
        return self._working.to_snapshot()

    def restore_working(self, snapshot: dict[str, Any]) -> None:
        """Restore working memory from a persistence snapshot.

        Args:
            snapshot: A dictionary previously created by get_working_snapshot().
        """
        self._working = WorkingMemory.from_snapshot(snapshot)
