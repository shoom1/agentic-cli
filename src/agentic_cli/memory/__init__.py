"""Memory module for session-scoped and persistent storage."""

from agentic_cli.memory.working import MemoryEntry, WorkingMemory
from agentic_cli.memory.longterm import (
    LongTermMemory,
    MemoryEntry as LongTermMemoryEntry,
    MemoryType,
)
from agentic_cli.memory.manager import MemoryManager, MemorySearchResult

__all__ = [
    "MemoryEntry",
    "WorkingMemory",
    "LongTermMemory",
    "LongTermMemoryEntry",
    "MemoryType",
    "MemoryManager",
    "MemorySearchResult",
]
