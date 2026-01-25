"""Memory module for session-scoped and persistent storage."""

from agentic_cli.memory.working import MemoryEntry, WorkingMemory
from agentic_cli.memory.longterm import (
    LongTermMemory,
    MemoryEntry as LongTermMemoryEntry,
    MemoryType,
)
from agentic_cli.memory.manager import MemoryManager, MemorySearchResult
from agentic_cli.memory.tools import (
    working_memory_tool,
    long_term_memory_tool,
    reset_working_memory,
)

__all__ = [
    "MemoryEntry",
    "WorkingMemory",
    "LongTermMemory",
    "LongTermMemoryEntry",
    "MemoryType",
    "MemoryManager",
    "MemorySearchResult",
    # Tools for agent use
    "working_memory_tool",
    "long_term_memory_tool",
    "reset_working_memory",
]
