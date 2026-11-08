"""The memory store.

Until the switch to memory files, ``_legacy`` holds 0.6's store; the new one
is built beside it (``models``, ``_files``, ``_scope``, ``store``).
"""

from agentic_cli.memory.store._legacy import ForgettingPolicy, MemoryItem, MemoryStore

__all__ = ["ForgettingPolicy", "MemoryItem", "MemoryStore"]
