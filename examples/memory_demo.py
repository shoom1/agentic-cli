#!/usr/bin/env python
"""Standalone demo for the memory system.

This demo tests the memory system components:
1. Working memory (session-scoped key-value storage with tags)
2. Long-term memory (persistent storage with types)
3. Memory manager (unified interface)
4. Cross-tier search
5. Serialization and snapshots

Usage:
    conda run -n agenticcli python examples/memory_demo.py
"""

import sys
import tempfile
from pathlib import Path

from agentic_cli.memory.working import WorkingMemory
from agentic_cli.memory.longterm import LongTermMemory, MemoryEntry as LongTermMemoryEntry, MemoryType
from agentic_cli.memory.manager import MemoryManager, MemorySearchResult
from agentic_cli.config import BaseSettings


# =============================================================================
# Mock Settings for Demo
# =============================================================================


class DemoSettings(BaseSettings):
    """Settings for demo with temp directory."""

    # Override workspace_dir as a regular field
    workspace_dir: Path = Path(".")

    def __init__(self, temp_dir: Path, **kwargs):
        super().__init__(workspace_dir=temp_dir, **kwargs)


# =============================================================================
# Demo Functions
# =============================================================================


def demo_working_memory():
    """Demo working memory (session-scoped key-value store)."""
    print("\n" + "=" * 60)
    print("Working Memory Demo")
    print("=" * 60)

    memory = WorkingMemory()

    # Basic set/get
    print("\n  Basic set/get:")
    memory.set("user_name", "Alice")
    memory.set("task", "research")
    memory.set("iteration", 1)

    print(f"    user_name: {memory.get('user_name')}")
    print(f"    task: {memory.get('task')}")
    print(f"    iteration: {memory.get('iteration')}")

    # With tags (note: tags is a list, not a set)
    print("\n  Storing with tags:")
    memory.set("api_key", "sk-123", tags=["secret", "config"])
    memory.set("model", "gpt-4", tags=["config"])
    memory.set("temperature", 0.7, tags=["config", "tuning"])

    print(f"    api_key (tags: secret, config)")
    print(f"    model (tags: config)")
    print(f"    temperature (tags: config, tuning)")

    # Query by tag using list() with tags filter
    print("\n  Query by tag 'config':")
    config_items = memory.list(tags=["config"])
    for key in config_items:
        print(f"    - {key}: {memory.get(key)}")

    # List all keys
    print(f"\n  All keys: {memory.list()}")

    # Delete
    memory.delete("iteration")
    print(f"  After deleting 'iteration': {memory.list()}")

    # Check if key exists (using get with default)
    print(f"\n  'user_name' exists: {'user_name' in memory.list()}")
    print(f"  'iteration' exists: {'iteration' in memory.list()}")
    print()

    return memory


def demo_working_memory_serialization(memory: WorkingMemory):
    """Demo working memory serialization."""
    print("\n" + "=" * 60)
    print("Working Memory Serialization Demo")
    print("=" * 60)

    # Export snapshot
    snapshot = memory.to_snapshot()
    print(f"  Snapshot keys: {list(snapshot.keys())}")
    print(f"  Number of entries: {len(snapshot.get('entries', {}))}")

    # Create new memory from snapshot
    restored = WorkingMemory.from_snapshot(snapshot)
    print(f"\n  Restored memory keys: {restored.list()}")
    print(f"  Restored user_name: {restored.get('user_name')}")
    print()


def demo_longterm_memory(temp_dir: Path):
    """Demo long-term memory (persistent storage)."""
    print("\n" + "=" * 60)
    print("Long-Term Memory Demo")
    print("=" * 60)

    settings = DemoSettings(temp_dir)
    memory = LongTermMemory(settings)

    # Store different types of memories
    print("\n  Storing memories of different types:")

    # Store a fact
    fact_id = memory.store(
        type=MemoryType.FACT,
        content="The project uses Python 3.12",
        source="project_analysis",
        confidence=0.95,
    )
    print(f"    Stored FACT: {fact_id[:8]}...")

    # Store a learning (note: MemoryType has FACT, PREFERENCE, LEARNING, REFERENCE)
    learning_id = memory.store(
        type=MemoryType.LEARNING,
        content="Users prefer concise responses over verbose ones",
        source="user_feedback",
        confidence=0.8,
    )
    print(f"    Stored LEARNING: {learning_id[:8]}...")

    # Store a reference
    reference_id = memory.store(
        type=MemoryType.REFERENCE,
        content="See documentation in docs/architecture.md",
        source="session_2024_01_15",
    )
    print(f"    Stored REFERENCE: {reference_id[:8]}...")

    # Store a preference
    pref_id = memory.store(
        type=MemoryType.PREFERENCE,
        content="User prefers TypeScript over JavaScript",
        source="user_statement",
        confidence=1.0,
    )
    print(f"    Stored PREFERENCE: {pref_id[:8]}...")

    # Recall by query
    print("\n  Recall by query 'Python':")
    results = memory.recall("Python")
    for entry in results:
        print(f"    [{entry.type.value}] {entry.content[:50]}...")

    # Recall by type
    print("\n  Recall by type FACT:")
    facts = memory.recall("", type=MemoryType.FACT)
    for entry in facts:
        print(f"    - {entry.content[:50]}...")

    # Get specific entry
    print(f"\n  Get entry by ID ({fact_id[:8]}...):")
    entry = memory.get(fact_id)
    if entry:
        print(f"    Type: {entry.type.value}")
        print(f"    Content: {entry.content}")
        print(f"    Source: {entry.source}")
        print(f"    Confidence: {entry.confidence}")

    # List all (via recall with empty query)
    all_entries = memory.recall("")
    print(f"\n  Total memories stored: {len(all_entries)}")
    print()

    return memory


def demo_memory_manager(temp_dir: Path):
    """Demo unified memory manager."""
    print("\n" + "=" * 60)
    print("Memory Manager Demo")
    print("=" * 60)

    settings = DemoSettings(temp_dir)
    manager = MemoryManager(settings)

    # Use working memory through manager
    print("\n  Using working memory:")
    manager.working.set("current_context", "code review")
    manager.working.set("files_reviewed", ["app.py", "utils.py"])
    print(f"    current_context: {manager.working.get('current_context')}")
    print(f"    files_reviewed: {manager.working.get('files_reviewed')}")

    # Use long-term memory through manager
    print("\n  Using long-term memory:")
    manager.longterm.store(
        type=MemoryType.FACT,
        content="The codebase has good test coverage",
        source="analysis",
    )
    manager.longterm.store(
        type=MemoryType.LEARNING,
        content="Review process improved code quality",
        source="observation",
    )

    # Cross-tier search
    print("\n  Cross-tier search for 'code':")
    results = manager.search("code", include_working=True, include_longterm=True)

    print(f"    Working memory matches: {len(results.working_results)}")
    for key, value in results.working_results:
        print(f"      - {key}: {value}")

    print(f"    Long-term memory matches: {len(results.longterm_results)}")
    for entry in results.longterm_results:
        print(f"      - [{entry.type.value}] {entry.content[:40]}...")

    # Snapshot working memory
    print("\n  Working memory snapshot:")
    snapshot = manager.get_working_snapshot()
    print(f"    Entries in snapshot: {len(snapshot.get('entries', {}))}")

    # Clear working memory
    manager.clear_working()
    print(f"    After clear, entries: {len(manager.working.list())}")
    print()


def demo_memory_types():
    """Demo memory type enumeration."""
    print("\n" + "=" * 60)
    print("Memory Types Demo")
    print("=" * 60)

    print("\n  Available memory types:")
    for mem_type in MemoryType:
        print(f"    - {mem_type.value}: {mem_type.name}")
    print()

    print("  Use cases:")
    print("    FACT:       Verified information (e.g., 'Python version is 3.12')")
    print("    LEARNING:   Patterns or insights (e.g., 'Users prefer X over Y')")
    print("    REFERENCE:  Links to resources (e.g., 'See docs/architecture.md')")
    print("    PREFERENCE: User preferences (e.g., 'Prefers TypeScript')")
    print()


def main():
    """Run all demos."""
    print("\n" + "#" * 60)
    print("#  Memory System Demo")
    print("#" * 60)

    # Create temp directory for long-term memory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"\n  Using temp directory: {temp_path}")

        # Run demos
        demo_memory_types()
        memory = demo_working_memory()
        demo_working_memory_serialization(memory)
        demo_longterm_memory(temp_path)
        demo_memory_manager(temp_path)

    print("\n" + "#" * 60)
    print("#  Demo Complete!")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
