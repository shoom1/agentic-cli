#!/usr/bin/env python
"""Standalone demo for the simplified memory system.

Demonstrates:
1. Storing memories with optional tags
2. Searching memories by substring
3. Loading all memories for system prompt injection
4. Persistence across instances

Usage:
    conda run -n agenticcli python examples/memory_demo.py
"""

import tempfile
from pathlib import Path

from agentic_cli.memory.store import MemoryStore, MemoryItem
from agentic_cli.config import BaseSettings


class DemoSettings(BaseSettings):
    """Settings for demo with temp directory."""

    workspace_dir: Path = Path(".")

    def __init__(self, temp_dir: Path, **kwargs):
        super().__init__(workspace_dir=temp_dir, **kwargs)


def main():
    print("\n" + "#" * 60)
    print("#  Memory System Demo (Simplified)")
    print("#" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        settings = DemoSettings(temp_path)

        # --- Store memories ---
        print("\n  Storing memories:")
        store = MemoryStore(settings)

        id1 = store.store("User prefers markdown output", tags=["preference"])
        print(f"    Stored: {id1[:8]}... [preference]")

        id2 = store.store("Basel III requires 99% VaR confidence", tags=["fact", "finance"])
        print(f"    Stored: {id2[:8]}... [fact, finance]")

        id3 = store.store("Python 3.12 is the project runtime")
        print(f"    Stored: {id3[:8]}... (no tags)")

        # --- Search ---
        print("\n  Search for 'Basel':")
        for item in store.search("Basel"):
            print(f"    - {item.content}")

        print("\n  Search for 'Python':")
        for item in store.search("Python"):
            print(f"    - {item.content}")

        # --- Load all (system prompt injection) ---
        print("\n  All memories (for system prompt):")
        print(store.load_all())

        # --- Persistence ---
        print("\n  Persistence test:")
        store2 = MemoryStore(settings)
        results = store2.search("")
        print(f"    New instance loaded {len(results)} memories")
        assert len(results) == 3

    print("\n" + "#" * 60)
    print("#  Demo Complete!")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
