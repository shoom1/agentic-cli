#!/usr/bin/env python
"""Standalone demo of the memory store: markdown files in two scopes.

Demonstrates:
1. Saving memories to the user and the project scope
2. Searching them by keyword
3. The index the main agent gets each conversation
4. Updating one, then restoring its previous version
5. A proposal waiting for review, then accepted
6. Persistence across instances

Usage:
    conda run -n agenticcli python examples/memory_demo.py
"""

import tempfile
from pathlib import Path

from agentic_cli.memory import Candidate, MemoryStore


def main() -> None:
    print("\n" + "#" * 60)
    print("#  Memory Store Demo")
    print("#" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        project = root / "projects" / "demo" / "memory"
        store = MemoryStore(root / "user", project, project_name="demo")

        print("\n  Saving memories:")
        saved = [
            store.save("Prefers markdown output", "Answers in markdown, with short headings.",
                       "feedback", scope="user"),
            store.save("Basel III needs 99% VaR confidence", "The risk model reports VaR at 99%.", "project"),
            store.save("Runs on Python 3.12", "The project's runtime is Python 3.12.", "project"),
        ]
        for memory in saved:
            print(f"    {memory.id} [{memory.scope}/{memory.type}] {memory.description} -> {memory.path.name}")

        print("\n  Search for 'VaR':")
        for memory in store.search("VaR"):
            print(f"    - {memory.description}")

        print("\n  The index the main agent gets:")
        print(store.index())

        print("\n  Update, then restore the previous version:")
        runtime = saved[2]
        store.update(runtime.id, content="The project's runtime is Python 3.13.")
        [version] = store.history(runtime.id)
        store.restore(runtime.id, version.saved)
        print(f"    {store.read(runtime.id).content}")

        print("\n  A proposal waits for review:")
        proposal = store.propose(
            Candidate("new", None, "Uses metric units", "Report distances in km.", "user", "user")
        )
        print(f"    {len(store.proposals())} waiting: {proposal.description}")
        store.accept(proposal.id)
        print(f"    Accepted: {len(store.memories())} memories")

        print("\n  Persistence: a new store over the same folders")
        again = MemoryStore(root / "user", project)
        print(f"    It reads {len(again.memories())} memories")
        assert len(again.memories()) == 4

    print("\n" + "#" * 60)
    print("#  Demo Complete!")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
