"""Stores open on one folder keep each other's changes.

Every CLI session of an application opens the same memory store. Each kept
the memories it read when it opened and rewrote the files from them on every
change, so whichever session saved last dropped what the others had saved
since: with stores A and B open, a memory stored through A, then one through
B, left only B's. Each read or change now holds the folder's lock and reads
again first what another store wrote.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

from agentic_cli.memory import MemoryStore, MockEmbeddingService


def _contents(folder: Path) -> list[str]:
    return sorted(i.content for i in MemoryStore(folder)._items.values())


def test_memories_stored_through_two_open_stores_are_both_kept(tmp_path):
    a, b = MemoryStore(tmp_path), MemoryStore(tmp_path)

    a.store("from A")
    b.store("from B")

    assert _contents(tmp_path) == ["from A", "from B"]


def test_a_store_finds_a_memory_another_stored_after_it_opened(tmp_path):
    a, b = MemoryStore(tmp_path), MemoryStore(tmp_path)

    a.store("the otters nest by the weir")

    assert [i.content for i in b.search("otters")] == ["the otters nest by the weir"]
    assert "otters" in b.load_all()


def test_an_update_through_one_store_survives_a_write_through_the_other(tmp_path):
    a, b = MemoryStore(tmp_path), MemoryStore(tmp_path)
    item_id = a.store("draft")

    assert b.update(item_id, content="final")
    a.store("another")

    assert _contents(tmp_path) == ["another", "final"]


def test_a_delete_through_one_store_survives_a_write_through_the_other(tmp_path):
    a, b = MemoryStore(tmp_path), MemoryStore(tmp_path)
    item_id = a.store("to forget")

    assert b.delete(item_id, purge=True)
    a.store("to keep")

    assert _contents(tmp_path) == ["to keep"]


def test_embeddings_from_both_stores_are_kept(tmp_path):
    a = MemoryStore(tmp_path, MockEmbeddingService())
    b = MemoryStore(tmp_path, MockEmbeddingService())

    a.store("from A")
    b.store("from B")

    reopened = MemoryStore(tmp_path)
    assert all(i.embedding is not None for i in reopened._items.values())
    assert len(reopened._items) == 2


_WRITER = textwrap.dedent('''
    import sys
    from agentic_cli.memory import MemoryStore
    store = MemoryStore(sys.argv[1])
    for n in range(int(sys.argv[3])):
        store.store(f"{sys.argv[2]}-{n}")
''')


def test_processes_storing_at_once_lose_nothing(tmp_path):
    writers, each = 4, 25
    procs = [
        subprocess.Popen([sys.executable, "-c", _WRITER, str(tmp_path), f"w{w}", str(each)])
        for w in range(writers)
    ]
    assert [p.wait(timeout=120) for p in procs] == [0] * writers

    assert len(_contents(tmp_path)) == writers * each
