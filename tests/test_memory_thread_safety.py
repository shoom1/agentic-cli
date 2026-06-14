"""Thread-safety tests for MemoryStore.

LangGraph's ToolNode runs sync tools concurrently in executor threads, so
store/search (search is a writer) and the full-file rewrite in _save must not
interleave — otherwise iteration over self._items races mutation, raising
"dictionary changed size during iteration" or losing writes.
"""

import threading

from agentic_cli.config import BaseSettings
from agentic_cli.tools.memory_tools import MemoryStore


def test_concurrent_store_and_search_no_corruption(tmp_path):
    store = MemoryStore(BaseSettings(workspace_dir=tmp_path))  # substring mode
    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            for j in range(25):
                store.store(f"memory {i}-{j}", tags=["t"])
                store.search("memory")  # writer: bumps access_count + _save
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"threads raised: {errors}"

    # No writes lost: 8 workers * 25 stores = 200 items.
    results = store.search("memory", limit=100000)
    assert len(results) == 200

    # On-disk file is valid and complete after the concurrent churn.
    reloaded = MemoryStore(BaseSettings(workspace_dir=tmp_path))
    assert len(reloaded.search("memory", limit=100000)) == 200
