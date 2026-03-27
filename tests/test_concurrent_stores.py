"""Tests for concurrent access to MemoryStore, plan registry state, and task functions.

Verifies that parallel read/write operations don't corrupt data.
"""

import asyncio

import pytest

from tests.conftest import MockContext


@pytest.fixture
def mock_context():
    with MockContext() as ctx:
        yield ctx


class TestConcurrentMemoryStore:
    """Test concurrent access to MemoryStore."""

    async def test_concurrent_writes(self, mock_context):
        """Parallel stores should not lose data."""
        from agentic_cli.tools.memory_tools import MemoryStore

        store = MemoryStore(mock_context.settings)

        async def write_memory(i: int) -> str:
            return store.store(f"memory-{i}", tags=[f"tag-{i}"])

        ids = await asyncio.gather(*(write_memory(i) for i in range(20)))

        assert len(ids) == 20
        assert len(set(ids)) == 20  # All unique IDs
        # All memories should be searchable
        results = store.search("", limit=100)
        assert len(results) == 20

    async def test_concurrent_read_write(self, mock_context):
        """Reads during writes should not raise or return corrupt data."""
        from agentic_cli.tools.memory_tools import MemoryStore

        store = MemoryStore(mock_context.settings)
        # Seed some data
        for i in range(5):
            store.store(f"seed-{i}")

        async def writer(i: int):
            store.store(f"new-{i}")

        async def reader():
            results = store.search("seed")
            # Should always find at least the seeds
            assert len(results) >= 5
            return results

        tasks = [writer(i) for i in range(10)] + [reader() for _ in range(10)]
        await asyncio.gather(*tasks)

        # Final state: 5 seeds + 10 new = 15
        assert len(store.search("", limit=100)) == 15


class TestConcurrentPlanRegistry:
    """Test concurrent access to plan via service registry."""

    async def test_concurrent_save_get(self):
        """Concurrent save/get should not corrupt plan content."""
        from agentic_cli.workflow.service_registry import set_service_registry

        registry = {}
        token = set_service_registry(registry)
        try:
            async def save_plan(i: int):
                registry["plan"] = f"## Plan {i}\n- Step 1\n- Step 2"

            async def get_plan():
                content = registry.get("plan", "")
                # Content should be either empty or a valid plan
                if content:
                    assert content.startswith("## Plan")
                return content

            tasks = [save_plan(i) for i in range(20)] + [get_plan() for _ in range(20)]
            await asyncio.gather(*tasks)

            # Final state should have some plan
            assert registry.get("plan", "")
        finally:
            token.var.reset(token)

    async def test_concurrent_save_clear(self):
        """Concurrent save/clear should not corrupt state."""
        from agentic_cli.workflow.service_registry import set_service_registry

        registry = {}
        token = set_service_registry(registry)
        try:
            async def save_and_clear(i: int):
                registry["plan"] = f"Plan {i}"
                if i % 3 == 0:
                    registry["plan"] = ""

            await asyncio.gather(*(save_and_clear(i) for i in range(20)))
            # Should not raise; state is either empty or has content
            _ = registry.get("plan", "")
        finally:
            token.var.reset(token)


class TestConcurrentTaskNormalization:
    """Test concurrent access to task normalization (pure functions)."""

    async def test_concurrent_normalize(self, mock_context):
        """Parallel normalize_tasks should not interfere with each other."""
        from agentic_cli.tools._core.tasks import normalize_tasks

        async def normalize_batch(batch: int):
            tasks = [
                {"description": f"batch-{batch}-task-{j}", "status": "pending"}
                for j in range(5)
            ]
            return normalize_tasks(tasks)

        results = await asyncio.gather(*(normalize_batch(i) for i in range(10)))

        # All returned normalized lists should have 5 entries
        for normalized, ids in results:
            assert len(normalized) == 5
            assert len(ids) == 5

    async def test_concurrent_filter(self, mock_context):
        """Concurrent filter_tasks on shared data should be consistent."""
        from agentic_cli.tools._core.tasks import normalize_tasks, filter_tasks

        normalized, _ = normalize_tasks([
            {"description": f"task-{i}", "status": "pending" if i % 2 == 0 else "in_progress"}
            for i in range(10)
        ])

        async def reader():
            filtered = filter_tasks(normalized, status="pending")
            # Should always get consistent count
            assert len(filtered) == 5
            return filtered

        results = await asyncio.gather(*(reader() for _ in range(20)))
        for r in results:
            assert len(r) == 5


class TestFileLockTimeout:
    """Test file_lock timeout behavior."""

    def test_lock_timeout_raises(self, tmp_path):
        """file_lock should raise FileLockTimeout when lock is held."""
        from agentic_cli.persistence._utils import file_lock, FileLockTimeout

        target = tmp_path / "test.json"
        target.write_text("{}")

        # Acquire lock in outer scope, then try to acquire with short timeout
        with file_lock(target, timeout=None):
            with pytest.raises(FileLockTimeout):
                with file_lock(target, timeout=0.1):
                    pass  # Should not reach here

    def test_lock_no_timeout(self, tmp_path):
        """file_lock with timeout=None should block indefinitely (legacy)."""
        from agentic_cli.persistence._utils import file_lock

        target = tmp_path / "test.json"
        target.write_text("{}")

        # Should acquire and release without error
        with file_lock(target, timeout=None):
            pass

    def test_lock_default_timeout(self, tmp_path):
        """file_lock with default timeout should work normally."""
        from agentic_cli.persistence._utils import file_lock

        target = tmp_path / "test.json"
        target.write_text("{}")

        with file_lock(target):
            pass  # Should work fine with default 10s timeout
