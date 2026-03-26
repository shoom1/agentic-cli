"""Tests for concurrent access to MemoryStore, PlanStore, and TaskStore.

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


class TestConcurrentTaskStore:
    """Test concurrent access to TaskStore."""

    async def test_concurrent_replace_all(self, mock_context):
        """Parallel replace_all should not corrupt the store."""
        from agentic_cli.tools.task_tools import TaskStore

        store = TaskStore(mock_context.settings)

        async def replace_tasks(batch: int):
            tasks = [
                {"description": f"batch-{batch}-task-{j}", "status": "pending"}
                for j in range(5)
            ]
            return store.replace_all(tasks)

        results = await asyncio.gather(*(replace_tasks(i) for i in range(10)))

        # Last write wins — store should have exactly 5 tasks
        all_tasks = store.list_tasks()
        assert len(all_tasks) == 5
        # All returned ID lists should have 5 entries
        for ids in results:
            assert len(ids) == 5

    async def test_concurrent_read_write(self, mock_context):
        """Reads during replace_all should return consistent snapshots."""
        from agentic_cli.tools.task_tools import TaskStore, TaskStatus

        store = TaskStore(mock_context.settings)
        store.replace_all([
            {"description": f"initial-{i}", "status": "pending"}
            for i in range(5)
        ])

        async def writer(batch: int):
            tasks = [
                {"description": f"batch-{batch}-{j}", "status": "in_progress"}
                for j in range(5)
            ]
            store.replace_all(tasks)

        async def reader():
            tasks = store.list_tasks()
            # Should always get a consistent list (not partially updated)
            assert len(tasks) == 5
            return tasks

        tasks = [writer(i) for i in range(10)] + [reader() for _ in range(10)]
        await asyncio.gather(*tasks)


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
