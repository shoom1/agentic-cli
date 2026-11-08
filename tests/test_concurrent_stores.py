"""Tests for concurrent access to plan registry state and task functions.

Verifies that parallel read/write operations don't corrupt data.
"""

import asyncio

import pytest

from tests.conftest import MockContext


@pytest.fixture
def mock_context():
    with MockContext() as ctx:
        yield ctx


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
