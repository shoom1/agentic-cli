"""Concurrent initialize_services() must run the init body exactly once.

The guard at the top of BaseWorkflowManager.initialize_services() was
check-then-act: a user message arriving while background init is mid-flight
(manager._ensure_initialized → initialize_services) raced the background
call and ran the whole body twice (duplicate registry refresh, duplicate
service creation). An asyncio lock serializes them.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from agentic_cli.workflow.base_manager import BaseWorkflowManager


class _CountingManager(BaseWorkflowManager):
    """Minimal concrete manager counting _do_initialize runs."""

    def __init__(self, settings):
        super().__init__(agent_configs=[], settings=settings)
        self.do_init_calls = 0

    def _get_state_tools(self):
        return []

    @property
    def backend_type(self) -> str:
        return "test"

    async def _do_initialize(self) -> None:
        self.do_init_calls += 1
        # Yield so a concurrent initialize_services() can interleave
        await asyncio.sleep(0.02)

    async def process(self, message, user_id, session_id=None):
        raise NotImplementedError

    async def reinitialize(self, model=None, preserve_sessions=True):
        pass

    async def cleanup(self):
        pass


def _settings():
    s = MagicMock()
    s.app_name = "test-app"
    s.google_api_key = None
    s.anthropic_api_key = None
    return s


def _manager():
    m = _CountingManager(_settings())
    m._model_registry = MagicMock(refresh=AsyncMock())
    m._ensure_managers_initialized = lambda: None
    return m


async def test_concurrent_initialize_services_runs_once():
    m = _manager()

    await asyncio.gather(
        m.initialize_services(validate=False),
        m.initialize_services(validate=False),
    )

    assert m.do_init_calls == 1
    assert m.is_initialized


async def test_sequential_initialize_services_is_idempotent():
    m = _manager()

    await m.initialize_services(validate=False)
    await m.initialize_services(validate=False)

    assert m.do_init_calls == 1
