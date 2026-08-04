"""Lifecycle races between a turn, a cleanup, and a cancelled initialization.

Two defects:

1. ``process()`` initializes *before* taking the turn lock (that ordering is
   what keeps cleanup from deadlocking against a running turn). A cleanup that
   was already queued therefore ran in between, and the turn woke up holding
   admission to a manager whose runner had just been released — it then used
   ``None`` as a runner deep inside ADK.
2. Service construction runs on a worker thread (``asyncio.to_thread``) and
   wrote straight into ``self._services``. Cancelling the awaiting coroutine
   does not stop the thread, so a rolled-back initialization was followed,
   moments later, by that thread publishing services into a manager that had
   already been cleaned up — leaking a sandbox/job manager nobody would close.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("google.adk")

from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.base_manager import BaseWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.events import EventType, WorkflowEvent  # noqa: E402
from agentic_cli.workflow.service_registry import SANDBOX_MANAGER  # noqa: E402
from tests.conftest import MockContext  # noqa: E402


# ---------------------------------------------------------------------------
# 1. A turn admitted after a cleanup must not use released resources
# ---------------------------------------------------------------------------


class _AdmissionHarness:
    """An ADK manager whose stream and initialization the test drives."""

    def __init__(self, ctx) -> None:
        self.manager = GoogleADKWorkflowManager(
            agent_configs=[AgentConfig(name="a", prompt="p")], settings=ctx.settings
        )
        self.manager._event_processor = SimpleNamespace(model=None)
        self.inits = 0
        self.streams: list[str] = []
        self.runner_at_stream: list[object] = []

        async def _do_initialize() -> None:
            self.inits += 1
            self.manager._session_service = SimpleNamespace(
                get_session=self._get_session
            )
            self.manager._root_agent = SimpleNamespace(name="a")
            self.manager._runner = SimpleNamespace(name=f"runner-{self.inits}")

        async def _get_or_create(user_id, session_id):
            return SimpleNamespace(id=session_id)

        async def _stream(*, session_id, user_id, new_message, run_config):
            self.streams.append(session_id)
            self.runner_at_stream.append(self.manager._runner)
            yield WorkflowEvent(type=EventType.TEXT, content=session_id)

        self.manager._do_initialize = _do_initialize
        self.manager._get_or_create_session = _get_or_create
        self.manager._run_and_stream = _stream
        self.manager._model_registry = MagicMock(refresh=AsyncMock())
        self.manager._ensure_managers_initialized = lambda: None
        self.manager._validate_agent_graph = lambda: None

    @staticmethod
    async def _get_session(**kwargs):
        return SimpleNamespace(id=kwargs.get("session_id"))

    async def drain(self, session_id: str) -> list[str]:
        return [
            e.content
            async for e in self.manager.process("hi", "u", session_id=session_id)
        ]


def _admission_harness():
    ctx = MockContext(google_api_key="test-key")
    ctx.__enter__()
    return _AdmissionHarness(ctx), ctx


class TestTurnAdmissionRechecksReadiness:
    async def test_turn_queued_behind_cleanup_reinitializes(self):
        """Cleanup lands between the turn's init and its admission."""
        h, ctx = _admission_harness()
        try:
            await h.manager.initialize_services(validate=False)
            assert h.inits == 1

            # Hold the turn lock so the turn queues, then clean up behind it.
            await h.manager._turn_lock.acquire()
            turn = asyncio.create_task(h.drain("sess-a"))
            await asyncio.sleep(0.05)
            assert not turn.done()

            # Release the manager's resources while the turn waits for
            # admission (the turn already passed _ensure_initialized).
            await h.manager._release_resources()
            assert h.manager._runner is None
            h.manager._turn_lock.release()

            assert await asyncio.wait_for(turn, timeout=2) == ["sess-a"]
            assert h.inits == 2, "the turn ran against the released backend"
            assert h.runner_at_stream[-1] is not None
            assert h.runner_at_stream[-1].name == "runner-2"
        finally:
            ctx.__exit__(None, None, None)

    async def test_turn_fails_cleanly_when_the_backend_cannot_be_revived(self):
        """No silent AttributeError on a ``None`` runner."""
        h, ctx = _admission_harness()
        try:
            await h.manager.initialize_services(validate=False)

            async def _do_nothing() -> None:
                self_inits = None  # noqa: F841 - deliberately leaves it unready
                return None

            await h.manager._turn_lock.acquire()
            turn = asyncio.create_task(h.drain("sess-a"))
            await asyncio.sleep(0.05)

            await h.manager._release_resources()
            h.manager._do_initialize = _do_nothing
            h.manager._turn_lock.release()

            with pytest.raises(RuntimeError, match="initiali"):
                await asyncio.wait_for(turn, timeout=2)
            assert h.streams == [], "the turn streamed from a released backend"
        finally:
            ctx.__exit__(None, None, None)

    async def test_resume_turn_rechecks_too(self):
        h, ctx = _admission_harness()
        try:
            await h.manager.initialize_services(validate=False)
            record = SimpleNamespace(
                job_id="j1", session_id="sess-r", user_id="u", call_id="c1",
                call_name="t", tool="t", state=SimpleNamespace(value="succeeded"),
                exit_code=0, error=None,
            )

            await h.manager._turn_lock.acquire()

            async def _drain_resume():
                return [
                    e.content
                    async for e in h.manager.resume_with_job_result(record, "ok")
                ]

            resume = asyncio.create_task(_drain_resume())
            await asyncio.sleep(0.05)

            await h.manager._release_resources()
            h.manager._turn_lock.release()

            assert await asyncio.wait_for(resume, timeout=2) == ["sess-r"]
            assert h.inits == 2
        finally:
            ctx.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# 2. Worker-thread service construction is transactional
# ---------------------------------------------------------------------------


class _SlowServiceManager(BaseWorkflowManager):
    """Builds services on a worker thread, slowly, and records the closes."""

    def __init__(self, settings, gate: threading.Event) -> None:
        super().__init__(agent_configs=[], settings=settings)
        self._gate = gate
        self.built: list[object] = []
        self._required_managers = {"sandbox_manager"}

    def _get_state_tools(self):
        return []

    @property
    def backend_type(self) -> str:
        return "test"

    async def _do_initialize(self) -> None:
        return None

    async def process(self, message, user_id, session_id=None):
        raise NotImplementedError

    async def reinitialize(self, model=None, preserve_sessions=True):
        return None

    async def cleanup(self):
        await self._release_resources()

    def _make_sandbox_manager(self):
        # Called on the worker thread; blocks until the test releases it.
        self._gate.wait(timeout=5)
        service = MagicMock()
        service.cleanup = MagicMock()
        self.built.append(service)
        return service


def _slow_manager(gate: threading.Event) -> _SlowServiceManager:
    settings = MagicMock()
    settings.app_name = "test-app"
    settings.google_api_key = None
    settings.anthropic_api_key = None
    manager = _SlowServiceManager(settings, gate)
    manager._model_registry = MagicMock(refresh=AsyncMock())
    return manager


class TestWorkerThreadConstructionIsTransactional:
    async def test_cancelled_init_does_not_publish_services(self, monkeypatch):
        gate = threading.Event()
        manager = _slow_manager(gate)
        monkeypatch.setattr(
            "agentic_cli.tools.sandbox.manager.SandboxManager",
            lambda settings: manager._make_sandbox_manager(),
        )

        task = asyncio.create_task(manager.initialize_services(validate=False))
        await asyncio.sleep(0.05)  # let the worker thread start and block
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        gate.set()  # the thread finishes *after* the rollback
        for _ in range(100):
            if manager.built:
                break
            await asyncio.sleep(0.02)

        assert manager.services.get(SANDBOX_MANAGER) is None, (
            "a cancelled initialization published services into a live manager"
        )
        assert manager.is_initialized is False

    async def test_services_built_after_cancellation_are_released(self, monkeypatch):
        gate = threading.Event()
        manager = _slow_manager(gate)
        monkeypatch.setattr(
            "agentic_cli.tools.sandbox.manager.SandboxManager",
            lambda settings: manager._make_sandbox_manager(),
        )

        task = asyncio.create_task(manager.initialize_services(validate=False))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        gate.set()
        for _ in range(100):
            if manager.built and manager.built[0].cleanup.called:
                break
            await asyncio.sleep(0.02)

        assert manager.built, "the worker thread never finished"
        manager.built[0].cleanup.assert_called_once()

    async def test_successful_init_publishes_normally(self, monkeypatch):
        gate = threading.Event()
        gate.set()
        manager = _slow_manager(gate)
        monkeypatch.setattr(
            "agentic_cli.tools.sandbox.manager.SandboxManager",
            lambda settings: manager._make_sandbox_manager(),
        )

        await manager.initialize_services(validate=False)

        assert manager.services.get(SANDBOX_MANAGER) is manager.built[0]
        assert manager.is_initialized is True
