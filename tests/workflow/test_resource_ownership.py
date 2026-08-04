"""Owned async resources are closed exactly once, on every shutdown path.

``cleanup()`` used to drop the session service by assignment. The durable
``DatabaseSessionService`` owns a SQLAlchemy engine with an async ``close()``,
so dropping the reference leaked its connection pool. Cleanup now awaits the
close contract, stays idempotent, and never closes a service it handed over
(``reinitialize(preserve_sessions=True)``).
"""

from __future__ import annotations

import asyncio

from types import SimpleNamespace

import pytest

pytest.importorskip("google.adk")

from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402


class _AsyncClosable:
    """Stand-in for DatabaseSessionService: async close(), counted."""

    def __init__(self) -> None:
        self.closes = 0

    async def close(self) -> None:
        self.closes += 1


class _SyncClosable:
    def __init__(self) -> None:
        self.closes = 0

    def close(self) -> None:
        self.closes += 1


class _NotClosable:
    """Stand-in for InMemorySessionService: nothing to release."""


class _FailingClosable:
    async def close(self) -> None:
        raise RuntimeError("close blew up")


def _manager(session_service) -> GoogleADKWorkflowManager:
    mgr = GoogleADKWorkflowManager.__new__(GoogleADKWorkflowManager)
    mgr._settings = SimpleNamespace(app_name="test", default_user="u")
    mgr._app_name = "test"
    mgr._services = {}
    mgr._session_service = session_service
    mgr._runner = object()
    mgr._root_agent = object()
    mgr._initialized = True
    mgr._llm_logging_plugin = None
    mgr._model = "gemini-2.5-flash"
    mgr._model_resolved = True
    mgr._session_service_pinned = False
    mgr._lifecycle_lock = asyncio.Lock()
    mgr._turn_lock = asyncio.Lock()
    return mgr


class TestSessionServiceClose:
    async def test_cleanup_awaits_async_close(self):
        service = _AsyncClosable()
        mgr = _manager(service)

        await mgr.cleanup()

        assert service.closes == 1
        assert mgr._session_service is None
        assert mgr.is_initialized is False

    async def test_cleanup_is_idempotent(self):
        service = _AsyncClosable()
        mgr = _manager(service)

        await mgr.cleanup()
        await mgr.cleanup()

        assert service.closes == 1

    async def test_service_without_close_is_tolerated(self):
        mgr = _manager(_NotClosable())
        await mgr.cleanup()  # must not raise
        assert mgr._session_service is None

    async def test_failing_close_does_not_block_shutdown(self):
        mgr = _manager(_FailingClosable())
        await mgr.cleanup()  # swallowed and logged
        assert mgr._session_service is None

    async def test_sync_close_is_supported(self):
        service = _SyncClosable()
        mgr = _manager(service)
        await mgr.cleanup()
        assert service.closes == 1


def _real_manager(monkeypatch=None):
    """A manager built through ``__init__`` with the network stubbed out.

    ``_do_initialize`` is replaced by a faithful stand-in: it creates the
    session service only when one is not already held (which is exactly how
    ``reinitialize(preserve_sessions=True)`` avoids building a replacement).
    """
    from unittest.mock import AsyncMock, MagicMock

    from agentic_cli.workflow.config import AgentConfig
    from tests.conftest import MockContext

    ctx = MockContext(google_api_key="test-key")
    ctx.__enter__()
    mgr = GoogleADKWorkflowManager(
        agent_configs=[AgentConfig(name="a", prompt="p")], settings=ctx.settings
    )
    mgr._model_registry = MagicMock(refresh=AsyncMock(), discovery_complete=False)
    mgr._ensure_managers_initialized = lambda: None

    created: list[_AsyncClosable] = []

    def _make_service():
        service = _AsyncClosable()
        created.append(service)
        return service

    mgr._make_session_service = _make_service
    state = {"boom": False}

    async def _do_init():
        if mgr._session_service is None:
            mgr._session_service = mgr._make_session_service()
        if state["boom"]:
            raise RuntimeError("backend init failed")
        mgr._runner = object()
        mgr._root_agent = object()

    mgr._do_initialize = _do_init
    return mgr, created, state, ctx


class TestReinitializeTransaction:
    """reinitialize() either fully succeeds or leaves nothing half-built."""

    async def test_preserved_service_is_reused_not_replaced(self):
        mgr, created, _state, ctx = _real_manager()
        try:
            await mgr.initialize_services()
            original = mgr._session_service
            assert len(created) == 1

            await mgr.reinitialize(preserve_sessions=True)

            assert mgr._session_service is original, "the live service was swapped"
            assert len(created) == 1, "a replacement service was built and discarded"
            assert original.closes == 0, "the preserved service was closed"
            assert mgr.is_initialized is True
        finally:
            ctx.__exit__(None, None, None)

    async def test_discarded_service_is_closed_and_replaced(self):
        mgr, created, _state, ctx = _real_manager()
        try:
            await mgr.initialize_services()
            original = mgr._session_service

            await mgr.reinitialize(preserve_sessions=False)

            assert original.closes == 1
            assert mgr._session_service is not original
            assert len(created) == 2
        finally:
            ctx.__exit__(None, None, None)

    async def test_failed_reinit_keeps_preserved_service_and_uninitializes(self):
        mgr, created, state, ctx = _real_manager()
        try:
            await mgr.initialize_services()
            original = mgr._session_service

            state["boom"] = True
            with pytest.raises(RuntimeError, match="backend init failed"):
                await mgr.reinitialize(preserve_sessions=True)

            assert original.closes == 0, "the preserved service was lost"
            assert mgr._session_service is original
            assert mgr.is_initialized is False, "a failed reinit must not look ready"
            assert len(created) == 1
        finally:
            ctx.__exit__(None, None, None)

    async def test_failed_reinit_closes_the_replacement_it_created(self):
        """preserve_sessions=False: the new service must not leak on failure."""
        mgr, created, state, ctx = _real_manager()
        try:
            await mgr.initialize_services()
            original = mgr._session_service

            state["boom"] = True
            with pytest.raises(RuntimeError):
                await mgr.reinitialize(preserve_sessions=False)

            assert original.closes == 1  # discarded on purpose
            assert len(created) == 2
            assert created[1].closes == 1, "the replacement service leaked"
            assert mgr._session_service is None
            assert mgr.is_initialized is False
        finally:
            ctx.__exit__(None, None, None)


class TestDirectInitializationFailure:
    """A direct initialize_services() failure rolls its own resources back."""

    async def test_partial_initialization_is_rolled_back(self):
        mgr, created, state, ctx = _real_manager()
        try:
            state["boom"] = True
            with pytest.raises(RuntimeError, match="backend init failed"):
                await mgr.initialize_services()

            assert len(created) == 1
            assert created[0].closes == 1, "the session service leaked"
            assert mgr._session_service is None
            assert mgr.is_initialized is False
            assert mgr.services == {}
        finally:
            ctx.__exit__(None, None, None)

    async def test_manager_can_be_initialized_after_a_failure(self):
        mgr, created, state, ctx = _real_manager()
        try:
            state["boom"] = True
            with pytest.raises(RuntimeError):
                await mgr.initialize_services()

            state["boom"] = False
            await mgr.initialize_services()

            assert mgr.is_initialized is True
            assert mgr._session_service is created[-1]
        finally:
            ctx.__exit__(None, None, None)


class TestLifecycleSerialization:
    """close() and reinitialize() must not interleave."""

    async def test_concurrent_cleanup_and_reinitialize(self):
        mgr, created, _state, ctx = _real_manager()
        try:
            await mgr.initialize_services()

            order: list[str] = []
            real_do_init = mgr._do_initialize

            async def _slow_init():
                order.append("reinit-start")
                await asyncio.sleep(0.02)
                await real_do_init()
                order.append("reinit-end")

            mgr._do_initialize = _slow_init

            async def _cleanup():
                await asyncio.sleep(0.005)
                order.append("cleanup-start")
                await mgr.cleanup()
                order.append("cleanup-end")

            await asyncio.gather(mgr.reinitialize(preserve_sessions=True), _cleanup())

            # cleanup must wait for the whole reinit, never interleave with it
            assert order.index("reinit-end") < order.index("cleanup-end")
            assert mgr.is_initialized is False  # cleanup ran last
        finally:
            ctx.__exit__(None, None, None)


class TestOwnedServicesReleased:
    """_cleanup_managers releases only services this manager created."""

    async def test_job_manager_and_sandbox_are_closed(self):
        closed: list[str] = []
        mgr = _manager(_NotClosable())
        mgr._services = {
            "job_manager": SimpleNamespace(close=lambda: closed.append("jobs")),
            "sandbox_manager": SimpleNamespace(cleanup=lambda: closed.append("sandbox")),
        }

        await mgr.cleanup()

        assert sorted(closed) == ["jobs", "sandbox"]
        assert mgr._services == {}

    async def test_second_cleanup_finds_nothing_to_close(self):
        closed: list[str] = []
        mgr = _manager(_NotClosable())
        mgr._services = {"job_manager": SimpleNamespace(close=lambda: closed.append("jobs"))}

        await mgr.cleanup()
        await mgr.cleanup()

        assert closed == ["jobs"]


class TestPartialConstructionRollsBack:
    """A service constructor that raises must not strand its predecessors.

    ``_build_services`` builds into a local dict and hands it to the caller to
    publish. When a *later* constructor raised, that dict was simply dropped —
    so an already-built SandboxManager (a container/process pool) or JobManager
    (a thread pool) was never published and never closed: nothing could ever
    release it.
    """

    @staticmethod
    def _manager_needing(*services: str):
        from unittest.mock import MagicMock

        from agentic_cli.workflow.base_manager import BaseWorkflowManager

        class _Manager(BaseWorkflowManager):
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

        settings = MagicMock()
        settings.app_name = "test-app"
        settings.max_concurrent_jobs = 2
        mgr = _Manager(agent_configs=[], settings=settings)
        mgr._required_managers = set(services)
        return mgr

    def test_sandbox_is_released_when_a_later_constructor_raises(self, monkeypatch):
        sandbox = SimpleNamespace(cleanup=lambda: closed.append("sandbox"))
        closed: list[str] = []

        monkeypatch.setattr(
            "agentic_cli.tools.sandbox.manager.SandboxManager",
            lambda settings: sandbox,
        )
        monkeypatch.setattr(
            "agentic_cli.tools.jobs.JobManager",
            _raising_ctor("jobs blew up"),
        )

        mgr = self._manager_needing("sandbox_manager", "job_manager")
        with pytest.raises(RuntimeError, match="jobs blew up"):
            mgr._build_services()

        assert closed == ["sandbox"], "an already-built service was stranded"

    def test_job_manager_is_released_when_a_later_constructor_raises(
        self, monkeypatch
    ):
        closed: list[str] = []
        jobs = SimpleNamespace(close=lambda: closed.append("jobs"))

        monkeypatch.setattr("agentic_cli.tools.jobs.JobManager", lambda *a, **k: jobs)
        monkeypatch.setattr(
            "agentic_cli.tools.arxiv_source.ArxivSearchSource",
            _raising_ctor("arxiv blew up"),
        )

        mgr = self._manager_needing("job_manager", "arxiv_source")
        with pytest.raises(RuntimeError, match="arxiv blew up"):
            mgr._build_services()

        assert closed == ["jobs"]

    async def test_initialization_failure_leaves_nothing_published(
        self, monkeypatch
    ):
        closed: list[str] = []
        monkeypatch.setattr(
            "agentic_cli.tools.sandbox.manager.SandboxManager",
            lambda settings: SimpleNamespace(cleanup=lambda: closed.append("sandbox")),
        )
        monkeypatch.setattr(
            "agentic_cli.tools.jobs.JobManager", _raising_ctor("jobs blew up")
        )

        mgr = self._manager_needing("sandbox_manager", "job_manager")
        mgr._model_registry = SimpleNamespace(refresh=_noop_refresh)

        with pytest.raises(RuntimeError, match="jobs blew up"):
            await mgr.initialize_services(validate=False)

        assert closed == ["sandbox"]
        assert mgr.services == {}
        assert mgr.is_initialized is False


def _raising_ctor(message: str):
    def _ctor(*args, **kwargs):
        raise RuntimeError(message)

    return _ctor


async def _noop_refresh(**kwargs):
    return None


class TestCloserIsolation:
    """One resource's close failure must not skip the others."""

    async def test_failing_sync_closer_does_not_block_the_rest(self):
        closed: list[str] = []

        def _boom():
            raise RuntimeError("sandbox cleanup blew up")

        mgr = _manager(_AsyncClosable())
        mgr._services = {
            "sandbox_manager": SimpleNamespace(cleanup=_boom),
            "job_manager": SimpleNamespace(close=lambda: closed.append("jobs")),
        }

        await mgr.cleanup()

        assert closed == ["jobs"], "a failing closer skipped the next resource"
        assert mgr._services == {}

    async def test_failing_sync_closer_does_not_block_the_session_service(self):
        service = _AsyncClosable()
        mgr = _manager(service)
        mgr._services = {
            "sandbox_manager": SimpleNamespace(
                cleanup=lambda: (_ for _ in ()).throw(RuntimeError("boom"))
            )
        }

        await mgr.cleanup()

        assert service.closes == 1
