"""Owned async resources are closed exactly once, on every shutdown path.

``cleanup()`` used to drop the session service by assignment. The durable
``DatabaseSessionService`` owns a SQLAlchemy engine with an async ``close()``,
so dropping the reference leaked its connection pool. Cleanup now awaits the
close contract, stays idempotent, and never closes a service it handed over
(``reinitialize(preserve_sessions=True)``).

The Deep Agents backend owns its checkpointer's connection on the same terms
(the classes at the end).
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
    mgr._catalog = None
    mgr._session_service_pinned = False
    mgr._carried_service_keys = frozenset()
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
    from agentic_cli.workflow.config import AgentConfig
    from tests.conftest import MockContext

    ctx = MockContext(google_api_key="test-key")
    ctx.__enter__()
    mgr = GoogleADKWorkflowManager(
        agent_configs=[AgentConfig(name="a", prompt="p")], settings=ctx.settings
    )
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

        from tests.conftest import settings_double

        mgr = _Manager(agent_configs=[], settings=settings_double(max_concurrent_jobs=2))
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

        with pytest.raises(RuntimeError, match="jobs blew up"):
            await mgr.initialize_services(validate=False)

        assert closed == ["sandbox"]
        assert mgr.services == {}
        assert mgr.is_initialized is False


def _raising_ctor(message: str):
    def _ctor(*args, **kwargs):
        raise RuntimeError(message)

    return _ctor


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


# ---------------------------------------------------------------------------
# Deep Agents: the checkpointer's connection is owned the same way
# ---------------------------------------------------------------------------


class _CountedConnection:
    """Stands in for the context manager that owns a saver's connection."""

    def __init__(self, fail: bool = False) -> None:
        self.closes = 0
        self.fail = fail

    async def __aexit__(self, *exc) -> None:
        self.closes += 1
        if self.fail:
            raise RuntimeError("close blew up")


class _CountedMCP:
    """Stands in for a manager's MCP sessions (whose ``aclose`` never raises)."""

    def __init__(self) -> None:
        self.closes = 0

    async def aclose(self) -> None:
        self.closes += 1


def _deepagents_manager(connection, mcp=None):
    """A ready Deep Agents manager holding ``connection``, without __init__."""
    pytest.importorskip("deepagents")
    from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager

    mgr = DeepAgentsWorkflowManager.__new__(DeepAgentsWorkflowManager)
    mgr._mcp = mcp or _CountedMCP()
    mgr._pending_notices = []
    mgr._settings = SimpleNamespace(app_name="test", default_user="u")
    mgr._app_name = "test"
    mgr._services = {}
    mgr._checkpointer = object()
    mgr._checkpointer_cm = connection
    mgr._checkpointer_pinned = False
    mgr._sessions = object()
    mgr._root = object()
    mgr._root_name = "a"
    mgr._initialized = True
    mgr._catalog = None
    mgr._carried_service_keys = frozenset()
    mgr._lifecycle_lock = asyncio.Lock()
    mgr._turn_lock = asyncio.Lock()
    return mgr


class TestDeepAgentsCheckpointerClose:
    async def test_cleanup_closes_the_connection(self):
        connection = _CountedConnection()
        mgr = _deepagents_manager(connection)

        await mgr.cleanup()

        assert connection.closes == 1
        assert (mgr._checkpointer, mgr._checkpointer_cm, mgr._sessions) == (None, None, None)
        assert mgr.is_initialized is False

    async def test_cleanup_is_idempotent(self):
        connection = _CountedConnection()
        mgr = _deepagents_manager(connection)

        await mgr.cleanup()
        await mgr.cleanup()

        assert connection.closes == 1

    async def test_an_in_memory_saver_has_nothing_to_close(self):
        mgr = _deepagents_manager(None)
        await mgr.cleanup()  # must not raise
        assert mgr._checkpointer is None

    async def test_a_failing_close_does_not_block_shutdown(self):
        closed: list[str] = []
        mgr = _deepagents_manager(_CountedConnection(fail=True))
        mgr._services = {"job_manager": SimpleNamespace(close=lambda: closed.append("jobs"))}

        await mgr.cleanup()  # swallowed and logged

        assert mgr._checkpointer is None
        assert closed == ["jobs"]


class TestDeepAgentsMCPClose:
    """The MCP sessions close on every release, a preserving one included."""

    async def test_cleanup_closes_them_with_the_checkpointer(self):
        connection, mcp = _CountedConnection(), _CountedMCP()
        mgr = _deepagents_manager(connection, mcp)

        await mgr.cleanup()

        assert (mcp.closes, connection.closes) == (1, 1)

    async def test_a_release_that_keeps_the_checkpointer_still_closes_them(self):
        connection, mcp = _CountedConnection(), _CountedMCP()
        mgr = _deepagents_manager(connection, mcp)
        mgr._checkpointer_pinned = True  # the rollback inside a model switch

        await mgr._release_resources()
        await mgr._release_resources(keep_checkpointer=True)

        assert (mcp.closes, connection.closes) == (2, 0)

    async def test_their_notices_go_with_them(self):
        mgr = _deepagents_manager(None)
        mgr._pending_notices = [object()]

        await mgr.cleanup()

        assert mgr._pending_notices == []


@pytest.fixture
def deepagents_manager(monkeypatch):
    """A Deep Agents manager built through ``__init__``, on counted connections.

    ``open_checkpointer`` hands out an in-memory saver with a counted
    connection each time it is called; ``state["boom"]`` makes building the
    agents fail, after the checkpointer was opened.
    """
    pytest.importorskip("deepagents")
    from langgraph.checkpoint.memory import InMemorySaver

    from agentic_cli.workflow.config import AgentConfig
    from agentic_cli.workflow.deepagents import manager as manager_module
    from tests.conftest import MockContext
    from tests.workflow.deepagents.fakes import ScriptedModels

    created: list[_CountedConnection] = []

    async def _open(kind, **kwargs):
        connection = _CountedConnection()
        created.append(connection)
        return InMemorySaver(), connection

    state = {"boom": False}
    real_build = manager_module.build_agent_tree

    def _build(*args, **kwargs):
        if state["boom"]:
            raise RuntimeError("backend init failed")
        return real_build(*args, **kwargs)

    monkeypatch.setattr(manager_module, "open_checkpointer", _open)
    monkeypatch.setattr(manager_module, "build_agent_tree", _build)
    monkeypatch.setattr(manager_module, "build_chat_model", ScriptedModels(a=[]))
    with MockContext(google_api_key="test-key") as ctx:
        mgr = manager_module.DeepAgentsWorkflowManager(
            agent_configs=[AgentConfig(name="a", prompt="p")], settings=ctx.settings
        )
        yield mgr, created, state


class TestDeepAgentsReinitializeTransaction:
    """The same transaction as ADK's, over the checkpointer."""

    async def test_preserved_checkpointer_is_reused_not_replaced(self, deepagents_manager):
        mgr, created, _state = deepagents_manager
        await mgr.initialize_services()
        saver = mgr._checkpointer

        await mgr.reinitialize(preserve_sessions=True)

        assert mgr._checkpointer is saver, "the live checkpointer was swapped"
        assert len(created) == 1, "a replacement was opened"
        assert created[0].closes == 0, "the preserved connection was closed"
        assert mgr.is_initialized is True
        await mgr.cleanup()

    async def test_discarded_checkpointer_is_closed_and_replaced(self, deepagents_manager):
        mgr, created, _state = deepagents_manager
        await mgr.initialize_services()
        saver = mgr._checkpointer

        await mgr.reinitialize(preserve_sessions=False)

        assert created[0].closes == 1
        assert mgr._checkpointer is not saver
        assert len(created) == 2
        await mgr.cleanup()

    async def test_failed_reinit_keeps_the_preserved_checkpointer(self, deepagents_manager):
        mgr, created, state = deepagents_manager
        await mgr.initialize_services()
        saver = mgr._checkpointer

        state["boom"] = True
        with pytest.raises(RuntimeError, match="backend init failed"):
            await mgr.reinitialize(preserve_sessions=True)

        assert created[0].closes == 0, "the preserved checkpointer was lost"
        assert mgr._checkpointer is saver
        assert mgr.is_initialized is False, "a failed reinit must not look ready"
        assert len(created) == 1
        await mgr.cleanup()
        assert created[0].closes == 1

    async def test_failed_reinit_closes_the_replacement_it_opened(self, deepagents_manager):
        mgr, created, state = deepagents_manager
        await mgr.initialize_services()

        state["boom"] = True
        with pytest.raises(RuntimeError):
            await mgr.reinitialize(preserve_sessions=False)

        assert created[0].closes == 1  # discarded on purpose
        assert len(created) == 2
        assert created[1].closes == 1, "the replacement leaked"
        assert mgr._checkpointer is None
        assert mgr.is_initialized is False

    async def test_a_failed_initialization_is_rolled_back(self, deepagents_manager):
        mgr, created, state = deepagents_manager
        state["boom"] = True

        with pytest.raises(RuntimeError, match="backend init failed"):
            await mgr.initialize_services()

        assert len(created) == 1
        assert created[0].closes == 1, "the checkpointer's connection leaked"
        assert mgr._checkpointer is None
        assert mgr.is_initialized is False
        assert mgr.services == {}

    async def test_it_can_be_initialized_after_a_failure(self, deepagents_manager):
        mgr, created, state = deepagents_manager
        state["boom"] = True
        with pytest.raises(RuntimeError):
            await mgr.initialize_services()

        state["boom"] = False
        await mgr.initialize_services()

        assert mgr.is_initialized is True
        assert len(created) == 2 and created[1].closes == 0
        await mgr.cleanup()
