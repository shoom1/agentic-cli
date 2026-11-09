"""One lifecycle, in the base: cleanup, reinitialize and the release template.

``BaseWorkflowManager`` owns the lifecycle: ``cleanup()`` and
``reinitialize()`` are the base's, and ``_release_resources()`` resets the
base's state, asks the backend for what it holds (``_detach_backend``, and
``_detach_store`` unless a model switch keeps the store) and closes it all in
one task of its own, awaited under a shield. Three things follow, each once
for every backend:

- A backend that implements only the hooks works: a cleanup leaves it
  uninitialized, it initializes again fully, and a turn after a cleanup
  initializes it again instead of running on released resources.
- A cancel during a release (Ctrl+C during a ``/settings`` model switch, a
  cancelled ``async with manager``) stops only the wait: every resource the
  manager owned is still closed, exactly once. Deep Agents' own copy awaited
  its closes unshielded, so a cancel left its checkpointer's connection, the
  catalog and the services open.
- A preserving switch keeps the same store, after a successful and after a
  failed attempt, by one mechanism: nothing has to restore it.

Offline: catalogs never list (``tests/conftest.py``), and no model is called.
"""

from __future__ import annotations

import asyncio
import functools
import gc
import warnings
from collections import Counter
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.config import AgentConfig, AgentGraph, validate_agent_graph
from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.service_registry import (
    JOB_MANAGER,
    PERMISSION_ENGINE,
    SANDBOX_MANAGER,
)
from agentic_cli.workflow.sessions import SessionRef
from tests.conftest import MockContext

# ---------------------------------------------------------------------------
# A backend written to the hooks alone
# ---------------------------------------------------------------------------


class _HooksOnly(BaseWorkflowManager):
    """A backend with a graph and a store, and no cleanup or reinitialize of its own."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.builds = 0
        self.stores_opened = 0
        self.closed: list[str] = []
        self.graph: object | None = None
        self.store: object | None = None

    @property
    def backend_type(self) -> str:
        return "hooks-only"

    def _get_state_tools(self) -> list[Callable]:
        return []

    def _validate_agent_graph(self) -> AgentGraph:
        return validate_agent_graph(self._agent_configs, backend=self.backend_type)

    async def _do_initialize(self) -> None:
        self.builds += 1
        if self.store is None:
            self.stores_opened += 1
            self.store = SimpleNamespace(number=self.stores_opened)
        self.graph = SimpleNamespace(number=self.builds)
        # What the base resets on every release.
        self._budgets = {self._root_agent_name: object()}
        self._index_snapshots[SessionRef(self.app_name, "u", "s")] = "the index"

    def _backend_ready(self) -> bool:
        return self._initialized and self.graph is not None and self.store is not None

    def _detach_backend(self):
        graph, self.graph = self.graph, None
        if graph is None:
            return []
        return [("graph", functools.partial(self.closed.append, f"graph {graph.number}"))]

    def _detach_store(self):
        store, self.store = self.store, None
        if store is None:
            return []
        return [("store", functools.partial(self.closed.append, f"store {store.number}"))]

    async def process(self, message, user_id, session_id=None):
        async with self._turn_admission():
            with self._workflow_context(session_id=session_id, user_id=user_id):
                yield WorkflowEvent.text(f"graph {self.graph.number}", session_id)


@pytest.fixture
def ctx():
    with MockContext(google_api_key="test-key") as context:
        yield context


def _hooks_only(ctx) -> _HooksOnly:
    return _HooksOnly([AgentConfig(name="a", prompt="p")], settings=ctx.settings)


async def _reply(manager: BaseWorkflowManager) -> list[str]:
    return [event.content async for event in manager.process("hi", "u", "s")]


class TestABackendOfHooksOnly:
    async def test_cleanup_leaves_it_uninitialized_and_released(self, ctx):
        manager = _hooks_only(ctx)
        await manager.initialize_services()
        assert manager._budgets and manager._index_snapshots

        await manager.cleanup()

        assert manager.is_initialized is False
        assert manager._budgets == {}
        assert manager._index_snapshots == {}
        assert manager.catalog is None
        assert manager.services == {}
        assert manager.closed == ["graph 1", "store 1"]

    async def test_it_initializes_again_fully(self, ctx):
        manager = _hooks_only(ctx)
        await manager.initialize_services()
        await manager.cleanup()

        await manager.initialize_services()

        assert manager.builds == 2
        assert manager.is_initialized
        assert manager.catalog is not None
        assert PERMISSION_ENGINE in manager.services
        await manager.cleanup()

    async def test_a_turn_after_cleanup_initializes_it_again(self, ctx):
        manager = _hooks_only(ctx)
        assert await _reply(manager) == ["graph 1"]
        await manager.cleanup()

        assert await _reply(manager) == ["graph 2"]
        assert manager.builds == 2
        await manager.cleanup()

    async def test_cleanup_is_idempotent(self, ctx):
        manager = _hooks_only(ctx)
        await manager.initialize_services()

        await manager.cleanup()
        await manager.cleanup()

        assert manager.closed == ["graph 1", "store 1"]

    async def test_a_model_switch_keeps_the_store_and_the_services(self, ctx):
        manager = _hooks_only(ctx)
        await manager.initialize_services()
        store, engine = manager.store, manager.services[PERMISSION_ENGINE]

        await manager.reinitialize(preserve_sessions=True)

        assert manager.store is store
        assert manager.services[PERMISSION_ENGINE] is engine
        assert manager.closed == ["graph 1"]
        assert await _reply(manager) == ["graph 2"]
        await manager.cleanup()
        assert manager.closed == ["graph 1", "graph 2", "store 1"]

    async def test_a_switch_that_does_not_preserve_replaces_the_store(self, ctx):
        manager = _hooks_only(ctx)
        await manager.initialize_services()
        engine = manager.services[PERMISSION_ENGINE]

        await manager.reinitialize(preserve_sessions=False)

        assert manager.closed == ["graph 1", "store 1"]
        assert manager.store.number == 2
        assert manager.services[PERMISSION_ENGINE] is not engine
        await manager.cleanup()

    async def test_a_failed_switch_keeps_the_store_for_a_retry(self, ctx):
        manager = _hooks_only(ctx)
        await manager.initialize_services()
        store = manager.store

        async def _fails() -> None:
            raise RuntimeError("build failed")

        manager._do_initialize = _fails
        with pytest.raises(RuntimeError, match="build failed"):
            await manager.reinitialize(preserve_sessions=True)

        assert manager.is_initialized is False
        assert manager.store is store
        del manager._do_initialize
        await manager.reinitialize(preserve_sessions=True)
        assert manager.store is store and manager.stores_opened == 1
        await manager.cleanup()


class TestTheBaseReleaseResetsTheBaseState:
    """A backend that keeps a ``cleanup`` of its own and calls the base
    release (as the docstring once showed) is left uninitialized by it."""

    async def test_its_own_cleanup_through_the_base_release(self, ctx):
        class OwnCleanup(_HooksOnly):
            async def reinitialize(self, model=None, preserve_sessions=True) -> None:
                raise NotImplementedError

            async def cleanup(self) -> None:
                async with self._lifecycle_lock:
                    async with self._turn_lock:
                        await self._release_resources()

        manager = OwnCleanup([AgentConfig(name="a", prompt="p")], settings=ctx.settings)
        await manager.initialize_services()

        await manager.cleanup()

        assert manager.is_initialized is False
        assert (manager._budgets, manager._index_snapshots) == ({}, {})
        assert await _reply(manager) == ["graph 2"]
        await manager.cleanup()


class TestAHookMustBeSynchronous:
    async def test_an_async_hook_fails_the_release_loudly(self, ctx):
        """A hook written ``async def`` by mistake detaches nothing (its body
        never runs): the release says so, and still releases the rest."""

        class AsyncHook(_HooksOnly):
            async def _detach_backend(self):  # the mistake
                return super()._detach_backend()

        manager = AsyncHook([AgentConfig(name="a", prompt="p")], settings=ctx.settings)
        await manager.initialize_services()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(TypeError, match=r"_detach_backend must be synchronous"):
                await manager.cleanup()
            gc.collect()

        assert [str(w.message) for w in caught if "never awaited" in str(w.message)] == []
        assert manager.is_initialized is False
        assert manager.closed == ["store 1"]
        assert manager.catalog is None and manager.services == {}


# ---------------------------------------------------------------------------
# A cancel during a release closes everything anyway, on every backend
# ---------------------------------------------------------------------------


class _Gate:
    """Holds the one slow close until the test lets it finish."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()


class _Resource:
    """An owned resource with every async close protocol a backend uses."""

    def __init__(self, closes: Counter, name: str, gate: _Gate | None) -> None:
        self.closes, self.name, self.gate = closes, name, gate

    async def _close(self) -> None:
        if self.gate is not None:
            self.gate.started.set()
            await self.gate.release.wait()
        self.closes[self.name] += 1

    async def aclose(self) -> None:
        await self._close()

    async def __aexit__(self, *exc: object) -> None:
        await self._close()


class _Service:
    """A service, closed synchronously (the sandbox's ``cleanup``, the jobs' ``close``)."""

    def __init__(self, closes: Counter, name: str) -> None:
        self.closes, self.name = closes, name

    def cleanup(self) -> None:
        self.closes[self.name] += 1

    close = cleanup


def _adk_manager(settings) -> BaseWorkflowManager:
    pytest.importorskip("google.adk")
    from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager

    return GoogleADKWorkflowManager([AgentConfig(name="a", prompt="p")], settings=settings)


def _adk_hold(manager, backend: _Resource, store: _Resource) -> None:
    manager._runner, manager._root_agent = backend, SimpleNamespace(name="a")
    manager._session_service = store


def _deepagents_manager(settings) -> BaseWorkflowManager:
    pytest.importorskip("deepagents")
    from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager

    return DeepAgentsWorkflowManager([AgentConfig(name="a", prompt="p")], settings=settings)


def _deepagents_hold(manager, backend: _Resource, store: _Resource) -> None:
    manager._mcp, manager._root = backend, SimpleNamespace(name="a")
    manager._checkpointer, manager._checkpointer_cm = object(), store
    manager._sessions = SimpleNamespace()


#: Each backend: how to build it, and how it holds its backend resource (the
#: runner, the MCP sessions) and its store (the session service, the
#: checkpointer's connections).
_BACKENDS = {
    "adk": (_adk_manager, _adk_hold, lambda m: m._session_service),
    "deepagents": (_deepagents_manager, _deepagents_hold, lambda m: m._checkpointer_cm),
}

_EVERYTHING = {"backend": 1, "store": 1, "catalog": 1, "sandbox": 1, "jobs": 1}


@pytest.fixture(params=sorted(_BACKENDS))
def backend(request):
    return _BACKENDS[request.param]


class _Owned:
    """The resources one manager owns, counted, with one of them slow to close."""

    def __init__(self, slow: str) -> None:
        self.closes: Counter = Counter()
        self.gate = _Gate()
        self.slow = slow

    def resource(self, name: str) -> _Resource:
        return _Resource(self.closes, name, self.gate if name == self.slow else None)

    def services(self) -> dict[str, _Service]:
        return {
            SANDBOX_MANAGER: _Service(self.closes, "sandbox"),
            JOB_MANAGER: _Service(self.closes, "jobs"),
        }

    async def cancel_during_the_slow_close(self, task: asyncio.Task) -> None:
        await asyncio.wait_for(self.gate.started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        self.gate.release.set()

    async def settled(self, expected: dict[str, int]) -> dict[str, int]:
        """The closes once they reach ``expected``, or after two seconds."""
        for _ in range(200):
            if dict(self.closes) == expected:
                break
            await asyncio.sleep(0.01)
        return dict(self.closes)


class TestACancelDuringARelease:
    @pytest.mark.parametrize("slow", ["backend", "store"])
    async def test_cleanup_still_closes_everything_once(self, ctx, backend, slow):
        build, hold, _store_of = backend
        owned = _Owned(slow)
        manager = build(ctx.settings)
        hold(manager, owned.resource("backend"), owned.resource("store"))
        manager._catalog = owned.resource("catalog")
        manager._services = owned.services()
        manager._initialized = True

        await owned.cancel_during_the_slow_close(asyncio.create_task(manager.cleanup()))

        assert await owned.settled(_EVERYTHING) == _EVERYTHING
        assert manager.is_initialized is False
        assert manager.catalog is None and manager.services == {}
        await manager.cleanup()
        assert dict(owned.closes) == _EVERYTHING

    async def test_a_model_switch_still_closes_what_it_does_not_keep(self, ctx, backend):
        """Ctrl+C during a ``/settings`` model switch's release."""
        build, hold, store_of = backend
        owned = _Owned("backend")
        manager = build(ctx.settings)
        store = owned.resource("store")
        hold(manager, owned.resource("backend"), store)
        manager._catalog = owned.resource("catalog")
        manager._services = services = owned.services()
        manager._initialized = True

        await owned.cancel_during_the_slow_close(
            asyncio.create_task(manager.reinitialize(preserve_sessions=True))
        )

        released = {"backend": 1, "catalog": 1}
        assert await owned.settled(released) == released
        assert manager.is_initialized is False
        assert manager.catalog is None
        # What the switch keeps stays with the manager, for a retry.
        assert store_of(manager) is store
        assert manager.services == services
        await manager.cleanup()
        assert dict(owned.closes) == _EVERYTHING

    @pytest.mark.parametrize("slow", ["backend", "store"])
    async def test_the_rollback_of_a_failed_switch_still_closes_everything_once(
        self, ctx, backend, slow
    ):
        build, hold, _store_of = backend
        owned = _Owned(slow)
        manager = build(ctx.settings)

        async def _load_models() -> None:
            manager._catalog = owned.resource("catalog")

        async def _construct_services() -> None:
            manager._services.update(owned.services())

        async def _do_initialize() -> None:
            hold(manager, owned.resource("backend"), owned.resource("store"))
            raise RuntimeError("build failed")

        manager._load_models = _load_models
        manager._construct_services = _construct_services
        manager._do_initialize = _do_initialize

        await owned.cancel_during_the_slow_close(
            asyncio.create_task(manager.reinitialize(preserve_sessions=False))
        )

        assert await owned.settled(_EVERYTHING) == _EVERYTHING
        assert manager.is_initialized is False
        assert manager.catalog is None and manager.services == {}
        await manager.cleanup()
        assert dict(owned.closes) == _EVERYTHING


# ---------------------------------------------------------------------------
# What a preserving switch keeps: the same store, whatever the outcome
# ---------------------------------------------------------------------------


def _adk_store(manager) -> tuple:
    return (manager._session_service,)


def _deepagents_store(manager) -> tuple:
    return (manager._checkpointer, manager._checkpointer_cm, manager._sessions)


@pytest.fixture
def hermetic(tmp_path, monkeypatch):
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    with MockContext(google_api_key="test-key", session_store="sqlite") as context:
        yield context.settings


@pytest.fixture(params=["adk", "deepagents"])
def real_manager(request, hermetic, monkeypatch):
    """A real manager of each backend, initialized offline."""
    if request.param == "adk":
        return _adk_manager(hermetic), _adk_store
    manager = _deepagents_manager(hermetic)
    from agentic_cli.workflow.deepagents import manager as manager_module
    from tests.workflow.deepagents.fakes import ScriptedModels

    monkeypatch.setattr(manager_module, "build_chat_model", ScriptedModels(a=[]))
    return manager, _deepagents_store


def _same(left: tuple, right: tuple) -> bool:
    return len(left) == len(right) and all(a is b for a, b in zip(left, right))


class TestWhatAPreservingSwitchKeeps:
    async def test_the_same_store_after_a_switch_that_succeeds(self, real_manager):
        manager, store_of = real_manager
        await manager.initialize_services()
        store = store_of(manager)
        assert all(part is not None for part in store)

        await manager.reinitialize(preserve_sessions=True)

        assert _same(store_of(manager), store)
        await manager.cleanup()

    async def test_the_same_store_after_a_switch_that_fails(self, real_manager):
        """The rollback inside the switch already leaves the store in place:
        nothing has to put it back after."""
        manager, store_of = real_manager
        await manager.initialize_services()
        store = store_of(manager)
        real_initialize = manager._initialize_locked
        at_the_rollback: list[bool] = []

        async def _fails() -> None:
            raise RuntimeError("build failed")

        async def _watched(validate: bool = True) -> None:
            try:
                await real_initialize(validate)
            except BaseException:
                at_the_rollback.append(_same(store_of(manager), store))
                raise

        manager._initialize_locked = _watched
        manager._do_initialize = _fails
        with pytest.raises(RuntimeError, match="build failed"):
            await manager.reinitialize(preserve_sessions=True)

        assert at_the_rollback == [True]
        assert _same(store_of(manager), store)
        assert manager.is_initialized is False
        del manager._do_initialize, manager._initialize_locked
        await manager.reinitialize(preserve_sessions=True)
        assert _same(store_of(manager), store)
        await manager.cleanup()
