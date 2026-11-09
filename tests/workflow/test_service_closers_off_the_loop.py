"""A release closes the services on a worker thread, never on the event loop.

A sandbox's ``cleanup`` removes its containers (``docker rm -f``, through
``subprocess.run``) or shuts its kernels down, a job manager's ``close``
takes the jobs directory's lock, a knowledge base's ``close`` closes its
databases: each one blocks. A release ran them on the event loop, so a
cleanup (the app's exit, the old manager of an orchestrator swap, a model
switch that starts afresh, a failed start's rollback) froze the whole app,
Ctrl+C included, for as long as they took. They now run on a worker thread,
first in the release's closing task, before the backend's, the store's and
the catalog's closers: a cancel still stops only the wait. So do the
services a cancelled initialization's build thread finishes after the
cancel.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.service_registry import JOB_MANAGER, SANDBOX_MANAGER
from tests.conftest import MockContext, settings_double
from tests.loop_guard import no_blocking_calls


class _Blocking:
    """A service whose close blocks (a sleep), recording where it ran and in what order."""

    def __init__(self, name: str, closed: list[str], gate: threading.Event | None = None) -> None:
        self.name, self.closed, self.gate = name, closed, gate
        self.threads: list[threading.Thread] = []
        self.started = threading.Event()

    def _close(self) -> None:
        self.started.set()
        self.threads.append(threading.current_thread())
        if self.gate is not None:
            self.gate.wait(timeout=5)
        time.sleep(0.001)  # what docker rm -f, a kernel shutdown or a lock does
        self.closed.append(self.name)

    cleanup = close = _close


class _Closable:
    """A backend, store or catalog resource: closed asynchronously, on the loop."""

    def __init__(self, name: str, closed: list[str]) -> None:
        self.name, self.closed = name, closed

    async def aclose(self) -> None:
        self.closed.append(self.name)

    async def close(self) -> None:
        self.closed.append(self.name)

    async def __aexit__(self, *exc: object) -> None:
        self.closed.append(self.name)


class _HooksOnly(BaseWorkflowManager):
    """A backend written to the release hooks alone."""

    graph: Any = None
    store: Any = None

    @property
    def backend_type(self) -> str:
        return "hooks-only"

    def _get_state_tools(self) -> list[Callable]:
        return []

    async def _do_initialize(self) -> None:
        return None

    def _detach_backend(self):
        graph, self.graph = self.graph, None
        return [] if graph is None else [("graph", graph.aclose)]

    def _detach_store(self):
        store, self.store = self.store, None
        return [] if store is None else [("store", store.aclose)]

    async def process(self, message, user_id, session_id=None):
        raise NotImplementedError
        yield  # pragma: no cover


def _hooks_only(settings, backend, store) -> BaseWorkflowManager:
    manager = _HooksOnly([AgentConfig(name="a", prompt="p")], settings=settings)
    manager.graph, manager.store = backend, store
    return manager


def _adk(settings, backend, store) -> BaseWorkflowManager:
    pytest.importorskip("google.adk")
    from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager

    manager = GoogleADKWorkflowManager([AgentConfig(name="a", prompt="p")], settings=settings)
    manager._runner, manager._root_agent = backend, SimpleNamespace(name="a")
    manager._session_service = store
    return manager


def _deepagents(settings, backend, store) -> BaseWorkflowManager:
    pytest.importorskip("deepagents")
    from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager

    manager = DeepAgentsWorkflowManager([AgentConfig(name="a", prompt="p")], settings=settings)
    manager._mcp, manager._root = backend, SimpleNamespace(name="a")
    manager._checkpointer, manager._checkpointer_cm = object(), store
    manager._sessions = SimpleNamespace()
    return manager


@pytest.fixture(params=["hooks-only", "adk", "deepagents"])
def build(request) -> Callable[..., BaseWorkflowManager]:
    return {"hooks-only": _hooks_only, "adk": _adk, "deepagents": _deepagents}[request.param]


@pytest.fixture
def ctx():
    with MockContext(google_api_key="test-key") as context:
        yield context


class _Held:
    """One manager, ready, holding two blocking services, a backend, a store and a catalog."""

    def __init__(self, build, settings, gate: threading.Event | None = None) -> None:
        self.closed: list[str] = []
        self.sandbox = _Blocking("sandbox", self.closed, gate)
        self.jobs = _Blocking("jobs", self.closed)
        self.manager = build(
            settings, _Closable("backend", self.closed), _Closable("store", self.closed)
        )
        self.manager._catalog = _Closable("catalog", self.closed)
        self.manager._services = {SANDBOX_MANAGER: self.sandbox, JOB_MANAGER: self.jobs}
        self.manager._initialized = True

    def threads(self) -> list[threading.Thread]:
        return self.sandbox.threads + self.jobs.threads


EVERYTHING = ["sandbox", "jobs", "backend", "store", "catalog"]


class TestAReleaseClosesTheServicesOffTheLoop:
    async def test_a_cleanup(self, ctx, build):
        held = _Held(build, ctx.settings)

        with no_blocking_calls():
            await held.manager.cleanup()

        assert held.closed == EVERYTHING
        assert held.threads() and threading.main_thread() not in held.threads()
        assert held.manager.services == {}

    async def test_the_rollback_of_a_failed_start(self, ctx, build):
        held = _Held(build, ctx.settings)
        manager = held.manager
        services = dict(manager._services)
        manager._services = {}
        manager._initialized = False

        async def _load_models() -> None:
            return None

        async def _construct_services() -> None:
            manager._services.update(services)

        async def _do_initialize() -> None:
            raise RuntimeError("build failed")

        manager._load_models = _load_models
        manager._construct_services = _construct_services
        manager._do_initialize = _do_initialize
        manager._compute_budgets = lambda: {}

        with no_blocking_calls():
            with pytest.raises(RuntimeError, match="build failed"):
                await manager.initialize_services(validate=False)

        assert {"sandbox", "jobs"} <= set(held.closed)
        assert held.threads() and threading.main_thread() not in held.threads()
        assert manager.services == {}

    async def test_a_cancel_while_they_close_stops_only_the_wait(self, ctx, build):
        gate = threading.Event()
        held = _Held(build, ctx.settings, gate)

        task = asyncio.create_task(held.manager.cleanup())
        assert await asyncio.to_thread(held.sandbox.started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        gate.set()
        for _ in range(200):
            if held.closed == EVERYTHING:
                break
            await asyncio.sleep(0.01)

        assert held.closed == EVERYTHING
        assert held.manager.is_initialized is False
        assert held.manager.services == {}
        await held.manager.cleanup()
        assert held.closed == EVERYTHING


class _SlowBuild(BaseWorkflowManager):
    """Builds its sandbox on a worker thread, after the test lets it."""

    def __init__(self, settings, gate: threading.Event, closed: list[str]) -> None:
        super().__init__(agent_configs=[], settings=settings)
        self._gate = gate
        self.built: list[_Blocking] = []
        self.closed = closed
        self._required_managers = {"sandbox_manager"}

    @property
    def backend_type(self) -> str:
        return "test"

    def _get_state_tools(self) -> list[Callable]:
        return []

    async def _do_initialize(self) -> None:
        return None

    async def process(self, message, user_id, session_id=None):
        raise NotImplementedError
        yield  # pragma: no cover

    def make_sandbox(self) -> _Blocking:
        self._gate.wait(timeout=5)
        self.built.append(_Blocking("sandbox", self.closed))
        return self.built[-1]


async def test_what_a_cancelled_start_builds_after_the_cancel_closes_off_the_loop(monkeypatch):
    gate, closed = threading.Event(), []
    manager = _SlowBuild(settings_double(), gate, closed)
    monkeypatch.setattr(
        "agentic_cli.tools.sandbox.manager.SandboxManager",
        lambda settings: manager.make_sandbox(),
    )

    with no_blocking_calls():
        task = asyncio.create_task(manager.initialize_services(validate=False))
        await asyncio.sleep(0.05)  # the build thread waits at the gate
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        gate.set()
        for _ in range(200):
            if closed:
                break
            await asyncio.sleep(0.01)

    assert closed == ["sandbox"]
    assert threading.main_thread() not in manager.built[0].threads
    assert manager.services.get(SANDBOX_MANAGER) is None


def test_the_helper_that_builds_without_a_loop_still_closes_in_place(ctx):
    """``_cleanup_managers`` (LangGraph's release, and scripts) stays synchronous."""
    closed: list[str] = []
    manager = _hooks_only(ctx.settings, None, None)
    manager._services = {SANDBOX_MANAGER: _Blocking("sandbox", closed)}

    manager._cleanup_managers()

    assert closed == ["sandbox"]
    assert manager.services == {}
