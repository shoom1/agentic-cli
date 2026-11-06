"""Calls on one sandbox session run one at a time.

Both backends run a model step's synchronous calls in worker threads at once
(Deep Agents in LangChain's executor, ADK in its tool thread pool), so two
``sandbox_execute`` calls can reach one session together. A kernel runs one
cell at a time and its client is not thread-safe: two threads reading one
local kernel's output socket raised ``ValueError: b'<IDS|MSG>' is not in
list`` or took each other's output, and two first calls on the docker backend
both started a container under one name. ``SandboxManager`` holds a lock per
session across a call (staging, execution, output collection), and a reset or
cleanup that comes during a call leaves the session's teardown to that call.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from agentic_cli.tools.sandbox.backends.base import SandboxBackend
from agentic_cli.tools.sandbox.backends.detect import DockerAvailability
from agentic_cli.tools.sandbox.backends.jupyter_docker import JupyterDockerBackend
from agentic_cli.tools.sandbox.manager import SandboxManager
from agentic_cli.tools.sandbox.models import ExecutionResult
from tests.conftest import MockContext
from tests.tools.test_sandbox_docker_backend import FakeRuntime


class GatedBackend(SandboxBackend):
    """A backend whose calls run until the test opens ``gate``. It records the
    calls that entered, the most that ran at once in one session, and what
    was reset or cleaned up."""

    backend_name = "gated"

    def __init__(self, *, open_gate: bool = False) -> None:
        self.gate = threading.Event()
        if open_gate:
            self.gate.set()
        self.entered: list[str] = []
        self.most_at_once = 0
        self.reset_calls: list[str] = []
        self.cleanups = 0
        self._running: dict[str, int] = {}
        self._sessions: set[str] = set()
        self._count = threading.Lock()

    def execute(self, code, session_id, timeout_seconds=120, working_dir=None, inputs=None):
        with self._count:
            self._running[session_id] = self._running.get(session_id, 0) + 1
            self.most_at_once = max(self.most_at_once, self._running[session_id])
            self._sessions.add(session_id)
            self.entered.append(code)
        try:
            self.gate.wait(timeout=5)
            return ExecutionResult(success=True, stdout=f"{code}\n")
        finally:
            with self._count:
                self._running[session_id] -= 1

    def reset_session(self, session_id):
        self.reset_calls.append(session_id)
        self._sessions.discard(session_id)

    def cleanup(self):
        self.cleanups += 1
        self._sessions.clear()

    def has_session(self, session_id):
        return session_id in self._sessions


def until(condition, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


@pytest.fixture
def ctx():
    with MockContext(stateful_executor_backend="local") as context:
        yield context


def test_two_calls_on_one_local_session_each_get_their_own_output(ctx):
    from agentic_cli.tools.sandbox.backends.jupyter_local import JupyterLocalBackend

    mgr = SandboxManager(ctx.settings, backend=JupyterLocalBackend(interrupt_grace=1.0))
    try:
        assert mgr.execute("import time", session_id="s1").success  # the kernel is up
        with ThreadPoolExecutor(2) as pool:
            slow = pool.submit(
                mgr.execute,
                "for i in range(3):\n    print('A', i); time.sleep(0.3)\n'A-done'",
                session_id="s1", timeout_seconds=5,
            )
            time.sleep(0.2)
            quick = pool.submit(
                mgr.execute, "print('B'); 'B-done'", session_id="s1", timeout_seconds=5,
            )
            a, b = slow.result(), quick.result()
    finally:
        mgr.cleanup()

    assert (a.success, a.stdout, a.result, a.error) == (True, "A 0\nA 1\nA 2\n", "'A-done'", "")
    assert (b.success, b.stdout, b.result, b.error) == (True, "B\n", "'B-done'", "")


def test_calls_on_one_session_run_one_at_a_time(ctx):
    backend = GatedBackend()
    mgr = SandboxManager(ctx.settings, backend=backend)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(mgr.execute, "one", session_id="s1")
        assert until(lambda: backend.entered == ["one"])
        second = pool.submit(mgr.execute, "two", session_id="s1")
        time.sleep(0.2)  # the second call has reached the session
        backend.gate.set()
        results = [first.result(), second.result()]
    mgr.cleanup()

    assert backend.most_at_once == 1
    assert [r.stdout for r in results] == ["one\n", "two\n"]


def test_calls_on_two_sessions_run_side_by_side(ctx):
    backend = GatedBackend()
    mgr = SandboxManager(ctx.settings, backend=backend)
    with ThreadPoolExecutor(2) as pool:
        calls = [pool.submit(mgr.execute, code, session_id=code) for code in ("s1", "s2")]
        both_inside = until(lambda: len(backend.entered) == 2, timeout=2)
        backend.gate.set()
        assert all(call.result().success for call in calls)
    mgr.cleanup()

    assert both_inside


def test_calls_that_start_together_create_one_backend(ctx, monkeypatch):
    made: list[GatedBackend] = []

    def slow_backend(self, name):
        time.sleep(0.2)  # long enough for the other call to look too
        made.append(GatedBackend(open_gate=True))
        return made[-1]

    monkeypatch.setattr(SandboxManager, "_create_backend", slow_backend)
    mgr = SandboxManager(ctx.settings)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda sid: mgr.execute("x", session_id=sid), ["s1", "s2"]))
    mgr.cleanup()

    assert len(made) == 1
    assert all(r.success for r in results)


def test_max_sessions_holds_for_calls_that_start_together(monkeypatch):
    real_dir = SandboxManager._get_session_dir

    def slow_dir(self, session_id):
        time.sleep(0.2)  # between the count and the new session's entry
        return real_dir(self, session_id)

    monkeypatch.setattr(SandboxManager, "_get_session_dir", slow_dir)
    with MockContext(stateful_executor_backend="local", sandbox_max_sessions=1) as ctx:
        mgr = SandboxManager(ctx.settings, backend=GatedBackend(open_gate=True))
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda sid: mgr.execute("x", session_id=sid), ["s1", "s2"]))
        sessions = mgr.list_sessions()
        mgr.cleanup()

    assert sorted(r.success for r in results) == [False, True]
    assert "Maximum sessions (1) reached" in next(r.error for r in results if not r.success)
    assert len(sessions) == 1


def test_a_call_kept_waiting_past_its_timeout_is_told_the_session_is_busy(ctx):
    backend = GatedBackend()
    mgr = SandboxManager(ctx.settings, backend=backend)
    mgr.busy_margin = 0.0
    with ThreadPoolExecutor(1) as pool:
        first = pool.submit(mgr.execute, "one", session_id="s1")
        assert until(lambda: backend.entered == ["one"])
        started = time.monotonic()
        second = mgr.execute("two", session_id="s1", timeout_seconds=1)
        waited = time.monotonic() - started
        backend.gate.set()
        assert first.result().success
    mgr.cleanup()

    assert second.success is False
    assert "busy" in second.error and "'s1'" in second.error
    assert backend.entered == ["one"]
    assert 0.9 <= waited < 3


def test_a_timeout_past_what_a_lock_can_wait_still_runs(ctx):
    backend = GatedBackend(open_gate=True)
    mgr = SandboxManager(ctx.settings, backend=backend)
    result = mgr.execute("one", session_id="s1", timeout_seconds=10**12)
    mgr.cleanup()

    assert result.success, result.error


def test_a_reset_during_a_call_leaves_the_teardown_to_it(ctx):
    backend = GatedBackend()
    mgr = SandboxManager(ctx.settings, backend=backend)
    with ThreadPoolExecutor(1) as pool:
        running = pool.submit(mgr.execute, "one", session_id="s1")
        assert until(lambda: backend.entered == ["one"])
        started = time.monotonic()
        assert mgr.reset_session("s1") is True
        assert time.monotonic() - started < 0.5  # it never waits for the call
        assert backend.reset_calls == []  # nothing pulled out from under the call
        assert mgr.list_sessions() == []
        backend.gate.set()
        assert running.result().success
    assert backend.reset_calls == ["s1"]  # torn down as the call ended

    assert mgr.execute("two", session_id="s1").success  # a fresh session
    assert [row["execution_count"] for row in mgr.list_sessions()] == [1]
    mgr.cleanup()


def test_cleanup_during_a_call_leaves_its_session_to_it(ctx):
    backend = GatedBackend()
    mgr = SandboxManager(ctx.settings, backend=backend)
    with ThreadPoolExecutor(2) as pool:
        running = pool.submit(mgr.execute, "one", session_id="s1")
        assert until(lambda: backend.entered == ["one"])
        waiting = pool.submit(mgr.execute, "two", session_id="s1")
        time.sleep(0.2)  # the second call waits for the session
        started = time.monotonic()
        mgr.cleanup()
        assert time.monotonic() - started < 0.5
        assert (backend.reset_calls, backend.cleanups) == ([], 0)
        backend.gate.set()
        assert running.result().success
        refused = waiting.result()

    assert backend.reset_calls == ["s1"]
    # The waiting call starts no kernel that nothing would shut down.
    assert backend.entered == ["one"]
    assert refused.success is False and "shut down" in refused.error


def test_two_first_calls_on_docker_start_one_container():
    result = json.dumps({"type": "result", "success": True, "stdout": "", "stderr": "",
                         "result": None, "artifacts": [], "execution_time": 0.0, "error": ""})
    runtime = FakeRuntime()
    start = runtime.start

    def slow_start(spec):
        time.sleep(0.2)  # docker run takes a while
        handle = start(spec)
        for _ in range(2):
            handle.stdout.feed(result + "\n")
        return handle

    runtime.start = slow_start
    with MockContext(stateful_executor_backend="docker") as ctx:
        docker = JupyterDockerBackend(
            ctx.settings, runtime=runtime,
            detect_fn=lambda: DockerAvailability(True, "docker", "test"),
        )
        mgr = SandboxManager(ctx.settings, backend=docker)
        try:
            with ThreadPoolExecutor(2) as pool:
                results = list(pool.map(
                    lambda _: mgr.execute("x = 1", session_id="s1", timeout_seconds=5), range(2),
                ))
        finally:
            mgr.cleanup()

    assert len(runtime.started) == 1
    assert all(r.success for r in results)
