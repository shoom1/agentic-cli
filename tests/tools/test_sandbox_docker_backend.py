"""Tests for JupyterDockerBackend with a fake runtime + detect (no docker)."""

import json

import pytest

from agentic_cli.tools.sandbox.backends.detect import DockerAvailability
from agentic_cli.tools.sandbox.backends.container_runtime import ContainerSpec
from agentic_cli.tools.sandbox.backends.jupyter_docker import JupyterDockerBackend
from tests.conftest import MockContext
from tests.tools.test_sandbox_container_session import FakeHandle


class FakeRuntime:
    def __init__(self):
        self.started: list[ContainerSpec] = []
        self.killed: list[tuple[str, str | None]] = []
        self._handles: list[FakeHandle] = []
    def start(self, spec: ContainerSpec):
        self.started.append(spec)
        h = FakeHandle()
        h.name = spec.name
        h.stdout.feed(json.dumps({"type": "ready"}) + "\n")
        self._handles.append(h)
        return h
    def kill(self, name, signal=None):
        self.killed.append((name, signal))
        for h in self._handles:
            if h.name == name:
                h._dead = True
                h.stdout.eof()
    def inspect(self, name):
        return {}
    def remove(self, name):
        pass


def _backend(available=True):
    ctx = MockContext(sandbox_backend="jupyter_docker").__enter__()
    rt = FakeRuntime()
    detect_fn = lambda: DockerAvailability(available, "docker" if available else "", "test")
    backend = JupyterDockerBackend(ctx.settings, runtime=rt, detect_fn=detect_fn)
    return backend, rt, ctx


def test_fail_closed_when_docker_unavailable(tmp_path):
    backend, rt, ctx = _backend(available=False)
    try:
        result = backend.execute("print(1)", "s1", timeout_seconds=5, working_dir=tmp_path)
        assert result.success is False
        assert "unavailable" in result.error.lower()
        assert rt.started == []  # never tried to start a container
    finally:
        ctx.__exit__(None, None, None)


def test_execute_starts_container_with_isolation_spec(tmp_path):
    backend, rt, ctx = _backend()
    try:
        # feed a result after start: the session reads ready, then result
        orig_start = rt.start
        def start(spec):
            handle = orig_start(spec)
            handle.stdout.feed(json.dumps({"type": "result", "success": True, "stdout": "1\n",
                                           "stderr": "", "result": None, "artifacts": [],
                                           "execution_time": 0.1, "error": ""}) + "\n")
            return handle
        rt.start = start
        result = backend.execute("print(1)", "s1", timeout_seconds=5, working_dir=tmp_path)
        assert result.success is True
        spec = rt.started[0]
        assert spec.network == "none"
        assert spec.image  # from settings
        assert any(m.container == "/workspace" and not m.read_only for m in spec.mounts)
        assert any(m.container.endswith("/driver.py") and m.read_only for m in spec.mounts)
        assert any(m.container.endswith("/kernel_exec.py") and m.read_only for m in spec.mounts)
        assert spec.labels.get("agentic-session") == "s1"
        assert spec.labels.get("agentic-sandbox") == "1"
        assert spec.env.get("AGENTIC_SANDBOX_WORKSPACE") == "/workspace"
    finally:
        ctx.__exit__(None, None, None)


def test_reset_session_kills_container(tmp_path):
    backend, rt, ctx = _backend()
    try:
        # start a session via execute (feed ready + one result)
        orig = rt.start
        def start_with_result(spec):
            handle = orig(spec)
            handle.stdout.feed(json.dumps({"type": "result", "success": True, "stdout": "", "stderr": "",
                                           "result": None, "artifacts": [], "execution_time": 0.0, "error": ""}) + "\n")
            return handle
        rt.start = start_with_result
        backend.execute("x=1", "s1", timeout_seconds=5, working_dir=tmp_path)
        assert backend.has_session("s1")
        backend.reset_session("s1")
        assert not backend.has_session("s1")
        assert any(name == "agentic-sbx-s1" for name, _ in rt.killed)
    finally:
        ctx.__exit__(None, None, None)


def test_bad_startup_message_kills_container(tmp_path):
    backend, rt, ctx = _backend()
    try:
        def start_bad(spec):
            handle = FakeHandle()
            handle.name = spec.name
            rt._handles.append(handle)
            rt.started.append(spec)
            handle.stdout.feed(json.dumps({"type": "boom"}) + "\n")  # not "ready"
            return handle
        rt.start = start_bad
        result = backend.execute("print(1)", "s1", timeout_seconds=5, working_dir=tmp_path)
        assert result.success is False
        assert any(name == "agentic-sbx-s1" for name, _ in rt.killed)  # cleaned up, not orphaned
        assert not backend.has_session("s1")
    finally:
        ctx.__exit__(None, None, None)


def test_session_status_reports_backend(tmp_path):
    backend, rt, ctx = _backend()
    try:
        st = backend.session_status("absent")
        assert st.backend == "jupyter_docker"
        assert st.state == "absent"
    finally:
        ctx.__exit__(None, None, None)


# Fix 2a: OSError from runtime.start must not propagate
def test_start_oserror_returns_error_dict(tmp_path):
    backend, rt, ctx = _backend()
    try:
        def raise_oserror(spec):
            raise OSError("docker gone")
        rt.start = raise_oserror
        result = backend.execute("print(1)", "s1", timeout_seconds=5, working_dir=tmp_path)
        assert result.success is False
        assert "docker gone" in result.error or "failed" in result.error.lower()
    finally:
        ctx.__exit__(None, None, None)
