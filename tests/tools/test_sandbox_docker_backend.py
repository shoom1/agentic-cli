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
    ctx = MockContext(stateful_executor_backend="docker").__enter__()
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


def _feed_result(rt):
    """Make the fake runtime feed one successful result after 'ready'."""
    orig = rt.start
    def start_with_result(spec):
        h = orig(spec)
        h.stdout.feed(json.dumps({"type": "result", "success": True, "stdout": "", "stderr": "",
                                  "result": None, "artifacts": [], "execution_time": 0.0, "error": ""}) + "\n")
        return h
    rt.start = start_with_result


# The non-root container must be able to write the /workspace bind mount. Rather
# than make the host session dir world-writable (chmod 0777 — a multi-user-host
# exposure), run the container AS the host uid so files it writes are host-owned.
@pytest.mark.skipif(not hasattr(__import__("os"), "getuid"), reason="POSIX uid only")
def test_container_runs_as_host_uid_and_dir_not_world_writable(tmp_path):
    import os
    import stat
    backend, rt, ctx = _backend()
    try:
        _feed_result(rt)
        wd = tmp_path / "sess"
        wd.mkdir(mode=0o700)
        backend.execute("x = 1", "s1", timeout_seconds=5, working_dir=wd)
        assert rt.started[0].user == f"{os.getuid()}:{os.getgid()}"
        # session dir perms untouched — NOT made world-writable
        assert stat.S_IMODE(os.stat(wd).st_mode) == 0o700
    finally:
        ctx.__exit__(None, None, None)


def test_explicit_container_user_overrides_host_uid(tmp_path):
    ctx = MockContext(stateful_executor_backend="docker",
                      sandbox_container_user="1234:5678").__enter__()
    rt = FakeRuntime()
    backend = JupyterDockerBackend(
        ctx.settings, runtime=rt,
        detect_fn=lambda: DockerAvailability(True, "docker", "test"),
    )
    try:
        _feed_result(rt)
        backend.execute("x = 1", "s1", timeout_seconds=5, working_dir=tmp_path)
        assert rt.started[0].user == "1234:5678"
    finally:
        ctx.__exit__(None, None, None)


def test_execute_stages_inputs_into_session_inputs_dir(tmp_path):
    backend, rt, ctx = _backend()
    try:
        _feed_result(rt)
        src = tmp_path / "sales.csv"; src.write_text("x\n1\n")
        wd = tmp_path / "sess"; wd.mkdir()
        backend.execute("print(1)", "s1", timeout_seconds=5, working_dir=wd, inputs=[str(src)])
        assert (wd / "inputs" / "sales.csv").read_text() == "x\n1\n"
    finally:
        ctx.__exit__(None, None, None)


def test_build_spec_mounts_shared_outputs_dir(tmp_path):
    backend, rt, ctx = _backend()
    try:
        _feed_result(rt)
        backend.execute("print(1)", "s1", timeout_seconds=5, working_dir=tmp_path)
        spec = rt.started[0]
        outs = [m for m in spec.mounts if m.container == "/workspace/outputs"]
        assert outs and outs[0].read_only is False
    finally:
        ctx.__exit__(None, None, None)


def test_outputs_mountpoint_pre_created_as_host_user(tmp_path):
    """Docker must not create /workspace/outputs as root.
    The backend must pre-create <working_dir>/outputs before runtime.start() so
    the mount-point directory is owned by the host user (not root), which allows
    pytest teardown to remove it and keeps the session dir clean."""
    backend, rt, ctx = _backend()
    try:
        _feed_result(rt)
        wd = tmp_path / "sess"
        wd.mkdir()
        backend.execute("x = 1", "s1", timeout_seconds=5, working_dir=wd)
        assert (wd / "outputs").exists(), "outputs/ mount-point must exist after execute"
        assert (wd / "outputs").is_dir(), "outputs/ must be a directory, not a file"
    finally:
        ctx.__exit__(None, None, None)


def test_data_mount_name_cannot_escape_workspace(tmp_path):
    """A hostile data-mount name (traversal) must not remap the mount point
    outside /workspace/data/ inside the container."""
    import posixpath
    ctx = MockContext(stateful_executor_backend="docker",
                      sandbox_data_mounts=[f"{tmp_path}:../../etc"]).__enter__()
    rt = FakeRuntime()
    backend = JupyterDockerBackend(
        ctx.settings, runtime=rt,
        detect_fn=lambda: DockerAvailability(True, "docker", "test"),
    )
    try:
        _feed_result(rt)
        backend.execute("x = 1", "s1", timeout_seconds=5, working_dir=tmp_path)
        data = [m for m in rt.started[0].mounts if m.container.startswith("/workspace/data/")]
        assert data, "expected a data mount under /workspace/data/"
        for m in data:
            assert ".." not in m.container
            assert posixpath.normpath(m.container).startswith("/workspace/data/")
    finally:
        ctx.__exit__(None, None, None)
