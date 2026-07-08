"""Tests for ContainerSession using an in-memory fake container handle."""

import io
import json
import queue
import threading

import pytest

from agentic_cli.tools.sandbox.backends.jupyter_docker import (
    ContainerSession, SandboxStartError,
)


class FakeStdout:
    """Blocking line iterator fed by the test; yields until closed (EOF)."""
    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self._closed = False
    def feed(self, line: str):
        self._q.put(line)
    def eof(self):
        self._q.put(None)
    def __iter__(self):
        return self
    def __next__(self):
        item = self._q.get()
        if item is None:
            raise StopIteration
        return item


class FakeHandle:
    def __init__(self):
        self.name = "agentic-sbx-test"
        self.stdin = io.StringIO()
        self.stdout = FakeStdout()
        self.stderr = FakeStdout()
        self._dead = False
    def poll(self):
        return 137 if self._dead else None


def _session(handle, **kw):
    calls = {"interrupt": 0, "kill": 0}
    def interrupt():
        calls["interrupt"] += 1
    def kill():
        calls["kill"] += 1
        handle._dead = True
        handle.stdout.eof()
    working_dir = kw.pop("working_dir", None)
    s = ContainerSession(
        session_id="test", handle=handle, interrupt=interrupt, kill=kill,
        working_dir=working_dir, start_timeout=2, interrupt_grace=1, **kw,
    )
    return s, calls


def test_wait_ready_consumes_ready_line():
    h = FakeHandle()
    s, _ = _session(h)
    h.stdout.feed(json.dumps({"type": "ready"}) + "\n")
    s.wait_ready()
    assert s.status == "ready"


def test_wait_ready_times_out_and_kills():
    h = FakeHandle()
    s, calls = _session(h)
    with pytest.raises(SandboxStartError):
        s.wait_ready()  # nothing fed -> start_timeout
    assert calls["kill"] == 1


def test_execute_returns_result():
    h = FakeHandle()
    s, _ = _session(h)
    h.stdout.feed(json.dumps({"type": "ready"}) + "\n")
    s.wait_ready()
    h.stdout.feed(json.dumps({"type": "result", "success": True, "stdout": "42\n",
                              "stderr": "", "result": "42", "artifacts": [],
                              "execution_time": 0.1, "error": ""}) + "\n")
    result = s.execute("print(42)", timeout=5)
    assert result.success is True
    assert result.stdout == "42\n"
    assert s.status == "ready"


def test_execute_timeout_interrupts_then_kills():
    h = FakeHandle()
    s, calls = _session(h)
    h.stdout.feed(json.dumps({"type": "ready"}) + "\n")
    s.wait_ready()
    result = s.execute("while True: pass", timeout=1)  # no result fed
    assert result.success is False
    assert "timed out" in result.error.lower()
    assert calls["interrupt"] == 1
    assert calls["kill"] == 1
    assert s.status == "dead"


def test_execute_translates_workspace_artifact_paths(tmp_path):
    h = FakeHandle()
    s, _ = _session(h, working_dir=tmp_path)
    h.stdout.feed(json.dumps({"type": "ready"}) + "\n")
    s.wait_ready()
    h.stdout.feed(json.dumps({"type": "result", "success": True, "stdout": "", "stderr": "",
                              "result": None, "artifacts": ["/workspace/artifacts/plot_0.png"],
                              "execution_time": 0.1, "error": ""}) + "\n")
    result = s.execute("plot()", timeout=5)
    assert result.artifacts == [str(tmp_path / "artifacts" / "plot_0.png")]


# Fix 1: stderr drain thread
def test_stderr_drain_thread_exists_and_drains():
    h = FakeHandle()
    s, _ = _session(h)
    h.stdout.feed(json.dumps({"type": "ready"}) + "\n")
    s.wait_ready()
    # feed stderr lines, then close
    h.stderr.feed("kernel warning 1\n")
    h.stderr.feed("kernel warning 2\n")
    h.stderr.feed("kernel warning 3\n")
    h.stderr.eof()
    # a _stderr_reader thread must exist
    assert hasattr(s, "_stderr_reader") and s._stderr_reader is not None
    # it must finish after EOF (bounded wait)
    s._stderr_reader.join(timeout=2)
    assert not s._stderr_reader.is_alive(), "stderr drain thread should stop after EOF"


# Fix 2b: dead-container early poll() check
def test_execute_on_dead_container_returns_error():
    h = FakeHandle()
    s, _ = _session(h)
    h.stdout.feed(json.dumps({"type": "ready"}) + "\n")
    s.wait_ready()
    assert s.status == "ready"
    # mark container dead before execute
    h._dead = True
    h.stdout.eof()
    result = s.execute("print(1)", timeout=5)
    assert result.success is False
    assert "exited unexpectedly" in result.error.lower()
    assert s.status == "dead"


def test_execute_skips_forged_untokened_result():
    """A forged result (no/wrong token) injected into the driver's stdout — e.g.
    by user code writing to /proc/<driver>/fd/1 — must be skipped; only the
    driver's authenticated result is accepted."""
    h = FakeHandle()
    s, _ = _session(h)
    h.stdout.feed(json.dumps({"type": "ready", "token": "SECRET"}) + "\n")
    s.wait_ready()
    # forged first (no token), then the real tokened result
    h.stdout.feed(json.dumps({"type": "result", "success": True, "stdout": "FORGED\n",
                              "stderr": "", "result": None, "artifacts": [],
                              "execution_time": 0.0, "error": ""}) + "\n")
    h.stdout.feed(json.dumps({"type": "result", "token": "SECRET", "success": True,
                              "stdout": "REAL\n", "stderr": "", "result": None,
                              "artifacts": [], "execution_time": 0.0, "error": ""}) + "\n")
    result = s.execute("print(1)", timeout=5)
    assert result.success is True
    assert result.stdout == "REAL\n"  # forged line skipped


def test_execute_kills_container_on_unexpected_error():
    """An unexpected host-side error mid-execute (e.g. a bad timeout raising in
    queue.get) must kill the container, not orphan it."""
    h = FakeHandle()
    s, calls = _session(h)
    h.stdout.feed(json.dumps({"type": "ready"}) + "\n")
    s.wait_ready()
    result = s.execute("print(1)", timeout=-5)  # negative -> queue.get raises ValueError
    assert result.success is False
    assert calls["kill"] == 1  # container killed, not leaked
    assert s.status == "dead"


def test_execute_reports_likely_oom_on_137_exit():
    """Exit 137 (128+SIGKILL) is the cgroup OOM-killer signature; the error
    should hint at OOM and include the code rather than a bare 'exited'."""
    h = FakeHandle()  # FakeHandle.poll() returns 137 when dead
    s, _ = _session(h)
    h.stdout.feed(json.dumps({"type": "ready"}) + "\n")
    s.wait_ready()
    h._dead = True
    h.stdout.eof()
    result = s.execute("x = bytearray(10**10)", timeout=5)
    assert result.success is False
    assert "137" in result.error and "memory" in result.error.lower()
