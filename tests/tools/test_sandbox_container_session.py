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
