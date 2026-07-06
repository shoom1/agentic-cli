"""Tests for the shared kernel execution helper (no docker needed)."""

import pytest

pytest.importorskip("jupyter_client")

from jupyter_client import KernelManager

from agentic_cli.tools.sandbox.backends import kernel_exec


@pytest.fixture
def kernel():
    km = KernelManager()
    km.start_kernel()
    kc = km.blocking_client()
    kc.start_channels()
    kc.wait_for_ready(timeout=30)
    yield kc
    kc.stop_channels()
    km.shutdown_kernel(now=True)


def test_validate_rejects_shell_bang():
    ok, msg = kernel_exec.validate_code("!rm -rf /")
    assert ok is False
    assert "shell" in msg.lower()


def test_validate_allows_plain_code():
    assert kernel_exec.validate_code("x = 1\nprint(x)") == (True, "")


def test_collect_captures_stdout(kernel, tmp_path):
    msg_id = kernel.execute("print('hello')")
    out = kernel_exec.collect_execution(kernel, msg_id, timeout=30, working_dir=tmp_path)
    assert out["success"] is True
    assert "hello" in out["stdout"]
    assert out["error"] == ""


def test_collect_captures_error(kernel, tmp_path):
    msg_id = kernel.execute("raise ValueError('boom')")
    out = kernel_exec.collect_execution(kernel, msg_id, timeout=30, working_dir=tmp_path)
    assert out["success"] is False
    assert "boom" in out["error"]


def test_collect_caps_large_stdout(kernel, tmp_path):
    """Runaway output is bounded (host memory / LLM context) with a truncation
    marker, rather than accumulated unbounded."""
    cap = kernel_exec.MAX_STREAM_CHARS
    msg_id = kernel.execute(f"print('A' * {cap * 3})")
    out = kernel_exec.collect_execution(kernel, msg_id, timeout=30, working_dir=tmp_path)
    assert out["success"] is True
    assert len(out["stdout"]) <= cap + 200  # cap plus the short marker
    assert "truncated" in out["stdout"].lower()
