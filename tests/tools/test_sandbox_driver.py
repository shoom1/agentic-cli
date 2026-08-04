"""Tests for the in-container kernel driver (real kernel, no docker)."""

import io
import json
import signal
from unittest.mock import MagicMock

import pytest

pytest.importorskip("jupyter_client")

from agentic_cli.tools.sandbox.backends.driver import KernelDriver


@pytest.fixture
def driver(tmp_path):
    d = KernelDriver(stdin=io.StringIO(), stdout=io.StringIO(), workspace=str(tmp_path))
    d.start()
    yield d
    d.close()


def test_state_persists_across_requests(driver):
    driver.handle_request({"type": "execute", "code": "x = 41", "timeout": 30})
    r = driver.handle_request({"type": "execute", "code": "print(x + 1)", "timeout": 30})
    assert r["success"] is True
    assert "42" in r["stdout"]


def test_error_is_captured(driver):
    r = driver.handle_request({"type": "execute", "code": "1/0", "timeout": 30})
    assert r["success"] is False
    assert "ZeroDivisionError" in r["error"]


def test_run_emits_ready_then_result(tmp_path):
    stdin = io.StringIO(json.dumps({"type": "execute", "code": "print('hi')", "timeout": 30}) + "\n")
    stdout = io.StringIO()
    d = KernelDriver(stdin=stdin, stdout=stdout, workspace=str(tmp_path))
    try:
        d.run()  # returns at stdin EOF
        lines = [json.loads(l) for l in stdout.getvalue().splitlines() if l.strip()]
        assert lines[0]["type"] == "ready"
        assert lines[1]["type"] == "result"
        assert "hi" in lines[1]["stdout"]
    finally:
        d.close()


def test_messages_carry_consistent_session_token(tmp_path):
    """Every driver message is tagged with a per-session token so the host can
    reject forged lines; ready and result share the same non-empty token."""
    stdin = io.StringIO(json.dumps({"type": "execute", "code": "print('hi')", "timeout": 30}) + "\n")
    stdout = io.StringIO()
    d = KernelDriver(stdin=stdin, stdout=stdout, workspace=str(tmp_path))
    try:
        d.run()
        lines = [json.loads(l) for l in stdout.getvalue().splitlines() if l.strip()]
        assert lines[0]["type"] == "ready" and lines[0].get("token")
        assert lines[1]["type"] == "result"
        assert lines[1].get("token") == lines[0]["token"]
    finally:
        d.close()


def test_sigint_handler_interrupts_kernel(tmp_path):
    d = KernelDriver(stdin=io.StringIO(), stdout=io.StringIO(), workspace=str(tmp_path))
    d._km = MagicMock()
    d._on_sigint(signal.SIGINT, None)
    d._km.interrupt_kernel.assert_called_once()
