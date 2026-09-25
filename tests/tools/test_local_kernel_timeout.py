"""The local Jupyter backend's timeout is a deadline, and it stops the cell.

``collect_execution`` passed the timeout to every ``get_iopub_msg`` call, so it
measured the gap between two outputs: a cell that printed every few seconds
never timed out. When it did time out, nothing interrupted the kernel, so the
cell kept running and the next ``sandbox_execute`` in that session queued behind
it and timed out too. The Docker backend already owns a deadline → interrupt →
kill sequence on the host; the local backend now does the equivalent:
deadline → interrupt → wait a grace period → restart the kernel.
"""

from __future__ import annotations

import time

import pytest

from agentic_cli.tools.sandbox.backends import kernel_exec

pytest.importorskip("jupyter_client")
pytest.importorskip("ipykernel")


class _ChattyClient:
    """A kernel client whose cell prints every 0.1 s for 4 s, then goes idle
    (the end keeps a broken deadline from hanging the test)."""

    def __init__(self):
        self.sent = 0

    def get_iopub_msg(self, timeout=None):
        time.sleep(0.1 if timeout is None else min(0.1, timeout))
        self.sent += 1
        if self.sent > 40:
            return {
                "parent_header": {"msg_id": "m1"},
                "msg_type": "status",
                "content": {"execution_state": "idle"},
            }
        return {
            "parent_header": {"msg_id": "m1"},
            "msg_type": "stream",
            "content": {"name": "stdout", "text": "tick\n"},
        }


class TestCollectExecutionDeadline:
    def test_steady_output_does_not_extend_the_deadline(self):
        start = time.monotonic()
        data = kernel_exec.collect_execution(_ChattyClient(), "m1", 1.0, None)
        elapsed = time.monotonic() - start

        assert elapsed < 1.5
        assert data["timed_out"] is True
        assert data["success"] is False
        assert "timed out after 1.0s" in data["error"]
        assert "tick" in data["stdout"]

    def test_a_finished_cell_is_not_timed_out(self):
        class Done:
            def get_iopub_msg(self, timeout=None):
                return {
                    "parent_header": {"msg_id": "m1"},
                    "msg_type": "status",
                    "content": {"execution_state": "idle"},
                }

        data = kernel_exec.collect_execution(Done(), "m1", 1.0, None)
        assert data["timed_out"] is False
        assert data["success"] is True


@pytest.fixture
def backend():
    from agentic_cli.tools.sandbox.backends.jupyter_local import JupyterLocalBackend

    b = JupyterLocalBackend(interrupt_grace=3.0)
    yield b
    b.cleanup()


def _timed(fn):
    start = time.monotonic()
    result = fn()
    return result, time.monotonic() - start


class TestJupyterLocalTimeout:
    def test_a_cell_that_keeps_printing_still_times_out(self, backend, tmp_path):
        code = "import time\nfor i in range(100):\n    print(i, flush=True)\n    time.sleep(0.2)\n"
        result, elapsed = _timed(lambda: backend.execute(
            code, session_id="s", timeout_seconds=1, working_dir=tmp_path,
        ))

        assert result.success is False
        assert "timed out after 1s" in result.error
        assert elapsed < 6
        assert "0" in result.stdout  # output produced before the deadline is kept

    def test_the_cell_is_interrupted_and_the_session_is_kept(self, backend, tmp_path):
        backend.execute("x = 41", session_id="s", working_dir=tmp_path)

        result, elapsed = _timed(lambda: backend.execute(
            "while True:\n    pass\n", session_id="s", timeout_seconds=1, working_dir=tmp_path,
        ))
        assert result.success is False
        assert "interrupted" in result.error
        assert elapsed < 6

        after, elapsed = _timed(lambda: backend.execute(
            "x + 1", session_id="s", timeout_seconds=10, working_dir=tmp_path,
        ))
        assert after.success is True, after.error
        assert after.result == "42"
        assert elapsed < 5

    def test_a_cell_that_ignores_the_interrupt_gets_a_new_kernel(self, backend, tmp_path):
        backend.execute("x = 41", session_id="s", working_dir=tmp_path)
        stubborn = (
            "while True:\n"
            "    try:\n"
            "        while True:\n"
            "            pass\n"
            "    except KeyboardInterrupt:\n"
            "        pass\n"
        )

        result, elapsed = _timed(lambda: backend.execute(
            stubborn, session_id="s", timeout_seconds=1, working_dir=tmp_path,
        ))
        assert result.success is False
        assert "restarted" in result.error
        assert elapsed < 15

        after = backend.execute("print('alive')", session_id="s", timeout_seconds=30, working_dir=tmp_path)
        assert after.success is True, after.error
        assert "alive" in after.stdout
        gone = backend.execute("x", session_id="s", timeout_seconds=10, working_dir=tmp_path)
        assert gone.success is False  # the restart lost the session's variables
