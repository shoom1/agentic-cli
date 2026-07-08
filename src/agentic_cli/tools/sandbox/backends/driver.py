"""In-container kernel driver.

Runs INSIDE the sandbox container as `python .../driver.py`. Hosts a jupyter
kernel and speaks NDJSON over stdio: reads {"type":"execute","code","timeout"}
on stdin, writes {"type":"ready"} once and {"type":"result", ...} per request
on stdout. Interrupt is delivered as SIGINT to this process (PID 1); the handler
forwards it to the kernel so a running cell aborts without losing session state.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import sys
from pathlib import Path

# Dual import: top-level when bind-mounted beside driver.py in-container;
# package path in tests / in-repo.
try:  # pragma: no cover - import shim
    from kernel_exec import collect_execution, validate_code
except ImportError:  # pragma: no cover - import shim
    from agentic_cli.tools.sandbox.backends.kernel_exec import collect_execution, validate_code


class KernelDriver:
    def __init__(self, stdin, stdout, workspace: str | None = None) -> None:
        self._stdin = stdin
        self._stdout = stdout
        self._workspace = Path(workspace or os.environ.get("AGENTIC_SANDBOX_WORKSPACE", "/workspace"))
        self._km = None
        self._kc = None
        # Per-session secret tagging every protocol message. It lives only in
        # this (trusted) driver process's memory — never in env/files, and the
        # kernel can't ptrace the driver (default docker seccomp blocks it), so
        # untrusted kernel code can't learn it. The host rejects any result line
        # lacking this token, defeating forged writes to the driver's fd 1 via
        # /proc/<pid>/fd/1 (which the kernel can open, but only write-only).
        self._token = secrets.token_hex(16)

    def start(self) -> None:
        import subprocess
        from jupyter_client import KernelManager

        # Start the kernel in the workspace so relative file writes land in the
        # (writable, host-mounted) /workspace rather than the read-only image
        # WORKDIR (e.g. /home/jovyan), where they fail or don't persist. The
        # kernel inherits the driver's cwd; KernelManager.cwd is not honored by
        # all jupyter_client versions, so chdir here is the reliable path.
        try:
            os.chdir(self._workspace)
        except OSError:
            pass
        km = KernelManager()
        # Isolate the kernel's raw std fds (0/1/2) from the driver's. The driver
        # uses its own fd 0/1 for the NDJSON protocol with the host; if the
        # kernel inherited them, user code could os.write(1, ...) a forged
        # {"type":"result"} to desync the session, os.read(0, ...) pending host
        # requests, or os.write(2, ...) to spam the host debug log. The kernel
        # speaks ZMQ, so it needs none of these; real output/errors flow over
        # iopub.
        km.start_kernel(stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
        kc = km.blocking_client()
        kc.start_channels()
        kc.wait_for_ready(timeout=60)
        self._km, self._kc = km, kc
        signal.signal(signal.SIGINT, self._on_sigint)

    def _on_sigint(self, signum, frame) -> None:
        if self._km is not None:
            self._km.interrupt_kernel()

    def handle_request(self, req: dict) -> dict:
        code = req.get("code", "")
        ok, msg = validate_code(code)
        if not ok:
            return {"type": "result", "success": False, "stdout": "", "stderr": "",
                    "result": None, "artifacts": [], "execution_time": 0.0, "error": msg}
        msg_id = self._kc.execute(code)
        # The HOST owns the timeout/interrupt/kill state machine: it waits the
        # per-cell deadline, then sends a cooperative interrupt (SIGINT ->
        # _on_sigint -> interrupt_kernel), then hard-kills the container if that
        # fails. So the driver must block until the kernel actually goes idle
        # (natural completion) or the interrupt aborts the cell. If the driver
        # self-timed-out on the same deadline it would return a "timed out"
        # result while the cell keeps running in the kernel — the host would
        # consume that result WITHOUT interrupting, wedging the session for the
        # next request. Passing timeout=None makes collect_execution block.
        data = collect_execution(self._kc, msg_id, None, self._workspace)
        return {"type": "result", **data}

    def _write(self, obj: dict) -> None:
        obj = {**obj, "token": self._token}
        self._stdout.write(json.dumps(obj) + "\n")
        self._stdout.flush()

    def run(self) -> None:
        self.start()
        self._write({"type": "ready"})
        for line in self._stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                self._write({"type": "result", "success": False, "error": "invalid request json",
                             "stdout": "", "stderr": "", "result": None, "artifacts": [],
                             "execution_time": 0.0})
                continue
            self._write(self.handle_request(req))

    def close(self) -> None:
        try:
            if self._kc is not None:
                self._kc.stop_channels()
        except Exception:
            pass
        try:
            if self._km is not None:
                self._km.shutdown_kernel(now=True)
        except Exception:
            pass


if __name__ == "__main__":  # pragma: no cover - container entrypoint
    KernelDriver(sys.stdin, sys.stdout).run()
