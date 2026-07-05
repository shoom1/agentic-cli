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

    def start(self) -> None:
        from jupyter_client import KernelManager

        km = KernelManager()
        km.start_kernel()
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
        timeout = req.get("timeout", 120)
        ok, msg = validate_code(code)
        if not ok:
            return {"type": "result", "success": False, "stdout": "", "stderr": "",
                    "result": None, "artifacts": [], "execution_time": 0.0, "error": msg}
        msg_id = self._kc.execute(code)
        data = collect_execution(self._kc, msg_id, timeout, self._workspace)
        return {"type": "result", **data}

    def _write(self, obj: dict) -> None:
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
