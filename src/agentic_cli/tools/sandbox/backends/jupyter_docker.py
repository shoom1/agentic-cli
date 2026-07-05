"""Docker-backed stateful sandbox backend."""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

from agentic_cli.logging import Loggers
from agentic_cli.tools.sandbox.models import ExecutionResult

logger = Loggers.tools()

_EOF = object()


class SandboxStartError(Exception):
    """Raised when a container/kernel fails to become ready."""


class ContainerSession:
    """Owns one container: its stdio protocol, reader thread, and status."""

    def __init__(self, session_id, handle, interrupt, kill, working_dir,
                 start_timeout=180, interrupt_grace=10.0, backend_name="jupyter_docker"):
        self._session_id = session_id
        self._handle = handle
        self._interrupt = interrupt
        self._kill = kill
        self._working_dir = working_dir
        self._start_timeout = start_timeout
        self._interrupt_grace = interrupt_grace
        self._backend_name = backend_name
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._status = "starting"
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

    @property
    def status(self) -> str:
        return self._status

    @property
    def container_id(self) -> str:
        return getattr(self._handle, "name", "")

    def _read_stdout(self) -> None:
        try:
            for line in self._handle.stdout:
                self._queue.put(line)
        finally:
            self._queue.put(_EOF)

    def wait_ready(self) -> None:
        try:
            line = self._queue.get(timeout=self._start_timeout)
        except queue.Empty:
            self._kill()
            self._status = "dead"
            raise SandboxStartError(f"sandbox kernel not ready within {self._start_timeout}s")
        if line is _EOF:
            self._status = "dead"
            raise SandboxStartError("sandbox container exited before becoming ready")
        msg = json.loads(line)
        if msg.get("type") != "ready":
            self._status = "dead"
            raise SandboxStartError(f"unexpected startup message: {msg!r}")
        self._status = "ready"

    def execute(self, code: str, timeout: float) -> ExecutionResult:
        with self._lock:
            self._status = "busy"
            self._handle.stdin.write(json.dumps({"type": "execute", "code": code, "timeout": timeout}) + "\n")
            self._handle.stdin.flush()

            line = self._await(timeout)
            if line is None:  # timed out -> cooperative interrupt, then hard kill
                self._interrupt()
                line = self._await(self._interrupt_grace)
                if line is None:
                    self._kill()
                    self._status = "dead"
                    return ExecutionResult(success=False,
                                           error=f"Execution timed out after {timeout}s; container killed")
            if line is _EOF:
                self._status = "dead"
                return ExecutionResult(success=False, error="sandbox container exited unexpectedly")

            data = json.loads(line)
            self._status = "ready"
            return self._to_result(data)

    def _await(self, timeout: float):
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _to_result(self, data: dict) -> ExecutionResult:
        artifacts = [self._translate(p) for p in data.get("artifacts", [])]
        return ExecutionResult(
            success=data.get("success", False),
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            result=data.get("result"),
            artifacts=artifacts,
            execution_time=data.get("execution_time", 0.0),
            error=data.get("error", ""),
        )

    def _translate(self, container_path: str) -> str:
        """Map an in-container /workspace path to the host session dir."""
        prefix = "/workspace"
        if self._working_dir is not None and container_path.startswith(prefix):
            rel = container_path[len(prefix):].lstrip("/")
            return str(Path(self._working_dir) / rel)
        return container_path

    def close(self) -> None:
        try:
            self._kill()
        finally:
            self._status = "dead"
