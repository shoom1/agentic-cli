"""Docker-backed stateful sandbox backend."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path

from agentic_cli.logging import Loggers
from agentic_cli.tools.sandbox.models import ExecutionResult, SessionStatus
from agentic_cli.file_utils import sanitize_filename
from agentic_cli.tools.sandbox.backends.base import SandboxBackend
from agentic_cli.tools.sandbox.backends import kernel_exec
from agentic_cli.tools.sandbox.backends.container_runtime import (
    ContainerSpec, Mount, DockerContainerRuntime,
)
from agentic_cli.tools.sandbox.backends.detect import detect_docker

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
        self._token: str | None = None  # captured from the driver's ready message
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self._stderr_reader = None
        if getattr(handle, "stderr", None) is not None:
            self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
            self._stderr_reader.start()

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

    def _read_stderr(self) -> None:
        try:
            for line in self._handle.stderr:
                logger.debug("sandbox_stderr", line=line.rstrip())
        except Exception:
            pass

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
        # Capture the session token. This ready message is genuine: no user code
        # runs before it (the kernel executes nothing until an execute request).
        self._token = msg.get("token")
        self._status = "ready"

    def execute(self, code: str, timeout: float) -> ExecutionResult:
        with self._lock:
            if self._handle.poll() is not None:
                self._status = "dead"
                return ExecutionResult(success=False,
                                       error=f"sandbox container exited unexpectedly ({self._exit_detail()})")
            self._status = "busy"
            try:
                # The host owns the timeout/interrupt/kill state machine; the
                # driver blocks until idle, so the request carries only the code.
                self._handle.stdin.write(json.dumps({"type": "execute", "code": code}) + "\n")
                self._handle.stdin.flush()

                msg = self._read_authenticated(timeout)
                if msg is None:  # timed out -> cooperative interrupt, then hard kill
                    self._interrupt()
                    msg = self._read_authenticated(self._interrupt_grace)
                    if msg is None:
                        self._kill()
                        self._status = "dead"
                        return ExecutionResult(success=False,
                                               error=f"Execution timed out after {timeout}s; container killed")
                if msg is _EOF:
                    self._status = "dead"
                    return ExecutionResult(success=False,
                                           error=f"sandbox container exited unexpectedly ({self._exit_detail()})")

                self._status = "ready"
                return self._to_result(msg)
            except Exception as exc:
                # Don't leak the container on an unexpected host-side error
                # (e.g. a bad timeout raising in queue.get).
                try:
                    self._kill()
                except Exception:
                    pass
                self._status = "dead"
                return ExecutionResult(success=False, error=f"sandbox execution failed: {exc}")

    def _read_authenticated(self, timeout: float):
        """Return the next AUTHENTICATED protocol message within ``timeout``,
        skipping lines that don't carry the session token — those are forged by
        user code writing to the driver's fd 1 (e.g. via /proc/<pid>/fd/1).
        Returns the parsed dict, ``None`` on timeout, or ``_EOF``."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                line = self._queue.get(timeout=remaining)
            except queue.Empty:
                return None
            if line is _EOF:
                return _EOF
            try:
                msg = json.loads(line)
            except (ValueError, TypeError):
                continue  # non-JSON garbage — forged
            if self._token is not None and msg.get("token") != self._token:
                continue  # missing/wrong token — forged, skip it
            return msg

    def _exit_detail(self) -> str:
        """Describe why the container process exited, from the docker-run exit
        code (reliable even with --rm, which removes the container before it
        could be inspected). 137 = 128+SIGKILL, the cgroup OOM-killer signature."""
        code = self._handle.poll()
        if code == 137:
            return f"exit {code}; possibly out-of-memory (OOM-killed)"
        return f"exit {code}"

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


_DRIVER_DIR = "/opt/agentic_sandbox"


class JupyterDockerBackend(SandboxBackend):
    """Runs each session in its own network-isolated container."""

    backend_name = "jupyter_docker"

    def __init__(self, settings, runtime=None, detect_fn=None) -> None:
        self._settings = settings
        self._detect = detect_fn or detect_docker
        self._runtime = runtime  # lazily created so detect can pick docker/podman
        self._sessions: dict[str, ContainerSession] = {}

    def _ensure_runtime(self):
        if self._runtime is None:
            avail = self._detect()
            self._runtime = DockerContainerRuntime(avail.runtime or "docker")
        return self._runtime

    def _outputs_dir(self) -> Path:
        configured = getattr(self._settings, "sandbox_outputs_dir", "") or ""
        base = Path(configured) if configured else Path(self._settings.workspace_dir) / "artifacts"
        base.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(base, 0o777)  # container runs as host uid; keep writable across sessions
        except OSError:
            pass
        return base

    def _build_spec(self, session_id: str, working_dir) -> ContainerSpec:
        s = self._settings
        here = Path(__file__).parent
        # NOTE: /workspace is a writable host bind mount with NO disk quota —
        # Docker caps memory/CPU/PIDs but not bind-mount disk. Operators who need
        # a hard limit should place workspace_dir on a quota'd filesystem.
        mounts = [
            Mount(str(working_dir), "/workspace", read_only=False),
            Mount(str(self._outputs_dir()), "/workspace/outputs", read_only=False),
            Mount(str(here / "driver.py"), f"{_DRIVER_DIR}/driver.py", read_only=True),
            Mount(str(here / "kernel_exec.py"), f"{_DRIVER_DIR}/kernel_exec.py", read_only=True),
        ]
        for entry in s.sandbox_data_mounts:
            host, _, name = entry.partition(":")
            # Sanitize the mount name so a hostile '..'/absolute value can't
            # remap the mount outside /workspace/data/ (sanitize_filename maps
            # '/' and '.' to '_').
            name = sanitize_filename(name or Path(host).name) or "mount"
            mounts.append(Mount(host, f"/workspace/data/{name}", read_only=True))
        env = {
            "HOME": "/tmp", "MPLCONFIGDIR": "/tmp", "IPYTHONDIR": "/tmp",
            "JUPYTER_RUNTIME_DIR": "/tmp", "PYTHONDONTWRITEBYTECODE": "1",
            "AGENTIC_SANDBOX_WORKSPACE": "/workspace",
        }
        # Run as the host uid:gid by default so files the non-root kernel writes
        # to the /workspace bind mount are owned by the host user — no
        # world-writable chmod on the session dir needed. An explicit
        # sandbox_container_user overrides.
        user = s.sandbox_container_user
        if not user and hasattr(os, "getuid"):
            user = f"{os.getuid()}:{os.getgid()}"
        return ContainerSpec(
            image=s.sandbox_image,
            name=f"agentic-sbx-{sanitize_filename(session_id)}",
            command=["python", f"{_DRIVER_DIR}/driver.py"],
            network=s.sandbox_network,
            memory_mb=s.sandbox_memory_mb,
            cpus=s.sandbox_cpus,
            pids_limit=s.sandbox_pids_limit,
            user=user,
            env=env,
            mounts=mounts,
            labels={"agentic-sandbox": "1", "agentic-session": session_id},
        )

    def _start_session(self, session_id: str, working_dir) -> ContainerSession:
        # The container runs as the host uid:gid (see _build_spec), so the
        # host-owned session dir is writable by the kernel without loosening its
        # permissions. No chmod needed.
        runtime = self._ensure_runtime()
        spec = self._build_spec(session_id, working_dir)
        if working_dir is not None:
            # Pre-create the /workspace/outputs mount point as the host user.
            # Docker would otherwise create this nested bind-mount target as root,
            # which pollutes the session dir and breaks host-side cleanup.
            (Path(working_dir) / "outputs").mkdir(parents=True, exist_ok=True)
        handle = runtime.start(spec)
        name = spec.name
        session = ContainerSession(
            session_id=session_id,
            handle=handle,
            interrupt=lambda: runtime.kill(name, "INT"),
            kill=lambda: runtime.kill(name),
            working_dir=working_dir,
            start_timeout=self._settings.sandbox_start_timeout,
            backend_name=self.backend_name,
        )
        try:
            session.wait_ready()
        except Exception:
            session.close()
            raise
        self._sessions[session_id] = session
        return session

    def execute(self, code, session_id, timeout_seconds=120, working_dir=None, inputs=None) -> ExecutionResult:
        avail = self._detect()
        if not avail.available:
            return ExecutionResult(
                success=False,
                error=(f"Docker sandbox backend unavailable ({avail.detail}). "
                       "Refusing to fall back to an unsandboxed kernel."),
            )
        ok, msg = kernel_exec.validate_code(code)
        if not ok:
            return ExecutionResult(success=False, error=msg)
        if inputs:
            from agentic_cli.tools.sandbox.manager import stage_inputs
            try:
                stage_inputs(working_dir, inputs)
            except ValueError as exc:
                return ExecutionResult(success=False, error=f"input staging failed: {exc}")
        session = self._sessions.get(session_id)
        if session is None or session.status == "dead":
            try:
                session = self._start_session(session_id, working_dir)
            except Exception as exc:
                return ExecutionResult(success=False, error=f"Failed to start sandbox: {exc}")
        result = session.execute(code, timeout_seconds)
        if result.success:
            outs = self._outputs_dir()
            # NOTE: outputs/ is shared/session-independent; it accumulates across runs and sessions (v1 accepted simplification).
            extra = [str(p) for p in sorted(outs.iterdir()) if p.is_file()]
            if extra:
                result.artifacts = list(result.artifacts) + extra
        return result

    def reset_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close()

    def cleanup(self) -> None:
        for session_id in list(self._sessions):
            self.reset_session(session_id)

    def has_session(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        return session is not None and session.status != "dead"

    def session_status(self, session_id: str) -> SessionStatus:
        session = self._sessions.get(session_id)
        if session is None:
            return SessionStatus(session_id=session_id, state="absent", backend=self.backend_name)
        return SessionStatus(
            session_id=session_id, state=session.status, backend=self.backend_name,
            container_id=session.container_id,
        )
