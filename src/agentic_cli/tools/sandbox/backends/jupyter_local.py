"""Jupyter-based local sandbox backend.

Uses jupyter_client.KernelManager + BlockingKernelClient for stateful
Python execution with per-session kernels.
"""

from __future__ import annotations

import base64
import re
import time
from pathlib import Path
from typing import Any

from jupyter_client import KernelManager
from jupyter_client.blocking import BlockingKernelClient

from agentic_cli.logging import Loggers
from agentic_cli.tools.sandbox.backends.base import SandboxBackend
from agentic_cli.tools.sandbox.models import ExecutionResult

logger = Loggers.tools()

# Regex to strip ANSI escape codes from tracebacks
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Blocked shell escape patterns
_BLOCKED_MAGICS = frozenset({"pip", "system", "sx"})

# Initialization code injected into new kernels to block network modules
_SANDBOX_INIT_CODE = '''
import sys as _sys

_BLOCKED_MODULES = frozenset({
    # Network
    'requests', 'urllib', 'http', 'httpx', 'aiohttp',
    'socket', 'ssl', 'ftplib', 'smtplib', 'poplib', 'imaplib',
    'xmlrpc', 'socketserver',
    # Package management
    'pip', 'ensurepip', 'setuptools', 'distutils',
    # Process execution
    'subprocess', 'shlex',
    # System-level
    'ctypes',
})

import importlib.abc as _abc
import importlib.machinery as _mach

class _RestrictedImportFinder(_abc.MetaPathFinder):
    """Meta-path finder that blocks restricted modules."""
    def find_spec(self, fullname, path, target=None):
        top = fullname.split('.')[0]
        if top in _BLOCKED_MODULES:
            raise ImportError(
                f"Module \\'{fullname}\\' is not available in the sandbox. "
                f"Use the appropriate tool instead (e.g., web_fetch for HTTP requests)."
            )
        return None

_sys.meta_path.insert(0, _RestrictedImportFinder())

# Remove pre-imported blocked modules from sys.modules so future
# `import X` goes through the meta_path hook. Existing kernel internals
# keep their references via their own module namespaces.
for _mod_name in list(_sys.modules):
    _top = _mod_name.split(".")[0]
    if _top in _BLOCKED_MODULES:
        del _sys.modules[_mod_name]
del _mod_name, _top, _sys
'''


class JupyterLocalBackend(SandboxBackend):
    """Local Jupyter kernel backend for stateful code execution.

    Each session_id maps to a dedicated KernelManager + BlockingKernelClient
    pair. State (variables, imports) persists across calls within a session.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, tuple[KernelManager, BlockingKernelClient]] = {}

    @staticmethod
    def _validate_code(code: str) -> tuple[bool, str]:
        """Pre-scan code for blocked shell escapes and magics."""
        for line in code.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("!"):
                return False, "Shell commands (!) are not allowed in the sandbox"
            if stripped.startswith("%") and not stripped.startswith("%%"):
                magic = stripped.lstrip("%").split()[0] if stripped.lstrip("%") else ""
                if magic in _BLOCKED_MAGICS:
                    return False, f"Magic command '%{magic}' is not allowed in the sandbox"
        return True, ""

    def _get_or_create_session(
        self, session_id: str, working_dir: Path | None = None,
    ) -> tuple[KernelManager, BlockingKernelClient]:
        """Get existing session or start a new kernel."""
        if session_id in self._sessions:
            return self._sessions[session_id]

        km = KernelManager()
        if working_dir:
            km.cwd = str(working_dir)

        km.start_kernel()
        kc = km.blocking_client()
        kc.start_channels()
        kc.wait_for_ready(timeout=30)

        # Inject sandbox restrictions
        init_msg_id = kc.execute(_SANDBOX_INIT_CODE)
        while True:
            try:
                msg = kc.get_iopub_msg(timeout=10)
                if (
                    msg.get("parent_header", {}).get("msg_id") == init_msg_id
                    and msg.get("content", {}).get("execution_state") == "idle"
                ):
                    break
            except TimeoutError:
                break

        self._sessions[session_id] = (km, kc)
        logger.debug("jupyter_session_started", session_id=session_id)
        return km, kc

    def execute(
        self,
        code: str,
        session_id: str,
        timeout_seconds: int = 120,
        working_dir: Path | None = None,
    ) -> ExecutionResult:
        """Execute code in a Jupyter kernel session."""
        # Pre-scan for blocked patterns
        valid, error = self._validate_code(code)
        if not valid:
            return ExecutionResult(success=False, error=error)

        start = time.monotonic()
        _, kc = self._get_or_create_session(session_id, working_dir)

        msg_id = kc.execute(code)

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        result_value: str | None = None
        artifacts: list[str] = []
        error_text = ""

        # Collect iopub messages until kernel goes idle
        while True:
            try:
                msg = kc.get_iopub_msg(timeout=timeout_seconds)
            except TimeoutError:
                elapsed = time.monotonic() - start
                return ExecutionResult(
                    success=False,
                    stdout="".join(stdout_parts),
                    stderr="".join(stderr_parts),
                    error=f"Execution timed out after {timeout_seconds}s",
                    execution_time=elapsed,
                )

            # Only process messages from our execution
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg.get("msg_type", "")
            content: dict[str, Any] = msg.get("content", {})

            if msg_type == "stream":
                name = content.get("name", "stdout")
                text = content.get("text", "")
                if name == "stderr":
                    stderr_parts.append(text)
                else:
                    stdout_parts.append(text)

            elif msg_type == "execute_result":
                data = content.get("data", {})
                result_value = data.get("text/plain", "")

            elif msg_type == "display_data":
                data = content.get("data", {})
                if "image/png" in data and working_dir:
                    # Save image artifact
                    artifact_dir = working_dir / "artifacts"
                    artifact_dir.mkdir(parents=True, exist_ok=True)
                    n = len(artifacts)
                    path = artifact_dir / f"plot_{n}.png"
                    img_bytes = base64.b64decode(data["image/png"])
                    path.write_bytes(img_bytes)
                    artifacts.append(str(path))
                    logger.debug("artifact_saved", path=str(path))

            elif msg_type == "error":
                traceback_lines = content.get("traceback", [])
                raw = "\n".join(traceback_lines)
                error_text = _ANSI_RE.sub("", raw)

            elif msg_type == "status" and content.get("execution_state") == "idle":
                break

        elapsed = time.monotonic() - start
        return ExecutionResult(
            success=not error_text,
            stdout="".join(stdout_parts),
            stderr="".join(stderr_parts),
            result=result_value,
            artifacts=artifacts,
            execution_time=elapsed,
            error=error_text,
        )

    def reset_session(self, session_id: str) -> None:
        """Restart the kernel for a session."""
        if session_id not in self._sessions:
            return
        km, kc = self._sessions.pop(session_id)
        try:
            kc.stop_channels()
            km.shutdown_kernel(now=True)
        except Exception:
            logger.debug("jupyter_session_reset_error", session_id=session_id, exc_info=True)
        logger.debug("jupyter_session_reset", session_id=session_id)

    def cleanup(self) -> None:
        """Shut down all kernels."""
        for session_id in list(self._sessions):
            self.reset_session(session_id)

    def has_session(self, session_id: str) -> bool:
        """Check if a session exists."""
        return session_id in self._sessions
