"""Jupyter-based local sandbox backend.

Uses jupyter_client.KernelManager + BlockingKernelClient for stateful
Python execution with per-session kernels.
"""

from __future__ import annotations

from pathlib import Path

from jupyter_client import KernelManager
from jupyter_client.blocking import BlockingKernelClient

from agentic_cli.logging import Loggers
from agentic_cli.tools.sandbox.backends import kernel_exec
from agentic_cli.tools.sandbox.backends.base import SandboxBackend
from agentic_cli.tools.sandbox.models import ExecutionResult

logger = Loggers.tools()

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

    backend_name = "jupyter_local"

    def __init__(self) -> None:
        self._sessions: dict[str, tuple[KernelManager, BlockingKernelClient]] = {}

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
        inputs: list[str] | None = None,
    ) -> ExecutionResult:
        """Execute code in a Jupyter kernel session."""
        # Pre-scan for blocked patterns
        valid, error = kernel_exec.validate_code(code)
        if not valid:
            return ExecutionResult(success=False, error=error)

        if inputs:
            from agentic_cli.tools.sandbox.manager import stage_inputs
            try:
                stage_inputs(working_dir, inputs)
            except ValueError as exc:
                return ExecutionResult(success=False, error=f"input staging failed: {exc}")

        _, kc = self._get_or_create_session(session_id, working_dir)
        msg_id = kc.execute(code)
        data = kernel_exec.collect_execution(kc, msg_id, timeout_seconds, working_dir)
        return ExecutionResult(**data)

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
