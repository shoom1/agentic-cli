"""The local backend: a Jupyter kernel on the host, with no isolation.

Uses jupyter_client.KernelManager + BlockingKernelClient for stateful
Python execution with per-session kernels. The kernel runs as the user, with
the user's files, network and privileges; the code runs as written (shell
escapes and magics included). Isolation is the docker backend's job.
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


class JupyterLocalBackend(SandboxBackend):
    """Local Jupyter kernel backend for stateful code execution.

    Each session_id maps to a dedicated KernelManager + BlockingKernelClient
    pair. State (variables, imports) persists across calls within a session.
    Nothing is restricted: CPython offers no in-process sandbox, and a
    restriction that can be undone from the code it restricts only breaks
    libraries.
    """

    backend_name = "jupyter_local"

    def __init__(self, interrupt_grace: float = 10.0) -> None:
        self._sessions: dict[str, tuple[KernelManager, BlockingKernelClient]] = {}
        # How long a timed-out cell gets to stop after an interrupt before the
        # kernel is restarted (the docker backend's interrupt_grace).
        self._interrupt_grace = interrupt_grace

    def _get_or_create_session(
        self, session_id: str, working_dir: Path | None = None,
    ) -> tuple[KernelManager, BlockingKernelClient]:
        """Get existing session or start a new kernel."""
        if session_id in self._sessions:
            return self._sessions[session_id]

        km = KernelManager()
        start_kwargs: dict = {}
        if working_dir:
            start_kwargs["cwd"] = str(working_dir)
        km.start_kernel(**start_kwargs)
        kc = km.blocking_client()
        kc.start_channels()
        kc.wait_for_ready(timeout=30)

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
        if working_dir is not None:
            (Path(working_dir) / "outputs").mkdir(parents=True, exist_ok=True)

        if inputs:
            from agentic_cli.tools.sandbox.manager import stage_inputs
            try:
                stage_inputs(working_dir, inputs)
            except ValueError as exc:
                return ExecutionResult(success=False, error=f"input staging failed: {exc}")

        km, kc = self._get_or_create_session(session_id, working_dir)
        msg_id = kc.execute(code)
        data = kernel_exec.collect_execution(kc, msg_id, timeout_seconds, working_dir)
        if data.pop("timed_out"):
            data = self._stop_timed_out_cell(session_id, km, kc, msg_id, data, working_dir)
        return ExecutionResult(**data)

    def _stop_timed_out_cell(
        self, session_id: str, km, kc, msg_id: str, data: dict, working_dir,
    ) -> dict:
        """Stop a cell that ran past its deadline, so the session stays usable.

        Left running, the cell keeps the kernel busy and every later call in
        the session queues behind it. Interrupt it (KeyboardInterrupt in the
        cell) and give it ``interrupt_grace`` to finish; a cell that does not
        stop gets a new kernel, which loses the session's variables.
        """
        timeout_msg = data["error"]
        km.interrupt_kernel()
        rest = kernel_exec.collect_execution(kc, msg_id, self._interrupt_grace, working_dir)
        merged = {
            **data,
            "stdout": data["stdout"] + rest["stdout"],
            "stderr": data["stderr"] + rest["stderr"],
            "artifacts": data["artifacts"] + rest["artifacts"],
            "execution_time": data["execution_time"] + rest["execution_time"],
        }
        if not rest["timed_out"]:
            merged["error"] = (
                f"{timeout_msg}; the cell was interrupted and the session's "
                "variables are kept."
            )
            return merged
        logger.warning("jupyter_interrupt_ignored", session_id=session_id)
        self.reset_session(session_id)
        merged["error"] = (
            f"{timeout_msg}; the cell did not stop when interrupted, so the kernel "
            "was restarted and the session's variables are lost."
        )
        return merged

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
