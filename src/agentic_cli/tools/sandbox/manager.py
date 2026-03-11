"""Sandbox manager for stateful code execution.

Manages sandbox sessions and delegates execution to a pluggable backend
(default: JupyterLocalBackend).
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from agentic_cli.logging import Loggers
from agentic_cli.persistence._utils import sanitize_filename
from agentic_cli.tools.sandbox.models import ExecutionResult

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings
    from agentic_cli.tools.sandbox.backends.base import SandboxBackend

logger = Loggers.tools()


@dataclass
class SandboxSession:
    """Metadata for an active sandbox session."""

    session_id: str
    working_dir: Path
    execution_count: int = 0


class SandboxManager:
    """Manages sandbox sessions and delegates to a backend.

    Args:
        settings: Application settings instance.
        backend: Optional backend for test injection. If None, created
            lazily from settings.sandbox_backend.
    """

    def __init__(
        self,
        settings: "BaseSettings",
        backend: "SandboxBackend | None" = None,
    ) -> None:
        self._settings = settings
        self._backend = backend
        self._sessions: dict[str, SandboxSession] = {}

        atexit.register(self._atexit_cleanup)
        logger.debug("sandbox_manager_created")

    def _ensure_backend(self) -> "SandboxBackend":
        """Lazily create the backend if not injected."""
        if self._backend is None:
            self._backend = self._create_backend(self._settings.sandbox_backend)
        return self._backend

    def _create_backend(self, backend_name: str) -> "SandboxBackend":
        """Create a backend by name."""
        if backend_name == "jupyter_local":
            from agentic_cli.tools.sandbox.backends.jupyter_local import JupyterLocalBackend
            return JupyterLocalBackend()
        raise ValueError(f"Unknown sandbox backend: {backend_name!r}")

    def _get_session_dir(self, session_id: str) -> Path:
        """Get or create the working directory for a session."""
        base = Path(self._settings.workspace_dir)
        safe_id = sanitize_filename(session_id)
        session_dir = base / "sandbox" / safe_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def execute(
        self,
        code: str,
        session_id: str = "default",
        timeout_seconds: int | None = None,
    ) -> ExecutionResult:
        """Execute code in a sandbox session.

        Args:
            code: Python code to execute.
            session_id: Session identifier (default: "default").
            timeout_seconds: Execution timeout (uses settings default if None).

        Returns:
            ExecutionResult with output and metadata.
        """
        max_sessions = self._settings.sandbox_max_sessions
        if session_id not in self._sessions and len(self._sessions) >= max_sessions:
            return ExecutionResult(
                success=False,
                error=f"Maximum sessions ({max_sessions}) reached. Reset an existing session first.",
            )

        if timeout_seconds is None:
            timeout_seconds = self._settings.sandbox_timeout

        # Get or create session metadata
        if session_id not in self._sessions:
            working_dir = self._get_session_dir(session_id)
            self._sessions[session_id] = SandboxSession(
                session_id=session_id,
                working_dir=working_dir,
            )

        session = self._sessions[session_id]
        backend = self._ensure_backend()

        result = backend.execute(
            code=code,
            session_id=session_id,
            timeout_seconds=timeout_seconds,
            working_dir=session.working_dir,
        )

        session.execution_count += 1
        logger.debug(
            "sandbox_executed",
            session_id=session_id,
            execution_count=session.execution_count,
            success=result.success,
        )
        return result

    def reset_session(self, session_id: str) -> bool:
        """Reset a sandbox session.

        Args:
            session_id: Session to reset.

        Returns:
            True if the session was active and reset, False if not found.
        """
        backend = self._ensure_backend()
        was_active = session_id in self._sessions
        if was_active:
            del self._sessions[session_id]
            backend.reset_session(session_id)
            logger.debug("sandbox_session_reset", session_id=session_id)
        return was_active

    def list_sessions(self) -> list[dict[str, Any]]:
        """List active sandbox sessions.

        Returns:
            List of session metadata dicts.
        """
        return [
            {
                "session_id": s.session_id,
                "working_dir": str(s.working_dir),
                "execution_count": s.execution_count,
            }
            for s in self._sessions.values()
        ]

    def cleanup(self) -> None:
        """Clean up all sessions and the backend."""
        atexit.unregister(self._atexit_cleanup)
        if self._backend is not None:
            self._backend.cleanup()
        self._sessions.clear()
        logger.debug("sandbox_manager_cleaned_up")

    def _atexit_cleanup(self) -> None:
        """Safety net for process exit."""
        try:
            self.cleanup()
        except Exception:
            pass
