"""Abstract base class for sandbox backends."""

from abc import ABC, abstractmethod
from pathlib import Path

from agentic_cli.tools.sandbox.models import ExecutionResult, SessionStatus


class SandboxBackend(ABC):
    """Abstract base for sandbox execution backends."""

    backend_name: str = "unknown"

    @abstractmethod
    def execute(
        self,
        code: str,
        session_id: str,
        timeout_seconds: int = 120,
        working_dir: Path | None = None,
    ) -> ExecutionResult:
        """Execute code in the given session.

        Args:
            code: Python code to execute.
            session_id: Session identifier.
            timeout_seconds: Maximum execution time.
            working_dir: Working directory for the session.

        Returns:
            ExecutionResult with output and metadata.
        """
        ...

    @abstractmethod
    def reset_session(self, session_id: str) -> None:
        """Reset (restart) a session's kernel/state."""
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up all sessions and resources."""
        ...

    @abstractmethod
    def has_session(self, session_id: str) -> bool:
        """Check if a session exists."""
        ...

    def session_status(self, session_id: str) -> SessionStatus:
        """Default status derived from has_session(); backends may override."""
        state = "ready" if self.has_session(session_id) else "absent"
        return SessionStatus(session_id=session_id, state=state, backend=self.backend_name)
