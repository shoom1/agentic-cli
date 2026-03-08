"""Abstract base class for sandbox backends."""

from abc import ABC, abstractmethod
from pathlib import Path

from agentic_cli.tools.sandbox.models import ExecutionResult


class SandboxBackend(ABC):
    """Abstract base for sandbox execution backends."""

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
