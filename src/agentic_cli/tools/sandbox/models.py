"""Data models for sandbox execution."""

from dataclasses import dataclass, field


@dataclass
class ExecutionResult:
    """Result of a sandbox code execution."""

    success: bool = True
    stdout: str = ""
    stderr: str = ""
    result: str | None = None
    artifacts: list[str] = field(default_factory=list)
    execution_time: float = 0.0
    error: str = ""


@dataclass
class SessionStatus:
    """Live status of a sandbox session."""

    session_id: str
    state: str  # "starting" | "ready" | "busy" | "dead" | "absent"
    backend: str
    execution_count: int = 0
    container_id: str | None = None
    detail: str = ""
