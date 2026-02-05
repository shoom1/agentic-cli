"""Audit logging for shell command execution.

Layer 8: Audit Logging
- Log all commands for forensics and review
- JSONL format with daily rotation
- Query interface for searching history
"""

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from agentic_cli.tools.shell.models import ApprovalType, RiskLevel


@dataclass
class AuditEntry:
    """A single audit log entry for a shell command.

    Attributes:
        timestamp: When the command was executed/attempted.
        session_id: Unique identifier for the session.
        command_original: The original command string.
        command_normalized: Command after preprocessing (if different).
        risk_level: Assessed risk level.
        risk_factors: List of factors contributing to risk.
        paths_accessed: Paths the command would access.
        approval_type: Type of approval required.
        user_response: User's approval response (if prompted).
        executed: Whether the command was actually executed.
        exit_code: Exit code if executed.
        duration_ms: Execution duration in milliseconds.
        stdout_preview: First N chars of stdout.
        stderr_preview: First N chars of stderr.
        blocked_reason: Reason if command was blocked.
        encoding_detected: Any encoding/obfuscation detected.
    """
    timestamp: str  # ISO format
    session_id: str
    command_original: str
    risk_level: str
    approval_type: str
    executed: bool

    # Optional fields
    command_normalized: str | None = None
    risk_factors: list[str] = field(default_factory=list)
    paths_accessed: list[str] = field(default_factory=list)
    user_response: str | None = None
    exit_code: int | None = None
    duration_ms: int | None = None
    stdout_preview: str = ""
    stderr_preview: str = ""
    blocked_reason: str | None = None
    encoding_detected: list[str] = field(default_factory=list)
    working_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None and v != [] and v != ""}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEntry":
        """Create from dictionary."""
        return cls(
            timestamp=data.get("timestamp", ""),
            session_id=data.get("session_id", ""),
            command_original=data.get("command_original", ""),
            command_normalized=data.get("command_normalized"),
            risk_level=data.get("risk_level", "low"),
            risk_factors=data.get("risk_factors", []),
            paths_accessed=data.get("paths_accessed", []),
            approval_type=data.get("approval_type", "auto"),
            user_response=data.get("user_response"),
            executed=data.get("executed", False),
            exit_code=data.get("exit_code"),
            duration_ms=data.get("duration_ms"),
            stdout_preview=data.get("stdout_preview", ""),
            stderr_preview=data.get("stderr_preview", ""),
            blocked_reason=data.get("blocked_reason"),
            encoding_detected=data.get("encoding_detected", []),
            working_dir=data.get("working_dir"),
        )


@dataclass
class AuditConfig:
    """Configuration for audit logging.

    Attributes:
        enabled: Whether audit logging is enabled.
        log_dir: Directory for audit logs.
        retention_days: How long to keep logs.
        max_preview_length: Maximum length for stdout/stderr previews.
    """
    enabled: bool = True
    log_dir: str = "~/.local/share/agentic-cli/audit"
    retention_days: int = 30
    max_preview_length: int = 500

    def get_log_dir(self) -> Path:
        """Get resolved log directory path."""
        return Path(self.log_dir).expanduser()


class AuditLogger:
    """Logger for shell command audit entries.

    Writes entries in JSONL format with daily rotation.
    """

    def __init__(
        self,
        config: AuditConfig | None = None,
        session_id: str | None = None,
    ):
        """Initialize the audit logger.

        Args:
            config: Audit configuration.
            session_id: Session identifier for grouping entries.
        """
        self.config = config or AuditConfig()
        self.session_id = session_id or self._generate_session_id()
        self._ensure_log_dir()

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        import uuid
        return str(uuid.uuid4())[:8]

    def _ensure_log_dir(self) -> None:
        """Ensure the log directory exists."""
        if not self.config.enabled:
            return

        log_dir = self.config.get_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)

    def _get_log_file(self, date: datetime | None = None) -> Path:
        """Get the log file path for a given date."""
        if date is None:
            date = datetime.now()
        filename = f"shell_audit_{date.strftime('%Y-%m-%d')}.jsonl"
        return self.config.get_log_dir() / filename

    def log(self, entry: AuditEntry) -> None:
        """Write an audit entry to the log.

        Args:
            entry: The audit entry to log.
        """
        if not self.config.enabled:
            return

        log_file = self._get_log_file()

        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except OSError as e:
            # Don't fail command execution due to audit logging issues
            import warnings
            warnings.warn(f"Failed to write audit log: {e}")

    def log_command(
        self,
        command: str,
        risk_level: RiskLevel,
        approval_type: ApprovalType,
        executed: bool,
        working_dir: str | Path | None = None,
        normalized_command: str | None = None,
        risk_factors: list[str] | None = None,
        paths: list[str] | None = None,
        user_response: str | None = None,
        exit_code: int | None = None,
        duration_ms: int | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
        blocked_reason: str | None = None,
        encoding_detected: list[str] | None = None,
    ) -> AuditEntry:
        """Log a shell command execution.

        Convenience method that creates and logs an AuditEntry.

        Args:
            command: The original command.
            risk_level: Assessed risk level.
            approval_type: Type of approval required/given.
            executed: Whether the command was executed.
            working_dir: Working directory.
            normalized_command: Preprocessed command (if different).
            risk_factors: Risk factors identified.
            paths: Paths accessed by the command.
            user_response: User's approval response.
            exit_code: Exit code if executed.
            duration_ms: Duration in milliseconds.
            stdout: Standard output (will be truncated).
            stderr: Standard error (will be truncated).
            blocked_reason: Reason if blocked.
            encoding_detected: Encodings detected.

        Returns:
            The created AuditEntry.
        """
        max_len = self.config.max_preview_length

        entry = AuditEntry(
            timestamp=datetime.now().isoformat(),
            session_id=self.session_id,
            command_original=command,
            command_normalized=normalized_command if normalized_command != command else None,
            risk_level=risk_level.value,
            risk_factors=risk_factors or [],
            paths_accessed=paths or [],
            approval_type=approval_type.value,
            user_response=user_response,
            executed=executed,
            exit_code=exit_code,
            duration_ms=duration_ms,
            stdout_preview=stdout[:max_len] if stdout else "",
            stderr_preview=stderr[:max_len] if stderr else "",
            blocked_reason=blocked_reason,
            encoding_detected=encoding_detected or [],
            working_dir=str(working_dir) if working_dir else None,
        )

        self.log(entry)
        return entry

    def query(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        command_pattern: str | None = None,
        risk_level: RiskLevel | None = None,
        executed_only: bool = False,
        blocked_only: bool = False,
        session_id: str | None = None,
        limit: int = 100,
    ) -> Iterator[AuditEntry]:
        """Query audit log entries.

        Args:
            start_date: Start of date range.
            end_date: End of date range.
            command_pattern: Substring to match in commands.
            risk_level: Filter by risk level.
            executed_only: Only return executed commands.
            blocked_only: Only return blocked commands.
            session_id: Filter by session ID.
            limit: Maximum entries to return.

        Yields:
            Matching AuditEntry objects.
        """
        if not self.config.enabled:
            return

        log_dir = self.config.get_log_dir()
        if not log_dir.exists():
            return

        # Determine date range
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=7)

        # Iterate through log files in date range
        current = start_date
        count = 0

        while current <= end_date and count < limit:
            log_file = self._get_log_file(current)

            if log_file.exists():
                try:
                    with open(log_file) as f:
                        for line in f:
                            if count >= limit:
                                return

                            try:
                                data = json.loads(line.strip())
                                entry = AuditEntry.from_dict(data)

                                # Apply filters
                                if command_pattern and command_pattern not in entry.command_original:
                                    continue
                                if risk_level and entry.risk_level != risk_level.value:
                                    continue
                                if executed_only and not entry.executed:
                                    continue
                                if blocked_only and entry.blocked_reason is None:
                                    continue
                                if session_id and entry.session_id != session_id:
                                    continue

                                yield entry
                                count += 1

                            except json.JSONDecodeError:
                                continue  # Skip malformed lines
                except OSError:
                    pass  # Skip unreadable files

            current += timedelta(days=1)

    def cleanup_old_logs(self) -> int:
        """Remove logs older than retention period.

        Returns:
            Number of files removed.
        """
        if not self.config.enabled:
            return 0

        log_dir = self.config.get_log_dir()
        if not log_dir.exists():
            return 0

        cutoff = datetime.now() - timedelta(days=self.config.retention_days)
        removed = 0

        for log_file in log_dir.glob("shell_audit_*.jsonl"):
            try:
                # Extract date from filename
                date_str = log_file.stem.replace("shell_audit_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d")

                if file_date < cutoff:
                    log_file.unlink()
                    removed += 1
            except (ValueError, OSError):
                continue  # Skip files with unexpected format

        return removed

    def get_session_summary(self, session_id: str | None = None) -> dict[str, Any]:
        """Get summary statistics for a session.

        Args:
            session_id: Session to summarize (default: current session).

        Returns:
            Dictionary with summary statistics.
        """
        session_id = session_id or self.session_id
        entries = list(self.query(session_id=session_id, limit=10000))

        if not entries:
            return {
                "session_id": session_id,
                "total_commands": 0,
            }

        executed = [e for e in entries if e.executed]
        blocked = [e for e in entries if e.blocked_reason]

        risk_counts = {}
        for entry in entries:
            risk_counts[entry.risk_level] = risk_counts.get(entry.risk_level, 0) + 1

        return {
            "session_id": session_id,
            "total_commands": len(entries),
            "executed": len(executed),
            "blocked": len(blocked),
            "pending": len(entries) - len(executed) - len(blocked),
            "risk_distribution": risk_counts,
            "first_command": entries[0].timestamp if entries else None,
            "last_command": entries[-1].timestamp if entries else None,
        }
