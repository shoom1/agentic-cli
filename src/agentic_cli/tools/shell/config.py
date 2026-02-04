"""Configuration for shell security module.

Provides user-configurable settings for command allow/deny lists,
path boundaries, approval behavior, and resource limits.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agentic_cli.tools.shell.audit import AuditConfig
from agentic_cli.tools.shell.sandbox import ExecutionLimits


@dataclass
class ShellSecurityConfig:
    """Configuration for shell command security.

    Attributes:
        allowed_paths: Additional paths allowed beyond project root.
        allow_commands: Commands to always allow (override defaults).
        deny_commands: Commands to always deny (override defaults).
        allow_patterns: Regex patterns for allowed commands.
        deny_patterns: Regex patterns for denied commands.
        auto_approve_in_project: Auto-approve write ops in project directory.
        auto_approve_read_only: Auto-approve read-only commands.
        timeout_seconds: Default command timeout.
        max_output_bytes: Maximum output size before truncation.
        execution_limits: Resource limits for command execution.
        audit: Audit logging configuration.
        enable_preprocessing: Whether to run encoding detection.
        block_on_encoding: Whether to block commands with high obfuscation.
    """

    # Path boundaries
    allowed_paths: list[str] = field(default_factory=list)

    # Command lists
    allow_commands: list[str] = field(default_factory=list)
    deny_commands: list[str] = field(default_factory=list)
    allow_patterns: list[str] = field(default_factory=list)
    deny_patterns: list[str] = field(default_factory=list)

    # Approval settings
    auto_approve_in_project: bool = True
    auto_approve_read_only: bool = True

    # Resource limits (legacy - use execution_limits for full control)
    timeout_seconds: int = 60
    max_output_bytes: int = 50000

    # Extended configuration
    execution_limits: ExecutionLimits = field(default_factory=ExecutionLimits)
    audit: AuditConfig = field(default_factory=AuditConfig)

    # Preprocessing/encoding detection
    enable_preprocessing: bool = True
    block_on_encoding: bool = True  # Block high obfuscation score commands

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShellSecurityConfig":
        """Create config from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            ShellSecurityConfig instance.
        """
        # Parse execution limits if present
        limits_data = data.get("execution_limits", {})
        execution_limits = ExecutionLimits(
            timeout_seconds=limits_data.get("timeout_seconds", data.get("timeout_seconds", 60)),
            max_output_bytes=limits_data.get("max_output_bytes", data.get("max_output_bytes", 50000)),
            max_memory_mb=limits_data.get("max_memory_mb", 512),
            max_cpu_percent=limits_data.get("max_cpu_percent", 80),
            max_processes=limits_data.get("max_processes", 100),
            max_open_files=limits_data.get("max_open_files", 256),
        )

        # Parse audit config if present
        audit_data = data.get("audit", {})
        audit_config = AuditConfig(
            enabled=audit_data.get("enabled", True),
            log_dir=audit_data.get("log_dir", "~/.local/share/agentic-cli/audit"),
            retention_days=audit_data.get("retention_days", 30),
            max_preview_length=audit_data.get("max_preview_length", 500),
        )

        return cls(
            allowed_paths=data.get("allowed_paths", []),
            allow_commands=data.get("allow_commands", []),
            deny_commands=data.get("deny_commands", []),
            allow_patterns=data.get("allow_patterns", []),
            deny_patterns=data.get("deny_patterns", []),
            auto_approve_in_project=data.get("auto_approve_in_project", True),
            auto_approve_read_only=data.get("auto_approve_read_only", True),
            timeout_seconds=data.get("timeout_seconds", 60),
            max_output_bytes=data.get("max_output_bytes", 50000),
            execution_limits=execution_limits,
            audit=audit_config,
            enable_preprocessing=data.get("enable_preprocessing", True),
            block_on_encoding=data.get("block_on_encoding", True),
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ShellSecurityConfig":
        """Load config from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            ShellSecurityConfig instance.
        """
        path = Path(path)
        if not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    @classmethod
    def load_default(cls) -> "ShellSecurityConfig":
        """Load configuration from default location.

        Looks for config in:
        1. ~/.config/agentic-cli/shell_security.yaml
        2. ./shell_security.yaml (project local)

        Returns:
            ShellSecurityConfig instance.
        """
        # Check user config
        user_config = Path.home() / ".config" / "agentic-cli" / "shell_security.yaml"
        if user_config.exists():
            return cls.from_yaml(user_config)

        # Check project local config
        local_config = Path("shell_security.yaml")
        if local_config.exists():
            return cls.from_yaml(local_config)

        return cls()

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "allowed_paths": self.allowed_paths,
            "allow_commands": self.allow_commands,
            "deny_commands": self.deny_commands,
            "allow_patterns": self.allow_patterns,
            "deny_patterns": self.deny_patterns,
            "auto_approve_in_project": self.auto_approve_in_project,
            "auto_approve_read_only": self.auto_approve_read_only,
            "timeout_seconds": self.timeout_seconds,
            "max_output_bytes": self.max_output_bytes,
            "execution_limits": {
                "timeout_seconds": self.execution_limits.timeout_seconds,
                "max_output_bytes": self.execution_limits.max_output_bytes,
                "max_memory_mb": self.execution_limits.max_memory_mb,
                "max_cpu_percent": self.execution_limits.max_cpu_percent,
                "max_processes": self.execution_limits.max_processes,
                "max_open_files": self.execution_limits.max_open_files,
            },
            "audit": {
                "enabled": self.audit.enabled,
                "log_dir": self.audit.log_dir,
                "retention_days": self.audit.retention_days,
                "max_preview_length": self.audit.max_preview_length,
            },
            "enable_preprocessing": self.enable_preprocessing,
            "block_on_encoding": self.block_on_encoding,
        }

    def merge_with(self, other: "ShellSecurityConfig") -> "ShellSecurityConfig":
        """Merge this config with another (other takes precedence).

        Args:
            other: Config to merge with (takes precedence).

        Returns:
            New merged config.
        """
        return ShellSecurityConfig(
            allowed_paths=self.allowed_paths + other.allowed_paths,
            allow_commands=list(set(self.allow_commands) | set(other.allow_commands)),
            deny_commands=list(set(self.deny_commands) | set(other.deny_commands)),
            allow_patterns=self.allow_patterns + other.allow_patterns,
            deny_patterns=self.deny_patterns + other.deny_patterns,
            auto_approve_in_project=other.auto_approve_in_project,
            auto_approve_read_only=other.auto_approve_read_only,
            timeout_seconds=other.timeout_seconds,
            max_output_bytes=other.max_output_bytes,
            execution_limits=other.execution_limits,
            audit=other.audit,
            enable_preprocessing=other.enable_preprocessing,
            block_on_encoding=other.block_on_encoding,
        )


def get_strict_config() -> ShellSecurityConfig:
    """Get a strict security configuration.

    Prompts for ALL write operations, even in project directory.
    """
    return ShellSecurityConfig(
        auto_approve_in_project=False,
        auto_approve_read_only=True,
    )


def get_permissive_config() -> ShellSecurityConfig:
    """Get a permissive security configuration.

    Auto-approves most operations in project directory.
    """
    return ShellSecurityConfig(
        auto_approve_in_project=True,
        auto_approve_read_only=True,
        allowed_paths=["/tmp", str(Path.home() / ".cache")],
    )
