"""Settings mixins for application identity and CLI/UI configuration.

AppSettingsMixin: Application identity and disk layout (app_name, workspace, paths).
CLISettingsMixin: CLI/UI-specific display settings (logging, thinking output).

These modules live outside cli/ so that config.py can compose BaseSettings
without importing the cli package.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator


class AppSettingsMixin:
    """Settings for application identity and disk layout.

    Mixin class that provides:
    - Application name and workspace directory
    - Path expansion for workspace_dir
    - Workspace directory creation
    - Derived path properties (sessions, artifacts, knowledge base)

    Should be composed with BaseSettings via multiple inheritance.
    """

    # Application identity (domain projects should override)
    app_name: str = Field(
        default="agentic_cli",
        title="App Name",
        description="Application name for agent services",
        json_schema_extra={"ui_order": 200},  # Not typically shown in UI
    )

    # Paths (domain projects should override workspace_dir default)
    workspace_dir: Path = Field(
        default_factory=lambda: Path.home() / ".agentic",
        title="Workspace Directory",
        description="Directory for storing artifacts and sessions",
        json_schema_extra={"ui_order": 201},
    )

    @field_validator("workspace_dir", mode="before")
    @classmethod
    def expand_path(cls, v: str | Path) -> Path:
        """Expand ~ and environment variables in paths."""
        if isinstance(v, str):
            return Path(v).expanduser()
        return v

    def ensure_workspace_exists(self) -> None:
        """Create workspace directory if it doesn't exist."""
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    @property
    def sessions_dir(self) -> Path:
        """Directory for session storage."""
        return self.workspace_dir / "sessions"

    @property
    def artifacts_dir(self) -> Path:
        """Directory for artifact storage."""
        return self.workspace_dir / "workspace"

    @property
    def knowledge_base_dir(self) -> Path:
        """Directory for knowledge base storage."""
        return self.workspace_dir / "knowledge_base"

    @property
    def knowledge_base_documents_dir(self) -> Path:
        """Directory for knowledge base documents."""
        return self.knowledge_base_dir / "documents"

    @property
    def knowledge_base_embeddings_dir(self) -> Path:
        """Directory for knowledge base embeddings."""
        return self.knowledge_base_dir / "embeddings"


class CLISettingsMixin:
    """Settings for CLI/UI configuration.

    Mixin class that provides CLI-specific settings.
    Should be composed with BaseSettings via multiple inheritance.

    Note: This is a mixin, not a BaseSettings subclass, to avoid
    MRO issues when composed with other settings classes.
    """

    # Logging configuration
    log_level: Literal["debug", "info", "warning", "error"] = Field(
        default="warning",
        title="Log Level",
        description="Logging verbosity level",
        json_schema_extra={"ui_order": 50},
    )
    log_format: Literal["console", "json"] = Field(
        default="console",
        title="Log Format",
        description="Log output format (console for dev, json for production)",
        json_schema_extra={"ui_order": 51},
    )

    # Activity logging
    log_activity: bool = Field(
        default=False,
        title="Log Activity",
        description="Save conversation to file for audit purposes",
        json_schema_extra={"ui_order": 30},
    )

    # Thinking output control
    verbose_thinking: bool = Field(
        default=True,
        title="Verbose Thinking",
        description="Show detailed thinking/reasoning output in the UI",
        json_schema_extra={"ui_order": 35},
    )
