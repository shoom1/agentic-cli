"""Settings mixins for application identity and CLI/UI configuration.

AppSettingsMixin: Application identity and disk layout (app_name, workspace, paths).
CLISettingsMixin: CLI/UI display settings (logging, thinking output, theme, dialogs).

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

    @property
    def sessions_dir(self) -> Path:
        """Directory for session storage."""
        return self.workspace_dir / "sessions"

    @property
    def knowledge_base_dir(self) -> Path:
        """Directory for knowledge base storage."""
        return self.workspace_dir / "knowledge_base"


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


    # Thinking output control
    verbose_thinking: bool = Field(
        default=True,
        title="Verbose Thinking",
        description="Show detailed thinking/reasoning output in the UI",
        json_schema_extra={"ui_order": 35},
    )

    # Look and feel: the user's preferences (USER_PREFERENCE_KEYS). A project
    # file or a cwd .env cannot set them, and /settings shows them only when
    # an app lists them.
    ui_theme: Literal["dark", "light", "mono", "terminal", "auto"] = Field(
        default="dark",
        title="UI Theme",
        description="Color theme (auto: chosen from the terminal)",
        json_schema_extra={"ui_order": 36},
    )
    dialog_placement: Literal["box", "inline"] = Field(
        default="box",
        title="Dialog Placement",
        description="Dialogs in a box, or inline under the prompt",
        json_schema_extra={"ui_order": 37},
    )
