"""CLI settings mixin.

Provides settings for CLI/UI configuration.
These settings control logging, user identity, and activity tracking.
"""

from typing import Literal

from pydantic import Field


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

    # User identity
    default_user: str = Field(
        default="default_user",
        title="Default User",
        description="Default user identifier for sessions",
        json_schema_extra={"ui_order": 60},
    )
