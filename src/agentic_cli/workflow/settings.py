"""Workflow settings mixin.

Provides settings for agentic workflow configuration, independent of UI.
These settings control model selection, orchestration, retry behavior,
HITL (human-in-the-loop), and memory management.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import Field


class WorkflowSettingsMixin:
    """Settings for agentic workflow configuration.

    Mixin class that provides workflow-specific settings.
    Should be composed with BaseSettings via multiple inheritance.

    Note: This is a mixin, not a BaseSettings subclass, to avoid
    MRO issues when composed with other settings classes.
    """

    # Model configuration
    default_model: str | None = Field(
        default=None,
        title="Model",
        description="Default model to use (auto-detected if not set)",
        json_schema_extra={"ui_order": 10},
    )
    thinking_effort: Literal["none", "low", "medium", "high"] = Field(
        default="medium",
        title="Thinking Effort",
        description="Controls depth of reasoning for models that support it",
        json_schema_extra={"ui_order": 20},
    )

    # Orchestrator selection
    orchestrator: Literal["adk", "langgraph"] = Field(
        default="adk",
        title="Orchestrator",
        description="Workflow orchestrator backend",
        json_schema_extra={"ui_order": 100},  # Advanced setting
    )
    langgraph_checkpointer: Literal["memory", "postgres"] | None = Field(
        default="memory",
        title="LangGraph Checkpointer",
        description="LangGraph state persistence type",
        json_schema_extra={"ui_order": 101},
    )

    # Retry configuration
    retry_max_attempts: int = Field(
        default=3,
        title="Max Retry Attempts",
        description="Maximum retry attempts for transient errors",
        json_schema_extra={"ui_order": 110},
    )
    retry_initial_delay: float = Field(
        default=2.0,
        title="Retry Initial Delay",
        description="Initial delay in seconds before first retry",
        json_schema_extra={"ui_order": 111},
    )
    retry_backoff_factor: float = Field(
        default=2.0,
        title="Retry Backoff Factor",
        description="Multiplier for exponential backoff between retries",
        json_schema_extra={"ui_order": 112},
    )

    # Python executor
    python_executor_timeout: int = Field(
        default=30,
        title="Python Executor Timeout",
        description="Default timeout for Python execution (seconds)",
        json_schema_extra={"ui_order": 120},
    )

    # HITL (Human-in-the-Loop) settings
    hitl_enabled: bool = Field(
        default=True,
        title="HITL Enabled",
        description="Enable human-in-the-loop features (approvals, checkpoints)",
        json_schema_extra={"ui_order": 130},
    )
    hitl_checkpoint_enabled: bool = Field(
        default=True,
        title="Checkpoints Enabled",
        description="Enable checkpoint creation for review points",
        json_schema_extra={"ui_order": 131},
    )
    hitl_feedback_enabled: bool = Field(
        default=True,
        title="Feedback Enabled",
        description="Enable feedback collection at checkpoints",
        json_schema_extra={"ui_order": 132},
    )
    hitl_default_rules: list[dict[str, Any]] = Field(
        default_factory=list,
        title="Default Approval Rules",
        description="Default approval rules as dicts (tool, operations, auto_approve_patterns)",
        json_schema_extra={"ui_order": 133},
    )

    # Memory settings
    memory_persistence_path: Path | None = Field(
        default=None,
        title="Memory Persistence Path",
        description="Path for persisting long-term memory (None for default)",
        json_schema_extra={"ui_order": 140},
    )
