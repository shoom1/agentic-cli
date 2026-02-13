"""Workflow settings mixin.

Provides settings for agentic workflow configuration, independent of UI.
These settings control model selection, orchestration, retry behavior,
HITL (human-in-the-loop), and memory management.
"""

from enum import Enum
from typing import Literal

from pydantic import Field


class OrchestratorType(str, Enum):
    """Types of workflow orchestrators available."""

    ADK = "adk"
    LANGGRAPH = "langgraph"


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
    orchestrator: OrchestratorType = Field(
        default=OrchestratorType.ADK,
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
    python_executor_max_memory_mb: int = Field(
        default=512,
        title="Python Executor Memory Limit",
        description="Maximum memory for Python executor subprocess (MB, Unix only)",
        json_schema_extra={"ui_order": 121},
    )

    # HITL (Human-in-the-Loop) settings
    hitl_enabled: bool = Field(
        default=True,
        title="HITL Enabled",
        description="Enable human-in-the-loop features (approvals)",
        json_schema_extra={"ui_order": 130},
    )

    # Persistence settings (LangGraph)
    postgres_uri: str | None = Field(
        default=None,
        title="PostgreSQL URI",
        description="PostgreSQL connection URI for persistent storage",
        json_schema_extra={"ui_order": 145},
    )
    sqlite_uri: str | None = Field(
        default=None,
        title="SQLite URI",
        description="SQLite connection URI or file path for persistent storage",
        json_schema_extra={"ui_order": 146},
    )
    store_type: Literal["memory", "postgres"] | None = Field(
        default="memory",
        title="Store Type",
        description="Store type for long-term memory (memory or postgres)",
        json_schema_extra={"ui_order": 147},
    )

    # Shell execution settings (for shell middleware)
    shell_sandbox_type: Literal["host", "docker"] = Field(
        default="host",
        title="Shell Sandbox Type",
        description="Execution environment for shell commands",
        json_schema_extra={"ui_order": 148},
    )
    shell_docker_image: str = Field(
        default="python:3.12-slim",
        title="Shell Docker Image",
        description="Docker image to use for sandboxed shell execution",
        json_schema_extra={"ui_order": 149},
    )
    shell_timeout: int = Field(
        default=60,
        title="Shell Timeout",
        description="Default timeout in seconds for shell commands",
        json_schema_extra={"ui_order": 150},
    )

    # LLM debugging settings
    raw_llm_logging: bool = Field(
        default=False,
        title="Raw LLM Logging",
        description="Enable logging of raw LLM request/response traffic for debugging",
        json_schema_extra={"ui_order": 160},
    )
    prompt_caching_enabled: bool = Field(
        default=True,
        title="Prompt Caching",
        description="Enable prompt caching for supported models (reduces cost and latency)",
        json_schema_extra={"ui_order": 170},
    )
