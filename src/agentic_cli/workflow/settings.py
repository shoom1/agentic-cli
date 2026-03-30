"""Workflow settings mixin.

Provides settings for agentic workflow configuration, independent of UI.
These settings control model selection, orchestration, retry behavior,
HITL (human-in-the-loop), memory management, API keys, and tool configuration.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, TYPE_CHECKING

from pydantic import Field

from agentic_cli.workflow.models import ModelFamily, ModelRegistry

if TYPE_CHECKING:
    pass

# Thinking effort levels (module-level constant for backward compatibility)
THINKING_EFFORT_LEVELS = ModelRegistry.THINKING_EFFORT_LEVELS


class OrchestratorType(str, Enum):
    """Types of workflow orchestrators available."""

    ADK = "adk"
    LANGGRAPH = "langgraph"


class WorkflowSettingsMixin:
    """Settings for agentic workflow configuration.

    Mixin class that provides workflow-specific settings including:
    - Model selection and configuration
    - API keys and provider detection
    - Orchestrator and retry settings
    - Tool configuration (web search/fetch, KB, shell, executor)
    - HITL and persistence settings

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

    # Context window management
    context_window_enabled: bool = Field(
        default=False,
        title="Context Window Management",
        description="Enable automatic context window management to prevent overflow",
        json_schema_extra={"ui_order": 25},
    )
    context_window_trigger_tokens: int = Field(
        default=100_000,
        title="Context Trigger Tokens",
        description="Start trimming when context exceeds this token count",
        json_schema_extra={"ui_order": 26},
    )
    context_window_target_tokens: int = Field(
        default=80_000,
        title="Context Target Tokens",
        description="Target token count after trimming",
        json_schema_extra={"ui_order": 27},
    )

    # API Keys (common across all domains, never saved to JSON)
    google_api_key: str | None = Field(
        default=None,
        description="Google API key for Gemini models",
        validation_alias="GOOGLE_API_KEY",
    )
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key for Claude models",
        validation_alias="ANTHROPIC_API_KEY",
    )
    tavily_api_key: str | None = Field(
        default=None,
        description="Tavily API key for web search",
        validation_alias="TAVILY_API_KEY",
    )
    brave_api_key: str | None = Field(
        default=None,
        description="Brave Search API key for web search",
        validation_alias="BRAVE_API_KEY",
    )

    # Web search configuration
    search_backend: Literal["tavily", "brave"] | None = Field(
        default=None,
        title="Search Backend",
        description="Web search provider to use (tavily or brave)",
        json_schema_extra={"ui_order": 55},
    )

    # Web fetch configuration
    webfetch_model: str | None = Field(
        default=None,
        title="WebFetch Model",
        description="Model for summarizing fetched content (None = auto-detect)",
        json_schema_extra={"ui_order": 56},
    )
    webfetch_blocked_domains: list[str] = Field(
        default_factory=list,
        title="WebFetch Blocked Domains",
        description="Domains to block from fetching (supports wildcards like *.example.com)",
        json_schema_extra={"ui_order": 57},
    )
    webfetch_cache_ttl_seconds: int = Field(
        default=900,
        title="WebFetch Cache TTL",
        description="Cache TTL in seconds for fetched pages (default: 15 minutes)",
        json_schema_extra={"ui_order": 58},
    )
    webfetch_max_content_bytes: int = Field(
        default=102400,
        title="WebFetch Max Content",
        description="Maximum content size in bytes (default: 100KB)",
        json_schema_extra={"ui_order": 59},
    )
    webfetch_max_pdf_bytes: int = Field(
        default=5242880,
        title="WebFetch Max PDF Size",
        description="Maximum PDF size in bytes (default: 5MB). Separate from HTML limit because PDFs are larger but extracted text is compact.",
        json_schema_extra={"ui_order": 60},
    )

    # Knowledge Base configuration
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        title="Embedding Model",
        description="Sentence transformer model for embeddings",
        json_schema_extra={"ui_order": 150},
    )
    embedding_batch_size: int = Field(
        default=32,
        title="Embedding Batch Size",
        description="Batch size for embedding generation",
        json_schema_extra={"ui_order": 151},
    )
    knowledge_base_use_mock: bool = Field(
        default=False,
        title="Use Mock Knowledge Base",
        description="Use mock knowledge base (no ML dependencies required)",
        json_schema_extra={"ui_order": 152},
    )
    enable_tool_reflections: bool = Field(
        default=False,
        title="Enable Tool Reflections",
        description="Store and inject heuristics learned from tool failures",
        json_schema_extra={"ui_order": 153},
    )
    auto_extract_session_facts: bool = Field(
        default=False,
        title="Auto-Extract Session Facts",
        description="At session end, extract key facts into memory via LLM",
        json_schema_extra={"ui_order": 154},
    )

    # User identity (needed by workflow.process())
    default_user: str = Field(
        default="default_user",
        title="Default User",
        description="Default user identifier for sessions",
        json_schema_extra={"ui_order": 60},
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

    # Sandbox executor (stateful Jupyter-backed execution)
    sandbox_backend: str = Field(
        default="jupyter_local",
        title="Sandbox Backend",
        description="Backend for stateful sandbox execution",
        json_schema_extra={"ui_order": 122},
    )
    sandbox_timeout: int = Field(
        default=120,
        title="Sandbox Timeout",
        description="Default timeout for sandbox execution (seconds)",
        json_schema_extra={"ui_order": 123},
    )
    sandbox_max_sessions: int = Field(
        default=5,
        title="Sandbox Max Sessions",
        description="Maximum concurrent sandbox sessions",
        json_schema_extra={"ui_order": 124},
    )
    sandbox_packages: list[str] = Field(
        default_factory=list,
        title="Sandbox Packages",
        description="Additional pip packages to pre-install in sandbox sessions (informational for local backend, drives image build for Docker backend)",
        json_schema_extra={"ui_order": 125},
    )

    # OS-level sandboxing
    os_sandbox_enabled: bool = Field(
        default=False,
        title="OS Sandbox Enabled",
        description="Enable OS-level sandboxing for shell and Python execution (requires sandbox-exec on macOS or bwrap on Linux)",
        json_schema_extra={"ui_order": 130},
    )
    os_sandbox_writable_paths: list[str] = Field(
        default_factory=list,
        title="OS Sandbox Writable Paths",
        description="Additional paths the sandboxed process can write to (working directory is always writable)",
        json_schema_extra={"ui_order": 131},
    )
    os_sandbox_allow_network: bool = Field(
        default=False,
        title="OS Sandbox Allow Network",
        description="Allow network access from sandboxed processes",
        json_schema_extra={"ui_order": 132},
    )

    # HITL (Human-in-the-Loop) settings
    hitl_enabled: bool = Field(
        default=True,
        title="HITL Enabled",
        description="Enable human-in-the-loop features (approvals)",
        json_schema_extra={"ui_order": 135},
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

    # === Model registry (not a pydantic field) ===
    # Set by BaseWorkflowManager after refresh()
    _model_registry: ModelRegistry | None = None

    def set_model_registry(self, registry: ModelRegistry) -> None:
        """Attach a populated ModelRegistry to this settings instance."""
        object.__setattr__(self, "_model_registry", registry)

    # === API key properties ===

    @property
    def has_google_key(self) -> bool:
        """Check if Google API key is available."""
        return bool(self.google_api_key)

    @property
    def has_anthropic_key(self) -> bool:
        """Check if Anthropic API key is available."""
        return bool(self.anthropic_api_key)

    @property
    def has_any_api_key(self) -> bool:
        """Check if any API key is available."""
        return self.has_google_key or self.has_anthropic_key

    # === Model helpers ===

    @property
    def default_model_google(self) -> str:
        """Default Google model."""
        registry = self._get_registry()
        return registry.get_default(ModelFamily.GEMINI)

    @property
    def default_model_anthropic(self) -> str:
        """Default Anthropic model."""
        registry = self._get_registry()
        return registry.get_default(ModelFamily.CLAUDE)

    def _get_registry(self) -> ModelRegistry:
        """Get the model registry (creates a default if none attached)."""
        reg = getattr(self, "_model_registry", None)
        if reg is not None:
            return reg
        # Return a fresh (un-refreshed) registry for fallback usage
        return ModelRegistry()

    def get_model(self) -> str:
        """Get the model to use based on configuration and available keys.

        Resolution order:
        1. Explicitly configured default_model
        2. Google model (if Google API key available)
        3. Anthropic model (if Anthropic API key available)

        Returns:
            Model name string

        Raises:
            RuntimeError: If no API keys are available
        """
        if self.default_model:
            return self.default_model

        registry = self._get_registry()

        if self.has_google_key:
            return registry.get_default(ModelFamily.GEMINI)
        if self.has_anthropic_key:
            return registry.get_default(ModelFamily.CLAUDE)

        raise RuntimeError(
            "No API keys found. Please set GOOGLE_API_KEY or ANTHROPIC_API_KEY."
        )

    def get_available_models(self) -> list[str]:
        """Get list of models available based on configured API keys."""
        registry = self._get_registry()
        return registry.get_available_models(
            google_key=self.has_google_key,
            anthropic_key=self.has_anthropic_key,
        )

    def is_google_model(self, model: str | None = None) -> bool:
        """Check if the given model (or current model) is a Google model."""
        model = model or self.get_model()
        registry = self._get_registry()
        try:
            return registry.get_family(model) == ModelFamily.GEMINI
        except ValueError:
            return False

    def is_anthropic_model(self, model: str | None = None) -> bool:
        """Check if the given model (or current model) is an Anthropic model."""
        model = model or self.get_model()
        registry = self._get_registry()
        try:
            return registry.get_family(model) == ModelFamily.CLAUDE
        except ValueError:
            return False

    def supports_thinking_effort(self, model: str | None = None) -> bool:
        """Check if the model supports thinking effort configuration."""
        model = model or self.get_model()
        registry = self._get_registry()
        return registry.supports_thinking(model)

    def set_model(self, model: str) -> None:
        """Set the default model."""
        registry = self._get_registry()
        if registry.is_refreshed:
            # Validate and possibly resolve deprecated models
            resolved = registry.resolve_model(model)
            object.__setattr__(self, "default_model", resolved)
        else:
            # Pre-refresh: validate against fallback list
            available = self.get_available_models()
            if model not in available:
                raise ValueError(
                    f"Model '{model}' is not available. "
                    f"Available models: {', '.join(available)}"
                )
            object.__setattr__(self, "default_model", model)

    def set_thinking_effort(self, effort: str) -> None:
        """Set the thinking effort level."""
        if effort not in THINKING_EFFORT_LEVELS:
            raise ValueError(
                f"Invalid thinking effort '{effort}'. "
                f"Valid levels: {', '.join(THINKING_EFFORT_LEVELS)}"
            )
        object.__setattr__(self, "thinking_effort", effort)

    def export_api_keys_to_env(self) -> None:
        """Export API keys to environment variables."""
        import os

        if self.google_api_key and not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = self.google_api_key

        if self.anthropic_api_key and not os.environ.get("ANTHROPIC_API_KEY"):
            os.environ["ANTHROPIC_API_KEY"] = self.anthropic_api_key
