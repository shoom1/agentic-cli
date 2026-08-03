"""Workflow settings mixin.

Provides settings for agentic workflow configuration, independent of UI.
These settings control model selection, orchestration, retry behavior,
HITL (human-in-the-loop), memory management, API keys, and tool configuration.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, TYPE_CHECKING

from pydantic import AliasChoices, BaseModel, Field, field_validator

from agentic_cli.logging import Loggers
from agentic_cli.workflow.models import ModelFamily, ModelRegistry

if TYPE_CHECKING:
    pass

logger = Loggers.config()

# Thinking effort levels (module-level constant for backward compatibility)
THINKING_EFFORT_LEVELS = ModelRegistry.THINKING_EFFORT_LEVELS

# Which environment variable supplies each provider's credential (for error
# messages — the value is never echoed).
_PROVIDER_ENV_VAR = {
    ModelFamily.GEMINI: "GOOGLE_API_KEY",
    ModelFamily.CLAUDE: "ANTHROPIC_API_KEY",
}


class PermissionRuleConfig(BaseModel):
    """A single permission rule — serialised to settings.json as JSON."""

    capability: str
    target: str


class PermissionsConfig(BaseModel):
    """Nested settings section holding user-editable permission rules."""

    allow: list[PermissionRuleConfig] = []
    deny: list[PermissionRuleConfig] = []


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

    # API Keys (common across all domains, never saved to JSON).
    #
    # Each accepts BOTH the provider's environment variable name and its Python
    # field name (``AliasChoices``): the bare env alias made
    # ``BaseSettings(google_api_key=...)`` bind nothing at all — the value was
    # dropped by ``extra="ignore"`` and the field kept its default. The env name
    # is listed first, so a real environment variable still wins within a source.
    # Values are kept out of ``repr()`` and out of every persisted file (see
    # ``settings_persistence.SECRET_FIELDS``).
    google_api_key: str | None = Field(
        default=None,
        description="Google API key for Gemini models",
        validation_alias=AliasChoices("GOOGLE_API_KEY", "google_api_key"),
        repr=False,
    )
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key for Claude models",
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "anthropic_api_key"),
        repr=False,
    )
    tavily_api_key: str | None = Field(
        default=None,
        description="Tavily API key for web search",
        validation_alias=AliasChoices("TAVILY_API_KEY", "tavily_api_key"),
        repr=False,
    )
    brave_api_key: str | None = Field(
        default=None,
        description="Brave Search API key for web search",
        validation_alias=AliasChoices("BRAVE_API_KEY", "brave_api_key"),
        repr=False,
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
    embedding_device: str = Field(
        default="auto",
        title="Embedding Device",
        description=(
            "Device for sentence-transformers: 'auto' picks cuda → "
            "mps (Apple Silicon only) → cpu. Force 'cpu' on Intel Macs "
            "with discrete AMD GPUs (MPS command-buffer failures). "
            "Valid values: auto, cpu, mps, cuda."
        ),
        json_schema_extra={"ui_order": 152},
    )
    knowledge_base_use_mock: bool = Field(
        default=False,
        title="Use Mock Knowledge Base",
        description="Use mock knowledge base (no ML dependencies required)",
        json_schema_extra={"ui_order": 152},
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
    anthropic_request_timeout: float = Field(
        default=900.0,
        title="Anthropic Request Timeout",
        description=(
            "Overall timeout (seconds) for direct-API Claude requests. A "
            "non-default value lets high-thinking (large max_tokens) requests "
            "run without the SDK's streaming-required guard."
        ),
        json_schema_extra={"ui_order": 113},
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
    stateful_executor_backend: Literal["none", "local", "docker"] = Field(
        default="none",
        title="Stateful Executor Backend",
        description=(
            "Backend for the stateful sandbox_execute tool. 'none' disables it; "
            "'docker' runs in a network-isolated container (recommended); 'local' "
            "runs a Jupyter kernel with host privileges (NOT OS-sandboxed). Future: "
            "'modal', 'runpod'."
        ),
        json_schema_extra={"ui_order": 121},
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
    sandbox_image: str = Field(
        default="quay.io/jupyter/scipy-notebook:python-3.12",
        title="Sandbox Image",
        description="Container image for the jupyter_docker backend. Must contain ipykernel/jupyter_client. Pin to a digest in production.",
        json_schema_extra={"ui_order": 126},
    )
    sandbox_memory_mb: int = Field(
        default=2048,
        title="Sandbox Memory (MB)",
        description="Per-container memory cap for the docker backend; also disables swap.",
        json_schema_extra={"ui_order": 127},
    )
    sandbox_cpus: float = Field(
        default=2.0,
        title="Sandbox CPUs",
        description="Per-container CPU cap for the docker backend.",
        json_schema_extra={"ui_order": 128},
    )
    sandbox_pids_limit: int = Field(
        default=256,
        title="Sandbox PID Limit",
        description="Per-container process/thread cap for the docker backend.",
        json_schema_extra={"ui_order": 129},
    )
    sandbox_network: str = Field(
        default="none",
        title="Sandbox Network",
        description="Docker network mode for the docker backend. v1 supports 'none' only.",
        json_schema_extra={"ui_order": 130},
    )
    sandbox_container_user: str = Field(
        default="",
        title="Sandbox Container User",
        description="uid:gid to run the container as; empty uses the image default.",
        json_schema_extra={"ui_order": 131},
    )
    sandbox_data_mounts: list[str] = Field(
        default_factory=list,
        title="Sandbox Data Mounts",
        description="Read-only data staged into the container as 'host_path:mount_name' (mounted read-only under /data/).",
        json_schema_extra={"ui_order": 132},
    )
    sandbox_start_timeout: int = Field(
        default=180,
        title="Sandbox Start Timeout",
        description="Seconds to wait for container start + image pull + kernel readiness (docker backend).",
        json_schema_extra={"ui_order": 133},
    )
    sandbox_outputs_dir: str = Field(
        default="",
        title="Sandbox Outputs Dir",
        description="Shared host dir mounted at /workspace/outputs for FINAL deliverables (default: <workspace_dir>/artifacts).",
        json_schema_extra={"ui_order": 134},
    )

    @field_validator("sandbox_network")
    @classmethod
    def _validate_sandbox_network(cls, v: str) -> str:
        """Fail closed: the docker backend's no-egress isolation depends on
        --network none, so reject any other value rather than silently
        weakening it. (v1 supports 'none' only.)"""
        if v != "none":
            raise ValueError(
                f"sandbox_network must be 'none' (got {v!r}). The docker sandbox's "
                "network-isolation guarantee depends on it; other modes are not "
                "supported in v1."
            )
        return v

    # OS-level sandboxing
    os_sandbox_enabled: bool = Field(
        default=True,
        title="OS Sandbox Enabled",
        description=(
            "Wrap Python execution in an OS-level sandbox when a backend is "
            "available (sandbox-exec on macOS, bwrap on Linux). When no backend "
            "is present, execution falls back to the restricted in-process "
            "executor (see os_sandbox_strict) — only pure-computation modules "
            "are importable in that case."
        ),
        json_schema_extra={"ui_order": 134},
    )
    os_sandbox_strict: bool = Field(
        default=False,
        title="OS Sandbox Strict",
        description=(
            "Refuse to run code when OS sandboxing is enabled but no backend is "
            "available, instead of falling back to the in-process executor."
        ),
        json_schema_extra={"ui_order": 137},
    )
    os_sandbox_writable_paths: list[str] = Field(
        default_factory=list,
        title="OS Sandbox Writable Paths",
        description="Additional paths the sandboxed process can write to (working directory is always writable)",
        json_schema_extra={"ui_order": 135},
    )
    os_sandbox_allow_network: bool = Field(
        default=False,
        title="OS Sandbox Allow Network",
        description="Allow network access from sandboxed processes",
        json_schema_extra={"ui_order": 136},
    )

    # Permissions
    permissions: PermissionsConfig = Field(
        default_factory=PermissionsConfig,
        title="Permissions",
        description="Declarative allow/deny rules for tool capabilities.",
        json_schema_extra={"ui_order": 138},
    )
    permissions_enabled: bool = Field(
        default=True,
        title="Permissions Enabled",
        description="Master switch; when False, all tool calls are allowed.",
        json_schema_extra={"ui_order": 139},
    )
    max_concurrent_jobs: int = Field(
        default=4,
        ge=1,
        title="Max Concurrent Jobs",
        description="Maximum long-running jobs running at once; excess are queued.",
        json_schema_extra={"ui_order": 140},
    )
    job_auto_resume: bool = Field(
        default=False,
        title="Auto-resume Finished Jobs",
        description=(
            "When True, a finished long-running job that opted in "
            "(resume_on_complete) automatically resumes the agent with its "
            "result at the next turn boundary (or via /resume)."
        ),
        json_schema_extra={"ui_order": 141},
    )

    # Session persistence — durable conversations across restarts.
    # Drives BOTH backends: ADK uses DatabaseSessionService, LangGraph uses a
    # persistent checkpointer (both keyed by session_id). "memory" = ephemeral.
    session_store: Literal["memory", "sqlite", "postgres"] = Field(
        default="sqlite",
        title="Session Store",
        description=(
            "Where conversations are persisted: sqlite (default, a single file), "
            "postgres (shared/multi-instance via Postgres URI), or memory (ephemeral)."
        ),
        json_schema_extra={"ui_order": 145},
    )

    # Skills (Agent Skills / SKILL.md folders)
    skills_dirs: list[str] = Field(
        default_factory=list,
        title="Skills Directories",
        description="Directories searched for named skills (Agent Skills / SKILL.md folders)",
        json_schema_extra={"ui_order": 141},
    )
    skill_scripts_enabled: bool = Field(
        default=False,
        title="Skill Scripts Enabled",
        description="Allow executing scripts bundled with skills (requires a code executor; disabled by default)",
        json_schema_extra={"ui_order": 142},
    )

    # Persistence settings (LangGraph)
    postgres_uri: str | None = Field(
        default=None,
        title="PostgreSQL URI",
        description="PostgreSQL connection URI for persistent storage",
        json_schema_extra={"ui_order": 146},
    )
    sqlite_uri: str | None = Field(
        default=None,
        title="SQLite URI",
        description="SQLite connection URI or file path for persistent storage",
        json_schema_extra={"ui_order": 147},
    )
    store_type: Literal["memory", "postgres"] | None = Field(
        default="memory",
        title="Store Type",
        description="Store type for long-term memory (memory or postgres)",
        json_schema_extra={"ui_order": 148},
    )

    def session_db_url(self) -> str | None:
        """Async SQLAlchemy URL for the session store, or None when ephemeral.

        Shared by both backends so ADK's DatabaseSessionService and LangGraph's
        checkpointer persist to the same place. SQLite is the zero-config
        default (``{workspace}/sessions/sessions.db``); Postgres via uri.
        """
        store = getattr(self, "session_store", "sqlite")
        if store == "memory":
            return None
        if store == "postgres":
            uri = self.postgres_uri
            if not uri:
                raise ValueError("session_store='postgres' requires postgres_uri")
            return uri.replace("postgresql://", "postgresql+asyncpg://", 1)
        # sqlite (default)
        if self.sqlite_uri:
            return self.sqlite_uri.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
        path = self.sessions_dir / "sessions.db"
        return f"sqlite+aiosqlite:///{path}"

    # Shell execution settings (for shell middleware)
    shell_sandbox_type: Literal["host", "docker"] = Field(
        default="host",
        title="Shell Sandbox Type",
        description="Execution environment for shell commands",
        json_schema_extra={"ui_order": 149},
    )
    shell_docker_image: str = Field(
        default="python:3.12-slim",
        title="Shell Docker Image",
        description="Docker image to use for sandboxed shell execution",
        json_schema_extra={"ui_order": 150},
    )
    shell_timeout: int = Field(
        default=60,
        title="Shell Timeout",
        description="Default timeout in seconds for shell commands",
        json_schema_extra={"ui_order": 151},
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

    def check_model(self, model: str, *, label: str = "model") -> str:
        """Validate one model against credentials and discovery authority.

        The single rule set shared by ``set_model()`` and ``validate_settings()``
        so a model the setter accepts can never be rejected at startup (or the
        reverse):

        1. the provider must be derivable from the id;
        2. that provider's credential must be configured;
        3. a deprecated alias resolves to its replacement (warned);
        4. an unknown model is rejected only when that provider's listing is
           authoritative — a degraded/unattempted listing cannot disprove it,
           and is logged instead.

        Args:
            model: Model identifier to check.
            label: What is being checked, for the error message.

        Returns:
            The resolved model id (differs only for a deprecated alias).

        Raises:
            ValueError: With an actionable message; never includes a credential.
        """
        registry = self._get_registry()
        try:
            family = registry.get_family(model)
        except ValueError:
            raise ValueError(
                f"Model '{model}' ({label}) is not available: its provider "
                "cannot be determined from the model id."
            ) from None

        if not self._has_credential_for(family):
            env_var = _PROVIDER_ENV_VAR.get(family, "the provider API key")
            raise ValueError(
                f"Model '{model}' ({label}) is not available: it needs a "
                f"{family.value} credential. Set {env_var}."
            )

        resolved = registry.resolve_model(model)  # raises when authoritative
        if resolved == model and model not in self.get_available_models():
            # Not authoritative (else resolve_model would have raised), so the
            # static list simply lags reality.
            logger.warning("model_not_in_static_list", model=model, source=label)
        return resolved

    def _has_credential_for(self, family: ModelFamily) -> bool:
        """Whether the credential a model family needs is configured."""
        if family is ModelFamily.GEMINI:
            return self.has_google_key
        if family is ModelFamily.CLAUDE:
            return self.has_anthropic_key
        return False

    def set_model(self, model: str) -> None:
        """Set the default model, validating it exactly as startup would.

        Raises:
            ValueError: If the model is unusable (see :meth:`check_model`).
        """
        object.__setattr__(self, "default_model", self.check_model(model))

    def set_thinking_effort(self, effort: str) -> None:
        """Set the thinking effort level."""
        if effort not in THINKING_EFFORT_LEVELS:
            raise ValueError(
                f"Invalid thinking effort '{effort}'. "
                f"Valid levels: {', '.join(THINKING_EFFORT_LEVELS)}"
            )
        object.__setattr__(self, "thinking_effort", effort)

    def export_api_keys_to_env(self) -> None:
        """Export configured API keys to provider environment variables.

        Provider SDKs used by the orchestrators (ADK's AnthropicLlm/Gemini,
        LangChain clients) read credentials from process env vars. The export
        OVERWRITES the env from this settings instance: the key fields bind
        only via their env alias (real env vars are the highest-priority
        source), so settings and env diverge only when the process env
        changed after this instance loaded — e.g. an earlier manager's
        export, or a key loaded from a class-specific env_file — and then
        this instance's configured value must win. (The previous
        set-if-absent guard let the first exporting manager pin credentials
        for every later one.) A key unset in settings leaves the environment
        untouched. Credentials are process-global; SettingsContext does not
        isolate them.
        """
        import os

        if self.google_api_key:
            os.environ["GOOGLE_API_KEY"] = self.google_api_key

        if self.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = self.anthropic_api_key
