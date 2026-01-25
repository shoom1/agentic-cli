"""Base configuration for agentic CLI applications.

Provides BaseSettings class that domain-specific applications extend
with their own env_prefix and workspace directory.

Settings Management:
    The module provides both global singleton and context-based settings:

    1. Global singleton (simple cases):
        set_settings(my_settings)
        settings = get_settings()

    2. Context-based (isolated contexts, multi-tenant):
        with SettingsContext(my_settings):
            # Code here sees my_settings via get_settings()
            settings = get_settings()  # Returns my_settings
"""

from contextvars import ContextVar
from pathlib import Path
from typing import Literal, Generator
from contextlib import contextmanager

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings as PydanticBaseSettings, SettingsConfigDict

from agentic_cli.resolvers import (
    ModelResolver,
    PathResolver,
    GOOGLE_MODELS,
    ANTHROPIC_MODELS,
    ALL_MODELS,
    THINKING_EFFORT_LEVELS,
)

# Re-export for backward compatibility
__all__ = [
    "BaseSettings",
    "SettingsContext",
    "SettingsValidationError",
    "get_settings",
    "set_settings",
    "set_context_settings",
    "get_context_settings",
    "get_context_workflow",
    "set_context_workflow",
    "validate_settings",
    "reload_settings",
    "GOOGLE_MODELS",
    "ANTHROPIC_MODELS",
    "ALL_MODELS",
    "THINKING_EFFORT_LEVELS",
]


class BaseSettings(PydanticBaseSettings):
    """Base settings for agentic CLI applications.

    Domain-specific applications extend this class and override:
    - model_config: Set env_prefix and env_file for domain
    - app_name: Default application name
    - workspace_dir: Default workspace directory

    Settings are loaded from (in order of precedence):
    1. Environment variables (with domain-specific prefix)
    2. Domain-specific .env file
    3. Default values
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENTIC_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys (common across all domains)
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

    # Model Configuration
    default_model: str | None = Field(
        default=None,
        description="Default model to use (auto-detected if not set)",
    )
    thinking_effort: Literal["none", "low", "medium", "high"] = Field(
        default="medium",
        description="Thinking effort level for models that support extended thinking",
    )

    # Paths (domain projects should override workspace_dir default)
    workspace_dir: Path = Field(
        default_factory=lambda: Path.home() / ".agentic",
        description="Directory for storing artifacts and sessions",
    )

    # Logging
    log_level: Literal["debug", "info", "warning", "error"] = Field(
        default="warning",
        description="Logging level",
    )
    log_format: Literal["console", "json"] = Field(
        default="console",
        description="Log output format (console for dev, json for production)",
    )

    # Application (domain projects should override)
    app_name: str = Field(
        default="agentic_cli",
        description="Application name for agent services",
    )
    default_user: str = Field(
        default="default_user",
        description="Default user identifier",
    )

    # Orchestrator Selection
    orchestrator: Literal["adk", "langgraph"] = Field(
        default="adk",
        description="Workflow orchestrator backend (adk=Google ADK, langgraph=LangGraph)",
    )
    langgraph_checkpointer: Literal["memory", "postgres"] | None = Field(
        default="memory",
        description="LangGraph checkpointer type for state persistence",
    )

    # Activity Logging
    log_activity: bool = Field(
        default=False,
        description="Log conversation activity to file for audit purposes",
    )

    # Knowledge Base & Exploration
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings",
    )
    embedding_batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation",
    )
    knowledge_base_use_mock: bool = Field(
        default=False,
        description="Use mock knowledge base (no ML dependencies required)",
    )
    serper_api_key: str | None = Field(
        default=None,
        description="Serper.dev API key for web search",
        validation_alias="SERPER_API_KEY",
    )
    python_executor_timeout: int = Field(
        default=30,
        description="Default timeout for Python execution (seconds)",
    )

    @field_validator("workspace_dir", mode="before")
    @classmethod
    def expand_path(cls, v: str | Path) -> Path:
        """Expand ~ and environment variables in paths."""
        if isinstance(v, str):
            return Path(v).expanduser()
        return v

    # Lazy-initialized resolvers
    _model_resolver: ModelResolver | None = None
    _path_resolver: PathResolver | None = None

    @property
    def model_resolver(self) -> ModelResolver:
        """Get the model resolver instance (lazy-initialized)."""
        if self._model_resolver is None:
            object.__setattr__(
                self,
                "_model_resolver",
                ModelResolver(
                    google_api_key=self.google_api_key,
                    anthropic_api_key=self.anthropic_api_key,
                    default_model=self.default_model,
                ),
            )
        return self._model_resolver

    @property
    def path_resolver(self) -> PathResolver:
        """Get the path resolver instance (lazy-initialized)."""
        if self._path_resolver is None:
            object.__setattr__(
                self,
                "_path_resolver",
                PathResolver(workspace_dir=self.workspace_dir),
            )
        return self._path_resolver

    # === Model-related properties (delegated to ModelResolver) ===

    @property
    def has_google_key(self) -> bool:
        """Check if Google API key is available."""
        return self.model_resolver.has_google_key

    @property
    def has_anthropic_key(self) -> bool:
        """Check if Anthropic API key is available."""
        return self.model_resolver.has_anthropic_key

    @property
    def has_any_api_key(self) -> bool:
        """Check if any API key is available."""
        return self.model_resolver.has_any_api_key

    @property
    def default_model_google(self) -> str:
        """Default Google model."""
        return ModelResolver.DEFAULT_GOOGLE_MODEL

    @property
    def default_model_anthropic(self) -> str:
        """Default Anthropic model."""
        return ModelResolver.DEFAULT_ANTHROPIC_MODEL

    def get_model(self) -> str:
        """Get the model to use based on configuration and available keys.

        Returns:
            Model name string

        Raises:
            RuntimeError: If no API keys are available
        """
        return self.model_resolver.get_model()

    def get_available_models(self) -> list[str]:
        """Get list of models available based on configured API keys."""
        return self.model_resolver.get_available_models()

    def is_google_model(self, model: str | None = None) -> bool:
        """Check if the given model (or current model) is a Google model."""
        return self.model_resolver.is_google_model(model)

    def is_anthropic_model(self, model: str | None = None) -> bool:
        """Check if the given model (or current model) is an Anthropic model."""
        return self.model_resolver.is_anthropic_model(model)

    def supports_thinking_effort(self, model: str | None = None) -> bool:
        """Check if the model supports thinking effort configuration."""
        return self.model_resolver.supports_thinking_effort(model)

    def set_model(self, model: str) -> None:
        """Set the default model."""
        self.model_resolver.validate_model(model)
        object.__setattr__(self, "default_model", model)
        # Invalidate resolver cache to pick up new model
        object.__setattr__(self, "_model_resolver", None)

    def set_thinking_effort(self, effort: str) -> None:
        """Set the thinking effort level."""
        if effort not in THINKING_EFFORT_LEVELS:
            raise ValueError(
                f"Invalid thinking effort '{effort}'. "
                f"Valid levels: {', '.join(THINKING_EFFORT_LEVELS)}"
            )
        object.__setattr__(self, "thinking_effort", effort)

    def get_api_key_status(self) -> dict[str, bool]:
        """Get status of all API keys."""
        return self.model_resolver.get_api_key_status()

    def export_api_keys_to_env(self) -> None:
        """Export API keys to environment variables."""
        import os

        if self.google_api_key and not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = self.google_api_key

        if self.anthropic_api_key and not os.environ.get("ANTHROPIC_API_KEY"):
            os.environ["ANTHROPIC_API_KEY"] = self.anthropic_api_key

    # === Path-related properties (delegated to PathResolver) ===

    def ensure_workspace_exists(self) -> None:
        """Create workspace directory if it doesn't exist."""
        self.path_resolver.ensure_workspace_exists()

    @property
    def sessions_dir(self) -> Path:
        """Directory for session storage."""
        return self.path_resolver.sessions_dir

    @property
    def artifacts_dir(self) -> Path:
        """Directory for artifact storage."""
        return self.path_resolver.artifacts_dir

    @property
    def templates_dir(self) -> Path:
        """Directory for report templates."""
        return self.path_resolver.templates_dir

    @property
    def reports_dir(self) -> Path:
        """Directory for generated reports."""
        return self.path_resolver.reports_dir

    @property
    def knowledge_base_dir(self) -> Path:
        """Directory for knowledge base storage."""
        return self.path_resolver.knowledge_base_dir

    @property
    def knowledge_base_documents_dir(self) -> Path:
        """Directory for knowledge base documents."""
        return self.path_resolver.knowledge_base_documents_dir

    @property
    def knowledge_base_embeddings_dir(self) -> Path:
        """Directory for knowledge base embeddings."""
        return self.path_resolver.knowledge_base_embeddings_dir


# Context variable for settings (takes precedence over global singleton)
_settings_context: ContextVar[BaseSettings | None] = ContextVar(
    "settings_context", default=None
)

# Global settings instance holder (fallback when no context)
_settings_instance: BaseSettings | None = None


def get_settings() -> BaseSettings:
    """Get the current settings instance.

    Settings resolution order:
    1. Context variable (set via SettingsContext or set_context_settings)
    2. Global singleton (set via set_settings)
    3. Fresh BaseSettings instance (created on first access)

    This allows tools to work with the correct settings regardless of
    how they're called:
    - Direct calls use global settings
    - Calls within WorkflowManager.process use context settings

    Returns:
        BaseSettings instance for the current context
    """
    # Check context first (enables isolated contexts for tools)
    context_settings = _settings_context.get()
    if context_settings is not None:
        return context_settings

    # Fall back to global singleton
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = BaseSettings()
    return _settings_instance


def set_settings(settings: BaseSettings) -> None:
    """Set the global settings instance.

    Allows domain applications to configure the global settings used
    by tools and other components.

    Note: For isolated contexts (e.g., testing, multi-tenant),
    prefer using SettingsContext instead.

    Args:
        settings: Settings instance to use globally
    """
    global _settings_instance
    _settings_instance = settings


def set_context_settings(settings: BaseSettings | None) -> None:
    """Set settings for the current context.

    This sets settings that will be returned by get_settings() for the
    current async context (and any code it calls). Useful for:
    - Multi-tenant scenarios
    - Testing with isolated settings
    - Workflow-specific configurations

    Args:
        settings: Settings to use in current context, or None to clear
    """
    _settings_context.set(settings)


def get_context_settings() -> BaseSettings | None:
    """Get settings from current context (if any).

    Returns:
        Settings from current context, or None if not set
    """
    return _settings_context.get()


# Context variable for workflow manager (allows tools to request user input)
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentic_cli.workflow.manager import WorkflowManager

_workflow_context: ContextVar[Any] = ContextVar("workflow_context", default=None)


def set_context_workflow(workflow: "WorkflowManager | None") -> None:
    """Set the workflow manager for the current context.

    This allows tools to access the workflow manager for operations
    like requesting user input.

    Args:
        workflow: WorkflowManager instance, or None to clear
    """
    _workflow_context.set(workflow)


def get_context_workflow() -> "WorkflowManager | None":
    """Get the workflow manager from the current context.

    Returns:
        WorkflowManager instance, or None if not in a workflow context
    """
    return _workflow_context.get()


@contextmanager
def SettingsContext(settings: BaseSettings) -> Generator[BaseSettings, None, None]:
    """Context manager for isolated settings.

    Use this to run code with specific settings that won't affect
    the global settings or other contexts.

    Example:
        with SettingsContext(test_settings) as s:
            # All get_settings() calls here return test_settings
            result = my_tool()  # Tool will use test_settings

    Args:
        settings: Settings to use within the context

    Yields:
        The settings instance
    """
    token = _settings_context.set(settings)
    try:
        yield settings
    finally:
        _settings_context.reset(token)


def reload_settings() -> BaseSettings:
    """Reload settings (clears global singleton cache).

    Note: This does not affect context-based settings.

    Returns:
        Fresh BaseSettings instance
    """
    global _settings_instance
    _settings_instance = None
    # Also clear context for a complete reset
    _settings_context.set(None)
    return get_settings()


class SettingsValidationError(Exception):
    """Raised when settings validation fails."""

    pass


def validate_settings(settings: BaseSettings) -> None:
    """Validate settings for runtime use.

    Performs validation that can only be done at runtime:
    - API key availability
    - Model compatibility
    - Path accessibility

    Args:
        settings: Settings to validate

    Raises:
        SettingsValidationError: If validation fails
    """
    errors = []

    if not settings.has_any_api_key:
        errors.append(
            "No API keys configured. Set GOOGLE_API_KEY or ANTHROPIC_API_KEY."
        )

    if settings.default_model:
        available = settings.get_available_models()
        if settings.default_model not in available:
            errors.append(
                f"Configured model '{settings.default_model}' is not available. "
                f"Available models: {', '.join(available) if available else 'none'}"
            )

    if errors:
        raise SettingsValidationError("\n".join(errors))
