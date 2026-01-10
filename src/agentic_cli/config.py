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
from functools import lru_cache
from pathlib import Path
from typing import Literal, Generator
from contextlib import contextmanager

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings as PydanticBaseSettings, SettingsConfigDict


# Available models by provider
GOOGLE_MODELS = [
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]

ANTHROPIC_MODELS = [
    "claude-opus-4-5",
    "claude-opus-4",
    "claude-sonnet-4-5",
    "claude-sonnet-4",
]

ALL_MODELS = GOOGLE_MODELS + ANTHROPIC_MODELS

# Thinking effort levels (for models that support it)
THINKING_EFFORT_LEVELS = ["none", "low", "medium", "high"]


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

    @property
    def default_model_google(self) -> str:
        """Default Google model."""
        return "gemini-3-flash-preview"

    @property
    def default_model_anthropic(self) -> str:
        """Default Anthropic model."""
        return "claude-sonnet-4-5"

    def get_model(self) -> str:
        """Get the model to use based on configuration and available keys.

        Returns:
            Model name string

        Raises:
            RuntimeError: If no API keys are available
        """
        if self.default_model:
            return self.default_model

        if self.has_google_key:
            return self.default_model_google
        if self.has_anthropic_key:
            return self.default_model_anthropic

        raise RuntimeError(
            "No API keys found. Please set GOOGLE_API_KEY or ANTHROPIC_API_KEY "
            f"in your environment or in {self.workspace_dir}/.env file."
        )

    def get_available_models(self) -> list[str]:
        """Get list of models available based on configured API keys."""
        models = []
        if self.has_google_key:
            models.extend(GOOGLE_MODELS)
        if self.has_anthropic_key:
            models.extend(ANTHROPIC_MODELS)
        return models

    def is_google_model(self, model: str | None = None) -> bool:
        """Check if the given model (or current model) is a Google model."""
        model = model or self.get_model()
        return model in GOOGLE_MODELS or model.startswith("gemini")

    def is_anthropic_model(self, model: str | None = None) -> bool:
        """Check if the given model (or current model) is an Anthropic model."""
        model = model or self.get_model()
        return model in ANTHROPIC_MODELS or model.startswith("claude")

    def supports_thinking_effort(self, model: str | None = None) -> bool:
        """Check if the model supports thinking effort configuration."""
        model = model or self.get_model()
        return (
            self.is_anthropic_model(model)
            or "gemini-2.5" in model
            or "gemini-3" in model
        )

    def set_model(self, model: str) -> None:
        """Set the default model."""
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

    def get_api_key_status(self) -> dict[str, bool]:
        """Get status of all API keys."""
        return {
            "google": self.has_google_key,
            "anthropic": self.has_anthropic_key,
        }

    def export_api_keys_to_env(self) -> None:
        """Export API keys to environment variables."""
        import os

        if self.google_api_key and not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = self.google_api_key

        if self.anthropic_api_key and not os.environ.get("ANTHROPIC_API_KEY"):
            os.environ["ANTHROPIC_API_KEY"] = self.anthropic_api_key

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
    def templates_dir(self) -> Path:
        """Directory for report templates."""
        return self.workspace_dir / "templates"

    @property
    def reports_dir(self) -> Path:
        """Directory for generated reports."""
        return self.workspace_dir / "reports"

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
