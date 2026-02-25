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

Settings Loading Priority (highest to lowest):
    1. Environment variables (AGENTIC_* prefix)
    2. Project config (./.{app_name}/settings.json)
    3. User config (~/.{app_name}/settings.json)
    4. .env file
    5. Default values
"""

from contextvars import ContextVar, Token
from pathlib import Path
from typing import Literal, Generator, Any, Tuple, Type
from contextlib import contextmanager

from pydantic import Field, field_validator
from pydantic_settings import (
    BaseSettings as PydanticBaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
)

from agentic_cli.resolvers import (
    GOOGLE_MODELS,
    ANTHROPIC_MODELS,
    ALL_MODELS,
    THINKING_EFFORT_LEVELS,
    DEFAULT_GOOGLE_MODEL,
    DEFAULT_ANTHROPIC_MODEL,
)
from agentic_cli.workflow.settings import WorkflowSettingsMixin
from agentic_cli.cli.settings import CLISettingsMixin

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


def _get_json_config_source(
    settings_cls: Type[PydanticBaseSettings],
    json_file: Path,
) -> PydanticBaseSettingsSource | None:
    """Create a JSON config source if the file exists.

    Args:
        settings_cls: The settings class
        json_file: Path to JSON config file

    Returns:
        JsonConfigSettingsSource if file exists, None otherwise
    """
    if not json_file.exists():
        return None

    try:
        from pydantic_settings import JsonConfigSettingsSource
        return JsonConfigSettingsSource(settings_cls, json_file=json_file)
    except ImportError:
        # Older pydantic-settings without JsonConfigSettingsSource
        return None


class BaseSettings(WorkflowSettingsMixin, CLISettingsMixin, PydanticBaseSettings):
    """Base settings for agentic CLI applications.

    Domain-specific applications extend this class and override:
    - model_config: Set env_prefix and env_file for domain
    - app_name: Default application name
    - workspace_dir: Default workspace directory

    Settings are loaded from (in order of precedence):
    1. Environment variables (with domain-specific prefix)
    2. Project config (./.{app_name}/settings.json)
    3. User config (~/.{app_name}/settings.json)
    4. Domain-specific .env file
    5. Default values

    Mixins provide organized settings:
    - WorkflowSettingsMixin: Model, orchestrator, retry settings
    - CLISettingsMixin: Logging, activity, user settings
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENTIC_",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
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

    # Knowledge Base & Exploration (optional feature settings)
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

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[PydanticBaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """Customize settings sources for layered JSON configuration.

        Priority (highest to lowest):
            1. init_settings (constructor arguments)
            2. env_settings (environment variables)
            3. project_json (./.app_name/settings.json)
            4. user_json (~/.app_name/settings.json)
            5. dotenv_settings (.env file)

        Note: JSON sources are only included if the files exist.
        """
        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            env_settings,
        ]

        # Get app_name from class default or model_fields
        app_name = "agentic_cli"
        if hasattr(cls, "model_fields") and "app_name" in cls.model_fields:
            field_info = cls.model_fields["app_name"]
            if field_info.default and field_info.default != ...:
                app_name = field_info.default

        # Add project-level JSON config (./.app_name/settings.json)
        project_json = _get_json_config_source(
            settings_cls,
            Path.cwd() / f".{app_name}" / "settings.json",
        )
        if project_json:
            sources.append(project_json)

        # Add user-level JSON config (~/.app_name/settings.json)
        user_json = _get_json_config_source(
            settings_cls,
            Path.home() / f".{app_name}" / "settings.json",
        )
        if user_json:
            sources.append(user_json)

        # Add dotenv settings
        sources.append(dotenv_settings)

        return tuple(sources)

    @field_validator("workspace_dir", mode="before")
    @classmethod
    def expand_path(cls, v: str | Path) -> Path:
        """Expand ~ and environment variables in paths."""
        if isinstance(v, str):
            return Path(v).expanduser()
        return v

    # === Model-related properties ===

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
        return DEFAULT_GOOGLE_MODEL

    @property
    def default_model_anthropic(self) -> str:
        """Default Anthropic model."""
        return DEFAULT_ANTHROPIC_MODEL

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

        if self.has_google_key:
            return DEFAULT_GOOGLE_MODEL
        if self.has_anthropic_key:
            return DEFAULT_ANTHROPIC_MODEL

        raise RuntimeError(
            "No API keys found. Please set GOOGLE_API_KEY or ANTHROPIC_API_KEY."
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

    def export_api_keys_to_env(self) -> None:
        """Export API keys to environment variables."""
        import os

        if self.google_api_key and not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = self.google_api_key

        if self.anthropic_api_key and not os.environ.get("ANTHROPIC_API_KEY"):
            os.environ["ANTHROPIC_API_KEY"] = self.anthropic_api_key

    # === Path-related properties ===

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
    - Calls within workflow manager's process use context settings

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


def set_context_settings(settings: BaseSettings | None) -> Token:
    """Set settings for the current context.

    This sets settings that will be returned by get_settings() for the
    current async context (and any code it calls). Useful for:
    - Multi-tenant scenarios
    - Testing with isolated settings
    - Workflow-specific configurations

    Args:
        settings: Settings to use in current context, or None to clear

    Returns:
        Token that can be used to reset the context variable.
    """
    return _settings_context.set(settings)


def get_context_settings() -> BaseSettings | None:
    """Get settings from current context (if any).

    Returns:
        Settings from current context, or None if not set
    """
    return _settings_context.get()


# Workflow context — canonical implementation lives in workflow.context;
# re-exported here for backward compatibility.
from agentic_cli.workflow.context import set_context_workflow, get_context_workflow


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
    """Reload settings (clears global singleton and context cache).

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
