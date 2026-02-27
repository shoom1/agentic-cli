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
from typing import Generator, Any, Tuple, Type
from contextlib import contextmanager

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
)
from agentic_cli.workflow.settings import WorkflowSettingsMixin
from agentic_cli.settings_mixins import AppSettingsMixin, CLISettingsMixin

# Re-export for backward compatibility
__all__ = [
    "BaseSettings",
    "SettingsContext",
    "SettingsValidationError",
    "get_settings",
    "set_settings",
    "set_context_settings",
    "get_context_settings",
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


class BaseSettings(WorkflowSettingsMixin, AppSettingsMixin, CLISettingsMixin, PydanticBaseSettings):
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
    - WorkflowSettingsMixin: Model, orchestrator, API keys, tool config
    - AppSettingsMixin: Application identity and disk layout
    - CLISettingsMixin: Logging and display settings
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENTIC_",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
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
