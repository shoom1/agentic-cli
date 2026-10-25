"""Base configuration for agentic CLI applications.

Provides BaseSettings class that domain-specific applications extend
with their own env_prefix and workspace directory.

Settings Management:
    The module provides both global singleton and context-based settings:

    1. Global singleton (simple cases):
        set_settings(my_settings)
        settings = get_settings()

    2. Context-based (isolated settings lookup, e.g. tests or per-request
       overrides):
        with SettingsContext(my_settings):
            # Code here sees my_settings via get_settings()
            settings = get_settings()  # Returns my_settings

       Note: isolation covers settings *lookup* only. API credentials are
       exported to process-global env vars at manager initialization
       (provider SDKs read them from the environment), so a SettingsContext
       does not isolate credentials between contexts in one process.

Settings Loading Priority (highest to lowest):
    1. Environment variables (AGENTIC_* prefix)
    2. Project config (./.{app_name}/settings.json)
    3. User config (~/.{app_name}/settings.json)
    4. .env file
    5. Default values
"""

import re
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Generator, Any, Tuple, Type
from contextlib import contextmanager

from pydantic import AliasChoices
from pydantic_settings import (
    BaseSettings as PydanticBaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
)

from agentic_cli.workflow.settings import WorkflowSettingsMixin
from agentic_cli.settings_mixins import AppSettingsMixin, CLISettingsMixin
# PROJECT_SETTABLE_KEYS lives in settings_persistence so the save-side split
# and the load-side filter below share one definition (and because the reverse
# import would be circular). See its definition for the trust rationale.
from agentic_cli.settings_persistence import (
    CREDENTIAL_KEY_RE as _CREDENTIAL_KEY_RE,
    PROJECT_SETTABLE_KEYS as _PROJECT_SETTABLE_KEYS,
    PROJECT_PRESERVED_KEYS as _PROJECT_PRESERVED_KEYS,
    get_project_config_path,
    get_user_config_path,
)
from agentic_cli.logging import Loggers

logger = Loggers.config()

#: ``update_setting("model_role:fast", ref)`` maps one role in ``model_roles``.
MODEL_ROLE_KEY_PREFIX = "model_role:"

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
]


class _AllowlistFilterSource(PydanticBaseSettingsSource):
    """Wrap an untrusted settings source, keeping only allowlisted keys.

    Applied to the project ``settings.json`` and a cwd-relative ``.env``. Any
    non-allowlisted key is dropped and logged (one warning per key) so a cloned
    repo cannot flip a security boundary. Drops (never raises) a non-allowlisted
    key, so a repo cannot flip a boundary by *adding* keys. (Malformed JSON or a
    bad-typed allowlisted value is still rejected upstream, as before P0-1 —
    this narrows, not removes, that pre-existing surface.)

    A key that names no setting at all (neither a field nor an alias) could
    set nothing anyway, so it is dropped without a log: a project file saved
    by an older version still holds settings since removed (0.6's
    ``context_window_*``), and settings are built before logging is
    configured, when structlog prints even a debug log above the UI. The
    ``permissions`` section (``PROJECT_PRESERVED_KEYS``) is passed over
    without a log too: it sets no setting, the permission engine reads its
    deny rules from the file itself.
    """

    def __init__(
        self,
        settings_cls: Type[PydanticBaseSettings],
        inner: PydanticBaseSettingsSource,
        label: str,
    ) -> None:
        super().__init__(settings_cls)
        self._inner = inner
        self._label = label

    def get_field_value(self, field: Any, field_name: str) -> Tuple[Any, str, bool]:
        # Unused: __call__ is overridden to filter the inner source's output.
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        kept: dict[str, Any] = {}
        settings = _accepted_input_names(self.settings_cls)
        for key, value in self._inner().items():
            if key in _PROJECT_SETTABLE_KEYS:
                kept[key] = value
            elif key in _PROJECT_PRESERVED_KEYS:
                continue  # the permission engine reads it (deny rules only)
            elif key in settings:
                logger.warning(
                    "untrusted_project_setting_ignored", key=key, source=self._label
                )
        return kept


# Constructor kwargs matching _CREDENTIAL_KEY_RE (defined in
# settings_persistence, which also uses it to never save credentials) are
# credentials; an unrecognised one must fail loudly instead of being swallowed
# by ``extra="ignore"``.


def _accepted_input_names(settings_cls: Type[PydanticBaseSettings]) -> set[str]:
    """Every name the model accepts for a field: its own name and any aliases."""
    names: set[str] = set()
    for field_name, field in settings_cls.model_fields.items():
        names.add(field_name)
        alias = field.validation_alias
        if isinstance(alias, str):
            names.add(alias)
        elif isinstance(alias, AliasChoices):
            names.update(c for c in alias.choices if isinstance(c, str))
        if isinstance(field.alias, str):
            names.add(field.alias)
    return names


def _reject_unknown_credential_kwargs(
    settings_cls: Type[PydanticBaseSettings], values: dict[str, Any]
) -> None:
    """Raise on a credential-shaped kwarg the model would silently drop.

    Raises:
        ValueError: If a kwarg looks like a credential but matches no field or
            alias. The message names the key only — never its value.
    """
    accepted = _accepted_input_names(settings_cls)
    # Leading underscore = pydantic-settings' own kwargs (_env_file,
    # _secrets_dir, …), not settings fields.
    unknown = [k for k in values if not k.startswith("_") and k not in accepted]
    bad = [k for k in unknown if _CREDENTIAL_KEY_RE.search(k)]
    if not bad:
        return
    known = sorted(n for n in _accepted_input_names(settings_cls) if _CREDENTIAL_KEY_RE.search(n))
    raise ValueError(
        f"Unknown credential setting(s): {', '.join(sorted(bad))}. "
        f"{settings_cls.__name__} accepts: {', '.join(known)}."
    )


def _get_json_config_source(
    settings_cls: Type[PydanticBaseSettings],
    json_file: Path,
    *,
    untrusted: bool = False,
) -> PydanticBaseSettingsSource | None:
    """Create a JSON config source if the file exists.

    Args:
        settings_cls: The settings class
        json_file: Path to JSON config file
        untrusted: When True (the project file), keep only allowlisted
            (non-security) keys so a cloned workspace cannot flip a security
            boundary.

    Returns:
        JsonConfigSettingsSource if file exists, None otherwise
    """
    if not json_file.exists():
        return None

    try:
        from pydantic_settings import JsonConfigSettingsSource
    except ImportError:
        # Older pydantic-settings without JsonConfigSettingsSource
        return None

    if not untrusted:
        return JsonConfigSettingsSource(settings_cls, json_file=json_file)

    # Untrusted (the project file): keep only allowlisted keys so a cloned
    # workspace cannot flip a security boundary.
    return _AllowlistFilterSource(
        settings_cls,
        JsonConfigSettingsSource(settings_cls, json_file=json_file),
        "project settings.json",
    )


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
        # Settings hold credentials, and a rejected value can be one: a key
        # written inline in a model_endpoints entry, or a password in its
        # base_url. Pydantic honours this only on the outermost model, so the
        # nested EndpointSpec's own setting does not cover it here. Errors
        # still name the setting and say what was expected.
        hide_input_in_errors=True,
    )

    def __init__(self, **values: Any) -> None:
        """Construct settings, rejecting credential kwargs that would be dropped.

        ``extra="ignore"`` (needed so config files may carry keys a given app
        does not define) means a mistyped constructor argument vanishes without
        a word. That is tolerable for an ordinary setting and dangerous for a
        credential — the app then runs unauthenticated, or silently on a
        different key. Only credential-shaped unknown kwargs raise.
        """
        _reject_unknown_credential_kwargs(type(self), values)
        super().__init__(**values)

    def update_setting(self, key: str, value: Any) -> None:
        """Update a single setting, using dedicated setters where required.

        Args:
            key: Setting name. ``model`` sets ``default_model``, and
                ``model_role:<role>`` maps one role in ``model_roles`` (the
                ``/settings`` model pickers' keys).
            value: New value.

        Raises:
            ValueError: If the value is invalid for the given setting.
        """
        if key == "model":
            self.set_model(value)
        elif key.startswith(MODEL_ROLE_KEY_PREFIX):
            self.set_model_role(key.removeprefix(MODEL_ROLE_KEY_PREFIX), value)
        elif key == "thinking_effort":
            self.set_thinking_effort(value)
        else:
            # Validate through Pydantic instead of object.__setattr__ so the
            # value is type-checked and constraint-checked before it lands on
            # the instance (and is later persisted to settings.json). An
            # invalid value or unknown key raises ValidationError, which is a
            # subclass of ValueError — exactly what callers already handle.
            self.__pydantic_validator__.validate_assignment(self, key, value)

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

        # Get app_name from init kwargs (constructor), then class default
        app_name = None
        if hasattr(init_settings, "init_kwargs"):
            app_name = init_settings.init_kwargs.get("app_name")

        if not app_name:
            if hasattr(cls, "model_fields") and "app_name" in cls.model_fields:
                field_info = cls.model_fields["app_name"]
                if field_info.default and field_info.default != ...:
                    app_name = field_info.default

        app_name = app_name or "agentic_cli"

        # Add project-level JSON config (./.app_name/settings.json). Untrusted:
        # strip security-sensitive keys (a cloned repo can ship this file).
        project_json = _get_json_config_source(
            settings_cls,
            get_project_config_path(app_name),
            untrusted=True,
        )
        if project_json:
            sources.append(project_json)

        # Add user-level JSON config (~/.app_name/settings.json)
        user_json = _get_json_config_source(
            settings_cls,
            get_user_config_path(app_name),
        )
        if user_json:
            sources.append(user_json)

        # dotenv: a cwd-relative env_file is an untrusted project source (a
        # cloned repo can ship ./.env), so filter it like project settings.json.
        # We inspect EVERY configured env_file (a str/Path or a list of them):
        # if ANY entry is cwd-relative after ~ expansion, the whole dotenv
        # source is filtered. DotEnvSettingsSource merges all files into one
        # dict, so we cannot filter per-file; over-filtering fails safe. An
        # absolute/user-level env_file (including a "~/..." path) and real
        # environment variables stay trusted. Consequence: secrets/keys placed
        # in a cwd .env are dropped — put them in real env vars or an
        # absolute/user-level file.
        env_file = settings_cls.model_config.get("env_file")
        if env_file is None:
            _env_entries: list = []
        elif isinstance(env_file, (list, tuple)):
            _env_entries = list(env_file)
        else:
            _env_entries = [env_file]
        _has_cwd_relative = any(
            e is not None and not Path(e).expanduser().is_absolute()
            for e in _env_entries
        )
        if _has_cwd_relative:
            sources.append(
                _AllowlistFilterSource(settings_cls, dotenv_settings, "cwd .env")
            )
        else:
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


def validate_settings(
    settings: BaseSettings, agent_configs: Any | None = None
) -> None:
    """Check, before any network call, that the model settings can work.

    The static half of model validation:

    - every reference parses: ``default_model``, ``model_roles``,
      ``pinned_models`` and each agent's ``model`` (a role must map to a
      model, never another role);
    - every endpoint a reference names is defined in ``model_endpoints``;
    - every provider a model will run on has a key: the default model's,
      each agent's, and that of each role an agent names;
    - some provider has a key (or an endpoint is defined).

    Whether a model exists, and which thinking modes it takes, need the
    provider's listing: a workflow manager checks them when it resolves its
    agents against the model catalog at startup.

    Args:
        settings: Settings to validate.
        agent_configs: Optional agent configs whose ``model`` is checked too.

    Raises:
        SettingsValidationError: Listing every problem, one per line.
    """
    errors = _model_settings_problems(settings, agent_configs)
    if errors:
        raise SettingsValidationError("\n".join(errors))


def _model_settings_problems(
    settings: BaseSettings,
    agent_configs: Any | None = None,
    *,
    manager_model: str | None = None,
) -> list[str]:
    """:func:`validate_settings`' findings, without raising.

    Internal. ``manager_model`` is a workflow manager's own model override,
    which takes the ``default`` role for that manager; with it,
    ``default_model`` is not what runs there, so its key is not required.
    """
    from agentic_cli.models import (
        PROVIDER_KEY_ENV,
        ModelRef,
        ModelSpecError,
        ProviderKind,
        RoleRef,
        parse_ref,
        parse_spec,
    )
    from agentic_cli.workflow import model_services

    errors: list[str] = []
    endpoints = model_services.endpoints_from_settings(settings)
    if not any(endpoint.has_credentials for endpoint in endpoints.values()):
        variables = list(PROVIDER_KEY_ENV.values())
        errors.append(
            f"No model provider is configured: set {', '.join(variables[:-1])} or "
            f"{variables[-1]}, or define an endpoint in model_endpoints (user settings)."
        )

    # Every reference a setting or an agent names, and those a model runs on.
    named: list[tuple[str, ModelRef]] = []
    runs: list[tuple[str, ModelRef]] = []

    def setting(who: str, spec: str | None) -> ModelRef | None:
        try:
            ref = parse_ref(spec)
        except ModelSpecError as exc:
            errors.append(f"{who}: {exc}")
            return None
        named.append((who, ref))
        return ref

    parsed_so_far = len(errors)
    if manager_model:
        ref = setting("The manager's model", manager_model)
        if ref is not None:
            runs.append(("Default model", ref))
    if settings.default_model:
        ref = setting("default_model", settings.default_model)
        if ref is not None and not manager_model:
            runs.append(("default_model", ref))
    for role, spec in settings.model_roles.items():
        setting(f"model_roles[{role!r}]", spec)
    roles = None
    if len(errors) == parsed_so_far:
        try:
            roles = model_services.role_map_from_settings(settings, manager_model)
        except ModelSpecError as exc:  # a role name, or a mapping of default
            errors.append(str(exc))
    for index, spec in enumerate(settings.pinned_models):
        setting(f"pinned_models[{index}]", spec)

    for config in agent_configs or []:
        spec = getattr(config, "model", None)
        if not spec:
            continue
        who = f"Agent {getattr(config, 'name', '?')!r}"
        try:
            parsed = parse_spec(spec)
        except ModelSpecError as exc:
            errors.append(f"{who}: {exc}")
            continue
        if isinstance(parsed, RoleRef):
            # A mapped role runs on its model; an unmapped one is computed
            # from the catalog at startup.
            mapped = roles.spec_for(parsed.role) if roles is not None else None
            if mapped is not None and parsed.role != "default":
                runs.append((f"{who} (role {parsed.role!r})", mapped))
            continue
        named.append((who, parsed))
        runs.append((who, parsed))

    defined = ", ".join(
        sorted(n for n, e in endpoints.items() if e.kind is ProviderKind.OPENAI_COMPATIBLE)
    )
    for who, ref in named:
        if ref.provider not in endpoints:
            errors.append(
                f"{who}: model {str(ref)!r} names endpoint {ref.provider!r}, which is not "
                f"defined in model_endpoints (user settings); defined endpoints: "
                f"{defined or 'none'}."
            )
    for who, ref in runs:
        endpoint = endpoints.get(ref.provider)
        if endpoint is not None and not endpoint.has_credentials:
            variable = PROVIDER_KEY_ENV.get(ref.provider, "the provider's API key")
            errors.append(
                f"{who}: model {str(ref)!r} needs the {ref.provider} API key; set {variable}."
            )
        spec = settings.model_endpoints.get(ref.provider)
        if (
            endpoint is not None
            and endpoint.kind is ProviderKind.OPENAI_COMPATIBLE
            and spec is not None
            and spec.api_key_env
            and not endpoint.api_key
        ):
            # Otherwise its requests carry a placeholder key: a bare 401.
            errors.append(
                f"{who}: model {str(ref)!r} runs on endpoint {ref.provider!r}, which reads "
                f"its key from {spec.api_key_env}; {spec.api_key_env} is not set (an "
                "endpoint's key is read from the environment only, not from a .env file)."
            )
    return errors
