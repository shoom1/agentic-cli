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

Settings Loading Priority (highest to lowest; settings_sources.py):
    1. Constructor arguments
    2. Environment variables (AGENTIC_* prefix)
    3. Project config (./.{app_name}/settings.json), allowlisted keys only
    4. User config (~/.{app_name}/settings.json)
    5. .env file (a cwd-relative one allowlisted keys only)
    6. Default values

    app_name comes only from the constructor or the class (or the command
    line, for an app that enables pydantic-settings' cli_parse_args).
"""

import os
from contextvars import ContextVar, Token
from typing import Generator, Any, Type
from contextlib import contextmanager

from pydantic import PrivateAttr
from pydantic_settings import (
    BaseSettings as PydanticBaseSettings,
    SettingsConfigDict,
)
from pydantic_settings.sources import ENV_FILE_SENTINEL

# The sources (settings_customise_sources) and their trust filters come with
# the mixins (settings_sources.SettingsSourcesMixin), so a class composed from
# them over pydantic-settings' own BaseSettings reads what this one reads.
from agentic_cli.workflow.settings import WorkflowSettingsMixin
from agentic_cli.settings_mixins import AppSettingsMixin, CLISettingsMixin
from agentic_cli.settings_persistence import CREDENTIAL_KEY_RE as _CREDENTIAL_KEY_RE
from agentic_cli.settings_sources import _accepted_input_names

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


# Constructor kwargs matching _CREDENTIAL_KEY_RE (defined in
# settings_persistence, which also uses it to never save credentials) are
# credentials; an unrecognised one must fail loudly instead of being swallowed
# by ``extra="ignore"``.


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


class BaseSettings(WorkflowSettingsMixin, AppSettingsMixin, CLISettingsMixin, PydanticBaseSettings):
    """Base settings for agentic CLI applications.

    Domain-specific applications extend this class and override:
    - model_config: Set env_prefix and env_file for domain
    - app_name: Default application name
    - workspace_dir: Default workspace directory

    Settings are loaded from (in order of precedence; the mixins'
    ``settings_sources.SettingsSourcesMixin``):
    1. Constructor arguments
    2. Environment variables (with domain-specific prefix)
    3. Project config (./.{app_name}/settings.json), allowlisted keys only
    4. User config (~/.{app_name}/settings.json)
    5. Domain-specific .env file (a cwd-relative one allowlisted keys only)
    6. Default values

    ``app_name`` chooses the files, so only the constructor or the class sets
    it.

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
        # written inline in a model_providers entry, or a password in its
        # base_url. Pydantic honours this only on the outermost model, so the
        # nested ProviderSpec's own setting does not cover it here. Errors
        # still name the setting and say what was expected.
        hide_input_in_errors=True,
    )

    # The env_file the dotenv source was given (see env_files).
    _env_file_used: Any = PrivateAttr(default=None)

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
        # As pydantic-settings decides which env files the dotenv source reads.
        given = values.get("_env_file", ENV_FILE_SENTINEL)
        self._env_file_used = (
            given if given != ENV_FILE_SENTINEL else type(self).model_config.get("env_file")
        )

    @property
    def env_files(self) -> tuple[str | os.PathLike[str], ...]:
        """The env files these settings were read from, as configured, in order.

        The constructor's ``_env_file`` when it was given one (None: none),
        else the class's ``env_file``. Entries are as written: ``~`` not
        expanded, cwd-relative ones included. A configured model provider's
        key is also looked up in the user-level ones
        (``workflow.model_services.user_env_files``).
        """
        env_file = self._env_file_used
        if env_file is None:
            return ()
        if isinstance(env_file, (str, os.PathLike)):
            return (env_file,)
        return tuple(env_file)

    def update_setting(self, key: str, value: Any) -> None:
        """Update a single setting, using dedicated setters where required.

        A hand-edited setting (``settings_persistence.HAND_EDITED_KEYS``:
        ``models``, ``model_providers``) takes the value as given, for this
        session only: the app's shipped entries are not merged in, and no
        save writes it.

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

    - every reference parses: ``default_model``, ``model_roles`` and each
      agent's ``model`` (a role must map to a model, never another role);
      ``models`` keys are checked when the settings load;
    - every provider a reference names is built in or defined in
      ``model_providers``;
    - every provider a model will run on can be called: the default model's,
      each agent's, and that of each role an agent names (a built-in has its
      key; a configured provider's ``api_key_env`` is set, in the environment
      or the app's user-level env files, or it names none);
    - a built-in role that is computed, not mapped (``default`` without a
      ``default_model``, a ``fast`` or ``strong`` an agent names), has a model
      to be computed from: only a Google, Anthropic or OpenAI model has a
      tier, so without one of their keys it has none (said once, as startup
      would say it);
    - no ``models`` entry asks an ``openrouter`` provider for chat-template
      thinking;
    - some provider can be called (else the message names the built-ins'
      key variables and each configured provider's, with where it is read).

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
        parse_entry_key,
        parse_ref,
        parse_spec,
    )
    from agentic_cli.models.providers import undefined_provider_message
    from agentic_cli.models.roles import unfilled_role_message
    from agentic_cli.workflow import model_services

    errors: list[str] = []
    providers = model_services.providers_from_settings(settings)
    # Where a configured provider's key is read, for the messages naming one.
    files = " or ".join(str(path) for path in model_services.user_env_files(settings))
    where = f"the environment or in {files}" if files else "the environment"
    if not any(provider.has_credentials for provider in providers.values()):
        variables = list(PROVIDER_KEY_ENV.values())
        builtin = f"set {', '.join(variables[:-1])} or {variables[-1]}"
        # None can be called, so each configured one names a key it lacks.
        configured = " or ".join(
            f"{provider.key_env} for {name!r}"
            for name, provider in providers.items()
            if not provider.builtin
        )
        if configured:
            errors.append(
                f"No model provider is configured: {builtin}, or {configured} "
                f"(model_providers), in {where}."
            )
        else:
            errors.append(
                f"No model provider is configured: {builtin}, or define a provider in "
                "model_providers (user settings)."
            )
    elif not any(p.builtin and p.has_credentials for p in providers.values()):
        # Only a built-in provider's model has a tier, so a role computed from
        # the tiers has no model: say so now, once, not at startup per agent.
        computed = model_services.computed_roles(settings, agent_configs or (), manager_model)
        errors.extend(unfilled_role_message(role) for role in computed)

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

    for who, ref in named:
        if ref.provider not in providers:
            errors.append(undefined_provider_message(who, ref, providers))
    for who, ref in runs:
        provider = providers.get(ref.provider)
        if provider is None or provider.has_credentials:
            continue
        if provider.builtin:
            errors.append(
                f"{who}: model {str(ref)!r} needs the {ref.provider} API key; "
                f"set {provider.key_env}."
            )
        else:
            # Otherwise its requests would carry a placeholder key: a bare 401.
            errors.append(
                f"{who}: model {str(ref)!r} runs on {ref.provider!r}, which reads its key "
                f"from {provider.key_env}; set {provider.key_env} in {where}."
            )

    # OpenRouter switches thinking by its reasoning parameter, never by the
    # chat template. Checked here, not when the settings load: an entry's
    # provider kind is known only with the providers.
    for key, entry in settings.models.items():
        provider = providers.get(parse_entry_key(key).provider)
        if (
            provider is not None
            and provider.kind is ProviderKind.OPENROUTER
            and entry.thinking == "chat_template"
        ):
            errors.append(
                f"models[{key!r}]: thinking 'chat_template' is not available on an "
                "openrouter provider; use 'none' or 'effort'."
            )
    return errors
