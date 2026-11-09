"""Where agentic-cli's settings become arguments for ``agentic_cli.models``.

The models package takes plain arguments (provider keys, provider specs and
the environment their keys are read from, model entries, role mappings, the
models the settings and the agents name, a cache path). This module is the
one place that reads them from settings, so the package never sees a settings
object. A workflow manager builds its catalog here (``build_catalog``) when it
initializes, and so do the ``/settings`` pickers and ``/models`` before it is
ready.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

# python-dotenv, which pydantic-settings reads the same files with.
from dotenv import dotenv_values

from agentic_cli.models import (
    BUILTIN_ROLES,
    CatalogView,
    ModelCatalog,
    ModelEntry,
    ModelRef,
    ModelResolutionError,
    ModelSpecError,
    Provider,
    RoleMap,
    RoleRef,
    build_providers,
    parse_entry_key,
    parse_ref,
    parse_spec,
)

if TYPE_CHECKING:
    import httpx

    from agentic_cli.config import BaseSettings
    from agentic_cli.workflow.config import AgentConfig

#: python-dotenv's logger, which reports each line it cannot parse at every read.
_DOTENV_LOGGER = logging.getLogger("dotenv.main")


class _Quiet(logging.Filter):
    """Drops every record; one instance per read, so reads never share one."""

    def filter(self, record: logging.LogRecord) -> bool:
        return False


def provider_keys(settings: "BaseSettings") -> dict[str, str | None]:
    """Each built-in provider's key, by provider name."""
    return {
        "google": settings.google_api_key,
        "anthropic": settings.anthropic_api_key,
        "openai": settings.openai_api_key,
    }


def user_env_files(settings: "BaseSettings") -> tuple[Path, ...]:
    """The settings' user-level env files, in order: each of their env files
    (``BaseSettings.env_files``) that is absolute once ``~`` is expanded, such
    as the research demo's ``~/.research_demo/.env``.

    A cwd-relative one (``.env``) is never among them: a cloned repository can
    ship it, so it supplies no key (the settings' sources,
    ``settings_sources.SettingsSourcesMixin``, filter it the same way).

    Settings composed from the mixins over pydantic-settings' own
    ``BaseSettings`` have no ``env_files``; their class's ``env_file`` is used.
    The mixins bring the same sources and filters, so a cwd-relative entry
    defines no provider there either.
    """
    entries = getattr(settings, "env_files", None)
    if entries is None:
        env_file = type(settings).model_config.get("env_file")
        if env_file is None:
            entries = ()
        elif isinstance(env_file, (str, os.PathLike)):
            entries = (env_file,)
        else:
            entries = tuple(env_file)
    files = (Path(entry).expanduser() for entry in entries)
    return tuple(path for path in files if path.is_absolute())


def provider_env(settings: "BaseSettings") -> dict[str, str]:
    """Where a configured provider's ``api_key_env`` is looked up: the user-level
    env files (``user_env_files``; a later file wins, an entry without a value
    is left out), then the process environment over them all."""
    env: dict[str, str] = {}
    files = [path for path in user_env_files(settings) if path.is_file()]
    if files:
        encoding = type(settings).model_config.get("env_file_encoding") or "utf-8"
        # The settings' own load read these files before the UI started and
        # reported any line it could not parse then; this read runs at every
        # validation, /settings and /models, over the UI, so it reports none.
        quiet = _Quiet()
        _DOTENV_LOGGER.addFilter(quiet)
        try:
            for path in files:
                values = dotenv_values(path, encoding=encoding)
                env.update({name: v for name, v in values.items() if v is not None})
        finally:
            _DOTENV_LOGGER.removeFilter(quiet)
    env.update(os.environ)
    return env


def providers_from_settings(
    settings: "BaseSettings", env: Mapping[str, str] | None = None
) -> dict[str, Provider]:
    """The built-in providers, then the ``model_providers``.

    A built-in's key is its setting. A configured provider's is read from
    ``env`` (``provider_env(settings)`` by default: the environment, then the
    app's user-level env files) under the variable its ``api_key_env`` names,
    and from nowhere else.
    """
    return build_providers(
        keys=provider_keys(settings),
        specs=settings.model_providers,
        env=provider_env(settings) if env is None else env,
    )


def entries_from_settings(settings: "BaseSettings") -> dict[ModelRef, ModelEntry]:
    """The ``models`` entries by reference, in their order: the app's first,
    then the user's, the project's and the environment's additions."""
    return {parse_entry_key(key): entry for key, entry in settings.models.items()}


def model_references(
    settings: "BaseSettings",
    agent_configs: Iterable["AgentConfig"] = (),
    manager_model: str | None = None,
) -> tuple[ModelRef, ...]:
    """Every model the settings, the agents and the manager name, once each.

    In order: the manager's own model (``manager_model``), ``default_model``,
    each ``model_roles`` value (so the model of each mapped role an agent
    names), then each agent's model that is a reference or a bare model id.
    One that does not parse is left out: validation reports it.
    """
    return tuple(_named_references(settings, agent_configs, manager_model))


def reference_settings(
    settings: "BaseSettings",
    agent_configs: Iterable["AgentConfig"] = (),
    manager_model: str | None = None,
) -> dict[ModelRef, str]:
    """Where each model ``model_references`` gives is named, in its order:
    ``the manager's model``, ``default_model``, ``model_roles['fast']``, ``agent
    'writer'``, those that name one model joined with "and". The catalog's
    warning that a provider does not serve a model says which setting to change.
    """
    named = _named_references(settings, agent_configs, manager_model)
    return {ref: " and ".join(where) for ref, where in named.items()}


def _named_references(
    settings: "BaseSettings",
    agent_configs: Iterable["AgentConfig"],
    manager_model: str | None,
) -> dict[ModelRef, list[str]]:
    """Each model the settings, the agents and the manager name, with where."""
    named: dict[ModelRef, list[str]] = {}
    specs = [
        ("the manager's model", manager_model),
        ("default_model", settings.default_model),
        *((f"model_roles[{role!r}]", spec) for role, spec in settings.model_roles.items()),
    ]
    for where, spec in specs:
        if not spec:
            continue
        try:
            named.setdefault(parse_ref(spec), []).append(where)
        except ModelSpecError:
            continue
    for config in agent_configs:
        parsed = _agent_model(config)
        if isinstance(parsed, ModelRef):
            named.setdefault(parsed, []).append(f"agent {config.name!r}")
    return named


def computed_roles(
    settings: "BaseSettings",
    agent_configs: Iterable["AgentConfig"] = (),
    manager_model: str | None = None,
) -> tuple[str, ...]:
    """The built-in roles whose model is computed, in ``BUILTIN_ROLES`` order.

    ``default`` when neither the manager's own model nor ``default_model`` is
    set, and each other built-in role an agent names that ``model_roles``
    does not map. (An app's own role nobody maps runs on the default model.)
    """
    computed: set[str] = set()
    if not manager_model and not settings.default_model:
        computed.add("default")
    for parsed in _agent_models(agent_configs):
        if (
            isinstance(parsed, RoleRef)
            and parsed.role in BUILTIN_ROLES
            and parsed.role != "default"
            and parsed.role not in settings.model_roles
        ):
            computed.add(parsed.role)
    return tuple(role for role in BUILTIN_ROLES if role in computed)


def _agent_models(agent_configs: Iterable["AgentConfig"]) -> list[ModelRef | RoleRef]:
    """Each agent's model that names one (None runs on the default) and parses."""
    parsed = (_agent_model(config) for config in agent_configs)
    return [model for model in parsed if model is not None]


def _agent_model(config: "AgentConfig") -> ModelRef | RoleRef | None:
    """The agent's model, when it names one (None runs on the default) that parses."""
    spec = getattr(config, "model", None)
    if not spec:
        return None
    try:
        return parse_spec(spec)
    except ModelSpecError:
        return None


def role_map_from_settings(settings: "BaseSettings", manager_model: str | None = None) -> RoleMap:
    """The roles the settings map: ``default_model`` and ``model_roles``.

    ``manager_model`` is a manager's own model override (``Manager(model=...)``
    or ``reinitialize(model=...)``), which wins for ``default`` in that manager.

    Raises:
        ModelSpecError: If a value names a role or is malformed; the message
            names the setting.
    """
    return RoleMap(
        default_model=settings.default_model,
        model_roles=settings.model_roles,
        manager_model=manager_model,
    )


def role_models(
    settings: "BaseSettings",
    catalog: CatalogView,
    providers: Mapping[str, Provider],
    default: str | None = None,
) -> dict[str, list[str]]:
    """The roles' models, each with the roles it fills, in role order.

    The models of ``default`` (``default`` when given: a manager's own
    model), ``fast`` and ``strong``, mapped or computed, and of each mapped
    app role; a role nothing can fill is left out. The list offers these
    models (``ModelCatalog.offered(extra=...)``) in the ``/settings`` pickers
    and in ``/models`` alike, whichever pickers an app shows.
    """
    try:
        role_map = role_map_from_settings(settings)
    except ModelSpecError:
        return {}
    models: dict[str, list[str]] = {}
    for role in dict.fromkeys([*BUILTIN_ROLES, *settings.model_roles]):
        if role == "default" and default:
            ref = default
        else:
            try:
                ref = str(role_map.resolve(role, catalog=catalog, providers=providers))
            except (ModelResolutionError, ModelSpecError):
                continue
        models.setdefault(ref, []).append(role)
    return models


def model_cache_path(settings: "BaseSettings") -> Path:
    """The model-list cache: ``~/.{app_name}/model_cache.json``."""
    return Path.home() / f".{settings.app_name}" / "model_cache.json"


def build_catalog(
    settings: "BaseSettings",
    providers: Mapping[str, Provider],
    *,
    agent_configs: Iterable["AgentConfig"] = (),
    manager_model: str | None = None,
    transport: "httpx.AsyncBaseTransport | None" = None,
) -> ModelCatalog:
    """The catalog a workflow manager loads: ``providers``' models, cached in
    ``model_cache_path(settings)``, with the ``models`` entries, the models
    and computed roles that decide which providers are in use and what the
    list offers, and where each model is named (``reference_settings``), for
    the warning that its provider does not serve it.

    Args:
        settings: Give the entries, the references and the roles, and name
            the app, and so the cache file.
        providers: Every provider, built in and configured
            (``providers_from_settings``).
        agent_configs: The agents, whose models are named too.
        manager_model: A manager's own model override, if any.
        transport: The HTTP transport for listings; httpx's own by default.
    """
    agent_configs = tuple(agent_configs)
    named_by = reference_settings(settings, agent_configs, manager_model)
    return ModelCatalog(
        providers=providers,
        entries=entries_from_settings(settings),
        references=tuple(named_by),
        named_by=named_by,
        computed_roles=computed_roles(settings, agent_configs, manager_model),
        cache_path=model_cache_path(settings),
        transport=transport,
    )
