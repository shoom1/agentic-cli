"""Where agentic-cli's settings become arguments for ``agentic_cli.models``.

The models package takes plain arguments (provider keys, provider specs and
the environment their keys are read from, model entries, role mappings, a
cache path). This module is the one place that reads them from settings, so
the package never sees a settings object. A workflow manager builds its
catalog here (``build_catalog``) when it initializes.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

# python-dotenv, which pydantic-settings reads the same files with.
from dotenv import dotenv_values

from agentic_cli.models import (
    ModelCatalog,
    ModelEntry,
    ModelRef,
    Provider,
    RoleMap,
    build_providers,
    parse_entry_key,
)

if TYPE_CHECKING:
    import httpx

    from agentic_cli.config import BaseSettings

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
    ship it, so it supplies no key (``BaseSettings.settings_customise_sources``
    filters it the same way).
    """
    files = (Path(entry).expanduser() for entry in settings.env_files)
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


def model_cache_path(settings: "BaseSettings") -> Path:
    """The model-list cache: ``~/.{app_name}/model_cache.json``."""
    return Path.home() / f".{settings.app_name}" / "model_cache.json"


def build_catalog(
    settings: "BaseSettings",
    providers: Mapping[str, Provider],
    *,
    transport: "httpx.AsyncBaseTransport | None" = None,
) -> ModelCatalog:
    """The catalog a workflow manager loads: ``providers``' models, cached in
    ``model_cache_path(settings)``, with the ``models`` entries.

    Args:
        settings: Give the entries, and name the app, and so the cache file.
        providers: Every provider, built in and configured
            (``providers_from_settings``).
        transport: The HTTP transport for listings; httpx's own by default.
    """
    return ModelCatalog(
        providers=providers,
        entries=entries_from_settings(settings),
        cache_path=model_cache_path(settings),
        transport=transport,
    )
