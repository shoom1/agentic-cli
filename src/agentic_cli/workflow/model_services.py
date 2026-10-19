"""Where agentic-cli's settings become arguments for ``agentic_cli.models``.

The models package takes plain arguments (provider keys, endpoint specs, role
mappings, a cache path). This module is the one place that reads them from
settings, so the package never sees a settings object. A workflow manager
builds its catalog here (``build_catalog``) when it initializes.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from agentic_cli.models import Endpoint, ModelCatalog, RoleMap, build_endpoints

if TYPE_CHECKING:
    import httpx

    from agentic_cli.config import BaseSettings


def provider_keys(settings: "BaseSettings") -> dict[str, str | None]:
    """Each built-in provider's key, by provider name."""
    return {
        "google": settings.google_api_key,
        "anthropic": settings.anthropic_api_key,
        "openai": settings.openai_api_key,
        "openrouter": settings.openrouter_api_key,
    }


def endpoints_from_settings(
    settings: "BaseSettings", env: Mapping[str, str] | None = None
) -> dict[str, Endpoint]:
    """The built-in providers and the user's ``model_endpoints``.

    An endpoint's key is read from ``env`` (the process environment by
    default) under the variable its ``api_key_env`` names, and from nowhere
    else.
    """
    return build_endpoints(
        keys=provider_keys(settings),
        specs=settings.model_endpoints,
        env=os.environ if env is None else env,
    )


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
    endpoints: Mapping[str, Endpoint],
    *,
    transport: "httpx.AsyncBaseTransport | None" = None,
) -> ModelCatalog:
    """The catalog a workflow manager loads: ``endpoints``' models, cached in
    ``model_cache_path(settings)``.

    Args:
        settings: Name the app, and so the cache file.
        endpoints: Every provider and endpoint (``endpoints_from_settings``).
        transport: The HTTP transport for listings; httpx's own by default.
    """
    return ModelCatalog(
        endpoints=endpoints, cache_path=model_cache_path(settings), transport=transport
    )
