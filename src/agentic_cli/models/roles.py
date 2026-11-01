"""Roles: a model named by what it is for, mapped to a reference.

``default`` is ``default_model`` and nothing else (a manager's own model
override comes first). ``fast`` and ``strong`` are mapped in ``model_roles``.
An app adds a role by naming it in an agent's model, and may ship a default
mapping by overriding the ``model_roles`` default in its settings class.

A built-in role nobody mapped takes the newest model of its tier from the
first provider with a key, in the order Google, Anthropic, OpenAI: Flash-Lite /
Flash / Pro, Haiku / Sonnet / Opus, mini / standard / pro. A configured
provider's models (``model_providers``) never fill a role on their own; the
user maps them. An app role nobody mapped falls back to ``default``, with a
warning once per process.

A role's value is always a model, never another role.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from agentic_cli.models.providers import PROVIDER_KEY_ENV, Provider
from agentic_cli.models.refs import (
    BUILTIN_ROLES,
    ModelRef,
    ModelSpecError,
    parse_ref,
    validate_role_name,
)
from agentic_cli.models.resolve import ModelResolutionError
from agentic_cli.models.table import newest

if TYPE_CHECKING:
    from agentic_cli.models.resolve import CatalogView

logger = logging.getLogger(__name__)

#: The providers that fill an unmapped built-in role, in order.
_ROLE_PROVIDERS = ("google", "anthropic", "openai")
#: App roles already warned about as unmapped. Once per process: a manager
#: builds a new map on every reinitialize, and the user was told already.
_WARNED_ROLES: set[str] = set()


class RoleMap:
    """Maps role names to model references.

    Args:
        default_model: The ``default`` role (``settings.default_model``).
        model_roles: Every other role's model (``settings.model_roles``).
        manager_model: A manager's own model override, which wins for
            ``default`` in that manager only.

    Raises:
        ModelSpecError: If a value is malformed or names a role, a role name
            is malformed, or ``model_roles`` maps ``default``. The message
            names the setting.
    """

    def __init__(
        self,
        *,
        default_model: str | None,
        model_roles: Mapping[str, str],
        manager_model: str | None = None,
    ) -> None:
        self._default = _ref_setting("default_model", default_model)
        self._manager = _ref_setting("the manager's model", manager_model)
        roles: dict[str, ModelRef] = {}
        for role, spec in model_roles.items():
            key = f"model_roles[{role!r}]"
            if role == "default":
                raise ModelSpecError(
                    f"{key}: the default role is default_model; set default_model instead."
                )
            try:
                validate_role_name(role)
            except ModelSpecError as exc:
                raise ModelSpecError(f"{key}: {exc}") from None
            roles[role] = _ref_setting(key, spec)
        self._roles = roles

    def spec_for(self, role: str) -> ModelRef | None:
        """The reference mapped to ``role``, or None when it is computed later."""
        if role == "default":
            return self._manager or self._default
        return self._roles.get(role)

    def resolve(
        self, role: str, *, catalog: "CatalogView", providers: Mapping[str, Provider]
    ) -> ModelRef:
        """The model ``role`` names now.

        Raises:
            ModelSpecError: If ``role`` is not a valid role name.
            ModelResolutionError: If a built-in role is unmapped and no
                provider with a key has a model of its tier.
        """
        validate_role_name(role)
        mapped = self.spec_for(role)
        if mapped is not None:
            return mapped
        if role in BUILTIN_ROLES:
            return self._computed(role, catalog, providers)
        if role not in _WARNED_ROLES:
            _WARNED_ROLES.add(role)
            logger.warning(
                "Role %r is not mapped; using the default model. "
                "Map it in model_roles to choose one.",
                role,
            )
        return self.resolve("default", catalog=catalog, providers=providers)

    def _computed(
        self, tier: str, catalog: "CatalogView", providers: Mapping[str, Provider]
    ) -> ModelRef:
        for name in _ROLE_PROVIDERS:
            provider = providers.get(name)
            if provider is None or not provider.has_credentials:
                continue
            candidates = [r for r in catalog.records(name) if r.tier == tier]
            if candidates:
                return newest(candidates).ref
        variables = ", ".join(PROVIDER_KEY_ENV[p] for p in _ROLE_PROVIDERS)
        fix = "set default_model" if tier == "default" else f"map it in model_roles[{tier!r}]"
        raise ModelResolutionError(
            f"No model for role {tier!r}: {fix}, or set a Google, Anthropic or OpenAI key "
            f"({variables}) so the newest {tier}-tier model is used."
        )


def _ref_setting(key: str, spec: str | None) -> ModelRef | None:
    if not spec:
        return None
    try:
        return parse_ref(spec)
    except ModelSpecError as exc:
        raise ModelSpecError(f"{key}: {exc}") from None
