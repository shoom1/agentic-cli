"""How an agent names its model: a ``provider:model`` reference or a role.

The same strings work in ``AgentConfig.model``, agent YAML, ``default_model``
and role values:

- ``provider:model``, split on the first colon only, so
  ``openrouter:qwen/qwen3-235b-a22b:free`` keeps its ``:free``. The provider
  is a built-in (``google``, ``anthropic``, ``openai``, ``openrouter``) or the
  name of an endpoint the user defined.
- A bare ``gemini-…``, ``claude-…``, ``gpt-…`` or ``o<digit>…`` id maps to its
  provider, as model ids did before references existed.
- Any other bare string is a role name (``default``, ``fast``, ``strong``, or
  one an app adds). Role names are validated so they cannot look like a model
  id.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

BUILTIN_PROVIDERS: tuple[str, ...] = ("google", "anthropic", "openai", "openrouter")
BUILTIN_ROLES: tuple[str, ...] = ("default", "fast", "strong")

# Provider, endpoint and role names share one shape.
_NAME = re.compile(r"[a-z][a-z0-9_-]{0,63}")
# What a bare model id starts with, and the provider it belongs to.
_BARE_ID_PREFIXES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"gemini-"), "google"),
    (re.compile(r"claude-"), "anthropic"),
    (re.compile(r"gpt-"), "openai"),
    (re.compile(r"o\d"), "openai"),
)
# A role name may not start like a model id, even without the hyphen
# ("gemini", "gpt5"), so a typo in an id never silently becomes a role.
_ID_LIKE = re.compile(r"gemini|claude|gpt|o\d")
# Nor contain a digit: a bare "qwen3-32b" is a model id missing its provider,
# and as a role it would run on the default model, perhaps another provider's.
_DIGIT = re.compile(r"\d")


class ModelSpecError(ValueError):
    """A model reference, role name or endpoint name is malformed."""


@dataclass(frozen=True)
class ModelRef:
    """One model of one provider (a built-in provider or a user endpoint)."""

    provider: str
    model: str

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass(frozen=True)
class RoleRef:
    """A role (``default``, ``fast``, ``strong`` or an app's own) to resolve later."""

    role: str


def parse_spec(spec: str) -> ModelRef | RoleRef:
    """Parse a reference, a bare model id or a role name.

    Raises:
        ModelSpecError: For an empty or whitespace-bearing spec, an empty
            provider or model, a malformed provider name, or a bare string that
            is neither a model id nor a valid role name. The message names the
            spec.
    """
    if not isinstance(spec, str) or not spec:
        raise ModelSpecError(f"Model {spec!r} is empty; give a provider:model reference or a role.")
    if any(ch.isspace() for ch in spec):
        raise ModelSpecError(f"Model {spec!r} contains whitespace.")

    provider, sep, model = spec.partition(":")
    if sep:
        if not _NAME.fullmatch(provider):
            raise ModelSpecError(
                f"Model {spec!r}: {provider!r} is not a provider or endpoint name "
                "(lowercase letters, digits, '_' and '-', starting with a letter)."
            )
        if not model:
            raise ModelSpecError(f"Model {spec!r} names no model after {provider + ':'!r}.")
        return ModelRef(provider, model)

    for prefix, bare_provider in _BARE_ID_PREFIXES:
        if prefix.match(spec):
            return ModelRef(bare_provider, spec)

    if _DIGIT.search(spec):
        raise ModelSpecError(
            f"Model {spec!r} has no provider: write it as provider:model, such as "
            f"openrouter:{spec}, or <endpoint>:{spec} for one of your model_endpoints. "
            "(A role name has no digits.)"
        )
    try:
        return RoleRef(validate_role_name(spec))
    except ModelSpecError:
        raise ModelSpecError(
            f"Model {spec!r} is neither a provider:model reference, a gemini-, claude-, "
            "gpt- or o<digit> model id, nor a role name (lowercase letters, '_' and '-')."
        ) from None


def parse_ref(spec: str) -> ModelRef:
    """Parse a reference or bare model id; a role is an error.

    Settings that map a role (``model_roles`` values, ``default_model``) must
    name a model, never another role.
    """
    parsed = parse_spec(spec)
    if isinstance(parsed, RoleRef):
        raise ModelSpecError(
            f"Model {spec!r} is a role; give a provider:model reference or a model id here."
        )
    return parsed


def validate_role_name(name: str) -> str:
    """Return ``name`` if it is a valid role name.

    Raises:
        ModelSpecError: If it is malformed, contains a digit, or starts like a
            model id (``gemini``, ``claude``, ``gpt``, ``o<digit>``).
    """
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ModelSpecError(
            f"Role {name!r} is not a role name (lowercase letters, '_' and '-', "
            "starting with a letter, at most 64 characters)."
        )
    if _DIGIT.search(name):
        raise ModelSpecError(
            f"Role {name!r} contains a digit, so it looks like a model id; "
            "choose a name without digits."
        )
    if _ID_LIKE.match(name):
        raise ModelSpecError(f"Role {name!r} starts like a model id; choose another name.")
    return name


def validate_endpoint_name(name: str) -> str:
    """Return ``name`` if it can name a user endpoint.

    Raises:
        ModelSpecError: If it is malformed or reuses a built-in provider name.
    """
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ModelSpecError(
            f"Endpoint {name!r} is not an endpoint name (lowercase letters, digits, '_' and '-', "
            "starting with a letter, at most 64 characters)."
        )
    if name in BUILTIN_PROVIDERS:
        raise ModelSpecError(
            f"Endpoint {name!r} reuses a built-in provider name ({', '.join(BUILTIN_PROVIDERS)})."
        )
    return name
