"""Provider kinds, the built-in providers and the configured ones.

Three providers are built in: Google, Anthropic and OpenAI, on their SDK's
default host, each with its key setting. Every other provider is configured
by name in ``model_providers``: an OpenAI-compatible host (Together,
Fireworks, a local vLLM and the like) or OpenRouter (``kind: openrouter``,
which adds OpenRouter's listing, attribution headers and reasoning
parameter). An app ships its providers in its settings class and a user adds
to them in their *user* settings: a provider holds a base URL, so a project
file that could define one could send the user's key to its own server.

A configured provider's key is read from the environment variable its
``api_key_env`` names (the workflow also looks in the app's user-level env
files), never stored. Without one the client sends a placeholder key, which
is what a local server without authentication expects. What is said about a
provider's models is a ``models`` entry (``entries.ModelEntry``), keyed
``<provider>:<model>``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agentic_cli.models.refs import BUILTIN_PROVIDERS, ModelRef, validate_provider_name


class ProviderKind(str, Enum):
    """Which API a provider speaks."""

    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    OPENAI_COMPATIBLE = "openai_compatible"


#: The environment variable each built-in provider's SDK and documentation
#: name for its key, for messages that tell the user what to set (the key
#: itself comes from its setting).
PROVIDER_KEY_ENV: Mapping[str, str] = MappingProxyType(
    {
        "google": "GOOGLE_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
)

# Each built-in provider and its kind; the provider name is the kind's value.
_BUILTIN_KINDS: tuple[ProviderKind, ...] = (
    ProviderKind.GOOGLE,
    ProviderKind.ANTHROPIC,
    ProviderKind.OPENAI,
)

_ENV_VAR = r"^[A-Z_][A-Z0-9_]*$"


class ProviderSpec(BaseModel):
    """One ``model_providers`` entry, as an app or a user writes it in settings."""

    # A rejected base_url may hold a password: keep input values out of errors.
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    kind: Literal["openai_compatible", "openrouter"] = "openai_compatible"
    base_url: str
    api_key_env: str | None = Field(default=None, pattern=_ENV_VAR)

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, url: str) -> str:
        """Plain ``http(s)://host[:port][/path]``; nothing that could carry a secret.

        A key belongs in ``api_key_env``, so userinfo and a query string are
        refused rather than sent along with every request. The error never
        echoes the URL, which may hold a password.
        """
        try:
            parts = urlsplit(url)
            hostname = parts.hostname
            parts.port  # noqa: B018 - raises ValueError on a malformed port
        except ValueError:
            raise ValueError("base_url is not a valid URL") from None
        if parts.scheme not in ("http", "https"):
            raise ValueError("base_url must start with http:// or https://")
        if not hostname:
            raise ValueError("base_url must name a host")
        if parts.username is not None or parts.password is not None:
            raise ValueError(
                "base_url must not carry user info; name the key's variable in api_key_env"
            )
        if parts.query or "?" in url:
            raise ValueError("base_url must not have a query string")
        if parts.fragment or "#" in url:
            raise ValueError("base_url must not have a fragment")
        return url.rstrip("/")

    @model_validator(mode="after")
    def _check_openrouter_key(self) -> ProviderSpec:
        """OpenRouter takes no request but its model list without a key, so the
        provider names the variable that holds it."""
        if self.kind == "openrouter" and self.api_key_env is None:
            raise ValueError(
                "kind 'openrouter' needs api_key_env, the variable holding the OpenRouter key"
            )
        return self


@dataclass(frozen=True)
class Provider:
    """A provider the workflow can send requests to: a built-in or a configured one.

    ``base_url`` is None for Google, Anthropic and OpenAI (the SDK's default
    host). ``api_key`` never appears in ``repr``.
    """

    name: str
    kind: ProviderKind
    base_url: str | None
    api_key: str | None = field(default=None, repr=False)
    #: The variable the key is read from: PROVIDER_KEY_ENV's for a built-in,
    #: the spec's api_key_env for a configured provider (None: no key needed).
    key_env: str | None = None

    @property
    def builtin(self) -> bool:
        """Whether this is Google, Anthropic or OpenAI."""
        return self.kind in _BUILTIN_KINDS

    @property
    def has_credentials(self) -> bool:
        """Whether requests can be sent: a key, or a configured provider that
        names no key variable (the client sends a placeholder key)."""
        return bool(self.api_key) or (not self.builtin and self.key_env is None)


def build_providers(
    *, keys: Mapping[str, str | None], specs: Mapping[str, ProviderSpec], env: Mapping[str, str]
) -> dict[str, Provider]:
    """The three built-in providers, then one provider per spec, in order.

    Args:
        keys: Each built-in provider's key (``google``, ``anthropic``,
            ``openai``); a missing or empty one means none.
        specs: The configured providers, by name.
        env: Where a configured provider's ``api_key_env`` is looked up (the
            process environment and the app's user-level env files, in the
            workflow). Nothing else supplies a configured provider's key.

    Raises:
        ModelSpecError: If a provider name is malformed or is a built-in
            provider's.
    """
    providers: dict[str, Provider] = {}
    for kind in _BUILTIN_KINDS:
        providers[kind.value] = Provider(
            name=kind.value,
            kind=kind,
            base_url=None,
            api_key=keys.get(kind.value) or None,
            key_env=PROVIDER_KEY_ENV[kind.value],
        )
    for name, spec in specs.items():
        validate_provider_name(name)
        providers[name] = Provider(
            name=name,
            kind=ProviderKind(spec.kind),
            base_url=spec.base_url,
            api_key=(env.get(spec.api_key_env) or None) if spec.api_key_env else None,
            key_env=spec.api_key_env,
        )
    return providers


def undefined_provider_message(who: str, ref: ModelRef, providers: Mapping[str, Provider]) -> str:
    """What to say when ``ref`` names a provider ``providers`` lacks: which
    providers exist, and where one is defined."""
    defined = ", ".join(sorted(name for name, p in providers.items() if not p.builtin))
    return (
        f"{who}: model {str(ref)!r} names provider {ref.provider!r}, which is neither a "
        f"built-in provider ({', '.join(BUILTIN_PROVIDERS)}) nor defined in model_providers "
        f"(user settings); defined: {defined or 'none'}."
    )
