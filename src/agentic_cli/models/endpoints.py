"""Provider kinds and named endpoints.

Four providers are built in: Google, Anthropic and OpenAI on their SDK's
default host, and OpenRouter. A user adds OpenAI-compatible servers (vLLM and
the like) as named endpoints in their *user* settings (``model_endpoints``):
an endpoint holds a base URL, so a project file that could define one could
send the user's key to its own server.

An endpoint's key is read from the environment variable its ``api_key_env``
names, never stored. Without one the client sends a placeholder key, which is
what a local server without authentication expects.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentic_cli.models.refs import validate_endpoint_name


class ProviderKind(str, Enum):
    """Which API a provider speaks."""

    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    OPENAI_COMPATIBLE = "openai_compatible"


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Each built-in provider and its kind; the provider name is the kind's value.
_BUILTIN_KINDS: tuple[ProviderKind, ...] = (
    ProviderKind.GOOGLE,
    ProviderKind.ANTHROPIC,
    ProviderKind.OPENAI,
    ProviderKind.OPENROUTER,
)
_BUILTIN_BASE_URLS: Mapping[ProviderKind, str] = {ProviderKind.OPENROUTER: OPENROUTER_BASE_URL}

_LEVEL = re.compile(r"[a-z][a-z0-9_]*")
_ENV_VAR = r"^[A-Z_][A-Z0-9_]*$"


class DeclaredModel(BaseModel):
    """What the user declares about one model an endpoint serves.

    An endpoint's ``/v1/models`` listing gives little more than the served
    name and context length, so what a model can do comes from here.
    """

    model_config = ConfigDict(extra="forbid")

    context_window: int | None = Field(default=None, ge=1)
    max_output: int | None = Field(default=None, ge=1)
    tools: bool = True
    image_input: bool = False
    #: ``chat_template``: thinking switched by ``chat_template_kwargs.enable_thinking``
    #: (Qwen3 and similar on vLLM); ``effort``: a ``reasoning_effort`` parameter.
    thinking: Literal["none", "chat_template", "effort"] = "none"
    #: The effort ladder, lowest first; used only when ``thinking == "effort"``.
    effort_levels: list[str] = Field(default_factory=lambda: ["low", "medium", "high"])
    reasoning_text: bool = True

    @field_validator("effort_levels")
    @classmethod
    def _check_levels(cls, levels: list[str]) -> list[str]:
        if not levels:
            raise ValueError("effort_levels must name at least one level")
        bad = [level for level in levels if not _LEVEL.fullmatch(level)]
        if bad:
            raise ValueError(f"effort level(s) {bad} must be lowercase names")
        if len(set(levels)) != len(levels):
            raise ValueError("effort_levels must not repeat a level")
        return levels


class EndpointSpec(BaseModel):
    """One ``model_endpoints`` entry, as the user writes it in settings."""

    # A rejected base_url may hold a password: keep input values out of errors.
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    kind: Literal["openai_compatible"] = "openai_compatible"
    base_url: str
    api_key_env: str | None = Field(default=None, pattern=_ENV_VAR)
    models: dict[str, DeclaredModel] = Field(default_factory=dict)

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

    @field_validator("models")
    @classmethod
    def _check_model_ids(cls, models: dict[str, DeclaredModel]) -> dict[str, DeclaredModel]:
        for model_id in models:
            if not model_id or any(ch.isspace() for ch in model_id):
                raise ValueError(
                    f"model id {model_id!r} must be a non-empty name without whitespace"
                )
        return models


@dataclass(frozen=True)
class Endpoint:
    """A provider the workflow can send requests to.

    ``base_url`` is None for Google, Anthropic and OpenAI (the SDK's default
    host). ``api_key`` never appears in ``repr``.
    """

    name: str
    kind: ProviderKind
    base_url: str | None
    api_key: str | None = field(default=None, repr=False)
    models: Mapping[str, DeclaredModel] = field(default_factory=dict)

    @property
    def has_credentials(self) -> bool:
        """Whether requests can be sent: a key, or an endpoint (which may need none)."""
        if self.kind is ProviderKind.OPENAI_COMPATIBLE:
            return True
        return bool(self.api_key)


def build_endpoints(
    *, keys: Mapping[str, str | None], specs: Mapping[str, EndpointSpec], env: Mapping[str, str]
) -> dict[str, Endpoint]:
    """The four built-in providers plus one endpoint per spec.

    Args:
        keys: Each built-in provider's key (``google``, ``anthropic``,
            ``openai``, ``openrouter``); a missing or empty one means none.
        specs: The user's endpoints, by name.
        env: Where an endpoint's ``api_key_env`` is looked up (the process
            environment, in the workflow). Nothing else supplies an endpoint
            key.

    Raises:
        ModelSpecError: If an endpoint name is malformed or reuses a built-in
            provider name.
    """
    endpoints: dict[str, Endpoint] = {}
    for kind in _BUILTIN_KINDS:
        endpoints[kind.value] = Endpoint(
            name=kind.value,
            kind=kind,
            base_url=_BUILTIN_BASE_URLS.get(kind),
            api_key=keys.get(kind.value) or None,
        )
    for name, spec in specs.items():
        validate_endpoint_name(name)
        endpoints[name] = Endpoint(
            name=name,
            kind=ProviderKind.OPENAI_COMPATIBLE,
            base_url=spec.base_url,
            api_key=(env.get(spec.api_key_env) or None) if spec.api_key_env else None,
            models=MappingProxyType(dict(spec.models)),
        )
    return endpoints
