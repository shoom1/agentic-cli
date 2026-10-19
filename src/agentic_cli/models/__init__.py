"""Model knowledge for every backend: references, endpoints and records.

Provisional in 0.7.0: names and signatures may still change.

The package is data and logic only. It imports nothing from the rest of
agentic-cli and no provider SDK (``tests/models/test_boundary.py`` checks
both); each backend's adapter turns what it resolves into SDK objects.

- ``refs``: ``provider:model`` references, bare model ids and role names.
- ``endpoints``: provider kinds, the built-in providers and the user's
  OpenAI-compatible endpoints.
- ``records``: ``ModelRecord``, what is known about one model, and its
  ``ThinkingRule``.
"""

from agentic_cli.models.endpoints import (
    OPENROUTER_BASE_URL,
    DeclaredModel,
    Endpoint,
    EndpointSpec,
    ProviderKind,
    build_endpoints,
)
from agentic_cli.models.records import (
    CacheMechanism,
    ModelRecord,
    ThinkingMechanism,
    ThinkingRule,
)
from agentic_cli.models.refs import (
    BUILTIN_PROVIDERS,
    BUILTIN_ROLES,
    ModelRef,
    ModelSpecError,
    RoleRef,
    parse_ref,
    parse_spec,
    validate_endpoint_name,
    validate_role_name,
)

__all__ = [
    "BUILTIN_PROVIDERS",
    "BUILTIN_ROLES",
    "CacheMechanism",
    "DeclaredModel",
    "Endpoint",
    "EndpointSpec",
    "ModelRecord",
    "ModelRef",
    "ModelSpecError",
    "OPENROUTER_BASE_URL",
    "ProviderKind",
    "RoleRef",
    "ThinkingMechanism",
    "ThinkingRule",
    "build_endpoints",
    "parse_ref",
    "parse_spec",
    "validate_endpoint_name",
    "validate_role_name",
]
