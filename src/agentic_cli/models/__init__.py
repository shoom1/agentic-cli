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
- ``table``: the shipped table (``data/table.json``): family rules, per-model
  numbers and the fallback model lists.
- ``roles``: ``RoleMap``, role names to references, computed from the
  providers with a key when unmapped.
- ``resolve``: ``resolve_model``, one agent's model, thinking, ``max_tokens``
  and sampling settings as a ``ResolvedModel``.
- ``discovery``: one lister per provider, over its REST API with httpx.
- ``catalog``: ``ModelCatalog``, the records merged from declarations,
  listings and the table; which ids each provider rejects; the listing cache
  and its background refresh.
"""

from agentic_cli.models.catalog import ModelCatalog, ProviderState, ProviderStatus
from agentic_cli.models.endpoints import (
    OPENROUTER_BASE_URL,
    PROVIDER_KEY_ENV,
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
from agentic_cli.models.resolve import (
    CatalogView,
    GenerationSettings,
    ModelResolutionError,
    NativeThinking,
    ResolvedModel,
    ThinkingIntent,
    resolve_model,
)
from agentic_cli.models.roles import RoleMap
from agentic_cli.models.table import ShippedTable, load_table, version_key

__all__ = [
    "BUILTIN_PROVIDERS",
    "BUILTIN_ROLES",
    "CacheMechanism",
    "CatalogView",
    "DeclaredModel",
    "Endpoint",
    "EndpointSpec",
    "GenerationSettings",
    "ModelCatalog",
    "ModelRecord",
    "ModelRef",
    "ModelResolutionError",
    "ModelSpecError",
    "NativeThinking",
    "OPENROUTER_BASE_URL",
    "PROVIDER_KEY_ENV",
    "ProviderKind",
    "ProviderState",
    "ProviderStatus",
    "ResolvedModel",
    "RoleMap",
    "RoleRef",
    "ShippedTable",
    "ThinkingIntent",
    "ThinkingMechanism",
    "ThinkingRule",
    "build_endpoints",
    "load_table",
    "parse_ref",
    "parse_spec",
    "resolve_model",
    "validate_endpoint_name",
    "validate_role_name",
    "version_key",
]
