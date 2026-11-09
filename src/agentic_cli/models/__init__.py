"""Model knowledge for every backend: references, providers and records.

Provisional in 0.7.0: names and signatures may still change.

The package is data and logic only. It imports nothing from the rest of
agentic-cli and no provider SDK (``tests/models/test_boundary.py`` checks
both); each backend's adapter turns what it resolves into SDK objects.

- ``refs``: ``provider:model`` references, bare model ids and role names.
- ``providers``: provider kinds, the built-in providers (Google, Anthropic,
  OpenAI) and the configured ones (``model_providers``).
- ``entries``: ``ModelEntry``, what the ``models`` setting says about one
  model, keyed by its ``provider:model`` reference.
- ``records``: ``ModelRecord``, what is known about one model, and its
  ``ThinkingRule``.
- ``table``: the shipped table (``data/table.json``): family rules, per-model
  numbers and the fallback model lists.
- ``roles``: ``RoleMap``, role names to references; an unmapped built-in
  role is computed from the list, else from the providers with a key.
- ``resolve``: ``resolve_model``, one agent's model, thinking, ``max_tokens``
  and sampling settings as a ``ResolvedModel``.
- ``discovery``: one lister per provider, over its REST API with httpx.
- ``catalog``: ``ModelCatalog``, the records merged from ``models`` entries,
  listings and the table; which ids each provider rejects; the providers in
  use, the list the model pickers offer, and search over full listings; the
  background refresh. It builds records with ``listing_records`` and keeps
  listings in the file ``listing_cache`` reads and writes (neither is
  exported).
"""

from agentic_cli.models.catalog import ModelCatalog, ProviderState, ProviderStatus
from agentic_cli.models.entries import ModelEntry, parse_entry_key
from agentic_cli.models.providers import (
    PROVIDER_KEY_ENV,
    Provider,
    ProviderKind,
    ProviderSpec,
    build_providers,
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
    validate_provider_name,
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
    "GenerationSettings",
    "ModelCatalog",
    "ModelEntry",
    "ModelRecord",
    "ModelRef",
    "ModelResolutionError",
    "ModelSpecError",
    "NativeThinking",
    "PROVIDER_KEY_ENV",
    "Provider",
    "ProviderKind",
    "ProviderSpec",
    "ProviderState",
    "ProviderStatus",
    "ResolvedModel",
    "RoleMap",
    "RoleRef",
    "ShippedTable",
    "ThinkingIntent",
    "ThinkingMechanism",
    "ThinkingRule",
    "build_providers",
    "load_table",
    "parse_entry_key",
    "parse_ref",
    "parse_spec",
    "resolve_model",
    "validate_provider_name",
    "validate_role_name",
    "version_key",
]
