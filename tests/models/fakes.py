"""In-memory stand-ins for the models tests: a catalog view and endpoints."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from agentic_cli.models import (
    Endpoint,
    EndpointSpec,
    ModelRecord,
    ModelRef,
    ModelResolutionError,
    ProviderKind,
    build_endpoints,
    load_table,
)


def endpoints(
    *, specs: Mapping[str, EndpointSpec] | None = None, env: Mapping[str, str] | None = None, **keys: str
) -> dict[str, Endpoint]:
    """The built-in providers with the given keys (``google="k"``), plus ``specs``."""
    all_keys = {"google": None, "anthropic": None, "openai": None, "openrouter": None, **keys}
    return build_endpoints(keys=all_keys, specs=specs or {}, env=env or {})


def table_records(kind: ProviderKind, *models: str, provider: str | None = None) -> list[ModelRecord]:
    """The shipped table's records for ``models``."""
    table = load_table()
    return [table.record_for(provider or kind.value, kind, model) for model in models]


class FakeCatalog:
    """A ``CatalogView`` over fixed records.

    An id it does not hold is rejected when its provider is ``authoritative``
    (as after a fresh listing), and otherwise gets the shipped table's record.
    """

    def __init__(
        self,
        records: Iterable[ModelRecord] = (),
        *,
        authoritative: Iterable[str] = (),
        kinds: Mapping[str, ProviderKind] | None = None,
    ) -> None:
        self._records = {record.ref: record for record in records}
        self._authoritative = set(authoritative)
        self._kinds = dict(kinds or {})

    def record(self, ref: ModelRef) -> ModelRecord:
        if ref in self._records:
            return self._records[ref]
        if ref.provider in self._authoritative:
            available = ", ".join(r.model for r in self.records(ref.provider)) or "none"
            raise ModelResolutionError(
                f"Model {ref} is not available. Available {ref.provider} models: {available}"
            )
        kind = self._kinds.get(ref.provider) or _builtin_kind(ref.provider)
        return load_table().record_for(ref.provider, kind, ref.model)

    def records(self, provider: str) -> Sequence[ModelRecord]:
        return [record for record in self._records.values() if record.provider == provider]


def _builtin_kind(provider: str) -> ProviderKind:
    try:
        return ProviderKind(provider)
    except ValueError:
        return ProviderKind.OPENAI_COMPATIBLE
