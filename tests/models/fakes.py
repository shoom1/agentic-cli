"""Stand-ins for the models tests: a catalog view, endpoints and the provider APIs."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx

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


# --- the provider APIs -------------------------------------------------------

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "model_listings"

_HOSTS = {
    "generativelanguage.googleapis.com": "google",
    "api.anthropic.com": "anthropic",
    "api.openai.com": "openai",
    "openrouter.ai": "openrouter",
}


def recorded(name: str) -> dict[str, Any]:
    """A recorded listing payload: ``google``, ``anthropic``, ``openai``,
    ``openrouter`` or ``vllm``."""
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


async def hang(request: httpx.Request) -> httpx.Response:
    """An answer that never comes."""
    await asyncio.sleep(3600)
    raise AssertionError("unreachable")


def refuse(request: httpx.Request) -> httpx.Response:
    """The network is down."""
    raise httpx.ConnectError("connection refused", request=request)


class ProviderAPI:
    """Every provider's model-list API behind one fake transport.

    Each provider answers with its recorded payload unless ``answers``
    overrides it: a payload, an HTTP status, or a handler (sync or async)
    taking the request. Requests to any other host go to ``endpoint``
    (``vllm.json``). ``calls`` records which provider each request reached.
    """

    def __init__(self, **answers: Any) -> None:
        self.answers: dict[str, Any] = {
            "google": recorded("google"),
            "anthropic": recorded("anthropic"),
            "openai": recorded("openai"),
            "openrouter": recorded("openrouter"),
            "endpoint": recorded("vllm"),
            **answers,
        }
        self.calls: list[str] = []

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        provider = _HOSTS.get(request.url.host, "endpoint")
        self.calls.append(provider)
        answer = self.answers[provider]
        if callable(answer):
            answer = answer(request)
            if inspect.isawaitable(answer):
                answer = await answer
        if isinstance(answer, httpx.Response):
            return answer
        if isinstance(answer, int):
            return httpx.Response(answer, json={"error": "fake"})
        return httpx.Response(200, json=answer)


class Clock:
    """A clock tests move by hand."""

    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now
