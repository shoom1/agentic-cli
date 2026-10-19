"""Model listings: what each provider says it serves.

One lister per provider kind, each over the provider's REST API with httpx,
never a provider SDK (only the backend adapters touch those):

- ``google``: ``GET /v1beta/models`` (``x-goog-api-key``), the text models
  0.6 kept;
- ``anthropic``: ``GET /v1/models`` (``x-api-key``), with each model's
  thinking types and effort levels, and whether thinking can be turned off;
  dated ids become their alias;
- ``openai``: ``GET /v1/models`` (bearer key), ids only;
- ``openrouter``: the public ``GET /api/v1/models`` (the key is sent when
  there is one), with tool and reasoning support;
- ``openai_compatible``: ``GET {base_url}/models`` on a user endpoint, the
  served names and ``max_model_len``.

``list_models`` never raises for an HTTP or network failure: it returns a
failed ``Listing`` whose ``error`` is a short reason that never contains the
key. A listing is bounded by one deadline across all its pages, so a hung
server costs at most ``timeout`` seconds.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from agentic_cli.models.endpoints import Endpoint, ProviderKind

logger = logging.getLogger(__name__)

#: Seconds one provider's listing may take, all pages together.
DEFAULT_TIMEOUT = 10.0
#: Pages followed before a listing that keeps paginating is given up on.
MAX_PAGES = 20


@dataclass(frozen=True)
class ListedModel:
    """One model as a provider's listing describes it; None means not said."""

    id: str
    display_name: str = ""
    context_window: int | None = None
    max_output: int | None = None
    image_input: bool | None = None
    tools: bool | None = None
    #: Whether the model thinks: OpenRouter's ``reasoning`` parameter,
    #: Google's ``thinking`` flag, Anthropic's thinking capability.
    reasoning: bool | None = None
    #: Anthropic: the supported thinking types, of ``adaptive`` and ``enabled``.
    thinking_types: tuple[str, ...] = ()
    #: Anthropic: the supported effort levels, lowest first.
    effort_levels: tuple[str, ...] = ()
    #: Anthropic: whether thinking can be turned off (the ``disabled`` thinking
    #: type); None when the listing does not say.
    can_disable: bool | None = None


@dataclass(frozen=True)
class Listing:
    """The outcome of listing one provider's models.

    Attributes:
        provider: The provider or endpoint name.
        ok: Whether the provider answered with a model list.
        models: The models listed (empty when ``ok`` is False).
        fetched_at: When the listing finished (``clock()`` seconds).
        error: Why it failed: a short reason, never the key.
    """

    provider: str
    ok: bool
    models: tuple[ListedModel, ...]
    fetched_at: float
    error: str | None = None


class ListingError(Exception):
    """A response a lister cannot use; the message is a short, key-free reason."""


async def list_models(
    endpoint: Endpoint,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    clock: Callable[[], float] = time.time,
) -> Listing:
    """List the models ``endpoint`` serves.

    Args:
        endpoint: The provider or user endpoint to ask.
        transport: The HTTP transport (a fake one in tests). The caller keeps
            ownership: it is not closed here, so one transport can serve
            several listings at once.
        timeout: Seconds for the whole listing, every page included.
        clock: Gives ``fetched_at``.

    Returns:
        The listing; a failed one for any HTTP, network or response-shape
        problem, never an exception.
    """
    fetch = _fetcher(endpoint.kind)
    try:
        async with asyncio.timeout(timeout):
            async with httpx.AsyncClient(
                transport=_Borrowed(transport) if transport is not None else None,
                timeout=timeout,
            ) as client:
                models = await fetch(client, endpoint)
    except (TimeoutError, httpx.TimeoutException):
        error = f"timed out after {timeout:g}s"
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        error = f"HTTP {status} {exc.response.reason_phrase}".rstrip()
    except httpx.TransportError as exc:
        error = f"connection failed ({type(exc).__name__})"
    except ListingError as exc:
        error = str(exc)
    except Exception as exc:  # noqa: BLE001 - a listing never fails startup
        logger.debug("Listing %s raised", endpoint.name, exc_info=True)
        error = f"unexpected response ({type(exc).__name__})"
    else:
        return Listing(provider=endpoint.name, ok=True, models=tuple(models), fetched_at=clock())

    error = _redact(error, endpoint.api_key)
    logger.debug("Listing %s failed: %s", endpoint.name, error)
    return Listing(provider=endpoint.name, ok=False, models=(), fetched_at=clock(), error=error)


# --- helpers for the listers ---------------------------------------------------


async def get_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: Mapping[str, str],
    params: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """GET ``url`` and return its JSON object.

    Raises:
        httpx.HTTPStatusError: For a non-2xx status.
        ListingError: If the body is not a JSON object.
    """
    response = await client.get(url, headers=dict(headers), params=params)
    response.raise_for_status()
    try:
        document = response.json()
    except ValueError:
        raise ListingError("invalid JSON in the response") from None
    if not isinstance(document, Mapping):
        raise ListingError("unexpected response: not a JSON object")
    return document


def entries(document: Mapping[str, Any], key: str) -> Iterator[Mapping[str, Any]]:
    """Each object in the list ``document[key]``.

    An entry that is not an object is skipped; the list itself must exist.

    Raises:
        ListingError: If ``document[key]`` is not a list.
    """
    items = document.get(key)
    if not isinstance(items, list):
        raise ListingError(f"unexpected response: {key!r} is not a list")
    for item in items:
        if isinstance(item, Mapping):
            yield item


def text(value: Any) -> str | None:
    """A non-empty string, else None."""
    return value if isinstance(value, str) and value else None


def count(value: Any) -> int | None:
    """A positive integer (a token count), else None."""
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def child(value: Any, key: str) -> Mapping[str, Any]:
    """``value[key]`` when both are objects, else an empty mapping."""
    if isinstance(value, Mapping):
        found = value.get(key)
        if isinstance(found, Mapping):
            return found
    return {}


async def follow_pages(
    fetch_page: Callable[[str | None], Awaitable[str | None]],
) -> None:
    """Call ``fetch_page(cursor)`` from ``None`` until it returns no next cursor.

    Raises:
        ListingError: If a cursor repeats or the pages exceed ``MAX_PAGES``.
    """
    cursor: str | None = None
    seen: set[str] = set()
    for _ in range(MAX_PAGES):
        cursor = await fetch_page(cursor)
        if not cursor:
            return
        if cursor in seen:
            break
        seen.add(cursor)
    raise ListingError("unexpected response: pagination did not end")


# --- internals ---------------------------------------------------------------


class _Borrowed(httpx.AsyncBaseTransport):
    """A caller's transport, used without being closed: the caller owns it."""

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        return None


def _fetcher(
    kind: ProviderKind,
) -> Callable[[httpx.AsyncClient, Endpoint], Awaitable[list[ListedModel]]]:
    # Imported here: the listers import this module's types and helpers.
    from agentic_cli.models.discovery import (
        anthropic,
        google,
        openai,
        openai_compatible,
        openrouter,
    )

    return {
        ProviderKind.GOOGLE: google.fetch,
        ProviderKind.ANTHROPIC: anthropic.fetch,
        ProviderKind.OPENAI: openai.fetch,
        ProviderKind.OPENROUTER: openrouter.fetch,
        ProviderKind.OPENAI_COMPATIBLE: openai_compatible.fetch,
    }[kind]


def _redact(message: str, secret: str | None) -> str:
    """``message`` without ``secret``: belt and braces, as no reason quotes a response."""
    if secret and secret in message:
        return message.replace(secret, "***")
    return message


__all__ = ["DEFAULT_TIMEOUT", "ListedModel", "Listing", "ListingError", "list_models"]
