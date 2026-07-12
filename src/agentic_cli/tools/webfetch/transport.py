"""SSRF-safe HTTP transport: resolve → validate → pin every request."""

from __future__ import annotations

import httpx

from agentic_cli.tools.webfetch.validator import URLValidator, BlockedAddressError


class PinnedTransport(httpx.AsyncBaseTransport):
    """Connect every request to a validated pinned IP.

    Wraps an inner transport (default httpx.AsyncHTTPTransport(); a
    httpx.MockTransport in tests). Resolves the host (all A+AAAA), requires all
    addresses globally routable, and rewrites the request to connect to the
    literal validated IP while preserving the Host header and TLS SNI — so cert
    verification stays bound to the hostname and the validated address IS the
    connected address (no TOCTOU). Sitting at the transport layer, it covers the
    initial fetch, every manually-issued redirect hop, and the robots fetch.
    """

    def __init__(
        self,
        validator: URLValidator,
        inner: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._validator = validator
        self._inner = inner if inner is not None else httpx.AsyncHTTPTransport()

    def _pin(self, request: httpx.Request) -> None:
        host = request.url.host
        port = request.url.port or (443 if request.url.scheme == "https" else 80)
        pinned_ip = self._validator.resolve_and_validate(host, port)  # raises on block
        # The client set the Host header from the original URL before this
        # transport runs; rewriting url.host to the IP does not touch it. Keep
        # Host + TLS SNI bound to the hostname.
        request.url = request.url.copy_with(host=pinned_ip)
        request.headers.setdefault("Host", host if port in (80, 443) else f"{host}:{port}")
        request.extensions["sni_hostname"] = host

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self._pin(request)
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        # No-op: this transport is SHARED and long-lived (one per fetcher, used
        # by both the fetcher and the robots checker). Each webfetch call opens
        # a short-lived httpx.AsyncClient(transport=self) and closes it on exit,
        # which would otherwise tear down the shared inner pool under a
        # concurrent call. The inner pool persists for the fetcher's lifetime
        # (which also enables connection keep-alive across fetches).
        return None
