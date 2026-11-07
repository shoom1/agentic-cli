"""URL validation and SSRF protection."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from agentic_cli.urls import canonical_host, canonical_url


class BlockedAddressError(httpx.RequestError):
    """A host resolves to a non-global / blocked address (or cannot be
    resolved). Subclasses httpx.RequestError so the fetcher's existing
    ``except httpx.RequestError`` maps it to a FetchResult error."""


@dataclass
class ValidationResult:
    """Result of URL validation."""

    valid: bool
    error: str | None = None


# Ranges ipaddress.is_global marks global but that are SSRF-risky.
_SUPPLEMENTARY_BLOCKED = [
    ipaddress.ip_network("64:ff9b::/96"),    # NAT64 well-known prefix (embeds v4)
    ipaddress.ip_network("64:ff9b:1::/48"),  # NAT64 local-use prefix
    ipaddress.ip_network("192.88.99.0/24"),  # 6to4 relay anycast (deprecated)
]


def _ip_is_safe(ip_obj: ipaddress._BaseAddress) -> bool:
    """True only if the address is globally routable and not SSRF-risky."""
    if not ip_obj.is_global:
        return False
    for net in _SUPPLEMENTARY_BLOCKED:
        if ip_obj in net:
            return False
    mapped = getattr(ip_obj, "ipv4_mapped", None)
    if mapped is not None and not _ip_is_safe(mapped):
        return False
    return True


def _as_ip_literal(host: str) -> ipaddress._BaseAddress | None:
    """Return the IP if host is an IP literal (brackets stripped), else None."""
    try:
        return ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return None


class URLValidator:
    """Validates URLs for SSRF protection and policy compliance.

    ``validate`` performs non-DNS policy (scheme, blocked domains, IP-literal
    safety). Authoritative resolution + pinning happens in
    ``resolve_and_validate`` (awaited by PinnedTransport), which resolves every
    A+AAAA and requires all of them to be globally routable.
    """

    ALLOWED_SCHEMES = {"http", "https"}

    def __init__(self, blocked_domains: list[str] | None = None) -> None:
        self.blocked_domains = blocked_domains or []

    def validate(self, url: str) -> ValidationResult:
        """Scheme + blocked-domain + hostname-present + IP-literal safety.

        The URL is read as the HTTP client sends it (``canonical_url``): the
        blocked domains are compared with the host httpx resolves, so
        ``blocked.example.`` and ``blocked。example`` (U+3002, which httpx
        sends as ``.``) are ``blocked.example``. A URL httpx cannot send (an
        invalid IDNA host) is an error here, not an exception out of the fetch.

        Does NOT resolve DNS — hostname resolution is done (once) by the
        transport's resolve_and_validate, which pins the connection.
        """
        try:
            parsed = urlsplit(canonical_url(url))
        except ValueError as e:
            return ValidationResult(valid=False, error=f"Invalid URL: {e}")

        if parsed.scheme not in self.ALLOWED_SCHEMES:
            return ValidationResult(
                valid=False,
                error=f"Scheme '{parsed.scheme}' not allowed. Use http or https.",
            )

        hostname = parsed.hostname
        if not hostname:
            return ValidationResult(valid=False, error="URL must have a hostname")

        if self._is_domain_blocked(hostname):
            return ValidationResult(valid=False, error=f"Domain '{hostname}' is blocked by policy")

        literal = _as_ip_literal(hostname)
        if literal is not None and not _ip_is_safe(literal):
            return ValidationResult(
                valid=False,
                error=f"Private/internal IP address blocked: {literal}",
            )

        return ValidationResult(valid=True)

    async def resolve_and_validate(self, host: str, port: int) -> str:
        """Resolve every A+AAAA for host and return one validated pinned IP.

        Rejects the whole host (BlockedAddressError) if ANY resolved address is
        unsafe, so a split-horizon / rebinding resolver cannot smuggle a private
        address alongside a public one. An IP-literal host is validated directly.

        The lookup runs in a worker thread (the loop's ``getaddrinfo``), as
        httpx's own transport resolves: on the loop's thread a slow resolver
        froze every other task, the UI and Ctrl+C with it, for as long as each
        lookup took.
        """
        literal = _as_ip_literal(host)
        if literal is not None:
            if not _ip_is_safe(literal):
                raise BlockedAddressError(f"blocked non-global address: {host}")
            return str(literal)

        try:
            infos = await asyncio.get_running_loop().getaddrinfo(
                host, port, type=socket.SOCK_STREAM
            )
        except socket.gaierror as exc:
            raise BlockedAddressError(f"could not resolve {host}: {exc}") from exc

        # sockaddr[0] is the IP; strip any IPv6 zone id ("fe80::1%eth0").
        ips = [ipaddress.ip_address(info[4][0].split("%")[0]) for info in infos]
        if not ips:
            raise BlockedAddressError(f"no addresses for {host}")
        for ip_obj in ips:
            if not _ip_is_safe(ip_obj):
                raise BlockedAddressError(f"blocked non-global address {ip_obj} for {host}")
        return str(ips[0])

    def _is_domain_blocked(self, hostname: str) -> bool:
        """Whether a canonical ``hostname`` is a blocked domain or, for a
        ``*.domain`` entry, below it. Each entry is read as a host the client
        sends (``canonical_host``), so one written with a Unicode host, its
        dots or a trailing dot blocks the domain it names."""
        for pattern in self.blocked_domains:
            pattern = pattern.strip()
            wildcard = pattern.startswith("*.")
            domain = _blocked_domain(pattern[2:] if wildcard else pattern)
            if hostname.endswith(f".{domain}") if wildcard else hostname == domain:
                return True
        return False


def _blocked_domain(domain: str) -> str:
    """A blocked-domains entry as the host the client sends; one it cannot
    send (no URL's host can equal it) is compared as written, lowercased."""
    try:
        return canonical_host(domain)
    except ValueError:
        return domain.lower().removesuffix(".")
