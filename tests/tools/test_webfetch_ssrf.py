"""P0-5 SSRF tests: resolve-validate-pin. Fully offline — socket.getaddrinfo is
monkeypatched and httpx.MockTransport is the PinnedTransport inner."""

import socket

import httpx
import pytest

from agentic_cli.tools.webfetch.validator import (
    URLValidator,
    BlockedAddressError,
    _ip_is_safe,
    _as_ip_literal,
)


def stub_getaddrinfo(*ips):
    """Return a socket.getaddrinfo replacement yielding the given IP strings."""
    def _stub(host, port, *args, **kwargs):
        out = []
        for ip in ips:
            is6 = ":" in ip
            fam = socket.AF_INET6 if is6 else socket.AF_INET
            sa = (ip, port, 0, 0) if is6 else (ip, port)
            out.append((fam, socket.SOCK_STREAM, 6, "", sa))
        return out
    return _stub


import ipaddress


class TestIpIsSafe:
    def test_public_v4_and_v6_safe(self):
        assert _ip_is_safe(ipaddress.ip_address("8.8.8.8"))
        assert _ip_is_safe(ipaddress.ip_address("2606:4700:4700::1111"))

    @pytest.mark.parametrize("ip", [
        "127.0.0.1", "10.0.0.1", "192.168.1.1", "172.16.0.1", "169.254.169.254",
        "100.64.0.1", "192.0.2.1", "198.18.0.1", "240.0.0.1", "0.0.0.1",
        "::1", "fc00::1", "fe80::1",
        "::ffff:10.0.0.1",           # v4-mapped private
        "64:ff9b::a00:1",            # NAT64 embedding 10.0.0.1 (is_global=True)
        "192.88.99.1",               # 6to4 relay anycast (deprecated)
    ])
    def test_unsafe_addresses(self, ip):
        assert not _ip_is_safe(ipaddress.ip_address(ip))


class TestResolveAndValidate:
    def test_public_host_returns_ip(self, monkeypatch):
        monkeypatch.setattr(socket, "getaddrinfo", stub_getaddrinfo("93.184.216.34"))
        assert URLValidator().resolve_and_validate("example.com", 443) == "93.184.216.34"

    def test_private_host_rejected(self, monkeypatch):
        monkeypatch.setattr(socket, "getaddrinfo", stub_getaddrinfo("127.0.0.1"))
        with pytest.raises(BlockedAddressError):
            URLValidator().resolve_and_validate("localhost", 80)

    def test_mixed_public_and_private_rejected(self, monkeypatch):
        # split-horizon: a public + a private answer → reject the whole host
        monkeypatch.setattr(socket, "getaddrinfo", stub_getaddrinfo("93.184.216.34", "10.0.0.1"))
        with pytest.raises(BlockedAddressError):
            URLValidator().resolve_and_validate("rebind.test", 443)

    def test_ipv6_ula_rejected(self, monkeypatch):
        monkeypatch.setattr(socket, "getaddrinfo", stub_getaddrinfo("fc00::1"))
        with pytest.raises(BlockedAddressError):
            URLValidator().resolve_and_validate("v6.test", 443)

    def test_unresolvable_host_raises_blocked(self, monkeypatch):
        def boom(*a, **k):
            raise socket.gaierror("nope")
        monkeypatch.setattr(socket, "getaddrinfo", boom)
        with pytest.raises(BlockedAddressError):
            URLValidator().resolve_and_validate("nx.test", 443)

    def test_ip_literal_public_ok_private_blocked(self):
        assert URLValidator().resolve_and_validate("8.8.8.8", 443) == "8.8.8.8"
        with pytest.raises(BlockedAddressError):
            URLValidator().resolve_and_validate("169.254.169.254", 80)


class TestValidateNoDNS:
    def test_ip_literal_private_blocked_without_dns(self):
        # validate() must reject a private IP-literal with no DNS call
        r = URLValidator().validate("http://127.0.0.1/x")
        assert r.valid is False

    def test_public_hostname_passes_policy(self):
        # validate() no longer resolves; a normal hostname passes policy checks
        r = URLValidator().validate("https://example.com/x")
        assert r.valid is True


def mock_pinned_transport(handler, validator=None):
    """PinnedTransport whose inner is an httpx.MockTransport(handler)."""
    from agentic_cli.tools.webfetch.transport import PinnedTransport
    return PinnedTransport(validator or URLValidator(), inner=httpx.MockTransport(handler))


class TestPinnedTransport:
    @pytest.mark.asyncio
    async def test_pins_to_validated_ip_preserving_host_and_sni(self, monkeypatch):
        monkeypatch.setattr(socket, "getaddrinfo", stub_getaddrinfo("93.184.216.34"))
        seen = {}

        def handler(request):
            seen["url_host"] = request.url.host
            seen["host_header"] = request.headers.get("Host")
            seen["sni"] = request.extensions.get("sni_hostname")
            return httpx.Response(200, text="ok")

        transport = mock_pinned_transport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await client.get("https://example.com/page")

        assert resp.status_code == 200
        assert seen["url_host"] == "93.184.216.34"   # connected to the pinned IP
        assert seen["host_header"] == "example.com"  # Host preserved
        assert seen["sni"] == "example.com"          # TLS SNI bound to hostname

    @pytest.mark.asyncio
    async def test_blocked_host_raises_and_never_calls_inner(self, monkeypatch):
        monkeypatch.setattr(socket, "getaddrinfo", stub_getaddrinfo("169.254.169.254"))
        called = {"inner": False}

        def handler(request):
            called["inner"] = True
            return httpx.Response(200)

        transport = mock_pinned_transport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(BlockedAddressError):
                await client.get("http://metadata.test/latest")
        assert called["inner"] is False  # never connected


class TestRobotsThroughTransport:
    @pytest.mark.asyncio
    async def test_robots_fetch_for_private_host_is_refused_at_transport(self, monkeypatch):
        """A host resolving to a private IP: the robots fetch is refused by the
        transport (no connection), and can_fetch stays permissive."""
        from agentic_cli.tools.webfetch.robots import RobotsTxtChecker
        monkeypatch.setattr(socket, "getaddrinfo", stub_getaddrinfo("169.254.169.254"))
        inner_called = {"n": 0}
        def handler(req):
            inner_called["n"] += 1
            return httpx.Response(200, text="User-agent: *\nDisallow: /\n")
        checker = RobotsTxtChecker(transport=mock_pinned_transport(handler))
        allowed = await checker.can_fetch("http://metadata.test/x")
        assert inner_called["n"] == 0     # never connected to the private IP
        assert allowed is True            # permissive on the (blocked) fetch error
