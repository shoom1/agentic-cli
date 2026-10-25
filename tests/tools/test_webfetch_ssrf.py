"""P0-5 SSRF tests: resolve-validate-pin. Fully offline — socket.getaddrinfo is
monkeypatched and httpx.MockTransport is the PinnedTransport inner."""

import socket

import httpx
import pytest

from agentic_cli.tools.webfetch.transport import PinnedTransport
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

    def test_public_hostname_passes_policy(self, monkeypatch):
        # validate() must NOT resolve DNS — make resolution explode and assert
        # a normal hostname still passes policy.
        def _boom(*a, **k):
            raise AssertionError("validate() must not call DNS")
        monkeypatch.setattr(socket, "getaddrinfo", _boom)
        monkeypatch.setattr(socket, "gethostbyname", _boom)
        assert URLValidator().validate("https://example.com/x").valid is True


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


class _SpyInner(httpx.MockTransport):
    def __init__(self, handler):
        super().__init__(handler)
        self.closed = False

    async def aclose(self):
        self.closed = True
        await super().aclose()


class TestSharedTransportLifecycle:
    @pytest.mark.asyncio
    async def test_client_close_does_not_close_shared_inner(self, monkeypatch):
        monkeypatch.setattr(socket, "getaddrinfo", stub_getaddrinfo("93.184.216.34"))
        inner = _SpyInner(lambda req: httpx.Response(200, text="ok"))
        transport = PinnedTransport(URLValidator(), inner=inner)
        async with httpx.AsyncClient(transport=transport) as client:
            await client.get("https://example.com/x")
        assert inner.closed is False   # shared inner NOT closed by client exit
        # ...and the transport is still usable for a subsequent client
        async with httpx.AsyncClient(transport=transport) as client2:
            r = await client2.get("https://example.com/y")
        assert r.status_code == 200


class _Stream:
    """One fake TCP connection: records its TLS name and the Host of each request."""

    def __init__(self, n: int, log: list[tuple]):
        self.n, self.log, self.pending = n, log, 0

    async def read(self, max_bytes, timeout=None):
        import asyncio

        await asyncio.sleep(0)  # let parallel requests interleave
        if self.pending:
            self.pending -= 1
            return b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"
        return b""

    async def write(self, buffer, timeout=None):
        hosts = [line for line in buffer.split(b"\r\n") if line.lower().startswith(b"host:")]
        if hosts:  # h11 writes the end of the message separately
            self.log.append(("request", self.n, hosts[0].split(b":", 1)[1].strip().decode()))
            self.pending += 1

    async def aclose(self):
        pass

    async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        self.log.append(("tls", self.n, server_hostname))
        return self

    def get_extra_info(self, info):
        if info == "ssl_object":
            class _SSL:
                def selected_alpn_protocol(self):
                    return "http/1.1"
            return _SSL()
        return None


class _Network:
    """httpcore's network backend, faked: every connect is a new ``_Stream``."""

    def __init__(self):
        self.log: list[tuple] = []
        self.count = 0

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        self.count += 1
        return _Stream(self.count, self.log)

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise NotImplementedError

    async def sleep(self, seconds):
        pass

    def requests_by_tls_name(self) -> list[tuple[str, str]]:
        """(the name a connection's TLS was verified for, the Host it carried), per request."""
        tls = {n: name for kind, n, name in self.log if kind == "tls"}
        return [(tls[n], host) for kind, n, host in self.log if kind == "request"]


class TestOneConnectionPerRequest:
    """The pool is keyed by the pinned IP, so a connection kept alive for one
    hostname would carry a request for another hostname on that IP without its
    certificate ever being checked. Each request gets its own connection."""

    @pytest.fixture
    def network(self, monkeypatch):
        # Both names resolve to one public address (shared hosting, a CDN).
        monkeypatch.setattr(socket, "getaddrinfo", stub_getaddrinfo("93.184.216.34"))
        return _Network()

    def _transport(self, network: _Network) -> PinnedTransport:
        transport = PinnedTransport(URLValidator())  # its own inner, as webfetch builds it
        transport._inner._pool._network_backend = network  # only the network is faked
        return transport

    @pytest.mark.asyncio
    async def test_two_hosts_on_one_ip_each_get_their_own_tls_handshake(self, network):
        transport = self._transport(network)
        for url in ("https://first.example/", "https://second.example/"):
            async with httpx.AsyncClient(transport=transport) as client:  # as each fetch does
                assert (await client.get(url)).text == "ok"

        assert network.requests_by_tls_name() == [
            ("first.example", "first.example"),
            ("second.example", "second.example"),
        ]

    @pytest.mark.asyncio
    async def test_parallel_requests_each_go_over_a_connection_verified_for_their_host(
        self, network
    ):
        import asyncio

        transport = self._transport(network)
        urls = ["https://first.example/", "https://second.example/"] * 3
        async with httpx.AsyncClient(transport=transport) as client:
            for _ in range(2):
                await asyncio.gather(*(client.get(url) for url in urls))

        pairs = network.requests_by_tls_name()
        assert len(pairs) == 12
        assert all(tls_name == host for tls_name, host in pairs), pairs
        assert network.count == 12  # one connection per request

    def test_http2_stays_off(self):
        """An HTTP/2 connection carries many requests at once, whatever the
        keep-alive limit: one connection per request needs HTTP/1.1."""
        assert PinnedTransport(URLValidator())._inner._pool._http2 is False
