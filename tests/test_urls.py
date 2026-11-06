"""One canonical form of a URL: what the HTTP client sends.

The permission engine's URL rules and web_fetch's blocked domains compared
the URL as spelled (urllib's hostname, the path as written), while httpx
removes dot segments, maps the IDNA dots U+3002, U+FF0E and U+FF61 to ``.``
and encodes a Unicode host with IDNA 2008, and DNS treats ``host.`` as
``host``. So ``/x/../admin``, ``/%61dmin``, ``docs.example.com.`` and
``docs。example。com`` passed a deny written for ``docs.example.com/admin``.
``agentic_cli.urls.canonical_url`` is the one spelling every check compares.
"""

from __future__ import annotations

import pytest

from agentic_cli.urls import canonical_host, canonical_url


@pytest.mark.parametrize(
    "url, canonical",
    [
        # What httpx already does: dot segments, IDNA dots, case, default port.
        ("https://docs.example.com/x/../admin/k", "https://docs.example.com/admin/k"),
        ("https://docs。example。com/admin/k", "https://docs.example.com/admin/k"),
        ("https://docs．example．com/admin/k", "https://docs.example.com/admin/k"),
        ("https://docs｡example｡com/admin/k", "https://docs.example.com/admin/k"),
        ("HTTPS://DOCS.Example.COM:443/A?q=1#frag", "https://docs.example.com/A?q=1"),
        ("http://example.com:80/x", "http://example.com/x"),
        ("https://example.com:8443/x", "https://example.com:8443/x"),
        ("https://example.com", "https://example.com/"),
        # A Unicode host is the punycode httpx sends (IDNA 2008), not IDNA 2003's.
        ("https://faß.de/", "https://xn--fa-hia.de/"),
        ("https://xn--fa-hia.de/", "https://xn--fa-hia.de/"),
        # What httpx leaves alone: one trailing dot names the same DNS host.
        ("https://docs.example.com./admin/k", "https://docs.example.com/admin/k"),
        # Unreserved escapes decoded, other escapes' hex uppercased.
        ("https://docs.example.com/%61dmin/k", "https://docs.example.com/admin/k"),
        ("https://example.com/a%2fb/%7euser/%7E", "https://example.com/a%2Fb/~user/~"),
        ("https://example.com/%c3%a9", "https://example.com/%C3%A9"),
        ("https://example.com/é", "https://example.com/%C3%A9"),
        # A decoded dot is a dot segment like any other.
        ("https://example.com/x/%2e%2e/admin", "https://example.com/admin"),
        ("https://example.com/a/%2E/b", "https://example.com/a/b"),
        # IPv6 in brackets, written one way.
        ("https://[2001:DB8:0::1]:8443/a", "https://[2001:db8::1]:8443/a"),
        # Credentials and the fragment are not part of what is requested.
        ("https://user:pw@example.com/p#top", "https://example.com/p"),
        # The query is kept as httpx sends it.
        ("https://example.com/p?r=%7e&s=a b", "https://example.com/p?r=%7e&s=a%20b"),
        # An IPv4 address in another notation is the address the resolver
        # connects to (getaddrinfo reads it as inet_aton does).
        ("https://3405803783/a", "https://203.0.113.7/a"),
        ("https://0xcb007107/a", "https://203.0.113.7/a"),
        ("https://203.0.0x71.7/a", "https://203.0.113.7/a"),
        ("https://203.29447/a", "https://203.0.115.7/a"),
        ("https://0x7f.1/", "https://127.0.0.1/"),
        ("https://deadbeef.example/", "https://deadbeef.example/"),  # a name
        # An IPv4-mapped IPv6 address reaches the IPv4 host (a dual-stack
        # socket delivers it there): it is that address too.
        ("https://[::ffff:203.0.113.7]/a", "https://203.0.113.7/a"),
        ("https://[::ffff:cb00:7107]:8443/a", "https://203.0.113.7:8443/a"),
        ("https://[0:0:0:0:0:FFFF:203.0.113.7]/a", "https://203.0.113.7/a"),
        ("https://[64:ff9b::cb00:7107]/a", "https://[64:ff9b::cb00:7107]/a"),  # NAT64: v6
        # A rule's wildcards are left alone.
        ("https://docs.example.com/**", "https://docs.example.com/**"),
        ("https://Docs.Example.com./a/*/b/**?x=*", "https://docs.example.com/a/*/b/**?x=*"),
    ],
)
def test_canonical_url(url, canonical):
    assert canonical_url(url) == canonical
    assert canonical_url(canonical) == canonical  # idempotent


@pytest.mark.parametrize(
    "url",
    [
        "https://i❤.ws/",  # not a valid IDNA 2008 name
        "https://bl­ocked.example/",  # a soft hyphen
        "https://ｂｌｏｃｋｅｄ.example/",  # fullwidth letters
        "https://example.com:99999/",  # no such port
        "https://[notip]/",
        "https:///no-host",
        "not a url",
        "https://exa\tmple.com/",
    ],
)
def test_a_url_the_client_cannot_send_raises_value_error(url):
    with pytest.raises(ValueError):
        canonical_url(url)


@pytest.mark.parametrize(
    "host, canonical",
    [
        ("blocked.example", "blocked.example"),
        ("BLOCKED.example.", "blocked.example"),
        ("blocked。example", "blocked.example"),
        ("blocked．example", "blocked.example"),
        ("faß.de", "xn--fa-hia.de"),
        ("2001:DB8:0::1", "2001:db8::1"),
        ("::ffff:127.0.0.1", "127.0.0.1"),
        ("::FFFF:7f00:1", "127.0.0.1"),
    ],
)
def test_canonical_host(host, canonical):
    assert canonical_host(host) == canonical


@pytest.mark.parametrize("host", ["i❤.ws", "ｂｌｏｃｋｅｄ.example", "", "a/b"])
def test_an_invalid_host_raises_value_error(host):
    with pytest.raises(ValueError):
        canonical_host(host)
