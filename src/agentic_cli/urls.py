"""The single interpretation of a URL: what the HTTP client sends.

A URL is judged by the permission engine (its ``http.*`` rules), by
``web_fetch``'s blocked domains, and then requested: resolved, connected to,
verified over TLS. Each of them read the host its own way (urllib's hostname,
httpx's IDNA 2008 host, Python's IDNA 2003 codec) and the path as spelled, so
a page one of them refused could be named so that another let it through:
``/x/../admin`` and ``/%61dmin`` for ``/admin``, ``host.`` and ``host。com``
(U+3002, which httpx sends as ``.``) for ``host.com``. Each of them now
compares :func:`canonical_url` or :func:`canonical_host`.

The form is httpx's parse of the URL:

* the scheme lowercased; the host as httpx sends it (UTS 46 dots mapped, a
  Unicode host as its IDNA 2008 punycode), lowercased, with one trailing dot
  dropped (``host.`` is the same DNS name); an IPv6 address compressed, in
  brackets; an IPv4 address in another notation (``0x7f.1``, which the
  resolver reads as an address, or the IPv4-mapped ``[::ffff:127.0.0.1]``)
  dotted-decimal; the scheme's default port dropped; no credentials;
* the path with percent-escapes of unreserved characters (``A-Z a-z 0-9 - .
  _ ~``) decoded and the others' hex uppercased, then dot segments removed
  (RFC 3986 §5.2.4; httpx removes the literal ones, so the decoded ``%2e``
  ones are what is left);
* the query as httpx sends it; no fragment.

``*``, ``?`` in a query and ``[`` are left alone, so a rule pattern keeps its
wildcards. httpx is imported inside the functions: the permission engine
loads at startup.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import string

__all__ = ["canonical_host", "canonical_url"]

_DEFAULT_PORTS = {"http": 80, "https": 443}
_UNRESERVED = frozenset(string.ascii_letters + string.digits + "-._~")
_ESCAPE = re.compile(r"%([0-9A-Fa-f]{2})")
#: Characters a host given on its own may not hold: they end it in a URL.
_NOT_IN_HOST = frozenset("/?#@\\")
#: A host the resolver may read as an IPv4 address (``inet_aton``'s grammar:
#: decimal, octal or ``0x`` hex parts).
_NUMERIC_HOST = re.compile(r"[0-9a-fx.]*[0-9][0-9a-fx.]*")


def canonical_url(url: str) -> str:
    """``url`` as the HTTP client sends it (see the module docstring).

    Raises:
        ValueError: if the client cannot send it: httpx cannot parse it (an
            invalid IDNA host, a control character), or it has no scheme, no
            host or a port outside 0-65535.
    """
    import httpx  # noqa: PLC0415  (see the module docstring)

    try:
        parsed = httpx.URL(url)
    except (httpx.InvalidURL, TypeError) as exc:
        raise ValueError(str(exc)) from exc
    scheme = parsed.scheme
    if not scheme:
        raise ValueError("no scheme")
    host = _host(parsed.raw_host)
    if not host:
        raise ValueError("no host")
    port = parsed.port
    if port is not None and not 0 <= port <= 65535:
        raise ValueError(f"no such port: {port}")
    netloc = f"[{host}]" if ":" in host else host
    if port is not None and port != _DEFAULT_PORTS.get(scheme):
        netloc = f"{netloc}:{port}"
    path = _path(parsed.raw_path.decode("ascii").partition("?")[0])
    query = parsed.query.decode("ascii")
    return f"{scheme}://{netloc}{path}" + (f"?{query}" if query else "")


def canonical_host(host: str) -> str:
    """A host name alone (a blocked domain, say) as the HTTP client sends it:
    lowercased, IDNA-encoded, one trailing dot dropped; an IPv6 address
    compressed, without brackets.

    Raises:
        ValueError: if it is empty, holds a character that would end it in a
            URL, or is no host httpx can send (an invalid IDNA name).
    """
    import httpx  # noqa: PLC0415  (see the module docstring)

    if not host or any(ch in _NOT_IN_HOST or ch.isspace() for ch in host):
        raise ValueError(f"invalid host {host!r}")
    try:
        parsed = httpx.URL(scheme="https", host=host, path="/")
    except (httpx.InvalidURL, TypeError) as exc:
        raise ValueError(f"invalid host {host!r}: {exc}") from exc
    canonical = _host(parsed.raw_host)
    if not canonical:
        raise ValueError(f"invalid host {host!r}")
    return canonical


def _host(raw_host: bytes) -> str:
    """httpx's ``raw_host`` lowercased, without one trailing dot; an IPv6
    address compressed; an IPv4 address in another notation, which the
    connection reaches as that address, dotted-decimal: ``3405803783``,
    ``0xcb.0.113.7``, ``203.29447`` (httpx sends them as written, getaddrinfo
    reads them as inet_aton does) and the IPv4-mapped ``::ffff:203.0.113.7``
    (a dual-stack socket delivers it to the IPv4 host)."""
    host = raw_host.decode("ascii").lower().removesuffix(".")
    if ":" in host:
        try:
            address = ipaddress.IPv6Address(host)
        except ValueError:
            pass  # a zone id httpx accepts (``fe80::1%25en0``): kept as sent
        else:
            mapped = address.ipv4_mapped
            host = str(mapped) if mapped is not None else address.compressed
    elif _NUMERIC_HOST.fullmatch(host):
        try:
            host = socket.inet_ntoa(socket.inet_aton(host))
        except OSError:
            pass  # a name after all (``deadbeef.example``)
    return host


def _path(path: str) -> str:
    """Unreserved escapes decoded, the others' hex uppercased, then dot
    segments removed."""
    path = _ESCAPE.sub(_escape, path)
    return _remove_dot_segments(path) or "/"


def _escape(match: re.Match[str]) -> str:
    char = chr(int(match.group(1), 16))
    return char if char in _UNRESERVED else f"%{match.group(1).upper()}"


def _remove_dot_segments(path: str) -> str:
    """RFC 3986 §5.2.4, for an absolute path (httpx always gives one)."""
    output: list[str] = []
    rest = path
    while rest:
        if rest.startswith("../"):
            rest = rest[3:]
        elif rest.startswith("./"):
            rest = rest[2:]
        elif rest.startswith("/./"):
            rest = rest[2:]
        elif rest == "/.":
            rest = "/"
        elif rest.startswith("/../"):
            rest = rest[3:]
            if output:
                output.pop()
        elif rest == "/..":
            rest = "/"
            if output:
                output.pop()
        elif rest in (".", ".."):
            rest = ""
        else:
            end = rest.find("/", 1)
            segment = rest if end == -1 else rest[:end]
            output.append(segment)
            rest = rest[len(segment):]
    return "".join(output)
