"""The fetcher keeps a page's whole body; the summarizer sees what it saw before.

The fetcher stopped reading text at ``webfetch_max_content_bytes`` (100 KB), so
a saved page would be cut off: the Wikipedia article on Okapi BM25 is 144 KB.
It now reads text up to ``webfetch_max_download_bytes`` (5 MB) and keeps the
body as received in ``FetchResult.raw``. ``content``, which the summarizer
gets, is still the first ``webfetch_max_content_bytes`` decoded.
"""

from __future__ import annotations

import socket
from unittest.mock import AsyncMock

import httpx
import pytest

URL = "https://example.com/page"


def _fetcher(monkeypatch, handler, **limits):
    from agentic_cli.tools.webfetch.fetcher import ContentFetcher
    from agentic_cli.tools.webfetch.robots import RobotsTxtChecker
    from agentic_cli.tools.webfetch.transport import PinnedTransport
    from agentic_cli.tools.webfetch.validator import URLValidator

    def _gai(host, port, *a, **k):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

    monkeypatch.setattr(socket, "getaddrinfo", _gai)
    validator = URLValidator()
    transport = PinnedTransport(validator, inner=httpx.MockTransport(handler))
    fetcher = ContentFetcher(
        validator=validator, robots_checker=RobotsTxtChecker(), transport=transport, **limits,
    )
    fetcher._robots.can_fetch = AsyncMock(return_value=True)
    return fetcher


def _html(body: bytes, content_type: str = "text/html"):
    return lambda req: httpx.Response(200, content=body, headers={"content-type": content_type})


async def test_a_page_over_the_summary_limit_is_kept_whole(monkeypatch):
    body = b"x" * 3000
    fetcher = _fetcher(monkeypatch, _html(body), max_content_bytes=1000, max_download_bytes=10_000)
    result = await fetcher.fetch(URL)

    assert result.raw == body
    assert result.raw_truncated is False
    assert result.content == "x" * 1000 + "\n\n[Content truncated at 1000 bytes]"
    assert result.truncated is True


async def test_a_small_page_is_unchanged(monkeypatch):
    body = b"<p>small page</p>"
    result = await _fetcher(monkeypatch, _html(body)).fetch(URL)

    assert result.content == "<p>small page</p>"
    assert result.truncated is False
    assert result.raw == body


async def test_a_body_over_the_download_limit_is_cut_and_flagged(monkeypatch):
    fetcher = _fetcher(monkeypatch, _html(b"y" * 5000), max_content_bytes=1000, max_download_bytes=2000)
    result = await fetcher.fetch(URL)

    assert len(result.raw) == 2000
    assert result.raw_truncated is True
    assert result.truncated is True


async def test_a_download_limit_below_the_summary_limit_uses_the_larger(monkeypatch):
    fetcher = _fetcher(monkeypatch, _html(b"z" * 800), max_content_bytes=1000, max_download_bytes=500)
    result = await fetcher.fetch(URL)

    assert result.raw == b"z" * 800
    assert result.raw_truncated is False


async def test_charset_and_final_url_are_reported(monkeypatch):
    def handler(req):
        if req.url.path == "/page":
            return httpx.Response(302, headers={"location": "https://example.com/final"})
        return httpx.Response(
            200, content="café".encode("latin-1"),
            headers={"content-type": "text/html; charset=iso-8859-1"},
        )

    result = await _fetcher(monkeypatch, handler).fetch(URL)

    assert result.final_url == "https://example.com/final"
    assert result.charset == "iso-8859-1"
    assert result.raw == "café".encode("latin-1")
    assert result.content == "café"


async def test_a_cache_hit_returns_the_same_body(monkeypatch):
    body = b"w" * 3000
    fetcher = _fetcher(monkeypatch, _html(body), max_content_bytes=1000)
    first = await fetcher.fetch(URL)
    second = await fetcher.fetch(URL)

    assert second.from_cache is True
    assert second.raw == first.raw == body
    assert second.content == first.content
    assert second.truncated is True
    assert second.final_url == URL


async def test_a_pdf_is_still_returned_as_bytes(monkeypatch):
    body = b"%PDF-1.4 fake"
    result = await _fetcher(monkeypatch, _html(body, "application/pdf")).fetch(URL)

    assert result.content == body
    assert result.raw == body


def test_the_fetcher_gets_the_download_limit_from_settings(monkeypatch):
    from agentic_cli.config import BaseSettings
    from agentic_cli.tools import webfetch_tool

    monkeypatch.setattr(webfetch_tool, "_fetcher", None)
    monkeypatch.setattr(webfetch_tool, "_fetcher_settings_snapshot", None)
    first = webfetch_tool.get_or_create_fetcher(BaseSettings(webfetch_max_download_bytes=200_000))
    assert first._max_download_bytes == 200_000

    second = webfetch_tool.get_or_create_fetcher(BaseSettings(webfetch_max_download_bytes=300_000))
    assert second is not first
    assert second._max_download_bytes == 300_000


def test_the_download_limit_setting():
    from agentic_cli.config import BaseSettings
    from agentic_cli.settings_persistence import PROJECT_SETTABLE_KEYS

    assert BaseSettings().webfetch_max_download_bytes == 5242880
    assert "webfetch_max_download_bytes" in PROJECT_SETTABLE_KEYS


def test_an_out_of_range_download_limit_is_rejected():
    from pydantic import ValidationError

    from agentic_cli.config import BaseSettings

    with pytest.raises(ValidationError):
        BaseSettings(webfetch_max_download_bytes=0)
    with pytest.raises(ValidationError):
        BaseSettings(webfetch_max_download_bytes=104857600 + 1)


# -- I-2(a): only types that are saved get the generous download cap. --


async def test_a_type_that_is_not_saved_is_capped_at_the_content_limit(monkeypatch):
    body = b"\x89PNG" + b"z" * 5000
    fetcher = _fetcher(
        monkeypatch, _html(body, "image/png"), max_content_bytes=1000, max_download_bytes=10_000,
    )
    result = await fetcher.fetch(URL)

    assert len(result.raw) == 1000
    assert result.raw_truncated is True


async def test_a_type_that_is_saved_still_gets_the_download_cap(monkeypatch):
    body = b"<p>" + b"z" * 5000
    fetcher = _fetcher(
        monkeypatch, _html(body, "text/html"), max_content_bytes=1000, max_download_bytes=10_000,
    )
    result = await fetcher.fetch(URL)

    assert len(result.raw) == len(body)
    assert result.raw_truncated is False


# -- I-2(b): an expired cache entry is swept when a different URL is cached,
# not just when that same URL is re-requested. --


async def test_an_expired_entry_is_swept_when_another_url_is_fetched(monkeypatch):
    def handler(req):
        return httpx.Response(200, content=b"body", headers={"content-type": "text/html"})

    fetcher = _fetcher(monkeypatch, handler, cache_ttl_seconds=1000)
    await fetcher.fetch("https://example.com/a")
    fetcher._cache["https://example.com/a"].timestamp -= 2000  # older than the TTL

    await fetcher.fetch("https://example.com/b")

    assert "https://example.com/a" not in fetcher._cache
    assert "https://example.com/b" in fetcher._cache
