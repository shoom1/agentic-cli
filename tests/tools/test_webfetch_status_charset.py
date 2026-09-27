"""``web_fetch`` reports HTTP errors as failures and survives any charset label.

An error page (404, 500, ...) was returned as fetched content: ``web_fetch``
summarized it as if it were the page, ``kb_ingest_url`` stored it in the
knowledge base, and it stayed cached for the TTL. A ``charset`` label Python
does not know raised ``LookupError`` out of the fetcher. Offline: DNS is stubbed
and the transport is an ``httpx.MockTransport`` behind the real PinnedTransport.
"""

from __future__ import annotations

import socket
from unittest.mock import AsyncMock, patch

import httpx
import pytest

URL = "https://example.com/page"


def _fetcher(monkeypatch, handler):
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
        validator=validator, robots_checker=RobotsTxtChecker(), transport=transport,
    )
    fetcher._robots.can_fetch = AsyncMock(return_value=True)
    return fetcher


class TestHttpErrorStatus:
    @pytest.mark.parametrize("status", [403, 404, 410, 429, 500, 503])
    async def test_error_status_is_a_failure(self, monkeypatch, status):
        page = httpx.Response(status, html="<h1>Something went wrong</h1>")
        result = await _fetcher(monkeypatch, lambda req: page).fetch(URL)

        assert result.success is False
        assert str(status) in result.error
        assert result.status_code == status
        assert result.content is None

    async def test_error_page_is_not_cached(self, monkeypatch):
        calls = []

        def handler(req):
            calls.append(req)
            if len(calls) == 1:
                return httpx.Response(503, text="try later")
            return httpx.Response(200, html="<p>the real page</p>")

        fetcher = _fetcher(monkeypatch, handler)
        assert (await fetcher.fetch(URL)).success is False

        second = await fetcher.fetch(URL)
        assert second.success is True
        assert second.from_cache is False
        assert "the real page" in second.content
        assert len(calls) == 2

    async def test_success_reports_its_status(self, monkeypatch):
        result = await _fetcher(
            monkeypatch, lambda req: httpx.Response(200, html="<p>ok</p>"),
        ).fetch(URL)
        assert result.success is True
        assert result.status_code == 200

    async def test_web_fetch_does_not_summarize_an_error_page(self, monkeypatch):
        from agentic_cli.tools.webfetch_tool import web_fetch
        from agentic_cli.workflow.service_registry import LLM_SUMMARIZER, set_service_registry

        summarizer = AsyncMock()
        fetcher = _fetcher(monkeypatch, lambda req: httpx.Response(404, html="<h1>Not Found</h1>"))
        token = set_service_registry({LLM_SUMMARIZER: summarizer})
        try:
            with patch("agentic_cli.tools.webfetch_tool.get_or_create_fetcher", return_value=fetcher):
                result = await web_fetch(URL, "summarize")
        finally:
            token.var.reset(token)

        assert result["success"] is False
        assert "404" in result["error"]
        summarizer.summarize.assert_not_awaited()


class TestCharset:
    # base64 names a codec that is not a text encoding; undefined always raises.
    @pytest.mark.parametrize(
        "label", ["x-no-such-charset", "utf-99", "../../etc", "base64", "undefined"],
    )
    async def test_unknown_charset_decodes_as_utf8(self, monkeypatch, label):
        body = "café — naïve".encode("utf-8")
        page = httpx.Response(
            200, content=body, headers={"content-type": f"text/html; charset={label}"},
        )
        result = await _fetcher(monkeypatch, lambda req: page).fetch(URL)

        assert result.success is True
        assert result.content == "café — naïve"

    async def test_known_charset_is_still_honoured(self, monkeypatch):
        page = httpx.Response(
            200, content="café".encode("latin-1"),
            headers={"content-type": "text/html; charset=iso-8859-1"},
        )
        result = await _fetcher(monkeypatch, lambda req: page).fetch(URL)
        assert result.content == "café"

    async def test_robots_rules_apply_despite_an_unknown_charset(self, monkeypatch):
        """robots.txt treated a decode failure as 'no robots.txt' and allowed
        everything; its rules must still apply."""
        from agentic_cli.tools.webfetch.robots import RobotsTxtChecker
        from agentic_cli.tools.webfetch.transport import PinnedTransport
        from agentic_cli.tools.webfetch.validator import URLValidator

        def _gai(host, port, *a, **k):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

        monkeypatch.setattr(socket, "getaddrinfo", _gai)
        robots = httpx.Response(
            200, content=b"User-agent: *\nDisallow: /private/\n",
            headers={"content-type": "text/plain; charset=x-no-such-charset"},
        )
        checker = RobotsTxtChecker(
            transport=PinnedTransport(URLValidator(), inner=httpx.MockTransport(lambda req: robots)),
        )
        assert await checker.can_fetch("https://example.com/private/x") is False
